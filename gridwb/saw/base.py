import datetime
import locale
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pythoncom
import win32com

from ._exceptions import COMError, CommandNotRespectedError, Error, PowerWorldError
from ._helpers import (
    convert_df_to_variant,
    convert_list_to_variant,
    convert_nested_list_to_variant,
    convert_to_windows_path,
)
# Set up locale
locale.setlocale(locale.LC_ALL, "")

logging.basicConfig(format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)

# Listing of PowerWorld data types. I guess 'real' means float?
DATA_TYPES = ["Integer", "Real", "String"]
# Hard-code based on indices.
NUMERIC_TYPES = DATA_TYPES[:2]


# noinspection PyPep8Naming
class SAWBase(object):
    """Base class for the SimAuto Wrapper, containing core COM functionality."""

    POWER_FLOW_FIELDS = {
        "bus": ["BusNum", "BusName", "BusPUVolt", "BusAngle", "BusNetMW", "BusNetMVR"],
        "gen": ["BusNum", "GenID", "GenMW", "GenMVR"],
        "load": ["BusNum", "LoadID", "LoadMW", "LoadMVR"],
        "shunt": ["BusNum", "ShuntID", "ShuntMW", "ShuntMVR"],
        "branch": [
            "BusNum",
            "BusNum:1",
            "LineCircuit",
            "LineMW",
            "LineMW:1",
            "LineMVR",
            "LineMVR:1",
        ],
    }

    FIELD_LIST_COLUMNS = [
        "key_field",
        "internal_field_name",
        "field_data_type",
        "description",
        "display_name",
    ]

    FIELD_LIST_COLUMNS_OLD = FIELD_LIST_COLUMNS[0:-1]

    FIELD_LIST_COLUMNS_NEW = [
        "key_field",
        "internal_field_name",
        "field_data_type",
        "description",
        "display_name",
        "enterable",
    ]

    SPECIFIC_FIELD_LIST_COLUMNS = [
        "variablename:location",
        "field",
        "column header",
        "field description",
    ]

    SPECIFIC_FIELD_LIST_COLUMNS_NEW = [
        "variablename:location",
        "field",
        "column header",
        "field description",
        "enterable",
    ]

    SIMAUTO_PROPERTIES = {
        "CreateIfNotFound": bool,
        "CurrentDir": str,
        "UIVisible": bool,
    }

    def __init__(
        self,
        FileName,
        early_bind=False,
        UIVisible=False,
        object_field_lookup=("bus", "gen", "load", "shunt", "branch"),
        CreateIfNotFound: bool = False,
        UseDefinedNamesInVariables: bool = False,
        pw_order=False,
    ):
        """
        Initialize SimAuto wrapper.

        Parameters
        ----------
        FileName : str
            Path to the PowerWorld case file (.pwb).
        early_bind : bool, optional
            Whether to use early binding for COM.
        UIVisible : bool, optional
            Whether to make the PowerWorld UI visible.
        object_field_lookup : tuple, optional
            Object types to pre-fetch field lists for.
        CreateIfNotFound : bool, optional
            Whether to create objects if they are not found.
        UseDefinedNamesInVariables : bool, optional
            Whether to use defined names in variables.
        pw_order : bool, optional
            Whether to preserve PowerWorld's internal ordering.
        """
        self.log = logging.getLogger(self.__class__.__name__)
        locale_db = locale.localeconv()
        self.decimal_delimiter = locale_db["decimal_point"]
        pythoncom.CoInitialize()

        try:
            if early_bind:
                try:
                    self._pwcom = win32com.client.gencache.EnsureDispatch("pwrworld.SimulatorAuto")
                except AttributeError:  # pragma: no cover
                    self._pwcom = win32com.client.dynamic.Dispatch("pwrworld.SimulatorAuto")
            else:
                self._pwcom = win32com.client.dynamic.Dispatch("pwrworld.SimulatorAuto")
        except Exception as e:
            m = (
                "Unable to launch SimAuto. Please confirm that your PowerWorld license includes "
                "the SimAuto add-on, and that SimAuto has been successfully installed."
            )
            self.log.exception(m)
            raise e

        self.pwb_file_path = None
        self.set_simauto_property("CreateIfNotFound", CreateIfNotFound)
        self.set_simauto_property("UIVisible", UIVisible)
        self.pw_order = pw_order

        # Initialize temporary file for UI updates
        self.ntf = tempfile.NamedTemporaryFile(mode="w", suffix=".axd", delete=False)
        self.empty_aux = Path(self.ntf.name).as_posix()
        self.ntf.close()

        self.OpenCase(FileName=FileName)

        version_string, self.build_date = self.get_version_and_builddate()
        self.version = int(re.search(r"\d+", version_string)[0])

        if UseDefinedNamesInVariables:
            self.exec_aux(
                'CaseInfo_Options_Value (Option,Value)\n{"UseDefinedNamesInVariables" "YES"}'
            )

        self.lodf = None
        self._object_fields = {}
        self._object_key_fields = {}

        for obj in object_field_lookup:
            o = obj.lower()
            self.GetFieldList(o)
            self.get_key_fields_for_object_type(ObjectType=o)

    def change_and_confirm_params_multiple_element(self, ObjectType: str, command_df: pd.DataFrame) -> None:
        cleaned_df = self._change_parameters_multiple_element_df(
            ObjectType=ObjectType, command_df=command_df
        )
        df = self.GetParametersMultipleElement(ObjectType=ObjectType, ParamList=cleaned_df.columns.tolist())
        eq = self._df_equiv_subset_of_other(df1=cleaned_df, df2=df, ObjectType=ObjectType)

        if not eq:
            m = (
                "After calling ChangeParametersMultipleElement, not all parameters were actually changed "
                "within PowerWorld. Try again with a different parameter (e.g. use GenVoltSet "
                "instead of GenRegPUVolt)."
            )
            raise CommandNotRespectedError(m)

    def change_parameters_multiple_element_df(self, ObjectType: str, command_df: pd.DataFrame) -> None:
        self._change_parameters_multiple_element_df(ObjectType=ObjectType, command_df=command_df)

    def clean_df_or_series(
        self, obj: Union[pd.DataFrame, pd.Series], ObjectType: str
    ) -> Union[pd.DataFrame, pd.Series]:
        if isinstance(obj, pd.DataFrame):
            df_flag = True
            fields = obj.columns.to_numpy()
        elif isinstance(obj, pd.Series):
            df_flag = False
            fields = obj.index.to_numpy()
        else:
            raise TypeError("The given object is not a DataFrame or Series!")

        if not self.pw_order:
            self._clean_df(ObjectType, fields, obj, df_flag)
        return obj

    def _clean_df(self, ObjectType, fields, obj, df_flag):
        numeric = self.identify_numeric_fields(ObjectType=ObjectType, fields=fields)
        numeric_fields = fields[numeric]
        obj[numeric_fields] = self._to_numeric(obj[numeric_fields])

        nn_cols = fields[~numeric]
        obj[nn_cols] = obj[nn_cols].astype(str)
        obj[nn_cols] = (
            obj[nn_cols].apply(lambda x: x.str.strip()) if df_flag else obj[nn_cols].str.strip()
        )

        if df_flag:
            try:
                obj.sort_values(by="BusNum", axis=0, inplace=True)
            except KeyError:
                pass
            else:
                obj.index = np.arange(start=0, stop=obj.shape[0])

    def exit(self):
        os.unlink(self.ntf.name)
        self.CloseCase()
        del self._pwcom
        self._pwcom = None
        pythoncom.CoUninitialize()
        return None

    def get_key_fields_for_object_type(self, ObjectType: str) -> pd.DataFrame:
        obj_type = ObjectType.lower()
        try:
            return self._object_key_fields[obj_type]
        except KeyError:
            pass

        field_list = self.GetFieldList(ObjectType=obj_type, copy=False)
        key_field_mask = field_list["key_field"].str.match(r"\*[0-9]+[A-Z]*\*").to_numpy()
        key_field_df = field_list.loc[key_field_mask].copy()
        key_field_df["key_field"] = key_field_df["key_field"].str.replace(r"\*", "", regex=True)
        key_field_df["key_field"] = key_field_df["key_field"].str.replace("[A-Z]*", "", regex=True)
        key_field_df["key_field_index"] = self._to_numeric(key_field_df["key_field"]) - 1
        key_field_df.drop("key_field", axis=1, inplace=True)
        key_field_df.set_index(keys="key_field_index", drop=True, verify_integrity=True, inplace=True)
        key_field_df.sort_index(axis=0, inplace=True)

        assert np.array_equal(
            key_field_df.index.to_numpy(),
            np.arange(0, key_field_df.index.to_numpy()[-1] + 1),
        )

        self._object_key_fields[obj_type] = key_field_df
        return key_field_df

    def get_key_field_list(self, ObjectType: str) -> List[str]:
        obj_type = ObjectType.lower()
        try:
            key_field_df = self._object_key_fields[obj_type]
        except KeyError:
            key_field_df = self.get_key_fields_for_object_type(obj_type)
        return key_field_df["internal_field_name"].tolist()

    def get_version_and_builddate(self) -> tuple:
        return self._call_simauto(
            "GetParametersSingleElement",
            "PowerWorldSession",
            convert_list_to_variant(["Version", "ExeBuildDate"]),
            convert_list_to_variant(["", ""]),
        )

    def identify_numeric_fields(self, ObjectType: str, fields: Union[List, np.ndarray]) -> np.ndarray:
        field_list = self.GetFieldList(ObjectType=ObjectType, copy=False)
        idx = field_list["internal_field_name"].to_numpy().searchsorted(fields)

        try:
            ifn = field_list["internal_field_name"].to_numpy()[idx]
            if set(ifn) != set(fields):
                raise ValueError("The given object has fields which do not match a PowerWorld internal field name!")
        except IndexError as e:
            raise ValueError("The given object has fields which do not match a PowerWorld internal field name!") from e

        data_types = field_list["field_data_type"].to_numpy()[idx]
        return np.isin(data_types, NUMERIC_TYPES)

    def set_simauto_property(self, property_name: str, property_value: Union[str, bool]):
        """Sets a property on the underlying SimAuto COM object.

        :param property_name: The name of the property (e.g., 'UIVisible', 'CurrentDir').
        :param property_value: The value to assign to the property.
        :raises ValueError: If the property name is unsupported or the value type is incorrect.
        :raises AttributeError: If the property does not exist on the current SimAuto version.
        """
        if property_name not in self.SIMAUTO_PROPERTIES:
            raise ValueError(
                f"The given property_name, {property_name}, is not currently supported. "
                f"Valid properties are: {list(self.SIMAUTO_PROPERTIES.keys())}"
            )

        if not isinstance(property_value, self.SIMAUTO_PROPERTIES[property_name]):
            m = (
                f"The given property_value, {property_value}, is invalid. "
                f"It must be of type {self.SIMAUTO_PROPERTIES[property_name]}."
            )
            raise ValueError(m)

        if property_name == "CurrentDir" and not os.path.isdir(property_value):
            raise ValueError(f"The given path for CurrentDir, {property_value}, is not a valid path!")

        try:
            self._set_simauto_property(property_name=property_name, property_value=property_value)
        except AttributeError as e:
            if property_name == "UIVisible":
                self.log.warning(
                    "UIVisible attribute could not be set. Note this SimAuto property was not introduced "
                    "until Simulator version 20. Check your version with the get_simulator_version method."
                )
            else:
                raise e from None

    def _set_simauto_property(self, property_name, property_value):
        setattr(self._pwcom, property_name, property_value)

    def ChangeParameters(self, ObjectType: str, ParamList: list, Values: list) -> None:
        return self.ChangeParametersSingleElement(ObjectType, ParamList, Values)

    def ChangeParametersSingleElement(self, ObjectType: str, ParamList: list, Values: list) -> None:
        """Modifies parameters for a single object in PowerWorld.

        :param ObjectType: The PowerWorld object type (e.g., 'Bus', 'Gen').
        :param ParamList: A list of internal field names to modify.
        :param Values: A list of values corresponding to the parameters.
        """
        return self._call_simauto(
            "ChangeParametersSingleElement",
            ObjectType,
            convert_list_to_variant(ParamList),
            convert_list_to_variant(Values),
        )

    def ChangeParametersMultipleElement(self, ObjectType: str, ParamList: list, ValueList: list) -> None:
        """Modifies parameters for multiple objects using a nested list of values.

        :param ObjectType: The PowerWorld object type.
        :param ParamList: A list of internal field names to modify.
        :param ValueList: A list of lists, where each inner list contains values for one object.
        """
        return self._call_simauto(
            "ChangeParametersMultipleElement",
            ObjectType,
            convert_list_to_variant(ParamList),
            convert_nested_list_to_variant(ValueList),
        )

    def ChangeParametersMultipleElementRect(self, ObjectType: str, ParamList: list, df: pd.DataFrame) -> None:
        """Modifies parameters for multiple objects using a pandas DataFrame.

        :param ObjectType: The PowerWorld object type.
        :param ParamList: A list of internal field names to modify.
        :param df: A DataFrame containing the new values. Columns must match ParamList.
        """
        return self._call_simauto(
            "ChangeParametersMultipleElementRect",
            ObjectType,
            convert_list_to_variant(ParamList),
            convert_df_to_variant(df),
        )

    def ChangeParametersMultipleElementFlatInput(
        self, ObjectType: str, ParamList: list, NoOfObjects: int, ValueList: list
    ) -> None:
        if isinstance(ValueList[0], list):
            raise Error("The value list has to be a 1-D array")
        return self._call_simauto(
            "ChangeParametersMultipleElementFlatInput",
            ObjectType,
            convert_list_to_variant(ParamList),
            NoOfObjects,
            convert_list_to_variant(ValueList),
        )

    def CloseCase(self):
        """Closes the currently open PowerWorld case without exiting the application."""
        return self._call_simauto("CloseCase")

    def GetCaseHeader(self, filename: str = None) -> Tuple[str]:
        """Retrieves the header information from a PowerWorld case file.

        :param filename: Path to the .pwb file. Defaults to the currently open case.
        :return: A tuple containing header strings.
        """
        if filename is None:
            filename = self.pwb_file_path
        return self._call_simauto("GetCaseHeader", filename)

    def GetFieldList(self, ObjectType: str, copy=False) -> pd.DataFrame:
        object_type = ObjectType.lower()
        try:
            output = self._object_fields[object_type]
        except KeyError:
            result = self._call_simauto("GetFieldList", ObjectType)
            result_arr = np.array(result)

            try:
                output = pd.DataFrame(result_arr, columns=self.FIELD_LIST_COLUMNS)
            except ValueError as e:
                exp_base = r"\([0-9]+,\s"
                exp_end = r"{}\)"
                nf_old = len(self.FIELD_LIST_COLUMNS_OLD)
                nf_default = len(self.FIELD_LIST_COLUMNS)
                nf_new = len(self.FIELD_LIST_COLUMNS_NEW)
                r1 = re.search(exp_base + exp_end.format(nf_old), e.args[0])
                r2 = re.search(exp_base + exp_end.format(nf_default), e.args[0])
                r3 = re.search(exp_base + exp_end.format(nf_new), e.args[0])

                if (r1 is None) or (r2 is None):
                    if r3 is None:
                        raise e
                    else:
                        output = pd.DataFrame(result_arr, columns=self.FIELD_LIST_COLUMNS_NEW)
                else:
                    output = pd.DataFrame(result_arr, columns=self.FIELD_LIST_COLUMNS_OLD)

            output.sort_values(by=["internal_field_name"], inplace=True)
            self._object_fields[object_type] = output

        return output.copy(deep=True) if copy else output

    def GetParametersSingleElement(self, ObjectType: str, ParamList: list, Values: list) -> pd.Series:
        """Retrieves parameters for a single object identified by its primary keys.

        :param ObjectType: The PowerWorld object type.
        :param ParamList: A list of internal field names to retrieve.
        :param Values: A list containing the primary key values for the object.
        :return: A pandas Series containing the requested data.
        """
        assert len(ParamList) == len(Values), "The given ParamList and Values must have the same length."

        output = self._call_simauto(
            "GetParametersSingleElement",
            ObjectType,
            convert_list_to_variant(ParamList),
            convert_list_to_variant(Values),
        )

        s = pd.Series(output, index=ParamList)
        return self.clean_df_or_series(obj=s, ObjectType=ObjectType)

    def GetParametersMultipleElement(
        self, ObjectType: str, ParamList: list, FilterName: str = ""
    ) -> Union[pd.DataFrame, None]:
        """Retrieves parameters for all objects of a type, optionally filtered.

        :param ObjectType: The PowerWorld object type.
        :param ParamList: A list of internal field names to retrieve.
        :param FilterName: Optional name of a PowerWorld filter to apply.
        :return: A pandas DataFrame containing the requested data.
        """
        output = self._call_simauto(
            "GetParametersMultipleElement",
            ObjectType,
            convert_list_to_variant(ParamList),
            FilterName,
        )
        if output is None:
            return output

        df = pd.DataFrame(np.array(output).transpose(), columns=ParamList)
        return self.clean_df_or_series(obj=df, ObjectType=ObjectType)

    def GetParamsRectTyped(
        self, ObjectType: str, ParamList: list, FilterName: str = ""
    ) -> Union[pd.DataFrame, None]:
        output = self._call_simauto(
            "GetParamsRectTyped",
            ObjectType,
            convert_list_to_variant(ParamList),
            FilterName,
            pythoncom.VT_VARIANT,
        )
        if output is None:
            return output

        return pd.DataFrame(output, columns=ParamList)

    def GetParametersMultipleElementFlatOutput(
        self, ObjectType: str, ParamList: list, FilterName: str = ""
    ) -> Union[None, Tuple[str]]:
        result = self._call_simauto(
            "GetParametersMultipleElementFlatOutput",
            ObjectType,
            convert_list_to_variant(ParamList),
            FilterName,
        )

        if len(result) == 0:
            return None
        else:
            return result

    def GetParameters(self, ObjectType: str, ParamList: list, Values: list) -> pd.Series:
        return self.GetParametersSingleElement(ObjectType, ParamList, Values)

    def GetSpecificFieldList(self, ObjectType: str, FieldList: List[str]) -> pd.DataFrame:
        try:
            df = (
                pd.DataFrame(
                    self._call_simauto("GetSpecificFieldList", ObjectType, convert_list_to_variant(FieldList)),
                    columns=self.SPECIFIC_FIELD_LIST_COLUMNS,
                )
                .sort_values(by=self.SPECIFIC_FIELD_LIST_COLUMNS[0])
                .reset_index(drop=True)
            )
        except ValueError:
            df = (
                pd.DataFrame(
                    self._call_simauto("GetSpecificFieldList", ObjectType, convert_list_to_variant(FieldList)),
                    columns=self.SPECIFIC_FIELD_LIST_COLUMNS_NEW,
                )
                .sort_values(by=self.SPECIFIC_FIELD_LIST_COLUMNS_NEW[0])
                .reset_index(drop=True)
            )
        return df

    def GetSpecificFieldMaxNum(self, ObjectType: str, Field: str) -> int:
        return self._call_simauto("GetSpecificFieldMaxNum", ObjectType, Field)

    def ListOfDevices(self, ObjType: str, FilterName="") -> Union[None, pd.DataFrame]:
        """Retrieves a list of all objects of a specific type and their primary keys.

        :param ObjType: The PowerWorld object type.
        :param FilterName: Optional name of a PowerWorld filter to apply.
        :return: A pandas DataFrame containing the primary key fields for the objects.
        """
        kf = self.get_key_fields_for_object_type(ObjType)
        output = self._call_simauto("ListOfDevices", ObjType, FilterName)

        all_none = all(i is None for i in output)

        if all_none:
            return None

        df = pd.DataFrame(output).transpose()
        df.columns = kf["internal_field_name"].to_numpy()
        df = self.clean_df_or_series(obj=df, ObjectType=ObjType)
        return df

    def ListOfDevicesAsVariantStrings(self, ObjType: str, FilterName="") -> tuple:
        return self._call_simauto("ListOfDevicesAsVariantStrings", ObjType, FilterName)

    def ListOfDevicesFlatOutput(self, ObjType: str, FilterName="") -> tuple:
        return self._call_simauto("ListOfDevicesFlatOutput", ObjType, FilterName)

    def LoadState(self) -> None:
        return self._call_simauto("LoadState")

    def OpenCase(self, FileName: Union[str, None] = None) -> None:
        """Opens a PowerWorld case file.

        :param FileName: Path to the .pwb or .pwx file.
        """
        if FileName is None:
            if self.pwb_file_path is None:
                raise TypeError("When OpenCase is called for the first time, a FileName is required.")
        else:
            self.pwb_file_path = FileName
        return self._call_simauto("OpenCase", self.pwb_file_path)

    def OpenCaseType(self, FileName: str, FileType: str, Options: Union[list, str, None] = None) -> None:
        self.pwb_file_path = FileName
        if isinstance(Options, list):
            options = convert_list_to_variant(Options)
        elif isinstance(Options, str):
            options = Options
        else:
            options = ""
        return self._call_simauto("OpenCaseType", self.pwb_file_path, FileType, options)

    def ProcessAuxFile(self, FileName):
        """Executes a PowerWorld auxiliary (.aux) file.

        :param FileName: Path to the auxiliary file.
        """
        return self._call_simauto("ProcessAuxFile", FileName)

    def RunScriptCommand(self, Statements):
        """Executes one or more PowerWorld script statements.

        :param Statements: A string containing the script commands.
        """
        return self._call_simauto("RunScriptCommand", Statements)

    def RunScriptCommand2(self, Statements: str, StatusMessage: str):
        """Executes script statements and provides a status message for the PowerWorld UI.

        :param Statements: A string containing the script commands.
        :param StatusMessage: A message to display in the PowerWorld status bar.
        """
        return self._pwcom.RunScriptCommand2(Statements, StatusMessage)

    def SaveCase(self, FileName=None, FileType="PWB", Overwrite=True):
        if FileName is not None:
            f = convert_to_windows_path(FileName)
        elif self.pwb_file_path is None:
            raise TypeError("SaveCase was called without a FileName, but OpenCase has not yet been called.")
        else:
            f = convert_to_windows_path(self.pwb_file_path)

        return self._call_simauto("SaveCase", f, FileType, Overwrite)

    def SaveState(self) -> None:
        return self._call_simauto("SaveState")

    def SendToExcel(self, ObjectType: str, FilterName: str, FieldList) -> None:
        return self._call_simauto("SendToExcel", ObjectType, FilterName, FieldList)


    @property
    def CreateIfNotFound(self):
        return self._pwcom.CreateIfNotFound

    @property
    def CurrentDir(self) -> str:
        return self._pwcom.CurrentDir

    @property
    def ProcessID(self) -> int:
        return self._pwcom.ProcessID

    @property
    def RequestBuildDate(self) -> int:
        return self._pwcom.RequestBuildDate

    @property
    def UIVisible(self) -> bool:
        try:
            return self._pwcom.UIVisible
        except AttributeError:
            self.log.warning(
                "UIVisible attribute could not be accessed. Note this SimAuto property was not introduced "
                "until Simulator version 20. Check your version with the get_simulator_version method."
            )
            return False

    @property
    def ProgramInformation(self) -> Union[tuple, bool]:
        try:
            result = self._pwcom.ProgramInformation
            result = [list(x) for x in result]
            result[0][2] = datetime.datetime.fromtimestamp(result[0][2].timestamp(), tz=result[0][2].tzinfo)
            result = tuple(tuple(x) for x in result)
            return result
        except AttributeError:  # pragma: no cover
            self.log.warning(
                "ProgramInformation attribute could not be accessed. Note this SimAuto property was not "
                "introduced until Simulator version 21. Check your version with the get_simulator_version method."
            )
            return False

    def _call_simauto(self, func: str, *args):
        try:
            f = getattr(self._pwcom, func)
        except AttributeError:
            raise AttributeError(f"The given function, {func}, is not a valid SimAuto function.") from None

        try:
            output = f(*args)
        except Exception as e:
            m = f"An error occurred when trying to call {func} with {args}"
            self.log.exception(m)
            raise COMError(m) from e

        if output == ("",):
            return None

        try:
            if output is None or output[0] == "":
                pass
            elif "No data" not in output[0]:
                raise PowerWorldError(output[0])
        except TypeError as e:
            if "is not subscriptable" in e.args[0]:
                if output == -1:
                    m = (
                        f"PowerWorld simply returned -1 after calling '{func}' with '{args}'. "
                        "Unfortunately, that's all we can help you with. Perhaps the arguments are "
                        "invalid or in the wrong order - double-check the documentation."
                    )
                    raise PowerWorldError(m) from e
                elif isinstance(output, int):
                    return output
            raise e

        return output[1] if len(output) == 2 else output[1:]

    def _change_parameters_multiple_element_df(self, ObjectType: str, command_df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = self.clean_df_or_series(obj=command_df, ObjectType=ObjectType)
        self.ChangeParametersMultipleElement(
            ObjectType=ObjectType,
            ParamList=cleaned_df.columns.tolist(),
            ValueList=cleaned_df.to_numpy().tolist(),
        )
        return cleaned_df

    def _df_equiv_subset_of_other(self, df1: pd.DataFrame, df2: pd.DataFrame, ObjectType: str) -> bool:
        kf = self.get_key_fields_for_object_type(ObjectType=ObjectType)
        merged = pd.merge(
            left=df1,
            right=df2,
            how="inner",
            on=kf["internal_field_name"].tolist(),
            suffixes=("_in", "_out"),
            copy=False,
        )

        cols_in = merged.columns[merged.columns.str.endswith("_in")]
        cols_out = merged.columns[merged.columns.str.endswith("_out")]

        cols = cols_in.str.replace("_in", "")
        numeric_cols = self.identify_numeric_fields(ObjectType=ObjectType, fields=cols)
        str_cols = ~numeric_cols

        return np.allclose(
            merged[cols_in[numeric_cols]].to_numpy(),
            merged[cols_out[numeric_cols]].to_numpy(),
        ) and np.array_equal(merged[cols_in[str_cols]].to_numpy(), merged[cols_out[str_cols]].to_numpy())

    def _to_numeric(
        self, data: Union[pd.DataFrame, pd.Series], errors="ignore"
    ) -> Union[pd.DataFrame, pd.Series]:
        if isinstance(data, pd.DataFrame):
            df_flag = True
        elif isinstance(data, pd.Series):
            df_flag = False
        else:
            raise TypeError("data must be either a DataFrame or Series.")

        if self.decimal_delimiter != ".":
            if df_flag:
                data = data.apply(self._replace_decimal_delimiter)
            else:
                data = self._replace_decimal_delimiter(data)

        if df_flag:
            return data.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(data)
        else:
            return pd.to_numeric(data, errors='coerce').fillna(data)

    def _replace_decimal_delimiter(self, data: pd.Series):
        try:
            return data.str.replace(self.decimal_delimiter, ".")
        except AttributeError:
            return data

    def exec_aux(self, aux: str, use_double_quotes: bool = False):
        if use_double_quotes:
            aux = aux.replace("'", '"')
        file = tempfile.NamedTemporaryFile(mode="wt", suffix=".aux", delete=False)
        file.write(aux)
        file.close()
        self.ProcessAuxFile(file.name)
        os.unlink(file.name)

    def update_ui(self) -> None:
        return self.ProcessAuxFile(self.empty_aux)