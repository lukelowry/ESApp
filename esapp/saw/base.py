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

from ._exceptions import (
    COMError,
    CommandNotRespectedError,
    Error,
    PowerWorldError,
    RPC_S_UNKNOWN_IF,
    RPC_S_CALL_FAILED,
)
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
        CreateIfNotFound: bool = False,
        UseDefinedNamesInVariables: bool = False,
        pw_order=False,
    ) -> None:
        """Initializes the SimAuto Wrapper (SAW) and establishes a COM connection to PowerWorld Simulator.

        Parameters
        ----------
        FileName : str
            Absolute or relative path to the PowerWorld case file (.pwb or .pwx).
        early_bind : bool, optional
            If True, uses `gencache` for faster COM calls (requires admin/write access to site-packages).
            Defaults to False.
        UIVisible : bool, optional
            If True, makes the PowerWorld Simulator application window visible. Defaults to False.
        CreateIfNotFound : bool, optional
            Sets the SimAuto property to create new objects during `ChangeParameters` calls. Defaults to False.
        UseDefinedNamesInVariables : bool, optional
            If True, configures the case to use defined names instead of internal IDs. Defaults to False.
        pw_order : bool, optional
            If True, disables automatic sorting of DataFrames to match PowerWorld's internal memory order.
            Defaults to False.

        Raises
        ------
        Exception
            If the SimAuto COM server cannot be initialized (e.g., PowerWorld not installed or license issue).
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

    def change_and_confirm_params_multiple_element(self, ObjectType: str, command_df: pd.DataFrame) -> None:
        """Modifies parameters for multiple elements and verifies the change was successfully applied in PowerWorld.

        This method first attempts to change parameters using `ChangeParametersMultipleElement`,
        then immediately retrieves the same parameters from PowerWorld to confirm the changes.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type (e.g., 'Bus', 'Gen').
        command_df : pandas.DataFrame
            A DataFrame where columns are field names and rows are object data.
            It must include the primary key fields for the specified `ObjectType`.

        Raises
        ------
        CommandNotRespectedError
            If the values in PowerWorld after the call do not match the `command_df`,
            indicating that the change was not fully accepted by PowerWorld.
        PowerWorldError
            If the underlying SimAuto call fails.
        """
        cleaned_df = self._change_parameters_multiple_element_df(
            ObjectType=ObjectType, command_df=command_df
        )
        df = self.GetParametersMultipleElement(ObjectType=ObjectType, ParamList=cleaned_df.columns.tolist())
        
        # Verify changes by merging on key fields and comparing values
        kf = self.get_key_fields_for_object_type(ObjectType=ObjectType)
        merged = pd.merge(
            left=cleaned_df,
            right=df,
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
        
        eq = np.allclose(
            merged[cols_in[numeric_cols]].to_numpy(),
            merged[cols_out[numeric_cols]].to_numpy(),
        ) and np.array_equal(merged[cols_in[str_cols]].to_numpy(), merged[cols_out[str_cols]].to_numpy())
        
        if not eq:
            m = (
                "After calling ChangeParametersMultipleElement, not all parameters were actually changed "
                "within PowerWorld. Try again with a different parameter (e.g. use GenVoltSet "
                "instead of GenRegPUVolt)."
            )
            raise CommandNotRespectedError(m)

    def clean_df_or_series(
        self, obj: Union[pd.DataFrame, pd.Series], ObjectType: str
    ) -> Union[pd.DataFrame, pd.Series]:
        """Standardizes data types and formatting for data retrieved from PowerWorld.

        This internal helper converts numeric fields to appropriate Python numeric types
        and strips whitespace from string fields, based on PowerWorld's field metadata.

        Parameters
        ----------
        obj : Union[pandas.DataFrame, pandas.Series]
            The DataFrame or Series object to clean.
        ObjectType : str
            The PowerWorld object type associated with the data (e.g., 'Bus', 'Gen').
            Used to look up field metadata.

        Returns
        -------
        Union[pandas.DataFrame, pandas.Series]
            The cleaned object with proper numeric types and stripped strings.

        Raises
        ------
        TypeError
            If `obj` is not a pandas DataFrame or Series.
        """
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
        """Closes the PowerWorld case, deletes temporary files, and releases the COM object.

        This method should be called when the SimAuto session is no longer needed
        to ensure proper cleanup and resource release.
        """
        os.unlink(self.ntf.name)
        self.CloseCase()
        del self._pwcom
        self._pwcom = None
        pythoncom.CoUninitialize()
        return None

    def get_key_fields_for_object_type(self, ObjectType: str) -> pd.DataFrame:
        """Retrieves metadata about the primary key fields for a specific PowerWorld object type.

        This method identifies which fields uniquely identify an object of the given type
        and caches this information for future use.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type (e.g., 'Bus', 'Gen').

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing key field names, their data types, descriptions,
            and their 0-based index within the object's key field list.
        """
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
        
        # Convert to numeric, handling locale if needed
        key_field_series = key_field_df["key_field"]
        if self.decimal_delimiter != ".":
            try:
                key_field_series = key_field_series.str.replace(self.decimal_delimiter, ".")
            except AttributeError:
                pass
        key_field_df["key_field_index"] = pd.to_numeric(key_field_series, errors='coerce').fillna(key_field_df["key_field"]) - 1
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
        """Returns a list of internal field names that serve as primary keys for a given object type.

        This is a convenience method that extracts just the field names from the
        key field metadata.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type (e.g., 'Bus', 'Gen').

        Returns
        -------
        List[str]
            A list of strings representing the internal field names that are primary keys.
        """
        obj_type = ObjectType.lower()
        try:
            key_field_df = self._object_key_fields[obj_type]
        except KeyError:
            key_field_df = self.get_key_fields_for_object_type(obj_type)
        return key_field_df["internal_field_name"].tolist()

    def get_version_and_builddate(self) -> tuple:
        """Retrieves the PowerWorld Simulator version string and executable build date.

        This method queries the 'PowerWorldSession' object for its version and build date.

        Returns
        -------
        tuple
            A tuple containing:
            - str: The version string of PowerWorld Simulator (e.g., "22.0.0.0").
            - datetime.datetime: The build date of the PowerWorld Simulator executable.

        """
        return self._call_simauto(
            "GetParametersSingleElement",
            "PowerWorldSession",
            convert_list_to_variant(["Version", "ExeBuildDate"]),
            convert_list_to_variant(["", ""]),
        )

    def identify_numeric_fields(self, ObjectType: str, fields: Union[List, np.ndarray]) -> np.ndarray:
        """Determines which of the given fields are numeric based on PowerWorld's metadata.

        This method queries the field list for the specified object type to identify
        fields classified as 'Integer' or 'Real'.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type (e.g., 'Bus', 'Gen').
        fields : Union[List[str], numpy.ndarray]
            A list or NumPy array of internal field names to check.

        Returns
        -------
        numpy.ndarray
            A boolean NumPy array where True indicates a numeric field (Integer or Real)
            at the corresponding position in the input `fields` list.

        Raises
        ------
        ValueError
            If any field name in `fields` is not recognized for the specified `ObjectType`.
        """
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

        This method provides a controlled way to set various SimAuto properties,
        including validation of property names and value types.

        Parameters
        ----------
        property_name : str
            The name of the property to set (e.g., 'UIVisible', 'CurrentDir', 'CreateIfNotFound').
        property_value : Union[str, bool]
            The value to assign to the property. The type must match the expected type
            for the specific property.

        Raises
        ------
        ValueError
            If the `property_name` is unsupported, the `property_value` has an incorrect type,
            or if `CurrentDir` is set to an invalid path.
        AttributeError
            If the property does not exist on the current SimAuto version (e.g., `UIVisible`
            on older versions of Simulator).
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
        """Internal helper to directly set a SimAuto COM property."""
        setattr(self._pwcom, property_name, property_value)

    def ChangeParameters(self, ObjectType: str, ParamList: list, Values: list) -> None:
        """Alias for ChangeParametersSingleElement.

        This method is a direct pass-through to `ChangeParametersSingleElement`.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type (e.g., 'Bus', 'Gen').
        ParamList : List[str]
            A list of internal field names to modify.
        Values : List[Any]
            A list of values corresponding to the parameters in `ParamList`.
        """
        return self.ChangeParametersSingleElement(ObjectType, ParamList, Values)

    def ChangeParametersSingleElement(self, ObjectType: str, ParamList: list, Values: list) -> None:
        """Modifies parameters for a single object in PowerWorld.

        This method is used to update specific fields for a single PowerWorld object,
        identified by its primary key values (which must be included in `Values`).

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type (e.g., 'Bus', 'Gen').
        ParamList : List[str]
            A list of internal field names to modify. This list must include the
            primary key fields for the `ObjectType` to identify the target object.
        Values : List[Any]
            A list of values corresponding to the parameters in `ParamList`. The order
            and length must match `ParamList`.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., invalid object type, field name, or value).
        """
        return self._call_simauto(
            "ChangeParametersSingleElement",
            ObjectType,
            convert_list_to_variant(ParamList),
            convert_list_to_variant(Values),
        )

    def ChangeParametersMultipleElement(self, ObjectType: str, ParamList: list, ValueList: list) -> None:
        """Modifies parameters for multiple objects using a nested list of values.

        This method is suitable for updating a moderate number of objects where
        the data is structured as a list of lists. For very large datasets,
        `ChangeParametersMultipleElementRect` is generally more efficient.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type.
        ParamList : List[str]
            A list of internal field names to modify. This list must include the
            primary key fields for the `ObjectType` to identify the target objects.
        ValueList : List[List[Any]]
            A list of lists, where each inner list contains values for one object.
            The order of values in each inner list must match `ParamList`.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._call_simauto(
            "ChangeParametersMultipleElement",
            ObjectType,
            convert_list_to_variant(ParamList),
            convert_nested_list_to_variant(ValueList),
        )

    def ChangeParametersMultipleElementRect(self, ObjectType: str, ParamList: list, df: pd.DataFrame) -> None:
        """
        Modifies parameters for multiple objects using a pandas DataFrame (rectangular data structure).

        This is generally the most efficient way to update a large number of objects at once.
        The DataFrame must include the primary key fields for the object type to identify
        which objects to update.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type.
        ParamList : List[str]
            A list of internal field names being updated. These must correspond to the
            column names in the `df`.
        df : pandas.DataFrame
            A DataFrame containing the data to update. The column names of `df` must
            match the `ParamList`, and it must contain primary key columns.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
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
        """Modifies parameters for multiple objects using a flat, 1-D list of values.

        This method is an alternative to `ChangeParametersMultipleElement` for cases
        where the data is already flattened.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type.
        ParamList : List[str]
            A list of internal field names to modify.
        NoOfObjects : int
            The number of objects being updated.
        ValueList : List[Any]
            A flat list of values. Its length must be `NoOfObjects * len(ParamList)`.
            The values are ordered by object, then by parameter within each object.

        Returns
        -------
        None

        Raises
        ------
        Error
            If `ValueList` is not a 1-D array (i.e., it's a list of lists).
        PowerWorldError
            If the SimAuto call fails.
        """
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
        """Closes the currently open PowerWorld case without exiting the Simulator application.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._call_simauto("CloseCase")

    def GetCaseHeader(self, filename: str = None) -> Tuple[str]:
        """Retrieves the header information from a PowerWorld case file.

        Parameters
        ----------
        filename : str, optional
            Path to the .pwb or .pwx file. If None, the header of the currently
            open case is retrieved.

        Returns
        -------
        tuple
            A tuple of strings, where each string is a line from the case header.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., file not found).
        """
        if filename is None:
            filename = self.pwb_file_path
        return self._call_simauto("GetCaseHeader", filename)

    def GetFieldList(self, ObjectType: str, copy=False) -> pd.DataFrame:
        """Retrieves the complete list of available fields for a given PowerWorld object type.

        This method queries PowerWorld for all fields associated with an object type
        and caches the result for performance.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type (e.g., 'Bus', 'Gen').
        copy : bool, optional
            If True, returns a deep copy of the cached field list DataFrame.
            Defaults to False.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing columns like 'key_field', 'internal_field_name',
            'field_data_type', 'description', 'display_name', and 'enterable'.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., invalid object type).
        """
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

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type (e.g., 'Bus', 'Gen').
        ParamList : List[str]
            A list of internal field names to retrieve. This list must include the
            primary key fields for the `ObjectType` to identify the target object.
        Values : List[Any]
            A list containing the primary key values for the object, followed by
            empty strings or placeholders for other parameters in `ParamList` if they
            are not part of the key. The length must match `ParamList`.

        Returns
        -------
        pandas.Series
            A pandas Series containing the requested data, indexed by `ParamList`.

        Raises
        ------
        AssertionError
            If the length of `ParamList` and `Values` do not match.
        PowerWorldError
            If the SimAuto call fails.
        """
        assert len(ParamList) == len(Values), "The given ParamList and Values must have the same length."

        output = self._call_simauto(
            "GetParametersSingleElement",
            ObjectType,
            convert_list_to_variant(ParamList),
            convert_list_to_variant(Values),
        )

        s = pd.Series(output, index=ParamList)
        
        # Clean data if not using PowerWorld ordering
        if not self.pw_order:
            numeric = self.identify_numeric_fields(ObjectType=ObjectType, fields=s.index.to_numpy())
            numeric_fields = s.index.to_numpy()[numeric]
            for field in numeric_fields:
                s[field] = pd.to_numeric(s[field], errors='coerce')
            nn_fields = s.index.to_numpy()[~numeric]
            s[nn_fields] = s[nn_fields].astype(str).str.strip()
        
        return s

    def GetParametersMultipleElement(
        self, ObjectType: str, ParamList: list, FilterName: str = ""
    ) -> Union[pd.DataFrame, None]:
        """Retrieves parameters for multiple objects of a specific type, optionally filtered.

        This method is commonly used to fetch data for all objects of a given type
        or a subset defined by a PowerWorld filter.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type (e.g., 'Bus', 'Gen').
        ParamList : List[str]
            A list of internal field names to retrieve.
        FilterName : str, optional
            Optional name of a PowerWorld filter to restrict the result set.
            Defaults to an empty string, meaning no filter is applied.

        Returns
        -------
        Union[pandas.DataFrame, None]
            A pandas DataFrame where columns correspond to `ParamList`.
            Returns None if no objects are found matching the criteria.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., invalid object type or field names).
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
        
        # Clean data if not using PowerWorld ordering
        if not self.pw_order:
            numeric = self.identify_numeric_fields(ObjectType=ObjectType, fields=df.columns.to_numpy())
            numeric_fields = df.columns.to_numpy()[numeric]
            # Convert numeric fields - use coerce to turn errors into NaN, then fillna for originals
            for field in numeric_fields:
                df[field] = pd.to_numeric(df[field], errors='coerce')
            nn_fields = df.columns.to_numpy()[~numeric]
            df[nn_fields] = df[nn_fields].astype(str).apply(lambda x: x.str.strip())
            try:
                df.sort_values(by="BusNum", axis=0, inplace=True)
                df.reset_index(drop=True, inplace=True)
            except (KeyError, TypeError):
                # KeyError if BusNum doesn't exist, TypeError if can't sort mixed types
                pass
        
        return df

    def GetParamsRectTyped(
        self, ObjectType: str, ParamList: list, FilterName: str = ""
    ) -> Union[pd.DataFrame, None]:
        """Retrieves data in a rectangular format with PowerWorld's native variant typing preserved.

        This method is similar to `GetParametersMultipleElement` but attempts to preserve
        the original data types as returned by SimAuto, which can sometimes be more efficient
        or necessary for specific use cases.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type.
        ParamList : List[str]
            A list of internal field names to retrieve.
        FilterName : str, optional
            Optional name of a PowerWorld filter to apply. Defaults to an empty string.

        Returns
        -------
        Union[pandas.DataFrame, None]
            A pandas DataFrame containing the requested data. Returns None if no objects found.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
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
        """Retrieves data for multiple elements in a flat, 1-D output format.

        The data is returned as a single tuple of strings, where values for each
        object are concatenated. This format can be less convenient for direct
        DataFrame conversion but might be useful for specific parsing needs.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type.
        ParamList : List[str]
            A list of internal field names to retrieve.
        FilterName : str, optional
            Optional name of a PowerWorld filter to apply. Defaults to an empty string.

        Returns
        -------
        Union[None, Tuple[str]]
            A tuple of strings containing the data. Returns None if no data is found.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
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
        """Alias for GetParametersSingleElement.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type.
        ParamList : List[str]
            A list of internal field names to retrieve.
        Values : List[Any]
            A list containing the primary key values for the object, followed by
            empty strings or placeholders for other parameters in `ParamList`.

        Returns
        -------
        pandas.Series
            A pandas Series containing the requested data.
        """
        return self.GetParametersSingleElement(ObjectType, ParamList, Values)

    def GetSpecificFieldList(self, ObjectType: str, FieldList: List[str]) -> pd.DataFrame:
        """Retrieves detailed metadata for a specific subset of fields for a given object type.

        This method provides more detailed information about specific fields,
        including their display names and whether they are enterable.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type.
        FieldList : List[str]
            A list of internal field names to query metadata for.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns like 'variablename:location', 'field',
            'column header', 'field description', and 'enterable'.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
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
        """Retrieves the maximum index for a field that supports multiple entries (e.g., CustomFloat).

        Some PowerWorld fields, like 'CustomFloat', can have multiple instances
        (e.g., 'CustomFloat:1', 'CustomFloat:2'). This method returns the highest
        available index for such a field.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type.
        Field : str
            The base field name (e.g., 'CustomFloat').

        Returns
        -------
        int
            The maximum integer index available for the specified field.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._call_simauto("GetSpecificFieldMaxNum", ObjectType, Field)

    def ListOfDevices(self, ObjType: str, FilterName="") -> Union[None, pd.DataFrame]:
        """Retrieves a list of all objects of a specific type and their primary keys.

        This method is useful for getting an inventory of all objects of a certain type
        in the case, or a filtered subset.

        Parameters
        ----------
        ObjType : str
            The PowerWorld object type (e.g., 'Bus', 'Gen').
        FilterName : str, optional
            Optional name of a PowerWorld filter to apply. Defaults to an empty string.

        Returns
        -------
        Union[None, pandas.DataFrame]
            A pandas DataFrame containing the primary key fields for the objects.
            Returns None if no objects are found.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        kf = self.get_key_fields_for_object_type(ObjType)
        output = self._call_simauto("ListOfDevices", ObjType, FilterName)

        all_none = all(i is None for i in output)

        if all_none:
            return None

        df = pd.DataFrame(output).transpose()
        df.columns = kf["internal_field_name"].to_numpy()
        
        # Clean data if not using PowerWorld ordering
        if not self.pw_order:
            numeric = self.identify_numeric_fields(ObjectType=ObjType, fields=df.columns.to_numpy())
            numeric_fields = df.columns.to_numpy()[numeric]
            for field in numeric_fields:
                df[field] = pd.to_numeric(df[field], errors='coerce')
            nn_fields = df.columns.to_numpy()[~numeric]
            df[nn_fields] = df[nn_fields].astype(str).apply(lambda x: x.str.strip())
            try:
                df.sort_values(by="BusNum", axis=0, inplace=True)
                df.reset_index(drop=True, inplace=True)
            except (KeyError, TypeError):
                pass
        
        return df

    def ListOfDevicesAsVariantStrings(self, ObjType: str, FilterName="") -> tuple:
        """Retrieves a list of devices where primary keys are returned as variant strings.

        This method returns the primary keys as a tuple of strings, which might be
        useful for direct use in other SimAuto commands that expect string identifiers.

        Parameters
        ----------
        ObjType : str
            The PowerWorld object type.
        FilterName : str, optional
            Optional name of a PowerWorld filter to apply. Defaults to an empty string.

        Returns
        -------
        tuple
            A tuple of strings, where each string represents the primary key(s) of an object.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._call_simauto("ListOfDevicesAsVariantStrings", ObjType, FilterName)

    def ListOfDevicesFlatOutput(self, ObjType: str, FilterName="") -> tuple:
        """Retrieves a list of devices in a flat, 1-D output format.

        Similar to `ListOfDevicesAsVariantStrings`, but the output format might differ
        slightly depending on the SimAuto version.

        Parameters
        ----------
        ObjType : str
            The PowerWorld object type.
        FilterName : str, optional
            Optional name of a PowerWorld filter to apply. Defaults to an empty string.

        Returns
        -------
        tuple
            A tuple of strings.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._call_simauto("ListOfDevicesFlatOutput", ObjType, FilterName)

    def LoadState(self) -> None:
        """Loads the last saved state of the PowerWorld case."""
        return self._call_simauto("LoadState")

    def OpenCase(self, FileName: Union[str, None] = None) -> None:
        """Opens a PowerWorld case file.

        Parameters
        ----------
        FileName : Union[str, None], optional
            Path to the .pwb or .pwx file. If None, it attempts to reopen the
            last file path stored in `self.pwb_file_path`.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If `FileName` is None and no previous `pwb_file_path` is set.
        PowerWorldError
            If the SimAuto call fails (e.g., file not found).
        """
        if FileName is None:
            if self.pwb_file_path is None:
                raise TypeError("When OpenCase is called for the first time, a FileName is required.")
        else:
            self.pwb_file_path = FileName
        return self._call_simauto("OpenCase", self.pwb_file_path)

    def OpenCaseType(self, FileName: str, FileType: str, Options: Union[list, str, None] = None) -> None:
        """Opens a case file of a specific type (e.g., PTI, GE) with options.
        
        Parameters
        ----------
        FileName : str
            Path to the file.
            Different sets of optional parameters apply for the PTI and GE file formats.
            The LoadTransactions and Star bus parameters are available for writing to RAW files.
            MSLine, VarLimDead, and PostCTGAGC are for writing EPC files.
            See `OpenCase` in the Auxiliary File Format PDF for more details on options.
        FileType : str
            The file format (e.g., 'PTI', 'GE', 'EPC').
            Valid options include: PWB, PTI (latest version), PTI23-PTI35, GE (latest version),
            GE14-GE23, CF, AUX, UCTE, AREVAHDB, OPENNETEMS.
        Options : Union[list, str, None], optional
            A list or string of format-specific options. Defaults to None.
        """
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

        Auxiliary files contain script commands or data definitions that PowerWorld
        can process to modify the case or perform actions.

        Parameters
        ----------
        FileName : str
            Path to the auxiliary (.aux) file.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., file not found, syntax error in aux file).

        """
        return self._call_simauto("ProcessAuxFile", FileName)

    def RunScriptCommand(self, Statements):
        """Executes one or more PowerWorld script statements.

        Parameters
        ----------
        Statements : str
            A string containing one or more PowerWorld script commands, separated by semicolons.
            See the "SCRIPT Section" in the Auxiliary File Format PDF for command syntax.
        
        Returns
        -------
        None
        
        Raises
        ------
        PowerWorldError
            If any of the script commands fail.
        """
        return self._call_simauto("RunScriptCommand", Statements)

    def RunScriptCommand2(self, Statements: str, StatusMessage: str):
        """Executes script statements and provides a status message for the PowerWorld UI.

        This method is similar to `RunScriptCommand` but also allows displaying
        a custom message in the PowerWorld Simulator status bar.

        Parameters
        ----------
        Statements : str
            A string containing one or more PowerWorld script commands.
            See the "SCRIPT Section" in the Auxiliary File Format PDF for command syntax.

        StatusMessage : str
            A message to display in the PowerWorld Simulator status bar while the
            commands are being executed.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If any of the script commands fail.
        """
        return self._pwcom.RunScriptCommand2(Statements, StatusMessage)

    def SaveCase(self, FileName=None, FileType="PWB", Overwrite=True):
        """Saves the currently open PowerWorld case to a file.

        Parameters
        ----------
        FileName : str, optional
            Path to save the file. If None, the case is saved to its current path,
            potentially overwriting the original file.
        FileType : str, optional
            The file format to save as (e.g., "PWB", "PTI", "GE"). Defaults to "PWB".
        Overwrite : bool, optional
            If True, overwrites an existing file at `FileName` without prompting.
            Defaults to True.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If `FileName` is None and no case has been opened previously.
        PowerWorldError
            If the SimAuto call fails (e.g., invalid path, permission issues).
        """
        if FileName is not None:
            f = convert_to_windows_path(FileName)
        elif self.pwb_file_path is None:
            raise TypeError("SaveCase was called without a FileName, but OpenCase has not yet been called.")
        else:
            f = convert_to_windows_path(self.pwb_file_path)

        return self._call_simauto("SaveCase", f, FileType, Overwrite)

    def SaveState(self) -> None:
        """Saves the current state of the PowerWorld case.

        This creates an unnamed snapshot of the case that can be restored later
        using `LoadState`.
        """
        return self._call_simauto("SaveState")

    def SendToExcel(self, ObjectType: str, FilterName: str, FieldList) -> None:
        """Exports data for the specified objects directly to Microsoft Excel.

        This method requires Microsoft Excel to be installed on the system.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type (e.g., 'Bus', 'Gen').
        FilterName : str
            Optional PowerWorld filter name to apply.
        FieldList : List[str]
            A list of internal field names to export.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., Excel not installed, invalid parameters).
        """
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
        """Tuple property: Detailed information about the Simulator version and license."""
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
        """Internal helper to execute SimAuto COM methods and handle error codes.

        This method wraps all direct COM calls to PowerWorld Simulator, providing
        consistent error handling and unwrapping of results.

        Parameters
        ----------
        func : str
            The name of the SimAuto method to call (e.g., "OpenCase", "GetParametersMultipleElement").
        *args : Any
            Variable arguments to pass to the SimAuto method. These are typically
            converted to COM-compatible types (e.g., variants) before the call.

        Returns
        -------
        Any
            The data returned by SimAuto, unwrapped from the (Error, Result) tuple.
            Returns None if SimAuto indicates no data or an empty result.

        Raises
        ------
        AttributeError
            If `func` is not a valid SimAuto function.
        COMError
            If a COM-specific error occurs during the call.
        PowerWorldError
            If SimAuto returns an error message (e.g., invalid parameters, operation failed).
        """
        try:
            f = getattr(self._pwcom, func)
        except AttributeError:
            raise AttributeError(f"The given function, {func}, is not a valid SimAuto function.") from None

        try:
            output = f(*args)
        except Exception as e:
            # Handle specific RPC server unavailable/unknown interface errors
            msg = str(e)
            if hex(RPC_S_UNKNOWN_IF) in msg or hex(RPC_S_CALL_FAILED) in msg:
                m = f"SimAuto server crashed or is unresponsive during call to {func} with {args}. (RPC Error)"
                self.log.critical(m)
            m = f"An error occurred when trying to call {func} with {args}"
            self.log.exception(m)
            raise COMError(m) from e

        if output == ("",):
            return None

        try:
            if output is None or output[0] == "":
                pass
            elif "No data" not in output[0]:
                raise PowerWorldError.from_message(output[0])
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
        """Internal helper to prepare and execute `ChangeParametersMultipleElement` using a DataFrame.

        This method cleans the input DataFrame and then calls the SimAuto method
        to apply the changes.

        Parameters
        ----------
        ObjectType : str
            The PowerWorld object type.
        command_df : pandas.DataFrame
            The DataFrame containing the data to update.

        Returns
        -------
        pandas.DataFrame
            The cleaned DataFrame that was sent to PowerWorld.
        """
        cleaned_df = command_df.copy()
        
        # Clean data if not using PowerWorld ordering
        if not self.pw_order:
            numeric = self.identify_numeric_fields(ObjectType=ObjectType, fields=cleaned_df.columns.to_numpy())
            numeric_fields = cleaned_df.columns.to_numpy()[numeric]
            for field in numeric_fields:
                cleaned_df[field] = pd.to_numeric(cleaned_df[field], errors='coerce')
            nn_fields = cleaned_df.columns.to_numpy()[~numeric]
            cleaned_df[nn_fields] = cleaned_df[nn_fields].astype(str).apply(lambda x: x.str.strip())
            try:
                cleaned_df.sort_values(by="BusNum", axis=0, inplace=True)
                cleaned_df.reset_index(drop=True, inplace=True)
            except (KeyError, TypeError):
                pass
        
        self.ChangeParametersMultipleElement(
            ObjectType=ObjectType,
            ParamList=cleaned_df.columns.tolist(),
            ValueList=cleaned_df.to_numpy().tolist(),
        )
        return cleaned_df

    def _to_numeric(
        self, data: Union[pd.DataFrame, pd.Series], errors="ignore"
    ) -> Union[pd.DataFrame, pd.Series]:
        """Internal helper to convert DataFrame or Series columns to numeric types.

        Handles locale-specific decimal delimiters before conversion.

        Parameters
        ----------
        data : Union[pandas.DataFrame, pandas.Series]
            The data to convert.
        errors : str, optional
            How to handle errors during conversion ('ignore', 'raise', 'coerce').
            Defaults to 'ignore'.

        Returns
        -------
        Union[pandas.DataFrame, pandas.Series]
            The data with numeric columns converted.
        """
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
        """Internal helper to replace locale-specific decimal delimiters with '.' in a Series.

        Parameters
        ----------
        data : pandas.Series
            The Series whose string elements might contain locale-specific decimal delimiters.

        Returns
        -------
        pandas.Series
            A new Series with decimal delimiters replaced, or the original Series
            if it does not contain string data.
        """
        try:
            return data.str.replace(self.decimal_delimiter, ".")
        except AttributeError:
            return data

    def exec_aux(self, aux: str, use_double_quotes: bool = False):
        """Executes an auxiliary command string directly.

        This method writes the provided `aux` string to a temporary .aux file
        and then processes it using `ProcessAuxFile`.

        Parameters
        ----------
        aux : str
            The auxiliary command string to execute.
        use_double_quotes : bool, optional
            If True, single quotes in `aux` will be replaced with double quotes. Defaults to False.
        """
        if use_double_quotes:
            aux = aux.replace("'", '"')
        file = tempfile.NamedTemporaryFile(mode="wt", suffix=".aux", delete=False)
        file.write(aux)
        file.close()
        self.ProcessAuxFile(file.name)
        os.unlink(file.name)

    def update_ui(self) -> None:
        """Triggers a refresh of the PowerWorld Simulator user interface.

        This can be useful after making programmatic changes that might not immediately reflect in the GUI.
        """
        return self.ProcessAuxFile(self.empty_aux)