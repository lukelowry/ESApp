"""Case Actions specific functions."""
import os
from typing import List, Union

from ._enums import YesNo
from ._exceptions import PowerWorldError
from ._helpers import convert_list_to_variant, convert_to_windows_path, format_list


class CaseActionsMixin:
    """Mixin for Case Actions functions."""

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
        return self._com_call(
            "GetParametersSingleElement",
            "PowerWorldSession",
            convert_list_to_variant(["Version", "ExeBuildDate"]),
            convert_list_to_variant(["", ""]),
        )

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
        try:
            return self._com_call("OpenCase", self.pwb_file_path)
        except PowerWorldError as e:
            hints = [f"Failed to open case: '{self.pwb_file_path}'"]
            if not os.path.exists(self.pwb_file_path):
                hints.append(f"File does not exist at the specified path.")
            else:
                hints.append("The file exists but PowerWorld could not open it.")
                hints.append("Possible causes:")
                hints.append("  - PowerWorld Simulator is not licensed or the license has expired")
                hints.append("  - The file is corrupted or in an unsupported format")
                hints.append("  - The file is locked by another process")
            hints.append(f"Original error: {e.raw_message}")
            raise PowerWorldError("\n".join(hints)) from e

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
        try:
            return self._com_call("OpenCaseType", self.pwb_file_path, FileType, options)
        except PowerWorldError as e:
            hints = [f"Failed to open case: '{self.pwb_file_path}' (format: {FileType})"]
            if not os.path.exists(self.pwb_file_path):
                hints.append(f"File does not exist at the specified path.")
            else:
                hints.append("The file exists but PowerWorld could not open it.")
                hints.append("Possible causes:")
                hints.append("  - PowerWorld Simulator is not licensed or the license has expired")
                hints.append(f"  - The file format '{FileType}' does not match the actual file contents")
                hints.append("  - The file is corrupted or locked by another process")
            hints.append(f"Original error: {e.raw_message}")
            raise PowerWorldError("\n".join(hints)) from e

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
        return self._com_call("CloseCase")

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

        return self._com_call("SaveCase", f, FileType, Overwrite)

    def CaseDescriptionClear(self):
        """Clears the case description.

        Removes all text from the case's description field.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("CaseDescriptionClear")

    def CaseDescriptionSet(self, text: str, append: bool = False):
        """Sets or appends text to the case description.

        Parameters
        ----------
        text : str
            The text string to set or append to the case description.
        append : bool, optional
            If True, `text` is appended to the existing description. If False,
            `text` overwrites the existing description. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        app = YesNo.from_bool(append)
        return self._run_script("CaseDescriptionSet", f'"{text}"', app)

    def DeleteExternalSystem(self):
        """Deletes the part of the power system where the 'Equiv' field on buses is set to true.

        Removes all buses from the case that have their 'Equiv' field set to True.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("DeleteExternalSystem")

    def Equivalence(self):
        """Equivalences the power system based on Equiv_Options.

        This action applies the equivalence settings configured in PowerWorld
        to reduce the size of the system model.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("Equivalence")

    def LoadEMS(self, filename: str, filetype: str = "AREVAHDB"):
        """Opens an EMS (Energy Management System) file.

        Parameters
        ----------
        filename : str
            Path to the EMS file.
        filetype : str, optional
            The EMS file format (e.g., "AREVAHDB", "GE"). Defaults to "AREVAHDB".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., file not found, invalid format).
        """
        return self._run_script("LoadEMS", f'"{filename}"', filetype)

    def NewCase(self):
        """Clears the existing case and opens a new, empty one.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("NewCase")

    def Renumber3WXFormerStarBuses(self, filename: str, delimiter: str = "BOTH"):
        """Renumbers 3-winding transformer star buses based on user-specified values in a file.

        Parameters
        ----------
        filename : str
            Path to the renumbering file. This file should contain mappings
            for star bus renumbering.
        delimiter : str, optional
            The delimiter used in the renumbering file (e.g., "COMMA", "TAB", "BOTH").
            Defaults to "BOTH".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., file not found, format error).
        """
        return self._run_script("Renumber3WXFormerStarBuses", f'"{filename}"', delimiter)

    def RenumberAreas(self, custom_integer_index: int = 0):
        """Renumbers Areas using the value in the specified Custom Integer field.

        Parameters
        ----------
        custom_integer_index : int, optional
            The 0-based index of the Custom Integer field to use for renumbering.
            Defaults to 0 (CustomInteger:0).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("RenumberAreas", custom_integer_index)

    def RenumberBuses(self, custom_integer_index: int = 1):
        """Renumbers Buses using the value in the specified Custom Integer field.

        Parameters
        ----------
        custom_integer_index : int, optional
            The 0-based index of the Custom Integer field to use for renumbering.
            Defaults to 1 (CustomInteger:1).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("RenumberBuses", custom_integer_index)

    def RenumberMSLineDummyBuses(self, filename: str, delimiter: str = "BOTH"):
        """Renumbers dummy buses of multisection lines based on a provided file.

        Parameters
        ----------
        filename : str
            Path to the renumbering file.
        delimiter : str, optional
            The delimiter used in the renumbering file. Defaults to "BOTH".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., file not found, format error).
        """
        return self._run_script("RenumberMSLineDummyBuses", f'"{filename}"', delimiter)

    def RenumberSubs(self, custom_integer_index: int = 2):
        """Renumbers Substations using the value in the specified Custom Integer field.

        Parameters
        ----------
        custom_integer_index : int, optional
            The 0-based index of the Custom Integer field to use for renumbering.
            Defaults to 2 (CustomInteger:2).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("RenumberSubs", custom_integer_index)

    def RenumberZones(self, custom_integer_index: int = 3):
        """Renumbers Zones using the value in the specified Custom Integer field.

        Parameters
        ----------
        custom_integer_index : int, optional
            The 0-based index of the Custom Integer field to use for renumbering.
            Defaults to 3 (CustomInteger:3).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("RenumberZones", custom_integer_index)

    def RenumberCase(self):
        """Renumbers objects in the case according to the swap list in memory.

        This applies any pending renumbering operations that have been configured.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("RenumberCase")

    def SaveExternalSystem(self, filename: str, filetype: str = "PWB", with_ties: bool = False):
        """Saves only buses where 'Equiv' is set to 'External' to a new case file.

        This is useful for creating reduced models of external systems.

        Parameters
        ----------
        filename : str
            Path to save the external system case file.
        filetype : str, optional
            The file format (e.g., "PWB", "PTI"). Defaults to "PWB".
        with_ties : bool, optional
            If True, includes tie lines connecting the external system to the
            internal system. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        wt = YesNo.from_bool(with_ties)
        return self._run_script("SaveExternalSystem", f'"{filename}"', filetype, wt)

    def SaveMergedFixedNumBusCase(self, filename: str, filetype: str = "PWB"):
        """Saves the Merged FixedNumBus case.

        This action is typically used after a fixed-number bus merge operation.

        Parameters
        ----------
        filename : str
            Path to save the merged case file.
        filetype : str, optional
            The file format (e.g., "PWB", "PTI"). Defaults to "PWB".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("SaveMergedFixedNumBusCase", f'"{filename}"', filetype)

    def Scale(
        self,
        scale_type: str,
        based_on: str,
        parameters: List[float],
        scale_marker: str,
    ):
        """Scales load and generation in the system based on specified criteria.

        This action can be used to uniformly adjust load or generation values
        across the system or specific regions.

        Parameters
        ----------
        scale_type : str
            The type of object to scale ("LOAD", "GEN", "INJECTIONGROUP", or "BUSSHUNT").
        based_on : str
            The scaling basis ("MW" for absolute MW/MVAR values, or "FACTOR" for a multiplier).
        parameters : List[float]
            A list of values for scaling. If `based_on` is "MW", this can be
            `[MW, MVAR]` or `[MW]` for LOAD/INJECTIONGROUP, `[MW]` for GEN, or
            `[GMW, BCAPMVAR, BREAMVAR]` for BUSSHUNT. If `based_on` is "FACTOR",
            this is `[Factor]`. Can also be a field variable name to use
            values from another field.
        scale_marker : str
            The scope of the scaling ("BUS", "AREA", "ZONE", "OWNER", or "SYSTEM").

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        params = format_list(parameters, stringify=True)
        return self._run_script("Scale", scale_type, based_on, params, scale_marker)
