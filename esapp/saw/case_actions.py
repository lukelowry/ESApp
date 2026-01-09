"""Case Actions specific functions."""
from typing import List


class CaseActionsMixin:
    """Mixin for Case Actions functions."""

    def AppendCase(
        self,
        filename: str,
        filetype: str,
        star_bus: str = "NEAR",
        estimate_voltages: bool = True,
        ms_line: str = "MAINTAIN",
        var_lim_dead: float = 2.0,
        post_ctg_agc: bool = False,
    ):
        """Merges another case file into the currently open PowerWorld case.

        This action is used to combine the network and data from an external
        case file with the currently loaded case.

        Parameters
        ----------
        filename : str
            The file name of the case to be appended.
        filetype : str
            The format of the file to append (e.g., "PWB", "GE", "PTI", "CF", "AUX",
            "UCTE", "AREVAHDB", "OPENNETEMS").
        star_bus : str, optional
            For PTI RAW format, specifies how to handle star buses ("NEAR", "MAX", or a numeric value).
            Defaults to "NEAR".
        estimate_voltages : bool, optional
            For GE EPC or PTI RAW format, if True, estimates voltages and angles for new buses
            introduced by the append. Angle smoothing is done across new lines.
            Defaults to True.
        ms_line : str, optional
            For GE EPC format, specifies how to handle multisection lines ("MAINTAIN" or "EQUIVALENCE").
            Defaults to "MAINTAIN".
        var_lim_dead : float, optional
            For GE EPC format, sets the var limit deadband. Defaults to 2.0.
        post_ctg_agc : bool, optional
            For GE EPC format, if True, populates the generator field 'Post-CTG Prevent Response'
            based on the EPC file's generator base load flag. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., file not found, invalid parameters).
        """
        est = "YES" if estimate_voltages else "NO"
        pc_agc = "YES" if post_ctg_agc else "NO"

        if "PTI" in filetype.upper():
            args = f'"{filename}", {filetype}, [{star_bus}, {est}]'
        elif "GE" in filetype.upper():
            args = f'"{filename}", {filetype}, [{ms_line}, {var_lim_dead}, {pc_agc}, {est}]'
        else:
            args = f'"{filename}", {filetype}'

        return self.RunScriptCommand(f"AppendCase({args});")

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
        return self.RunScriptCommand("CaseDescriptionClear;")

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
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'CaseDescriptionSet("{text}", {app});')

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
        return self.RunScriptCommand("DeleteExternalSystem;")

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
        return self.RunScriptCommand("Equivalence;")

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
        return self.RunScriptCommand(f'LoadEMS("{filename}", {filetype});')

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
        return self.RunScriptCommand("NewCase;")

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
        return self.RunScriptCommand(f'Renumber3WXFormerStarBuses("{filename}", {delimiter});')

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
        return self.RunScriptCommand(f"RenumberAreas({custom_integer_index});")

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
        return self.RunScriptCommand(f"RenumberBuses({custom_integer_index});")

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
        return self.RunScriptCommand(f'RenumberMSLineDummyBuses("{filename}", {delimiter});')

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
        return self.RunScriptCommand(f"RenumberSubs({custom_integer_index});")

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
        return self.RunScriptCommand(f"RenumberZones({custom_integer_index});")

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
        return self.RunScriptCommand("RenumberCase;")

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
        wt = "YES" if with_ties else "NO"
        return self.RunScriptCommand(f'SaveExternalSystem("{filename}", {filetype}, {wt});')

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
        return self.RunScriptCommand(f'SaveMergedFixedNumBusCase("{filename}", {filetype});')

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
            A list of values for scaling.
            - If `based_on` is "MW":
                - For LOAD/INJECTIONGROUP: `[MW, MVAR]` or `[MW]` (for constant power factor).
                - For GEN: `[MW]`.
                - For BUSSHUNT: `[GMW, BCAPMVAR, BREAMVAR]`.
            - If `based_on` is "FACTOR": `[Factor]`.
            - Can also be a field variable name to use values from another field.
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
        params = "[" + ", ".join([str(p) for p in parameters]) + "]"
        return self.RunScriptCommand(f"Scale({scale_type}, {based_on}, {params}, {scale_marker});")