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

        :param filename: File name of the case to be appended.
        :param filetype: PWB, GE, PTI, CF.
        :param star_bus: NEAR, MAX, or Value (PTI only).
        :param estimate_voltages: Estimate voltages/angles for new buses.
        :param ms_line: MAINTAIN or EQUIVALENCE (GE only).
        :param var_lim_dead: GE var limit deadband (GE only).
        :param post_ctg_agc: Populate Post-CTG Prevent Response (GE only).
        :return: None or error string.
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
        :return: None or error string."""
        return self.RunScriptCommand("CaseDescriptionClear;")

    def CaseDescriptionSet(self, text: str, append: bool = False):
        """Sets or appends text to the case description.

        :param text: The text to set or append.
        :param append: If True, appends to existing description.
        :return: None or error string.
        """
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'CaseDescriptionSet("{text}", {app});')

    def DeleteExternalSystem(self):
        """Deletes buses where Equiv is true.
        :return: None or error string."""
        return self.RunScriptCommand("DeleteExternalSystem;")

    def Equivalence(self):
        """Equivalences the power system based on Equiv_Options.
        :return: None or error string."""
        return self.RunScriptCommand("Equivalence;")

    def LoadEMS(self, filename: str, filetype: str = "AREVAHDB"):
        """Opens an EMS file.

        :param filename: Path to the EMS file.
        :param filetype: The EMS format (default 'AREVAHDB').
        :return: None or error string.
        """
        return self.RunScriptCommand(f'LoadEMS("{filename}", {filetype});')

    def NewCase(self):
        """Clears the existing case and opens a new one.
        :return: None or error string."""
        return self.RunScriptCommand("NewCase;")

    def Renumber3WXFormerStarBuses(self, filename: str, delimiter: str = "BOTH"):
        """Renumbers star buses based on user-specified values in a file.

        :param filename: Path to the renumbering file.
        :param delimiter: File delimiter (default 'BOTH').
        :return: None or error string.
        """
        return self.RunScriptCommand(f'Renumber3WXFormerStarBuses("{filename}", {delimiter});')

    def RenumberAreas(self, custom_integer_index: int = 0):
        """Renumbers Areas using the value in the specified Custom Integer field.

        :param custom_integer_index: The index of the Custom Integer field to use.
        :return: None or error string.
        """
        return self.RunScriptCommand(f"RenumberAreas({custom_integer_index});")

    def RenumberBuses(self, custom_integer_index: int = 1):
        """Renumbers Buses using the value in the specified Custom Integer field.

        :param custom_integer_index: The index of the Custom Integer field to use.
        :return: None or error string.
        """
        return self.RunScriptCommand(f"RenumberBuses({custom_integer_index});")

    def RenumberMSLineDummyBuses(self, filename: str, delimiter: str = "BOTH"):
        """Renumbers dummy buses of multisection lines based on file.

        :param filename: Path to the renumbering file.
        :param delimiter: File delimiter.
        :return: None or error string.
        """
        return self.RunScriptCommand(f'RenumberMSLineDummyBuses("{filename}", {delimiter});')

    def RenumberSubs(self, custom_integer_index: int = 2):
        """Renumbers Substations using the value in the specified Custom Integer field.

        :param custom_integer_index: The index of the Custom Integer field to use.
        :return: None or error string.
        """
        return self.RunScriptCommand(f"RenumberSubs({custom_integer_index});")

    def RenumberZones(self, custom_integer_index: int = 3):
        """Renumbers Zones using the value in the specified Custom Integer field.

        :param custom_integer_index: The index of the Custom Integer field to use.
        :return: None or error string.
        """
        return self.RunScriptCommand(f"RenumberZones({custom_integer_index});")

    def RenumberCase(self):
        """Renumbers the object in the case according to the swap list in memory.
        :return: None or error string."""
        return self.RunScriptCommand("RenumberCase;")

    def SaveExternalSystem(self, filename: str, filetype: str = "PWB", with_ties: bool = False):
        """Saves only buses where Equiv is set to External.

        :param filename: Path to save the external system.
        :param filetype: File format (default 'PWB').
        :param with_ties: If True, includes tie lines.
        :return: None or error string.
        """
        wt = "YES" if with_ties else "NO"
        return self.RunScriptCommand(f'SaveExternalSystem("{filename}", {filetype}, {wt});')

    def SaveMergedFixedNumBusCase(self, filename: str, filetype: str = "PWB"):
        """Saves the Merged FixedNumBus case.

        :param filename: Path to save the case.
        :param filetype: File format.
        :return: None or error string.
        """
        return self.RunScriptCommand(f'SaveMergedFixedNumBusCase("{filename}", {filetype});')

    def Scale(
        self,
        scale_type: str,
        based_on: str,
        parameters: List[float],
        scale_marker: str,
    ):
        """Scales load and generation in the system.

        :param scale_type: LOAD, GEN, INJECTIONGROUP, or BUSSHUNT.
        :param based_on: MW or FACTOR.
        :param parameters: List of values [MW, MVAR] or [Factor].
        :param scale_marker: BUS, AREA, ZONE, OWNER.
        :return: None or error string.
        """
        params = "[" + ", ".join([str(p) for p in parameters]) + "]"
        return self.RunScriptCommand(f"Scale({scale_type}, {based_on}, {params}, {scale_marker});")