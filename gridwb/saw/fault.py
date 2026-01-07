"""Fault analysis specific functions."""


class FaultMixin:
    """Mixin for fault analysis functions."""

    def RunFault(
        self,
        element: str,
        fault_type: str,
        r: float = 0.0,
        x: float = 0.0,
        location: float = None,
    ):
        """Calculates fault currents.

        :param element: The fault element (e.g. '[BUS 1]' or '[BRANCH 1 2 1]').
        :param fault_type: SLG, LL, 3PB, DLG.
        :param r: Fault resistance.
        :param x: Fault reactance.
        :param location: Percentage distance for branch faults (0-100). Required if element is a branch.
        """
        if location is not None:
            return self.RunScriptCommand(
                f"Fault({element}, {location}, {fault_type}, {r}, {x});"
            )
        else:
            return self.RunScriptCommand(f"Fault({element}, {fault_type}, {r}, {x});")

    def FaultClear(self):
        """Clears a single fault that has been calculated."""
        return self.RunScriptCommand("FaultClear;")

    def FaultAutoInsert(self):
        """Inserts multiple fault definitions based on auto-insert options."""
        return self.RunScriptCommand("FaultAutoInsert;")

    def FaultMultiple(self, use_dummy_bus: bool = False):
        """Runs fault analysis on a list of defined faults."""
        dummy = "YES" if use_dummy_bus else "NO"
        return self.RunScriptCommand(f"FaultMultiple({dummy});")

    def LoadPTISEQData(self, filename: str, version: int = 33):
        """Loads sequence data in the PTI format."""
        return self.RunScriptCommand(f'LoadPTISEQData("{filename}", {version});')
