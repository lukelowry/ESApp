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
        """Calculates fault currents for a single fault.

        This method simulates a fault at a specified element and calculates
        the resulting currents and voltages.

        Parameters
        ----------
        element : str
            The fault element string (e.g., '[BUS 1]', '[BRANCH 1 2 1]').
        fault_type : str
            The type of fault: "SLG" (Single Line to Ground), "LL" (Line to Line),
            "3PB" (Three Phase Balanced), or "DLG" (Double Line to Ground).
        r : float, optional
            Fault resistance in per unit. Defaults to 0.0.
        x : float, optional
            Fault reactance in per unit. Defaults to 0.0.
        location : float, optional
            Percentage distance (0-100) along the branch for branch faults.
            Required if `element` is a branch. Defaults to None.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., invalid element, fault type, or location).
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
