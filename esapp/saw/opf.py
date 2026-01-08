"""Optimal Power Flow (OPF) specific functions."""


class OPFMixin:
    """Mixin for OPF analysis functions."""

    def SolvePrimalLP(self):
        """Attempts to solve a primal linear programming optimal power flow (LP OPF).

        This method finds the least-cost generation dispatch while satisfying
        system constraints.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails or the OPF does not converge.
        """
        return self.RunScriptCommand('SolvePrimalLP("", "", NO, NO);')

    def InitializePrimalLP(self):
        """Clears all structures and results of previous primal LP OPF solutions.

        This prepares the system for a new OPF calculation.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand('InitializePrimalLP("", "", NO, NO);')

    def SolveSinglePrimalLPOuterLoop(self):
        """Performs a single optimization iteration of LP OPF.

        This is typically used in iterative solution schemes.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand('SolveSinglePrimalLPOuterLoop("", "", NO, NO);')

    def SolveFullSCOPF(self):
        """Performs a full Security Constrained Optimal Power Flow (SCOPF).

        SCOPF finds the least-cost dispatch that satisfies both base-case and
        contingency constraints.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails or the SCOPF does not converge.
        """
        return self.RunScriptCommand('SolveFullSCOPF(OPF, "", "", NO, NO);')

    def OPFWriteResultsAndOptions(self, filename: str):
        """Writes out all information related to OPF analysis to an auxiliary file.

        Parameters
        ----------
        filename : str
            The path to the auxiliary file where the OPF information will be written.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f'OPFWriteResultsAndOptions("{filename}");')
