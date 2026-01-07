"""Optimal Power Flow (OPF) specific functions."""


class OPFMixin:
    """Mixin for OPF analysis functions."""

    def SolvePrimalLP(self):
        """Attempts to solve a primal linear programming optimal power flow (LP OPF)."""
        return self.RunScriptCommand('SolvePrimalLP("", "", NO, NO);')

    def InitializePrimalLP(self):
        """Clears all structures and results of previous primal LP OPF solutions."""
        return self.RunScriptCommand('InitializePrimalLP("", "", NO, NO);')

    def SolveSinglePrimalLPOuterLoop(self):
        """Performs a single optimization iteration of LP OPF."""
        return self.RunScriptCommand('SolveSinglePrimalLPOuterLoop("", "", NO, NO);')

    def SolveFullSCOPF(self):
        """Performs a full Security Constrained Optimal Power Flow (SCOPF)."""
        return self.RunScriptCommand('SolveFullSCOPF(OPF, "", "", NO, NO);')

    def OPFWriteResultsAndOptions(self, filename: str):
        """Writes out all information related to OPF analysis."""
        return self.RunScriptCommand(f'OPFWriteResultsAndOptions("{filename}");')
