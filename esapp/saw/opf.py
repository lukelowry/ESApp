"""Optimal Power Flow (OPF) specific functions."""


class OPFMixin:
    """Mixin for OPF analysis functions."""

    def SolvePrimalLP(self, on_success_aux: str = "", on_fail_aux: str = "", create_if_not_found1: bool = False, create_if_not_found2: bool = False):
        """Attempts to solve a primal linear programming optimal power flow (LP OPF).

        This method finds the least-cost generation dispatch while satisfying
        system constraints.

        Parameters
        ----------
        on_success_aux : str, optional
            Auxiliary file to load if the solution is successful.
        on_fail_aux : str, optional
            Auxiliary file to load if the solution is NOT successful.
        create_if_not_found1 : bool, optional
            If True, creates objects from `on_success_aux` if they don't exist.
        create_if_not_found2 : bool, optional
            If True, creates objects from `on_fail_aux` if they don't exist.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails or the OPF does not converge.
        """
        c1 = "YES" if create_if_not_found1 else "NO"
        c2 = "YES" if create_if_not_found2 else "NO"
        return self.RunScriptCommand(f'SolvePrimalLP("{on_success_aux}", "{on_fail_aux}", {c1}, {c2});')

    def InitializePrimalLP(self, on_success_aux: str = "", on_fail_aux: str = "", create_if_not_found1: bool = False, create_if_not_found2: bool = False):
        """Clears all structures and results of previous primal LP OPF solutions.

        This prepares the system for a new OPF calculation.

        Parameters
        ----------
        on_success_aux : str, optional
            Auxiliary file to load if initialization is successful.
        on_fail_aux : str, optional
            Auxiliary file to load if initialization is NOT successful.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        c1 = "YES" if create_if_not_found1 else "NO"
        c2 = "YES" if create_if_not_found2 else "NO"
        return self.RunScriptCommand(f'InitializePrimalLP("{on_success_aux}", "{on_fail_aux}", {c1}, {c2});')

    def SolveSinglePrimalLPOuterLoop(self, on_success_aux: str = "", on_fail_aux: str = "", create_if_not_found1: bool = False, create_if_not_found2: bool = False):
        """Performs a single optimization iteration of LP OPF.

        This is typically used in iterative solution schemes.

        Parameters
        ----------
        on_success_aux : str, optional
            Auxiliary file to load if the iteration is successful.
        on_fail_aux : str, optional
            Auxiliary file to load if the iteration is NOT successful.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        c1 = "YES" if create_if_not_found1 else "NO"
        c2 = "YES" if create_if_not_found2 else "NO"
        return self.RunScriptCommand(f'SolveSinglePrimalLPOuterLoop("{on_success_aux}", "{on_fail_aux}", {c1}, {c2});')

    def SolveFullSCOPF(self, bc_method: str = "OPF", on_success_aux: str = "", on_fail_aux: str = "", create_if_not_found1: bool = False, create_if_not_found2: bool = False):
        """Performs a full Security Constrained Optimal Power Flow (SCOPF).

        SCOPF finds the least-cost dispatch that satisfies both base-case and
        contingency constraints.

        Parameters
        ----------
        bc_method : str, optional
            Solution method for the base case ("POWERFLOW" or "OPF"). Defaults to "OPF".
        on_success_aux : str, optional
            Auxiliary file to load if the solution is successful.
        on_fail_aux : str, optional
            Auxiliary file to load if the solution is NOT successful.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails or the SCOPF does not converge.
        """
        c1 = "YES" if create_if_not_found1 else "NO"
        c2 = "YES" if create_if_not_found2 else "NO"
        return self.RunScriptCommand(f'SolveFullSCOPF({bc_method}, "{on_success_aux}", "{on_fail_aux}", {c1}, {c2});')

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
