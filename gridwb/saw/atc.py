"""Available Transfer Capability (ATC) specific functions."""
import pandas as pd


class ATCMixin:
    """Mixin for ATC analysis functions."""

    def DetermineATC(
        self,
        seller: str,
        buyer: str,
        distributed: bool = False,
        multiple_scenarios: bool = False,
    ):
        """Calculates Available Transfer Capability (ATC) between a specified seller and buyer.

        This method initiates an ATC calculation, ramping transfer between the
        seller and buyer until a system limit is reached.

        Parameters
        ----------
        seller : str
            The source object string (e.g., '[AREA "Top"]', '[BUS 1]').
        buyer : str
            The sink object string (e.g., '[AREA "Bottom"]', '[BUS 2]').
        distributed : bool, optional
            If True, uses the distributed ATC solution method. Defaults to False.
        multiple_scenarios : bool, optional
            If True, processes each defined scenario in the case. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., invalid seller/buyer, calculation error).
        """
        dist = "YES" if distributed else "NO"
        mult = "YES" if multiple_scenarios else "NO"
        return self.RunScriptCommand(
            f"ATCDetermine({seller}, {buyer}, {dist}, {mult});"
        )

    def DetermineATCMultipleDirections(
        self, distributed: bool = False, multiple_scenarios: bool = False
    ):
        """Calculates ATC for all directions defined within the PowerWorld case.

        This method is used when multiple transfer directions have been pre-configured
        in the Simulator.

        Parameters
        ----------
        distributed : bool, optional
            If True, uses the distributed ATC solution method. Defaults to False.
        multiple_scenarios : bool, optional
            If True, processes each defined scenario in the case. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., no directions defined, calculation error).
        """
        dist = "YES" if distributed else "NO"
        mult = "YES" if multiple_scenarios else "NO"
        return self.RunScriptCommand(
            f"ATCDetermineMultipleDirections({dist}, {mult});"
        )

    def GetATCResults(self, fields: list = None) -> pd.DataFrame:
        """Retrieves Transfer Limiter results from the case after an ATC calculation.

        This method fetches the detailed results of the ATC analysis, including
        the maximum flow, limiting contingency, and limiting element.

        Parameters
        ----------
        fields : List[str], optional
            A list of internal field names to retrieve for the 'TransferLimiter' object type.
            If None, a default set of common fields is retrieved.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the requested data for 'TransferLimiter' objects.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        if fields is None:
            fields = [
                "LimitingElement",
                "LimitingContingency",
                "MaxFlow",
                "TransferLimit",
                "LimitUsed",
                "PTDF",
                "OTDF",
            ]

        return self.GetParametersMultipleElement("TransferLimiter", fields)
