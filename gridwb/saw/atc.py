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
        """Calculates Available Transfer Capability (ATC) between a seller and a buyer.

        :param seller: The seller (source) (e.g. '[AREA "Top"]').
        :param buyer: The buyer (sink) (e.g. '[BUS 7]').
        :param distributed: Use distributed ATC solution method.
        :param multiple_scenarios: Process each defined scenario.
        :return: None or error string.
        :return: None or error string.
        """
        dist = "YES" if distributed else "NO"
        mult = "YES" if multiple_scenarios else "NO"
        return self.RunScriptCommand(
            f"ATCDetermine({seller}, {buyer}, {dist}, {mult});"
        )

    def DetermineATCMultipleDirections(
        self, distributed: bool = False, multiple_scenarios: bool = False
    ):
        """Calculates ATC for all defined directions.

        :param distributed: Use distributed ATC solution method.
        :param multiple_scenarios: Process each defined scenario.
        """
        dist = "YES" if distributed else "NO"
        mult = "YES" if multiple_scenarios else "NO"
        return self.RunScriptCommand(
            f"ATCDetermineMultipleDirections({dist}, {mult});"
        )

    def GetATCResults(self, fields: list = None) -> pd.DataFrame:
        """Retrieves Transfer Limiter results from the case.

        :param fields: List of fields to retrieve. Defaults to common fields.
        :return: DataFrame containing TransferLimiter objects.
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
