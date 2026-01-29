"""QV (Reactive Power-Voltage) Analysis specific functions."""
import os
from pathlib import Path

import pandas as pd

from esapp.saw._enums import YesNo
from ._helpers import get_temp_filepath, pack_args


class QVMixin:
    """Mixin for QV analysis functions."""

    def QVDataWriteOptionsAndResults(self, filename: str, append: bool = True, key_field: str = "PRIMARY"):
        """Writes out all information related to QV analysis, including options and results.

        Parameters
        ----------
        filename : str
            The path to the auxiliary file where the QV information will be written.
        append : bool, optional
            If True, appends to the file if it exists. If False, overwrites.
            Defaults to True.
        key_field : str, optional
            Identifier to use for the data ("PRIMARY", "SECONDARY", "LABEL").
            Defaults to "PRIMARY".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        app = YesNo.from_bool(append)
        args = pack_args(f'"{filename}"', app, key_field)
        return self.RunScriptCommand(f"QVDataWriteOptionsAndResults({args});")

    def QVDeleteAllResults(self):
        """Deletes all QV results from memory.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand("QVDeleteAllResults;")

    def RunQV(self, filename: str = None) -> pd.DataFrame:
        """Starts a QV (Reactive Power-Voltage) analysis.

        This method simulates the system's voltage stability by varying reactive power
        and observing voltage response.

        Parameters
        ----------
        filename : str, optional
            Optional path to a CSV file to save results to. If None, a temporary file
            is used, and the results are returned as a pandas DataFrame. Defaults to None.

        Returns
        -------
        pandas.DataFrame or None
            If `filename` is None, returns a DataFrame containing the QV analysis results.
            Otherwise, returns None.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails or the QV analysis does not complete successfully.
        """
        if filename:
            self.RunScriptCommand(f'QVRun("{filename}", YES, NO);')
            return None
        else:
            temp_path = get_temp_filepath(".csv")

            try:
                self.RunScriptCommand(f'QVRun("{temp_path}", YES, NO);')
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    return pd.read_csv(temp_path)
                else:
                    return pd.DataFrame()
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def QVSelectSingleBusPerSuperBus(self):
        """Modifies monitored buses for QV analysis to one per pnode (super bus).

        This simplifies the QV analysis by focusing on representative buses.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand("QVSelectSingleBusPerSuperBus;")

    def QVWriteCurves(self, filename: str, include_quantities: bool = True, filter_name: str = "", append: bool = False):
        """Saves QV curve points to a file.

        Parameters
        ----------
        filename : str
            The path to the output file.
        include_quantities : bool, optional
            If True, includes quantities (e.g., MW, Mvar) in the output. Defaults to True.
        filter_name : str, optional
            A PowerWorld filter name to apply to buses. Defaults to an empty string (all).
        append : bool, optional
            If True, appends to the file if it exists. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        iq = YesNo.from_bool(include_quantities)
        app = YesNo.from_bool(append)
        args = pack_args(f'"{filename}"', iq, f'"{filter_name}"', app)
        return self.RunScriptCommand(f"QVWriteCurves({args});")

    def QVWriteResultsAndOptions(self, filename: str, append: bool = True):
        """Writes out all information related to QV analysis to an auxiliary file.

        Parameters
        ----------
        filename : str
            The path to the auxiliary file.
        append : bool, optional
            If True, appends to the file if it exists. Defaults to True.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        app = YesNo.from_bool(append)
        args = pack_args(f'"{filename}"', app)
        return self.RunScriptCommand(f"QVWriteResultsAndOptions({args});")
