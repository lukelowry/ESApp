"""QV (Reactive Power-Voltage) Analysis specific functions."""
import os
import tempfile
from pathlib import Path

import pandas as pd


class QVMixin:
    """Mixin for QV analysis functions."""

    def QVDataWriteOptionsAndResults(self, filename: str, append: bool = True, key_field: str = "PRIMARY"):
        """Writes out all information related to QV analysis."""
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'QVDataWriteOptionsAndResults("{filename}", {app}, {key_field});')

    def QVDeleteAllResults(self):
        """Deletes all QV results."""
        return self.RunScriptCommand("QVDeleteAllResults;")

    def RunQV(self, filename: str = None) -> pd.DataFrame:
        """Starts a QV analysis.

        :param filename: Optional CSV file to save results to. If None, a temp file is used and results returned as DataFrame.
        :return: DataFrame of results if filename is None, otherwise None.
        """
        if filename:
            self.RunScriptCommand(f'QVRun("{filename}", YES, NO);')
            return None
        else:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                temp_path = Path(tmp.name).as_posix()

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
        """Modify monitored buses to one per pnode."""
        return self.RunScriptCommand("QVSelectSingleBusPerSuperBus;")

    def QVWriteCurves(self, filename: str, include_quantities: bool = True, filter_name: str = "", append: bool = False):
        """Save QV curve points."""
        iq = "YES" if include_quantities else "NO"
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'QVWriteCurves("{filename}", {iq}, "{filter_name}", {app});')

    def QVWriteResultsAndOptions(self, filename: str, append: bool = True):
        """Writes out all information related to QV analysis."""
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'QVWriteResultsAndOptions("{filename}", {app});')
