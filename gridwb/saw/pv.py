"""PV (Power-Voltage) Analysis specific functions."""


class PVMixin:
    """Mixin for PV analysis functions."""

    def PVClear(self):
        """Clear all results of the PV study."""
        return self.RunScriptCommand("PVClear;")

    def RunPV(self, source: str, sink: str):
        """Starts a PV analysis.

        :param source: The source of power (e.g. '[INJECTIONGROUP "Source"]').
        :param sink: The sink of power (e.g. '[INJECTIONGROUP "Sink"]').
        """
        return self.RunScriptCommand(f"PVRun({source}, {sink});")

    def PVDataWriteOptionsAndResults(self, filename: str, append: bool = True, key_field: str = "PRIMARY"):
        """Writes out all information related to PV analysis."""
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'PVDataWriteOptionsAndResults("{filename}", {app}, {key_field});')

    def PVDestroy(self):
        """Destroy the PV study."""
        return self.RunScriptCommand("PVDestroy;")

    def PVQVTrackSingleBusPerSuperBus(self):
        """Reduce monitored buses to one per super bus."""
        return self.RunScriptCommand("PVQVTrackSingleBusPerSuperBus;")

    def PVSetSourceAndSink(self, source: str, sink: str):
        """Specify the source and sink elements."""
        return self.RunScriptCommand(f"PVSetSourceAndSink({source}, {sink});")

    def PVStartOver(self):
        """Start over the PV study."""
        return self.RunScriptCommand("PVStartOver;")

    def PVWriteInadequateVoltages(self, filename: str, append: bool = True, inadequate_type: str = "LOW"):
        """Save PV Inadequate Voltages."""
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'PVWriteInadequateVoltages("{filename}", {app}, {inadequate_type});')

    def PVWriteResultsAndOptions(self, filename: str, append: bool = True):
        """Writes out all information related to PV analysis."""
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'PVWriteResultsAndOptions("{filename}", {app});')

    def RefineModel(self, object_type: str, filter_name: str, action: str, tolerance: float):
        """Refine the system model to fix modeling idiosyncrasies."""
        filt = f'"{filter_name}"' if filter_name else ""
        return self.RunScriptCommand(f'RefineModel({object_type}, {filt}, {action}, {tolerance});')
