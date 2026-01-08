"""PV (Power-Voltage) Analysis specific functions."""


class PVMixin:
    """Mixin for PV analysis functions."""

    def PVClear(self):
        """
        Clear all results of the PV study.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self.RunScriptCommand("PVClear;")

    def RunPV(self, source: str, sink: str):
        """
        Starts a PV analysis.
        
        Parameters
        ----------
        source : str
            The source of power (e.g. '[INJECTIONGROUP "Source"]').
        sink : str
            The sink of power (e.g. '[INJECTIONGROUP "Sink"]').

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self.RunScriptCommand(f"PVRun({source}, {sink});")

    def PVDataWriteOptionsAndResults(self, filename: str, append: bool = True, key_field: str = "PRIMARY"):
        """
        Writes out all information related to PV analysis.

        Parameters
        ----------
        filename : str
            The file to write to.
        append : bool, optional
            If True, appends to the file. Defaults to True.
        key_field : str, optional
            The key field to use. Defaults to "PRIMARY".

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'PVDataWriteOptionsAndResults("{filename}", {app}, {key_field});')

    def PVDestroy(self):
        """
        Destroy the PV study.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self.RunScriptCommand("PVDestroy;")

    def PVQVTrackSingleBusPerSuperBus(self):
        """
        Reduce monitored buses to one per super bus.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self.RunScriptCommand("PVQVTrackSingleBusPerSuperBus;")

    def PVSetSourceAndSink(self, source: str, sink: str):
        """
        Specify the source and sink elements.

        Parameters
        ----------
        source : str
            The source element.
        sink : str
            The sink element.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self.RunScriptCommand(f"PVSetSourceAndSink({source}, {sink});")

    def PVStartOver(self):
        """
        Start over the PV study.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self.RunScriptCommand("PVStartOver;")

    def PVWriteInadequateVoltages(self, filename: str, append: bool = True, inadequate_type: str = "LOW"):
        """
        Save PV Inadequate Voltages.

        Parameters
        ----------
        filename : str
            The file to write to.
        append : bool, optional
            If True, appends to the file. Defaults to True.
        inadequate_type : str, optional
            Type of inadequacy ("LOW", "HIGH", "BOTH"). Defaults to "LOW".

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'PVWriteInadequateVoltages("{filename}", {app}, {inadequate_type});')

    def PVWriteResultsAndOptions(self, filename: str, append: bool = True):
        """
        Writes out all information related to PV analysis.

        Parameters
        ----------
        filename : str
            The file to write to.
        append : bool, optional
            If True, appends to the file. Defaults to True.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'PVWriteResultsAndOptions("{filename}", {app});')

    def RefineModel(self, object_type: str, filter_name: str, action: str, tolerance: float):
        """
        Refine the system model to fix modeling idiosyncrasies.

        Parameters
        ----------
        object_type : str
            The type of object to refine.
        filter_name : str
            Filter to apply.
        action : str
            Action to perform.
        tolerance : float
            Tolerance value.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        filt = f'"{filter_name}"' if filter_name else ""
        return self.RunScriptCommand(f'RefineModel({object_type}, {filt}, {action}, {tolerance});')
