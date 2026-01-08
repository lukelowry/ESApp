"""Sensitivity analysis specific functions."""
from ._helpers import create_object_string


class SensitivityMixin:
    """Mixin for sensitivity analysis functions."""

    def CalculateFlowSense(self, flow_element: str, flow_type: str):
        """Calculates the sensitivity of the flow of a line or interface to bus injections.

        This method determines how much the flow on a specific element changes
        for a unit change in injection at each bus.

        Parameters
        ----------
        flow_element : str
            The flow element string (e.g., '[BRANCH 1 2 1]', '[INTERFACE "name"]').
        flow_type : str
            The type of flow to calculate sensitivity for ("MW", "MVAR", "MVA").

        Returns
        -------
        str
            The result of the script command.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f'CalculateFlowSense({flow_element}, {flow_type});')

    def CalculatePTDF(self, seller: str, buyer: str, method: str = "DC"):
        """Calculates the PTDF (Power Transfer Distribution Factor) values between a seller and a buyer.

        PTDFs indicate how much power flow on a specific branch changes for a
        unit power transfer between two points in the system.

        Parameters
        ----------
        seller : str
            The seller (source) object string (e.g., '[AREA "Top"]', '[BUS 7]').
        buyer : str
            The buyer (sink) object string (e.g., '[AREA "Bottom"]', '[BUS 8]').
        method : str, optional
            The linear method to use for calculation ("AC", "DC", "DCPS"). Defaults to "DC".

        Returns
        -------
        str
            The result of the script command.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f'CalculatePTDF({seller}, {buyer}, {method});')

    def CalculateLODF(self, branch: str, method: str = "DC", post_closure_lcdf: str = ""):
        """Calculates LODF (Line Outage Distribution Factors) for a specified branch outage.

        LODFs quantify how much power flow on other branches changes when a
        specific branch is outaged.

        Parameters
        ----------
        branch : str
            The branch element string to outage/close (e.g., '[BRANCH 1 2 1]').
        method : str, optional
            The linear method to use for calculation ("DC", "DCPS"). Defaults to "DC".
        post_closure_lcdf : str, optional
            Optional parameter ("YES" or "NO") to include LCDF (Line Closure Distribution Factor)
            calculation relative to post-closure flow. Defaults to "".

        Returns
        -------
        str
            The result of the script command.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        args = f'{branch}, {method}'
        if post_closure_lcdf:
            args += f', {post_closure_lcdf}'
        return self.RunScriptCommand(f'CalculateLODF({args});')

    def CalculateLODFAdvanced(self, include_phase_shifters: bool, file_type: str, max_columns: int, min_lodf: float, number_format: str, decimal_points: int, only_increasing: bool, filename: str, include_islanding: bool = True):
        """Performs an advanced LODF calculation with various output and filtering options.

        Parameters
        ----------
        include_phase_shifters : bool
            If True, includes phase shifters in the LODF calculation.
        file_type : str
            The output file type (e.g., "CSV", "AUX").
        max_columns : int
            Maximum number of columns in the output file.
        min_lodf : float
            Minimum LODF value to report.
        number_format : str
            Format for numeric output (e.g., "DECIMAL", "PERCENT").
        decimal_points : int
            Number of decimal points for numeric output.
        only_increasing : bool
            If True, only reports increasing LODF values.
        filename : str
            The path to the output file.
        include_islanding : bool, optional
            If True, includes islanding information in the output. Defaults to True.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        ValueError
            If `file_type` or `number_format` are invalid.

        Notes
        -----
        This method corresponds to the `CalculateLODFAdvanced` script command in PowerWorld.
        """
        ips = "YES" if include_phase_shifters else "NO"
        inc = "YES" if only_increasing else "NO"
        isl = "YES" if include_islanding else "NO"
        cmd = f'CalculateLODFAdvanced({ips}, {file_type}, {max_columns}, {min_lodf}, {number_format}, {decimal_points}, {inc}, "{filename}", {isl});'
        return self.RunScriptCommand(cmd)

    def CalculateLODFScreening(self, filter_process: str, filter_monitor: str, include_phase_shifters: bool, include_open_lines: bool, use_lodf_threshold: bool, lodf_threshold: float, use_overload_threshold: bool, overload_low: float, overload_high: float, do_save_file: bool, file_location: str, custom_high_lodf: int = 0, custom_high_lodf_line: int = 0, custom_high_overload: int = 0, custom_high_overload_line: int = 0, do_use_ctg_name: bool = False, custom_orig_ctg_name: int = 0):
        """Performs LODF Screening calculation to identify critical outages and overloads.

        This method is used to quickly assess the impact of numerous outages based on
        LODF and overload thresholds.

        Parameters
        ----------
        filter_process : str
            Filter for branches to outage/process.
        filter_monitor : str
            Filter for branches to monitor.
        include_phase_shifters : bool
            If True, includes phase shifters in the calculation.
        include_open_lines : bool
            If True, includes initially open lines in the calculation.
        use_lodf_threshold : bool
            If True, applies an LODF threshold for reporting.
        lodf_threshold : float
            The LODF threshold value.
        use_overload_threshold : bool
            If True, applies an overload threshold for reporting.
        overload_low : float
            Lower bound for overload threshold (e.g., 100 for 100% limit).
        overload_high : float
            Upper bound for overload threshold.
        do_save_file : bool
            If True, saves the screening results to a file.
        file_location : str
            The path to the output file if `do_save_file` is True.
        custom_high_lodf : int, optional
            Custom field index for high LODF. Defaults to 0.
        custom_high_lodf_line : int, optional
            Custom field index for high LODF line. Defaults to 0.
        custom_high_overload : int, optional
            Custom field index for high overload. Defaults to 0.
        custom_high_overload_line : int, optional
            Custom field index for high overload line. Defaults to 0.
        do_use_ctg_name : bool, optional
            If True, uses contingency name in output. Defaults to False.
        custom_orig_ctg_name : int, optional
            Custom field index for original contingency name. Defaults to 0.

        Returns
        -------
        None
        """
        ips = "YES" if include_phase_shifters else "NO"
        iol = "YES" if include_open_lines else "NO"
        ult = "YES" if use_lodf_threshold else "NO"
        uot = "YES" if use_overload_threshold else "NO"
        dsf = "YES" if do_save_file else "NO"
        duc = "YES" if do_use_ctg_name else "NO"
        cmd = f'CalculateLODFScreening({filter_process}, {filter_monitor}, {ips}, {iol}, {ult}, {lodf_threshold}, {uot}, {overload_low}, {overload_high}, {dsf}, "{file_location}", {custom_high_lodf}, {custom_high_lodf_line}, {custom_high_overload}, {custom_high_overload_line}, {duc}, {custom_orig_ctg_name});'
        return self.RunScriptCommand(cmd)

    def CalculateShiftFactors(self, flow_element: str, direction: str, transactor: str, method: str = "DC"):
        """Calculates Shift Factor Sensitivity values (formerly known as TLRs).

        Shift Factors quantify how much power flow on a specific element changes
        for a unit injection change at a particular transactor (e.g., area, bus).

        Parameters
        ----------
        flow_element : str
            The monitored flow element string (e.g., '[BRANCH 1 2 1]', '[INTERFACE "name"]').
        direction : str
            The direction of transfer ("BUYER" or "SELLER").
        transactor : str
            The transactor object string (e.g., '[AREA "Top"]', '[BUS 7]').
        method : str, optional
            The linear method to use for calculation ("AC", "DC", "DCPS"). Defaults to "DC".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(
            f'CalculateShiftFactors({flow_element}, {direction}, {transactor}, {method});'
        )

    def CalculateShiftFactorsMultipleElement(self, type_element: str, which_element: str, direction: str, transactor: str, method: str = "DC"):
        """Calculates Shift Factor Sensitivity values for multiple elements.

        This method extends `CalculateShiftFactors` to apply the calculation
        across a set of elements defined by `type_element` and `which_element`.

        Parameters
        ----------
        type_element : str
            The type of element to calculate shift factors for (e.g., "BRANCH", "INTERFACE").
        which_element : str
            A PowerWorld filter name or "ALL" to specify which elements to include.
        direction : str
            The direction of transfer ("BUYER" or "SELLER").
        transactor : str
            The transactor object string (e.g., '[AREA "Top"]', '[BUS 7]').
        method : str, optional
            The linear method to use for calculation ("AC", "DC", "DCPS"). Defaults to "DC".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f'CalculateShiftFactorsMultipleElement({type_element}, {which_element}, {direction}, {transactor}, {method});')

    def CalculateLODFMatrix(
        self,
        which_ones: str,
        filter_process: str,
        filter_monitor: str,
        monitor_only_closed: bool = True,
        linear_method: str = "DC",
        filter_monitor_interface: str = "",
        post_closure_lcdf: bool = True,
    ):
        """Calculates the Line Outage Distribution Factors (LODF) matrix.

        This method generates a matrix showing the impact of a set of outages/closures
        on a set of monitored elements.

        Parameters
        ----------
        which_ones : str
            Specifies whether to calculate for "OUTAGES" or "CLOSURES".
        filter_process : str
            A PowerWorld filter name for branches to outage/close.
        filter_monitor : str
            A PowerWorld filter name for branches to monitor.
        monitor_only_closed : bool, optional
            If True, only monitors initially closed branches. Defaults to True.
        linear_method : str, optional
            The linear method to use ("DC" or "DCPS"). Defaults to "DC".
        filter_monitor_interface : str, optional
            A PowerWorld filter name for interfaces to monitor. Defaults to "".
        post_closure_lcdf : bool, optional
            If True, calculates LCDF (Line Closure Distribution Factor) relative
            to post-closure flow. Defaults to True.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        mon_closed = "YES" if monitor_only_closed else "NO"
        post_lcdf = "YES" if post_closure_lcdf else "NO"
        cmd = f"CalculateLODFMatrix({which_ones}, {filter_process}, {filter_monitor}, {mon_closed}, {linear_method}, {filter_monitor_interface}, {post_lcdf});"
        return self.RunScriptCommand(cmd)

    def CalculateVoltToTransferSense(
        self, seller: str, buyer: str, transfer_type: str = "P", turn_off_avr: bool = False
    ):
        """Calculates the sensitivity of bus voltage to power transfer between a seller and buyer.

        This helps in understanding voltage impacts of inter-area power exchanges.

        Parameters
        ----------
        seller : str
            The seller (source) object string (e.g., '[AREA "Top"]', '[BUS 7]').
        buyer : str
            The buyer (sink) object string (e.g., '[AREA "Bottom"]', '[BUS 8]').
        transfer_type : str, optional
            The type of transfer ("P" for active power, "Q" for reactive power).
            Defaults to "P".
        turn_off_avr : bool, optional
            If True, turns off Automatic Voltage Regulators (AVRs) during the calculation.
            Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        avr = "YES" if turn_off_avr else "NO"
        return self.RunScriptCommand(f"CalculateVoltToTransferSense({seller}, {buyer}, {transfer_type}, {avr});")

    def CalculateLossSense(self, function_type: str, area_ref: str = "NO", island_ref: str = "EXISTING"):
        """Calculates loss sensitivity at each bus.

        Loss sensitivity indicates how much system losses change for a unit
        injection change at each bus.

        Parameters
        ----------
        function_type : str
            The type of function for loss calculation (e.g., "AREA", "ZONE", "BUS").
        area_ref : str, optional
            Area reference for calculation ("NO", or an area name). Defaults to "NO".
        island_ref : str, optional
            Island reference for calculation ("EXISTING", or an island name). Defaults to "EXISTING".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f'CalculateLossSense({function_type}, {area_ref}, {island_ref});')

    def LineLoadingReplicatorCalculate(self, flow_element: str, injection_group: str, agc_only: bool, desired_flow: float, implement: bool, linear_method: str = "DC", use_load_min_max: bool = True, max_mult: float = 1.0, min_mult: float = 1.0):
        """Calculates injection changes required to alter a line flow to a desired value.

        This tool helps in determining how to adjust generation or load to achieve
        a target flow on a specific element.

        Parameters
        ----------
        flow_element : str
            The flow element string (e.g., '[BRANCH 1 2 1]').
        injection_group : str
            The injection group string (e.g., '[INJECTIONGROUP "GenGroup"]').
        agc_only : bool
            If True, only considers generators participating in AGC for adjustments.
        desired_flow : float
            The desired flow value on the `flow_element`.
        implement : bool
            If True, immediately implements the calculated injection changes.
        linear_method : str, optional
            The linear method to use ("DC", "AC"). Defaults to "DC".
        use_load_min_max : bool, optional
            If True, respects load min/max limits during adjustments. Defaults to True.
        max_mult : float, optional
            Maximum multiplier for adjustments. Defaults to 1.0.
        min_mult : float, optional
            Minimum multiplier for adjustments. Defaults to 1.0.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        agc = "YES" if agc_only else "NO"
        imp = "YES" if implement else "NO"
        ulmm = "YES" if use_load_min_max else "NO"
        cmd = f'LineLoadingReplicatorCalculate({flow_element}, {injection_group}, {agc}, {desired_flow}, {imp}, {linear_method}, {ulmm}, {max_mult}, {min_mult});'
        return self.RunScriptCommand(cmd)

    def LineLoadingReplicatorImplement(self):
        """Applies the changes calculated by the Line Loading Replicator.

        This action modifies the system based on the previously determined
        injection adjustments.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand("LineLoadingReplicatorImplement;")

    def CalculateTapSense(self, filter_name: str = ""):
        """Forces voltage to tap sensitivity calculation.

        This determines how bus voltages respond to changes in transformer tap settings.

        Parameters
        ----------
        filter_name : str, optional
            A PowerWorld filter name to apply to transformers. Defaults to an empty string (all).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f'CalculateTapSense("{filter_name}");')

    def CalculateVoltSelfSense(self, filter_name: str = ""):
        """Calculates the sensitivity of a bus's voltage to injections at the same bus.

        Parameters
        ----------
        filter_name : str, optional
            A PowerWorld filter name to apply to buses. Defaults to an empty string (all).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f'CalculateVoltSelfSense("{filter_name}");')

    def CalculateVoltSense(self, bus_num: int):
        """Calculates the sensitivity of a bus's voltage to injections at all buses.

        Parameters
        ----------
        bus_num : int
            The bus number for which to calculate voltage sensitivity.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        bus_str = create_object_string("Bus", bus_num)
        return self.RunScriptCommand(f'CalculateVoltSense({bus_str});')

    def SetSensitivitiesAtOutOfServiceToClosest(self, filter_name: str = "", branch_dist_meas: str = ""):
        """Populates sensitivity values at out-of-service buses by interpolating from the closest in-service buses.

        Parameters
        ----------
        filter_name : str, optional
            A PowerWorld filter name to apply to buses. Defaults to an empty string (all).
        branch_dist_meas : str, optional
            The branch distance measurement to use (e.g., "X", "R", "Z"). Defaults to "".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = f'"{filter_name}"' if filter_name else ""
        return self.RunScriptCommand(f'SetSensitivitiesAtOutOfServiceToClosest({filt}, {branch_dist_meas});')

    def CalculatePTDFMultipleDirections(self, store_branches: bool = True, store_interfaces: bool = True, method: str = "DC"):
        """Calculates PTDF values between all directions specified in the case.

        Parameters
        ----------
        store_branches : bool, optional
            If True, stores PTDFs for branches. Defaults to True.
        store_interfaces : bool, optional
            If True, stores PTDFs for interfaces. Defaults to True.
        method : str, optional
            The linear method to use for calculation ("DC", "AC", "DCPS"). Defaults to "DC".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        sb = "YES" if store_branches else "NO"
        si = "YES" if store_interfaces else "NO"
        return self.RunScriptCommand(f'CalculatePTDFMultipleDirections({sb}, {si}, {method});')