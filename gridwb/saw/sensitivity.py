"""Sensitivity analysis specific functions."""


class SensitivityMixin:
    """Mixin for sensitivity analysis functions."""

    def CalculateFlowSense(self, flow_element: str, flow_type: str):
        """Calculates the sensitivity of the flow of a line or interface.

        :param flow_element: The flow element (e.g. '[BRANCH 1 2 1]' or '[INTERFACE "name"]').
        :param flow_type: The type of flow (MW, MVAR, MVA).
        """
        return self.RunScriptCommand(f'CalculateFlowSense({flow_element}, {flow_type});')

    def CalculatePTDF(self, seller: str, buyer: str, method: str = "DC"):
        """Calculates the PTDF values between a seller and a buyer.

        :param seller: The seller (source) (e.g. '[AREA "Top"]').
        :param buyer: The buyer (sink) (e.g. '[BUS 7]').
        :param method: Linear method (AC, DC, DCPS). Defaults to DC.
        """
        return self.RunScriptCommand(f'CalculatePTDF({seller}, {buyer}, {method});')

    def CalculateLODF(self, branch: str, method: str = "DC", post_closure_lcdf: str = ""):
        """Calculates Line Outage Distribution Factors.

        :param branch: The branch to outage/close (e.g. '[BRANCH 1 2 1]').
        :param method: Linear method (DC, DCPS). Defaults to DC.
        :param post_closure_lcdf: Optional YES/NO for LCDF calculation.
        """
        args = f'{branch}, {method}'
        if post_closure_lcdf:
            args += f', {post_closure_lcdf}'
        return self.RunScriptCommand(f'CalculateLODF({args});')

    def CalculateLODFAdvanced(self, include_phase_shifters: bool, file_type: str, max_columns: int, min_lodf: float, number_format: str, decimal_points: int, only_increasing: bool, filename: str, include_islanding: bool = True):
        """Perform advanced LODF calculation."""
        ips = "YES" if include_phase_shifters else "NO"
        inc = "YES" if only_increasing else "NO"
        isl = "YES" if include_islanding else "NO"
        cmd = f'CalculateLODFAdvanced({ips}, {file_type}, {max_columns}, {min_lodf}, {number_format}, {decimal_points}, {inc}, "{filename}", {isl});'
        return self.RunScriptCommand(cmd)

    def CalculateLODFScreening(self, filter_process: str, filter_monitor: str, include_phase_shifters: bool, include_open_lines: bool, use_lodf_threshold: bool, lodf_threshold: float, use_overload_threshold: bool, overload_low: float, overload_high: float, do_save_file: bool, file_location: str, custom_high_lodf: int = 0, custom_high_lodf_line: int = 0, custom_high_overload: int = 0, custom_high_overload_line: int = 0, do_use_ctg_name: bool = False, custom_orig_ctg_name: int = 0):
        """Perform LODF Screening calculation."""
        ips = "YES" if include_phase_shifters else "NO"
        iol = "YES" if include_open_lines else "NO"
        ult = "YES" if use_lodf_threshold else "NO"
        uot = "YES" if use_overload_threshold else "NO"
        dsf = "YES" if do_save_file else "NO"
        duc = "YES" if do_use_ctg_name else "NO"
        cmd = f'CalculateLODFScreening({filter_process}, {filter_monitor}, {ips}, {iol}, {ult}, {lodf_threshold}, {uot}, {overload_low}, {overload_high}, {dsf}, "{file_location}", {custom_high_lodf}, {custom_high_lodf_line}, {custom_high_overload}, {custom_high_overload_line}, {duc}, {custom_orig_ctg_name});'
        return self.RunScriptCommand(cmd)

    def CalculateShiftFactors(self, flow_element: str, direction: str, transactor: str, method: str = "DC"):
        """Calculates Shift Factor Sensitivity values (formerly CalculateTLR).

        :param flow_element: The monitored flow element (e.g. '[BRANCH 1 2 1]').
        :param direction: BUYER or SELLER.
        :param transactor: The transactor (e.g. '[AREA "Top"]').
        :param method: Linear method (AC, DC, DCPS). Defaults to DC.
        """
        return self.RunScriptCommand(
            f'CalculateShiftFactors({flow_element}, {direction}, {transactor}, {method});'
        )

    def CalculateShiftFactorsMultipleElement(self, type_element: str, which_element: str, direction: str, transactor: str, method: str = "DC"):
        """Calculates Shift Factor Sensitivity values for multiple elements."""
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
        """Calculate Line Outage Distribution Factors matrix.

        :param which_ones: OUTAGES or CLOSURES.
        :param filter_process: Filter for branches to outage/close.
        :param filter_monitor: Filter for branches to monitor.
        :param monitor_only_closed: Monitor only closed branches.
        :param linear_method: DC or DCPS.
        :param filter_monitor_interface: Filter for interfaces to monitor.
        :param post_closure_lcdf: Calculate LCDF relative to post-closure flow.
        """
        mon_closed = "YES" if monitor_only_closed else "NO"
        post_lcdf = "YES" if post_closure_lcdf else "NO"
        cmd = f"CalculateLODFMatrix({which_ones}, {filter_process}, {filter_monitor}, {mon_closed}, {linear_method}, {filter_monitor_interface}, {post_lcdf});"
        return self.RunScriptCommand(cmd)

    def CalculateVoltToTransferSense(
        self, seller: str, buyer: str, transfer_type: str = "P", turn_off_avr: bool = False
    ):
        """Calculates sensitivity of bus voltage to power transfer."""
        avr = "YES" if turn_off_avr else "NO"
        return self.RunScriptCommand(f"CalculateVoltToTransferSense({seller}, {buyer}, {transfer_type}, {avr});")

    def CalculateLossSense(self, function_type: str, area_ref: str = "NO", island_ref: str = "EXISTING"):
        """Calculates loss sensitivity at each bus."""
        return self.RunScriptCommand(f'CalculateLossSense({function_type}, {area_ref}, {island_ref});')

    def LineLoadingReplicatorCalculate(self, flow_element: str, injection_group: str, agc_only: bool, desired_flow: float, implement: bool, linear_method: str = "DC", use_load_min_max: bool = True, max_mult: float = 1.0, min_mult: float = 1.0):
        """Calculates injection changes to alter a line flow."""
        agc = "YES" if agc_only else "NO"
        imp = "YES" if implement else "NO"
        ulmm = "YES" if use_load_min_max else "NO"
        cmd = f'LineLoadingReplicatorCalculate({flow_element}, {injection_group}, {agc}, {desired_flow}, {imp}, {linear_method}, {ulmm}, {max_mult}, {min_mult});'
        return self.RunScriptCommand(cmd)

    def LineLoadingReplicatorImplement(self):
        """Applies the changes in the injection change list."""
        return self.RunScriptCommand("LineLoadingReplicatorImplement;")

    def CalculateTapSense(self, filter_name: str = ""):
        """Forces voltage to tap sensitivity calculation."""
        return self.RunScriptCommand(f'CalculateTapSense("{filter_name}");')

    def CalculateVoltSelfSense(self, filter_name: str = ""):
        """Calculates sensitivity of a bus's voltage to injections at the same bus."""
        return self.RunScriptCommand(f'CalculateVoltSelfSense("{filter_name}");')

    def CalculateVoltSense(self, bus_num: int):
        """Calculates sensitivity of a bus's voltage to injections at all buses."""
        return self.RunScriptCommand(f'CalculateVoltSense([BUS {bus_num}]);')

    def SetSensitivitiesAtOutOfServiceToClosest(self, filter_name: str = "", branch_dist_meas: str = ""):
        """Populate sensitivity values at out-of-service buses."""
        filt = f'"{filter_name}"' if filter_name else ""
        return self.RunScriptCommand(f'SetSensitivitiesAtOutOfServiceToClosest({filt}, {branch_dist_meas});')

    def CalculatePTDFMultipleDirections(self, store_branches: bool = True, store_interfaces: bool = True, method: str = "DC"):
        """Calculate PTDF values between all directions specified in the case."""
        sb = "YES" if store_branches else "NO"
        si = "YES" if store_interfaces else "NO"
        return self.RunScriptCommand(f'CalculatePTDFMultipleDirections({sb}, {si}, {method});')