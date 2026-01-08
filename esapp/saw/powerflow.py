from typing import List, Union
import pandas as pd

from ._exceptions import PowerWorldError


class PowerflowMixin:

    def SolvePowerFlow(self, SolMethod: str = "RECTNEWT") -> None:
        """Solves the power flow using the specified solution method.

        Parameters
        ----------
        SolMethod : str, optional
            The solution method to use. Valid options include "RECTNEWT" (Rectangular Newton-Raphson),
            "POLARNEWT" (Polar Newton-Raphson), "GAUSSSEIDEL" (Gauss-Seidel), "FASTDEC" (Fast Decoupled),
            "ROBUST", and "DC". Defaults to "RECTNEWT".
        
        Returns
        -------
        None
        
        Raises
        ------
        PowerWorldError
            If the SimAuto call fails or the power flow does not converge.
        """
        script_command = f"SolvePowerFlow({SolMethod.upper()})"
        return self.RunScriptCommand(script_command)

    def ClearPowerFlowSolutionAidValues(self):
        """Clear power flow solution aid values.

        This is a wrapper for the ``ClearPowerFlowSolutionAidValues``
        script command. It is useful for clearing values set by
        features like "Find" that can interfere with subsequent
        analyses.
        """
        self.RunScriptCommand("ClearPowerFlowSolutionAidValues;")

    def ResetToFlatStart(self):
        """Resets all bus voltages to 1.0 per unit and angles to 0.

        This is a wrapper for the ``ResetToFlatStart`` script command.
        """
        self.RunScriptCommand("ResetToFlatStart();")

    def get_power_flow_results(
        self, ObjectType: str, additional_fields: Union[None, List[str]] = None
    ) -> Union[None, pd.DataFrame]:
        """Get standard power flow results for a given object type."""
        object_type = ObjectType.lower()
        try:
            field_list = self.POWER_FLOW_FIELDS[object_type]
            if additional_fields:
                field_list = field_list[:]
                field_list += additional_fields
        except KeyError as e:
            raise ValueError(f"Unsupported ObjectType for power flow results, {ObjectType}.") from e

        return self.GetParametersMultipleElement(ObjectType=object_type, ParamList=field_list)

    def SolvePowerFlowWithRetry(self, SolMethod: str = "RECTNEWT") -> None:
        """Run the SolvePowerFlow command, with a retry mechanism.

        If the first attempt to solve the power flow fails, this method
        will reset the case to a flat start and try one additional time.

        Parameters
        ----------
        SolMethod : str, optional
            The solution method to use (e.g., "RECTNEWT"). Defaults to "RECTNEWT".
        """
        try:
            self.SolvePowerFlow(SolMethod)
        except PowerWorldError:
            self.log.warning("Power flow failed, resetting to flat start and retrying.")
            self.ResetToFlatStart()
            self.SolvePowerFlow(SolMethod)

    def SetMVATolerance(self, tol: float = 0.1) -> None:
        """Sets the MVA Tolerance for Newton-Raphson convergence.
        
        Parameters
        ----------
        tol : float, optional
            The MVA tolerance value. Defaults to 0.1.
        """
        self.ChangeParametersSingleElement("Sim_Solution_Options", ["ConvergenceTol:2"], [str(tol)])

    def SetDoOneIteration(self, enable: bool = True) -> None:
        """Sets the 'Do One Iteration' power flow option.
        
        Parameters
        ----------
        enable : bool, optional
            If True, power flow will only perform one iteration. Defaults to True.
        """
        value = "YES" if enable else "NO"
        self.ChangeParametersSingleElement("Sim_Solution_Options", ["DoOneIteration"], [value])

    def SetInnerLoopCheckMVars(self, enable: bool = True) -> None:
        """Sets the 'Check Mvar Limits Immediately' power flow option.
        
        Parameters
        ----------
        enable : bool, optional
            If True, the inner loop of the power flow will check Mvar limits
            before proceeding to the outer loop. Defaults to True.
        """
        value = "YES" if enable else "NO"
        self.ChangeParametersSingleElement("Sim_Solution_Options", ["ChkVars"], [value])

    def GetMinPUVoltage(self) -> float:
        """Gets the minimum per-unit voltage magnitude in the case.
        
        Returns
        -------
        float
            The minimum p.u. voltage.
        """
        s = self.GetParametersSingleElement("PWCaseInformation", ["BusPUVolt:1"], [""])
        return float(s.iloc[0])

    def GetBusMismatches(self) -> pd.DataFrame:
        """Gets the complex power bus mismatches.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with bus identifiers and their P, Q, and S
            mismatches in MW, Mvar, and MVA.
        """
        key_fields = self.get_key_field_list("Bus")
        return self.GetParametersMultipleElement(
            "Bus", key_fields + ["BusMismatchP", "BusMismatchQ", "BusMismatchS"]
        )

    def UpdateIslandsAndBusStatus(self):
        """Updates islands and bus status without requiring a power flow solution."""
        return self.RunScriptCommand("UpdateIslandsAndBusStatus;")

    def ZeroOutMismatches(self, object_type: str = "BUSSHUNT"):
        """Forces mismatches to zero by changing bus shunts or loads."""
        return self.RunScriptCommand(f"ZeroOutMismatches({object_type});")

    def ConditionVoltagePockets(self, voltage_threshold: float, angle_threshold: float, filter_name: str = "ALL"):
        """Finds pockets of buses that may have bad initial voltage estimates."""
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE", "ALL"] else filter_name
        return self.RunScriptCommand(
            f"ConditionVoltagePockets({voltage_threshold}, {angle_threshold}, {filt});"
        )

    def EstimateVoltages(self, filter_name: str):
        """Estimates voltages and angles at buses meeting the filter."""
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE", "ALL"] else filter_name
        return self.RunScriptCommand(f'EstimateVoltages({filt});')

    def GenForceLDC_RCC(self, filter_name: str = ""):
        """Forces generators onto line drop / reactive current compensation."""
        return self.RunScriptCommand(f'GenForceLDC_RCC("{filter_name}");')

    def SaveGenLimitStatusAction(self, filename: str):
        """Saves Mvar information about generators in a text file."""
        return self.RunScriptCommand(f'SaveGenLimitStatusAction("{filename}");')

    def DiffCaseClearBase(self):
        """Clears the base case for the difference flows abilities."""
        return self.RunScriptCommand("DiffCaseClearBase;")

    def DiffCaseSetAsBase(self):
        """Sets the present case as the base case for difference flows."""
        return self.RunScriptCommand("DiffCaseSetAsBase;")

    def DiffCaseKeyType(self, key_type: str):
        """Changes the key type used when comparing fields."""
        return self.RunScriptCommand(f"DiffCaseKeyType({key_type});")

    def DiffCaseShowPresentAndBase(self, show: bool):
        """Toggles 'Show Present|Base in Difference and Change Mode'."""
        yn = "YES" if show else "NO"
        return self.RunScriptCommand(f"DiffCaseShowPresentAndBase({yn});")

    def DiffCaseMode(self, mode: str):
        """Changes the mode for difference flows (PRESENT, BASE, DIFFERENCE, CHANGE)."""
        return self.RunScriptCommand(f"DiffCaseMode({mode});")

    def DiffCaseRefresh(self):
        """Refreshes the linking between the base case and the present case."""
        return self.RunScriptCommand("DiffCaseRefresh;")

    def DiffCaseWriteCompleteModel(self, filename: str, append: bool = False, save_added: bool = True, save_removed: bool = True, save_both: bool = True, key_fields: str = "PRIMARY", export_format: str = "", use_area_zone: bool = False, use_data_maintainer: bool = False, assume_base_meet: bool = True, include_clear_pf_aids: bool = True, delete_branches_flip: bool = False):
        """Creates an auxiliary file with difference case information."""
        app = "YES" if append else "NO"
        sa = "YES" if save_added else "NO"
        sr = "YES" if save_removed else "NO"
        sb = "YES" if save_both else "NO"
        uaz = "YES" if use_area_zone else "NO"
        udm = "YES" if use_data_maintainer else "NO"
        abm = "YES" if assume_base_meet else "NO"
        icp = "YES" if include_clear_pf_aids else "NO"
        dbf = "YES" if delete_branches_flip else "NO"
        
        cmd = f'DiffCaseWriteCompleteModel("{filename}", {app}, {sa}, {sr}, {sb}, {key_fields}, "{export_format}", {uaz}, {udm}, {abm}, {icp}, {dbf});'
        return self.RunScriptCommand(cmd)

    def DiffCaseWriteBothEPC(self, filename: str, ge_file_type: str = "GE", use_area_zone: bool = False, base_area_zone_meet: bool = True, append: bool = False, export_format: str = "", use_data_maintainer: bool = False):
        """Saves elements in both base and present cases in GE EPC format."""
        uaz = "YES" if use_area_zone else "NO"
        baz = "YES" if base_area_zone_meet else "NO"
        app = "YES" if append else "NO"
        udm = "YES" if use_data_maintainer else "NO"
        return self.RunScriptCommand(f'DiffCaseWriteBothEPC("{filename}", {ge_file_type}, {uaz}, {baz}, {app}, "{export_format}", {udm});')

    def DiffCaseWriteNewEPC(self, filename: str, ge_file_type: str = "GE", use_area_zone: bool = False, base_area_zone_meet: bool = True, append: bool = False, use_data_maintainer: bool = False):
        """Saves new elements in GE EPC format."""
        uaz = "YES" if use_area_zone else "NO"
        baz = "YES" if base_area_zone_meet else "NO"
        app = "YES" if append else "NO"
        udm = "YES" if use_data_maintainer else "NO"
        return self.RunScriptCommand(f'DiffCaseWriteNewEPC("{filename}", {ge_file_type}, {uaz}, {baz}, {app}, {udm});')

    def DiffCaseWriteRemovedEPC(self, filename: str, ge_file_type: str = "GE", use_area_zone: bool = False, base_area_zone_meet: bool = True, append: bool = False, use_data_maintainer: bool = False):
        """Saves removed elements in GE EPC format."""
        uaz = "YES" if use_area_zone else "NO"
        baz = "YES" if base_area_zone_meet else "NO"
        app = "YES" if append else "NO"
        udm = "YES" if use_data_maintainer else "NO"
        return self.RunScriptCommand(f'DiffCaseWriteRemovedEPC("{filename}", {ge_file_type}, {uaz}, {baz}, {app}, {udm});')

    def DoCTGAction(self, action: str):
        """Applies a contingency action."""
        return self.RunScriptCommand(f'DoCTGAction({action});')

    def InterfacesCalculatePostCTGMWFlows(self):
        """Updates Interface MW Flow fields on Contingent Interfaces."""
        return self.RunScriptCommand("InterfacesCalculatePostCTGMWFlows;")

    def VoltageConditioning(self):
        """Perform voltage conditioning based on the Voltage Conditioning tool options."""
        return self.RunScriptCommand("VoltageConditioning;")