"""
Transient Stability and Critical Clearing Time
==============================================

Automating the calculation of Critical Clearing Time (CCT) for a fault
and generating transient response plots to visualize system stability.
"""

from gridwb import GridWorkBench
from gridwb.grid.components import Branch, Bus
import os

case_path = os.environ.get("SAW_TEST_CASE", "case.pwb")

if os.path.exists(case_path):
    wb = GridWorkBench(case_path)

    wb.io.esa.TSInitialize()

    branches = wb[Branch, ['BusNum', 'BusNum:1', 'LineCircuit']]
    if not branches.empty:
        b = branches.iloc[0]
        branch_str = f"[BRANCH {b['BusNum']} {b['BusNum:1']} {b['LineCircuit']}]"
        
        wb.io.esa.TSCalculateCriticalClearTime(branch_str)
        
        # ParamList and Values must match length; include keys to identify the object
        fields = ["BusNum", "BusNum:1", "LineCircuit", "TSCritClearTime"]
        values = [b['BusNum'], b['BusNum:1'], b['LineCircuit'], ""]
        res = wb.io.esa.GetParametersSingleElement("Branch", fields, values)
        cct_val = res["TSCritClearTime"]
        print(f"Critical Clearing Time: {cct_val:.4f} seconds ({cct_val*60:.1f} cycles)")

    try:
        wb.io.esa.TSAutoSavePlots(
            plot_names=["Generator Frequencies", "Bus Voltages"],
            contingency_names=["Fault_at_Bus_1"],
            image_file_type="JPG",
            width=1280,
            height=720
        )
        print("Plots saved to the case's result directory.")
    except Exception as e:
        print(f"Plot generation skipped: {e}")

else:
    print(f"Case file not found at {case_path}. Please set SAW_TEST_CASE environment variable.")