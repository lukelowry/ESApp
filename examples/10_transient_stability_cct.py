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

    # 1. Initialize Transient Stability
    # This prepares the dynamic models and calculates the initial state
    print("Initializing Transient Stability...")
    wb.esa.TSInitialize()

    # 2. Calculate Critical Clearing Time (CCT)
    # We'll pick a major transmission line and find how long a fault can persist
    branches = wb[Branch, ['BusNum', 'BusNum:1', 'LineCircuit']]
    if not branches.empty:
        b = branches.iloc[0]
        branch_str = f"[BRANCH {b['BusNum']} {b['BusNum:1']} {b['LineCircuit']}]"
        
        print(f"\nCalculating CCT for a 3-phase fault on {branch_str}...")
        # This command iteratively runs simulations to find the stability boundary
        wb.esa.TSCalculateCriticalClearTime(branch_str)
        
        # Retrieve the calculated CCT from the branch field
        cct_val = wb.esa.GetParametersSingleElement(
            "Branch", ["TSCritClearTime"], [b['BusNum'], b['BusNum:1'], b['LineCircuit']]
        )
        print(f"Critical Clearing Time: {cct_val:.4f} seconds ({cct_val*60:.1f} cycles)")

    # 3. Generate Transient Response Plots
    # We can automate the generation of JPG plots for specific contingencies
    print("\nGenerating transient response plots for 'Fault_at_Bus_1'...")
    
    # Note: This assumes a transient contingency named 'Fault_at_Bus_1' is defined
    # and that plots are configured in the PowerWorld case.
    try:
        wb.esa.TSAutoSavePlots(
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