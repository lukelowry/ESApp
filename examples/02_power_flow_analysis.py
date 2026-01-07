"""
Power Flow Analysis
===================

Solving the AC power flow and inspecting system voltages.
"""

from gridwb import GridWorkBench
import os

case_path = os.environ.get("SAW_TEST_CASE", "case.pwb")

if os.path.exists(case_path):
    wb = GridWorkBench(case_path)

    # 1. Solve power flow and get complex voltages
    V = wb.pflow()
    
    # 2. Check convergence status
    if wb.io.esa.is_converged():
        print(f"Power flow converged successfully for {len(V)} buses.")
    else:
        print("Power flow failed to converge!")
    
    # 3. Check for low voltage violations (< 0.95 pu)
    low_v = V[abs(V) < 0.95]
    if not low_v.empty:
        print(f"Found {len(low_v)} low voltage violations:")
        print(low_v)
    else:
        print("No low voltage violations found.")

    # 4. Extract and summarize results
    results = wb[Bus, ['BusPUVolt', 'BusAngle']]
    print("\nVoltage Summary:")
    print(results.describe())

    # 5. Find the bus with the lowest voltage
    min_v_bus = results['BusPUVolt'].idxmin()
    min_v_val = results['BusPUVolt'].min()
    print(f"\nLowest voltage at Bus {min_v_bus}: {min_v_val:.4f} pu")