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

    V = wb.pflow()
    
    if wb.io.esa.is_converged():
        print(f"Power flow converged successfully for {len(V)} buses.")
    else:
        print("Power flow failed to converge!")
    
    low_v = V[abs(V) < 0.95]
    if not low_v.empty:
        print(f"Found {len(low_v)} low voltage violations:")
        print(low_v)
    else:
        print("No low voltage violations found.")

    results = wb[Bus, ['BusPUVolt', 'BusAngle']]
    print("\nVoltage Summary:")
    print(results.describe())

    min_v_bus = results['BusPUVolt'].idxmin()
    min_v_val = results['BusPUVolt'].min()
    print(f"\nLowest voltage at Bus {min_v_bus}: {min_v_val:.4f} pu")