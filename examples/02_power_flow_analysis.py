"""
Power Flow Analysis
===================
Solves the AC power flow and inspects system voltages for violations.
"""
from gridwb import GridWorkBench
from gridwb.grid.components import Bus
import os

case_path = os.environ.get("SAW_TEST_CASE", "case.pwb")

if os.path.exists(case_path):
    # %% Solve Power Flow
    wb = GridWorkBench(case_path)
    V = wb.pflow()
    
    # %% Check Violations
    low_v = V[abs(V) < 0.95]

    # %% Summary Statistics
    results = wb[Bus, ['BusPUVolt', 'BusAngle']]
    min_v_bus = results['BusPUVolt'].idxmin()
    print(f"Convergence: {wb.io.esa.is_converged()}")
    print(f"Min Voltage: {results['BusPUVolt'].min():.4f} at Bus {min_v_bus}")