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

    # Solve power flow and get complex voltages
    V = wb.pflow()
    
    print(f"Solved power flow for {len(V)} buses.")
    
    # Check for low voltage violations (< 0.95 pu)
    low_v = V[abs(V) < 0.95]
    print(f"Found {len(low_v)} low voltage violations.")