"""
GIC Analysis
============

Calculating Geomagnetically Induced Currents (GIC) for a uniform electric field.
"""

from gridwb import GridWorkBench
import os

case_path = os.environ.get("SAW_TEST_CASE", "case.pwb")

if os.path.exists(case_path):
    wb = GridWorkBench(case_path)

    # Calculate GIC for a 1 V/km field pointing East (90 degrees)
    print("Calculating GIC...")
    wb.func.calculate_gic(max_field=1.0, direction=90.0)
    
    # Retrieve transformer GICs
    xfmr_gic = wb.gic.gictool().gicxfmrs[['BusNum', 'BusNum:1', 'GICAmps']]
    print(xfmr_gic.head())