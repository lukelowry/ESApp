"""
Basic Data Access
=================

This example demonstrates how to open a case and retrieve basic bus and generator data
using the indexing syntax.
"""

from gridwb import GridWorkBench
from gridwb.grid.components import Bus, Gen
import os

# Use the test case from environment or a dummy path
case_path = os.environ.get("SAW_TEST_CASE", "case.pwb")

if os.path.exists(case_path):
    wb = GridWorkBench(case_path)

    # Retrieve all bus numbers and names
    buses = wb[Bus, 'BusName']
    print("Buses in system:")
    print(buses.head())

    # Retrieve generator MW output and status
    gens = wb[Gen, ['GenMW', 'GenStatus']]
    print("\nAll Generators:")
    print(gens.head())

    # Filter for online generators only
    online_gens = gens[gens['GenStatus'] == 'Closed']
    print(f"\nOnline Generators ({len(online_gens)} total):")
    print(online_gens.head())

    # Accessing data for a specific bus (e.g., Bus 1)
    bus_1_volt = wb[Bus, 1, 'BusPUVolt']
    print(f"\nBus 1 Voltage: {bus_1_volt} pu")

    # Modifying data: Increase all generator setpoints by 10%
    print("\nIncreasing all generator setpoints by 10%...")
    current_mw = wb[Gen, 'GenMW']
    wb[Gen, 'GenMW'] = current_mw * 1.1
    
    # Verify the change
    new_mw = wb[Gen, 'GenMW']
    print(f"New total generation: {new_mw.sum():.2f} MW")

    # Closing the case (optional, workbench handles cleanup)
    # wb.close()
else:
    print(f"Case file not found at {case_path}. Please set SAW_TEST_CASE environment variable.")