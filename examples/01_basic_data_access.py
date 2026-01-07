"""
Basic Data Access
=================

This example demonstrates how to open a case and retrieve basic bus and generator data
using the indexing syntax.
"""

from gridwb import GridWorkBench
from gridwb.grid.components import Bus, Gen
import os

case_path = os.environ.get("SAW_TEST_CASE", "case.pwb")

if os.path.exists(case_path):
    wb = GridWorkBench(case_path)

    buses = wb[Bus, 'BusName']
    print("Buses in system:")
    print(buses.head())

    gens = wb[Gen, ['GenMW', 'GenStatus']]
    print("\nAll Generators:")
    print(gens.head())

    online_gens = gens[gens['GenStatus'] == 'Closed']
    print(f"\nOnline Generators ({len(online_gens)} total):")
    print(online_gens.head())

    bus_1_volt = wb[Bus, 1, 'BusPUVolt']
    print(f"\nBus 1 Voltage: {bus_1_volt} pu")

    current_mw = wb[Gen, 'GenMW']
    wb[Gen, 'GenMW'] = current_mw * 1.1
    new_mw = wb[Gen, 'GenMW']
    print(f"New total generation: {new_mw.sum():.2f} MW")
else:
    print(f"Case file not found at {case_path}. Please set SAW_TEST_CASE environment variable.")