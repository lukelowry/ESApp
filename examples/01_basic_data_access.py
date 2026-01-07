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

    # Retrieve generator MW output
    gens = wb[Gen, 'GenMW']
    print("\nGenerators:")
    print(gens.head())