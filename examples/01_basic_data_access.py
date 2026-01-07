"""
Basic Data Access
=================
Demonstrates opening a case and retrieving component data using indexing.
"""
from gridwb import GridWorkBench
from gridwb.grid.components import Bus, Gen
import os

case_path = os.environ.get("SAW_TEST_CASE", "case.pwb")

if os.path.exists(case_path):
    # %% Initialize Workbench
    wb = GridWorkBench(case_path)

    # %% Retrieve Data
    buses = wb[Bus, 'BusName']
    gens = wb[Gen, ['GenMW', 'GenStatus']]
    online_gens = gens[gens['GenStatus'] == 'Closed']
    bus_1_volt = wb[Bus, 1, 'BusPUVolt']

    # %% Modify Data
    current_mw = wb[Gen, 'GenMW']
    wb[Gen, 'GenMW'] = current_mw * 1.1
    
    # %% Final State
    new_mw = wb[Gen, 'GenMW']
    print(f"Total generation scaled to: {new_mw.sum():.2f} MW")