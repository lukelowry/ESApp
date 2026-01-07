"""
Network Expansion and Topology Modification
===========================================

Demonstrates how to programmatically modify system topology by tapping 
existing lines and splitting buses. This is essential for planning 
studies where new substations or interconnections are evaluated.
"""

from gridwb import GridWorkBench
from gridwb.grid.components import Bus, Branch
import os

case_path = os.environ.get("SAW_TEST_CASE", "case.pwb")

if os.path.exists(case_path):
    wb = GridWorkBench(case_path)

    branches = wb[Branch, ['BusNum', 'BusNum:1', 'LineCircuit']]
    if not branches.empty:
        b = branches.iloc[0]
        new_bus_num = int(wb[Bus, 'BusNum'].max() + 100)
        
        wb.esa.TapTransmissionLine(
            [b['BusNum'], b['BusNum:1'], b['LineCircuit']],
            pos_along_line=50.0,
            new_bus_number=new_bus_num,
            new_bus_name="Tapped_Substation"
        )

    target_bus = 1
    split_bus_num = int(wb[Bus, 'BusNum'].max() + 1)
    
    wb.esa.SplitBus(
        target_bus, 
        new_bus_number=split_bus_num,
        insert_bus_tie_line=True,
        branch_device_type="Breaker"
    )

    wb.pflow()
    
    if wb.io.esa.is_converged():
        print("Power flow converged successfully.")
        print(f"Total buses now in system: {len(wb[Bus, :])}")
    else:
        print("Power flow failed after topology changes.")

else:
    print(f"Case file not found at {case_path}. Please set SAW_TEST_CASE environment variable.")