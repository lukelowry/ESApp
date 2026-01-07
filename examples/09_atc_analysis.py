"""
Available Transfer Capability (ATC) Studies
===========================================

Determining the maximum power that can be transferred between two areas 
without violating system limits, considering N-1 contingencies.
"""

from gridwb import GridWorkBench
from gridwb.grid.components import Area
import os

case_path = os.environ.get("SAW_TEST_CASE", "case.pwb")

if os.path.exists(case_path):
    wb = GridWorkBench(case_path)

    areas = wb[Area, ['AreaNum', 'AreaName']]
    if len(areas) >= 2:
        seller = f"[AREA {areas.iloc[0]['AreaNum']}]"
        buyer = f"[AREA {areas.iloc[1]['AreaNum']}]"
        
        wb.esa.SetData("ATC_Options", ["Method"], ["IteratedLinearThenFull"])
        wb.esa.DetermineATC(seller, buyer, do_distributed=False, do_multiple_scenarios=False)

        results = wb.esa.GetATCResults(["MaxFlow", "LimitingContingency", "LimitingElement"])
        
        print("\nATC Results:")
        if not results.empty:
            atc_val = results.iloc[0]['MaxFlow']
            limit_ctg = results.iloc[0]['LimitingContingency']
            limit_el = results.iloc[0]['LimitingElement']
            
            print(f"Maximum Transfer Capability: {atc_val:.2f} MW")
            print(f"Limiting Contingency: {limit_ctg}")
            print(f"Limiting Element: {limit_el}")
        else:
            print("No ATC results found. Check if transfer is feasible.")
    else:
        print("Not enough areas in the case to perform a transfer study.")

else:
    print(f"Case file not found at {case_path}. Please set SAW_TEST_CASE environment variable.")