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

    # 1. Identify Seller and Buyer
    areas = wb[Area, ['AreaNum', 'AreaName']]
    if len(areas) >= 2:
        seller = f"[AREA {areas.iloc[0]['AreaNum']}]"
        buyer = f"[AREA {areas.iloc[1]['AreaNum']}]"
        
        print(f"Calculating ATC from {areas.iloc[0]['AreaName']} to {areas.iloc[1]['AreaName']}...")

        # 2. Configure ATC Options (optional, uses defaults if not set)
        # We can set the solution method to 'Iterated Linear then Full' for accuracy
        wb.esa.SetData("ATC_Options", ["Method"], ["IteratedLinearThenFull"])

        # 3. Run the ATC Determination
        # This will ramp the transfer until a limit is hit in base case or any contingency
        wb.esa.DetermineATC(seller, buyer, do_distributed=False, do_multiple_scenarios=False)

        # 4. Retrieve and Display Results
        # The results are stored in the 'TransferLimiter' object type
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