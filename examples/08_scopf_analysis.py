"""
Security Constrained OPF (SCOPF)
===============================

Solving a full Security Constrained Optimal Power Flow. This analysis 
finds the least-cost generation dispatch that satisfies both base-case 
constraints and all N-1 contingency constraints simultaneously.
"""

from gridwb import GridWorkBench
from gridwb.grid.components import Area, Gen
import os

case_path = os.environ.get("SAW_TEST_CASE", "case.pwb")

if os.path.exists(case_path):
    wb = GridWorkBench(case_path)

    # 1. Initialize the OPF structures
    print("Initializing Primal LP OPF...")
    wb.esa.InitializePrimalLP()

    # 2. Ensure contingencies are defined
    # SCOPF needs a set of contingencies to enforce security constraints
    print("Auto-inserting N-1 contingencies for SCOPF...")
    wb.func.auto_insert_contingencies()

    # 3. Solve the Full SCOPF
    # This uses the Primal LP method to optimize dispatch while monitoring contingencies
    print("Solving Full SCOPF (this involves multiple iterations)...")
    wb.esa.SolveFullSCOPF()

    # 4. Analyze Results
    if wb.io.esa.is_converged():
        print("\nSCOPF Converged.")
        
        # Extract area-wise generation costs
        area_data = wb[Area, ['AreaName', 'AreaGenCost', 'AreaLambda']]
        print("\nEconomic Summary by Area:")
        print(area_data)
        
        # Check for binding constraints (generators at limits)
        gens = wb[Gen, ['BusNum', 'GenID', 'GenMW', 'GenPMax', 'GenPMin']]
        at_max = gens[gens['GenMW'] >= gens['GenPMax'] - 0.1]
        print(f"\nGenerators at PMax: {len(at_max)}")
        if not at_max.empty:
            print(at_max[['BusNum', 'GenMW', 'GenPMax']])
            
        print(f"\nTotal System Operating Cost: ${area_data['AreaGenCost'].sum():,.2f}/hr")
    else:
        print("SCOPF failed to converge. Check for infeasible constraints.")

else:
    print(f"Case file not found at {case_path}. Please set SAW_TEST_CASE environment variable.")