"""
Security Constrained OPF (SCOPF)
===============================
Finds least-cost dispatch satisfying base-case and N-1 constraints.
"""
from gridwb import GridWorkBench
from gridwb.grid.components import Area, Gen
import os

case_path = os.environ.get("SAW_TEST_CASE", "case.pwb")

if os.path.exists(case_path):
    # %% Setup SCOPF
    wb = GridWorkBench(case_path)
    wb.esa.InitializePrimalLP()
    wb.func.auto_insert_contingencies()
    
    # %% Solve
    wb.esa.SolveFullSCOPF()

    # %% Results
    if wb.io.esa.is_converged():
        area_data = wb[Area, ['AreaName', 'AreaGenCost', 'AreaLambda']]
        gens = wb[Gen, ['BusNum', 'GenID', 'GenMW', 'GenPMax', 'GenPMin']]
        at_max = gens[gens['GenMW'] >= gens['GenPMax'] - 0.1]
        
        print(f"SCOPF Converged. Total Cost: ${area_data['AreaGenCost'].sum():,.2f}/hr")
        print(f"Generators at PMax: {len(at_max)}")