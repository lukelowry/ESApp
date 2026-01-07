"""
Contingency Analysis
====================

Automating N-1 contingency analysis and retrieving results.
"""

from gridwb import GridWorkBench
import os

case_path = os.environ.get("SAW_TEST_CASE", "case.pwb")

if os.path.exists(case_path):
    wb = GridWorkBench(case_path)

    # Auto-insert N-1 contingencies for all lines
    wb.func.auto_insert_contingencies()
    
    # Solve all contingencies
    print("Solving contingencies...")
    wb.func.solve_contingencies()
    
    # Get mismatches to verify solutions
    mismatches = wb.func.mismatches()
    print(mismatches.head())