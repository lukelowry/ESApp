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

    # 1. Auto-insert N-1 contingencies for all lines
    print("Inserting N-1 contingencies...")
    wb.func.auto_insert_contingencies()
    
    # 2. Solve all contingencies
    print("Solving contingencies...")
    wb.func.solve_contingencies()
    
    # 3. Retrieve the violation matrix
    # This returns a DataFrame where rows are contingencies and columns are monitored elements
    violations = wb.func.get_contingency_violations()
    print(f"\nViolation Matrix Summary ({violations.shape[0]} CTGs, {violations.shape[1]} Violations):")
    print(violations.iloc[:5, :5]) # Show a snippet

    # 4. Get mismatches to verify solutions
    print("\nSolution Mismatches:")
    mismatches = wb.func.mismatches()
    print(mismatches.head())

    # 5. Find the worst contingency (highest mismatch or most violations)
    worst_ctg = mismatches.idxmax()
    print(f"\nWorst Contingency by Mismatch: {worst_ctg} ({mismatches.max():.2e})")