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

    wb.func.auto_insert_contingencies()
    wb.func.solve_contingencies()
    
    violations = wb.func.get_contingency_violations()
    
    if not violations.empty:
        print(f"\nViolation Matrix Summary ({violations.shape[0]} CTGs, {violations.shape[1]} Violations):")
        print(violations.iloc[:5, :5])
        
        most_violations_ctg = violations.count(axis=1).idxmax()
        num_violations = violations.count(axis=1).max()
        print(f"\nContingency with most violations: {most_violations_ctg} ({num_violations} violations)")
    else:
        print("\nNo contingency violations found.")

    mismatches = wb.func.mismatches()
    print(mismatches.head())

    worst_ctg = mismatches.idxmax()
    print(f"\nWorst Contingency by Mismatch: {worst_ctg} ({mismatches.max():.2e})")

    wb.func.command("Contingency(CLEARALL);")