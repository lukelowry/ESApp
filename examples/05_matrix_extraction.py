"""
Matrix Extraction
=================

Extracting system matrices (Y-Bus and Jacobian) for external mathematical analysis.
"""

from gridwb import GridWorkBench
import os

case_path = os.environ.get("SAW_TEST_CASE", "case.pwb")

if os.path.exists(case_path):
    wb = GridWorkBench(case_path)

    # Get the sparse Y-Bus matrix
    Y = wb.ybus()
    print(f"Y-Bus shape: {Y.shape}, non-zeros: {Y.nnz}")

    # Get the power flow Jacobian
    J = wb.io.esa.get_jacobian()
    print(f"Jacobian shape: {J.shape}")
    
    # J is a scipy sparse matrix, ready for linear algebra