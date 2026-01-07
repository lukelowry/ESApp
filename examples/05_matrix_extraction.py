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

    # 1. Get the sparse Y-Bus matrix
    Y = wb.ybus()
    print(f"Y-Bus shape: {Y.shape}, non-zeros: {Y.nnz}")

    # 2. Get the power flow Jacobian
    J = wb.io.esa.get_jacobian()
    print(f"Jacobian shape: {J.shape}")
    
    # 3. Perform a simple sensitivity analysis
    # Let's say we want to see the impact of a small change in power injection
    import numpy as np
    from scipy.sparse.linalg import spsolve

    # Create a delta-P vector (1 MW injection at Bus 2)
    dP = np.zeros(J.shape[0])
    dP[1] = 0.01 # 0.01 pu

    # Solve J * dX = dP  => dX = J^-1 * dP
    dX = spsolve(J, dP)
    print(f"\nVoltage/Angle sensitivity for injection at Bus 2 (first 5 elements):")
    print(dX[:5])