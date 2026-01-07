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

    Y = wb.ybus()
    print(f"Y-Bus shape: {Y.shape}, non-zeros: {Y.nnz}")

    J = wb.io.esa.get_jacobian()
    print(f"Jacobian shape: {J.shape}")
    
    density = Y.nnz / (Y.shape[0] * Y.shape[1])
    print(f"Y-Bus density: {density:.4%}")

    import numpy as np
    from scipy.sparse.linalg import spsolve

    dP = np.zeros(J.shape[0])
    dP[1] = 0.01

    dX = spsolve(J, dP)
    print(f"\nVoltage/Angle sensitivity for injection at Bus index 1 (first 5 elements):")
    print(dX[:5])

    A = wb.network.incidence()
    print(f"\nIncidence Matrix shape: {A.shape}")