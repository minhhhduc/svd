"""
Test script for SVD_parallel.dll
Tests SVD decomposition via ctypes
"""

import ctypes
import numpy as np
from pathlib import Path
import os
import sys

# Add bin directory to PATH for DLL dependencies
bin_path = Path(__file__).parent.parent / "bin"
os.environ['PATH'] = str(bin_path) + os.pathsep + os.environ['PATH']

# Load DLL with dependencies
dll_path = bin_path / "SVD_parallel.dll"
try:
    lib = ctypes.CDLL(str(dll_path), mode=ctypes.RTLD_GLOBAL)
except OSError as e:
    print(f"Error loading DLL from {dll_path}: {e}")
    print(f"Current PATH: {os.environ['PATH']}")
    sys.exit(1)

# Define C function signature
# void svd_decomposition_parallel(int m, int n, const double* A, double* U, double* S, double* V, int num_threads)
svd_decomposition_parallel = lib.svd_decomposition_parallel
svd_decomposition_parallel.argtypes = [
    ctypes.c_int,  # m
    ctypes.c_int,  # n
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # A
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # U
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # S
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # V
    ctypes.c_int  # num_threads
]
svd_decomposition_parallel.restype = None

def test_svd_3x2():
    """Test SVD decomposition on 3x2 matrix"""
    print("\n" + "="*70)
    print("Test 1: 3x2 Matrix SVD Decomposition")
    print("="*70)
    
    m, n = 3, 2
    A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ], dtype=np.float64, order='C')
    
    print("\nInput matrix A (3x2):")
    print(A)
    
    # Allocate output arrays
    U = np.zeros((m, n), dtype=np.float64)
    S = np.zeros(n, dtype=np.float64)
    V = np.zeros((n, n), dtype=np.float64)
    
    # Call C function with 4 threads
    svd_decomposition_parallel(m, n, A, U, S, V, 4)
    
    print("\nComputed singular values:", S)
    print("\nU matrix (3x2):")
    print(U)
    print("\nV matrix (2x2):")
    print(V)
    
    # Reconstruct A
    A_recon = U @ np.diag(S) @ V.T
    print("\nReconstructed A:")
    print(A_recon)
    
    recon_error = np.linalg.norm(A - A_recon)
    print(f"\nReconstruction error: {recon_error:.2e}")
    
    # Compare with NumPy
    U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
    print(f"NumPy singular values: {S_np}")
    
    if recon_error < 1e-10:
        print("✓ TEST PASSED")
        return True
    else:
        print("✗ TEST FAILED")
        return False

def test_svd_4x3():
    """Test SVD decomposition on 4x3 matrix"""
    print("\n" + "="*70)
    print("Test 2: 4x3 Matrix SVD Decomposition")
    print("="*70)
    
    m, n = 4, 3
    A = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ], dtype=np.float64, order='C')
    
    print("\nInput matrix A (4x3):")
    print(A)
    
    # Allocate output arrays
    U = np.zeros((m, n), dtype=np.float64)
    S = np.zeros(n, dtype=np.float64)
    V = np.zeros((n, n), dtype=np.float64)
    
    # Call C function
    svd_decomposition_parallel(m, n, A, U, S, V, 4)
    
    print("\nComputed singular values:", S)
    
    # Reconstruct A
    A_recon = U @ np.diag(S) @ V.T
    
    recon_error = np.linalg.norm(A - A_recon)
    print(f"Reconstruction error: {recon_error:.2e}")
    
    # Compare with NumPy
    U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
    print(f"NumPy singular values: {S_np}")
    
    # Calculate singular value error
    S_sorted = np.sort(S)[::-1]
    S_np_sorted = np.sort(S_np)[::-1]
    sv_error = np.linalg.norm(S_sorted - S_np_sorted)
    print(f"Singular value error vs NumPy: {sv_error:.2e}")
    
    if recon_error < 1e-9:
        print("✓ TEST PASSED")
        return True
    else:
        print("✗ TEST FAILED")
        return False

def test_svd_random():
    """Test SVD on random matrix"""
    print("\n" + "="*70)
    print("Test 3: Random 5x3 Matrix SVD Decomposition")
    print("="*70)
    
    m, n = 5, 3
    np.random.seed(42)
    A = np.random.randn(m, n)
    A = np.ascontiguousarray(A, dtype=np.float64)
    
    print("\nInput matrix A (5x3):")
    print(A)
    
    # Allocate output arrays
    U = np.zeros((m, n), dtype=np.float64)
    S = np.zeros(n, dtype=np.float64)
    V = np.zeros((n, n), dtype=np.float64)
    
    # Call C function
    svd_decomposition_parallel(m, n, A, U, S, V, 4)
    
    print("\nComputed singular values:", S)
    
    # Reconstruct A
    A_recon = U @ np.diag(S) @ V.T
    
    recon_error = np.linalg.norm(A - A_recon)
    print(f"\nReconstruction error: {recon_error:.2e}")
    
    # Compare with NumPy
    U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
    print(f"NumPy singular values: {S_np}")
    
    S_sorted = np.sort(S)[::-1]
    S_np_sorted = np.sort(S_np)[::-1]
    sv_error = np.linalg.norm(S_sorted - S_np_sorted)
    print(f"Singular value error vs NumPy: {sv_error:.2e}")
    
    if recon_error < 1e-9:
        print("✓ TEST PASSED")
        return True
    else:
        print("✗ TEST FAILED")
        return False

if __name__ == "__main__":
    print("\n╔════════════════════════════════════════════════════════════════════╗")
    print("║   Testing SVD_parallel.dll via Python ctypes                     ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    results = []
    try:
        results.append(("SVD 3x2", test_svd_3x2()))
        results.append(("SVD 4x3", test_svd_4x3()))
        results.append(("SVD Random", test_svd_random()))
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    passed_count = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed_count}/{len(results)} tests passed")
