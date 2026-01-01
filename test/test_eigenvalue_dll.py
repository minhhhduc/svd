"""
Test script for norm_reducing_jacobi_parallel.dll
Tests eigenvalue computation via ctypes
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
dll_path = bin_path / "norm_reducing_jacobi_parallel.dll"
try:
    lib = ctypes.CDLL(str(dll_path), mode=ctypes.RTLD_GLOBAL)
except OSError as e:
    print(f"Error loading DLL from {dll_path}: {e}")
    print(f"Current PATH: {os.environ['PATH']}")
    sys.exit(1)

# Define C function signature
# void compute_eigenvalues_parallel(int n, const double* A_in, double* w, double* V_out)
compute_eigenvalues_parallel = lib.compute_eigenvalues_parallel
compute_eigenvalues_parallel.argtypes = [
    ctypes.c_int,  # n
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # A_in
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # w
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")   # V_out
]
compute_eigenvalues_parallel.restype = None

def test_eigenvalues_3x3():
    """Test eigenvalue computation on 3x3 symmetric matrix"""
    print("\n" + "="*70)
    print("Test 1: 3x3 Symmetric Matrix Eigenvalue Decomposition")
    print("="*70)
    
    # Create 3x3 symmetric matrix
    n = 3
    A = np.array([
        [2.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0]
    ], dtype=np.float64, order='C')
    
    print("\nInput matrix A:")
    print(A)
    
    # Allocate output arrays
    w = np.zeros(n, dtype=np.float64)
    V = np.zeros((n, n), dtype=np.float64)
    
    # Call C function
    compute_eigenvalues_parallel(n, A, w, V)
    
    print("\nComputed eigenvalues:", w)
    print("\nEigenvectors (columns are eigenvectors):")
    print(V)
    
    # Verify with NumPy
    evals_np, evecs_np = np.linalg.eigh(A)
    print("\nNumPy eigenvalues:", np.sort(evals_np))
    
    # Check if computed eigenvalues match NumPy (may be in different order)
    w_sorted = np.sort(w)
    evals_sorted = np.sort(evals_np)
    error = np.linalg.norm(w_sorted - evals_sorted)
    print(f"\nError compared to NumPy: {error:.2e}")
    
    if error < 1e-10:
        print("✓ TEST PASSED")
        return True
    else:
        print("✗ TEST FAILED")
        return False

def test_eigenvalues_5x5():
    """Test eigenvalue computation on 5x5 symmetric matrix"""
    print("\n" + "="*70)
    print("Test 2: 5x5 Symmetric Matrix Eigenvalue Decomposition")
    print("="*70)
    
    n = 5
    # Create random symmetric matrix
    A_temp = np.random.randn(n, n)
    A = (A_temp + A_temp.T) / 2
    A = np.ascontiguousarray(A, dtype=np.float64)
    
    print("\nInput matrix A:")
    print(A)
    
    # Allocate output arrays
    w = np.zeros(n, dtype=np.float64)
    V = np.zeros((n, n), dtype=np.float64)
    
    # Call C function
    compute_eigenvalues_parallel(n, A, w, V)
    
    print("\nComputed eigenvalues:", w)
    
    # Verify with NumPy
    evals_np, evecs_np = np.linalg.eigh(A)
    print("NumPy eigenvalues:", evals_np)
    
    # Check error
    w_sorted = np.sort(w)
    evals_sorted = np.sort(evals_np)
    error = np.linalg.norm(w_sorted - evals_sorted)
    print(f"\nError compared to NumPy: {error:.2e}")
    
    if error < 1e-8:
        print("✓ TEST PASSED")
        return True
    else:
        print("✗ TEST FAILED")
        return False

def test_eigenvalues_diagonal():
    """Test on diagonal matrix (trivial case)"""
    print("\n" + "="*70)
    print("Test 3: Diagonal Matrix (Trivial Case)")
    print("="*70)
    
    n = 4
    A = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 4.0]
    ], dtype=np.float64, order='C')
    
    print("\nDiagonal matrix A:")
    print(A)
    
    # Allocate output arrays
    w = np.zeros(n, dtype=np.float64)
    V = np.zeros((n, n), dtype=np.float64)
    
    # Call C function
    compute_eigenvalues_parallel(n, A, w, V)
    
    print("\nComputed eigenvalues:", np.sort(w))
    print("Expected eigenvalues: [1. 2. 3. 4.]")
    
    expected = np.array([1.0, 2.0, 3.0, 4.0])
    w_sorted = np.sort(w)
    error = np.linalg.norm(w_sorted - expected)
    print(f"\nError: {error:.2e}")
    
    if error < 1e-12:
        print("✓ TEST PASSED")
        return True
    else:
        print("✗ TEST FAILED")
        return False

if __name__ == "__main__":
    print("\n╔════════════════════════════════════════════════════════════════════╗")
    print("║   Testing norm_reducing_jacobi_parallel.dll via Python ctypes    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    results = []
    try:
        results.append(("3x3 Eigenvalues", test_eigenvalues_3x3()))
        results.append(("5x5 Eigenvalues", test_eigenvalues_5x5()))
        results.append(("Diagonal Matrix", test_eigenvalues_diagonal()))
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
