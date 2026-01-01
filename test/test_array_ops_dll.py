"""
Test script for other_operating_parallel.dll
Tests parallel array operations via ctypes
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
dll_path = bin_path / "other_operating_parallel.dll"
try:
    lib = ctypes.CDLL(str(dll_path), mode=ctypes.RTLD_GLOBAL)
except OSError as e:
    print(f"Error loading DLL from {dll_path}: {e}")
    print(f"Current PATH: {os.environ['PATH']}")
    sys.exit(1)

# Define C function signatures
# double* transpose_parallel(const double* A, int m, int n, int num_threads)
transpose_parallel = lib.transpose_parallel
transpose_parallel.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # A
    ctypes.c_int,  # m
    ctypes.c_int,  # n
    ctypes.c_int   # num_threads
]
transpose_parallel.restype = ctypes.POINTER(ctypes.c_double)

# double* argsort_parallel(const double* arr, int n, int num_threads)
argsort_parallel = lib.argsort_parallel
argsort_parallel.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # arr
    ctypes.c_int,  # n
    ctypes.c_int   # num_threads
]
argsort_parallel.restype = ctypes.POINTER(ctypes.c_double)

# double* square_parallel(const double* arr, int n, int num_threads)
square_parallel = lib.square_parallel
square_parallel.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # arr
    ctypes.c_int,  # n
    ctypes.c_int   # num_threads
]
square_parallel.restype = ctypes.POINTER(ctypes.c_double)

def test_transpose():
    """Test matrix transpose operation"""
    print("\n" + "="*70)
    print("Test 1: Matrix Transpose")
    print("="*70)
    
    m, n = 3, 4
    A = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ], dtype=np.float64, order='C')
    
    print(f"\nOriginal matrix A ({m}x{n}):")
    print(A)
    
    # Call C function
    result_ptr = transpose_parallel(A, m, n, 4)
    At = np.ctypeslib.as_array(result_ptr, shape=(n, m))
    At = np.array(At)  # Make a copy
    
    print(f"\nTransposed matrix A^T ({n}x{m}):")
    print(At)
    
    # Expected result
    expected = A.T
    print("\nExpected (NumPy):")
    print(expected)
    
    error = np.linalg.norm(At - expected)
    print(f"\nError: {error:.2e}")
    
    # Free C memory
    ctypes.CDLL(None).free(result_ptr)
    
    if error < 1e-12:
        print("✓ TEST PASSED")
        return True
    else:
        print("✗ TEST FAILED")
        return False

def test_argsort():
    """Test argsort operation"""
    print("\n" + "="*70)
    print("Test 2: Argsort (Parallel Sorting)")
    print("="*70)
    
    arr = np.array([5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0], dtype=np.float64, order='C')
    n = len(arr)
    
    print(f"\nOriginal array: {arr}")
    
    # Call C function
    result_ptr = argsort_parallel(arr, n, 4)
    indices = np.ctypeslib.as_array(result_ptr, shape=(n,))
    indices = np.array(indices, dtype=int)  # Make a copy
    
    print(f"Sorted indices:  {indices}")
    print(f"Sorted values:   {arr[indices]}")
    
    # Expected result
    expected_indices = np.argsort(arr)
    print(f"\nExpected indices (NumPy): {expected_indices}")
    print(f"Expected values:          {arr[expected_indices]}")
    
    # Check if sorting is correct
    sorted_values = arr[indices]
    expected_values = arr[expected_indices]
    error = np.linalg.norm(sorted_values - expected_values)
    print(f"\nError: {error:.2e}")
    
    # Free C memory
    ctypes.CDLL(None).free(result_ptr)
    
    if error < 1e-12:
        print("✓ TEST PASSED")
        return True
    else:
        print("✗ TEST FAILED")
        return False

def test_square():
    """Test element-wise square operation"""
    print("\n" + "="*70)
    print("Test 3: Element-wise Square")
    print("="*70)
    
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64, order='C')
    n = len(arr)
    
    print(f"\nOriginal array: {arr}")
    
    # Call C function
    result_ptr = square_parallel(arr, n, 4)
    squared = np.ctypeslib.as_array(result_ptr, shape=(n,))
    squared = np.array(squared)  # Make a copy
    
    print(f"Squared array:  {squared}")
    
    # Expected result
    expected = arr ** 2
    print(f"Expected:       {expected}")
    
    error = np.linalg.norm(squared - expected)
    print(f"\nError: {error:.2e}")
    
    # Free C memory
    ctypes.CDLL(None).free(result_ptr)
    
    if error < 1e-12:
        print("✓ TEST PASSED")
        return True
    else:
        print("✗ TEST FAILED")
        return False

def test_large_transpose():
    """Test transpose on larger matrix"""
    print("\n" + "="*70)
    print("Test 4: Large Matrix Transpose (100x200)")
    print("="*70)
    
    m, n = 100, 200
    np.random.seed(42)
    A = np.random.randn(m, n)
    A = np.ascontiguousarray(A, dtype=np.float64)
    
    print(f"\nMatrix dimensions: {m}x{n}")
    
    # Call C function
    result_ptr = transpose_parallel(A, m, n, 4)
    At = np.ctypeslib.as_array(result_ptr, shape=(n, m))
    At = np.array(At)  # Make a copy
    
    # Expected result
    expected = A.T
    error = np.linalg.norm(At - expected)
    print(f"Transpose error: {error:.2e}")
    
    # Free C memory
    ctypes.CDLL(None).free(result_ptr)
    
    if error < 1e-12:
        print("✓ TEST PASSED")
        return True
    else:
        print("✗ TEST FAILED")
        return False

def test_large_array_operations():
    """Test array operations on larger arrays"""
    print("\n" + "="*70)
    print("Test 5: Large Array Operations (n=10000)")
    print("="*70)
    
    n = 10000
    np.random.seed(42)
    arr = np.random.randn(n) * 10
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    
    print(f"\nArray size: {n}")
    
    # Test square
    result_ptr = square_parallel(arr, n, 4)
    squared = np.ctypeslib.as_array(result_ptr, shape=(n,))
    squared = np.array(squared)
    
    expected_squared = arr ** 2
    sq_error = np.linalg.norm(squared - expected_squared)
    print(f"Square operation error: {sq_error:.2e}")
    
    ctypes.CDLL(None).free(result_ptr)
    
    # Test argsort
    result_ptr = argsort_parallel(arr, n, 4)
    indices = np.ctypeslib.as_array(result_ptr, shape=(n,))
    indices = np.array(indices, dtype=int)
    
    expected_indices = np.argsort(arr)
    sorted_values = arr[indices]
    expected_values = arr[expected_indices]
    sort_error = np.linalg.norm(sorted_values - expected_values)
    print(f"Argsort operation error: {sort_error:.2e}")
    
    ctypes.CDLL(None).free(result_ptr)
    
    if sq_error < 1e-12 and sort_error < 1e-12:
        print("✓ TEST PASSED")
        return True
    else:
        print("✗ TEST FAILED")
        return False

if __name__ == "__main__":
    print("\n╔════════════════════════════════════════════════════════════════════╗")
    print("║   Testing other_operating_parallel.dll via Python ctypes         ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    results = []
    try:
        results.append(("Transpose 3x4", test_transpose()))
        results.append(("Argsort", test_argsort()))
        results.append(("Square", test_square()))
        results.append(("Large Transpose", test_large_transpose()))
        results.append(("Large Array Ops", test_large_array_operations()))
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
