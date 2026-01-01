"""
Comprehensive Python Test Suite for Parallel Computing DLLs
Using test executables compiled from C code
"""

import subprocess
import sys
from pathlib import Path

def run_test(test_name, executable_path):
    """Run a C test executable and capture output"""
    print(f"\n{'='*70}")
    print(f"Running: {test_name}")
    print('='*70)
    
    try:
        result = subprocess.run(
            [str(executable_path)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        return success, result.stdout
    except Exception as e:
        print(f"✗ Error running test: {e}")
        return False, str(e)

def main():
    bin_dir = Path(__file__).parent.parent / "bin"
    
    print("\n╔════════════════════════════════════════════════════════════════════╗")
    print("║   Comprehensive DLL Test Suite via C Executables                  ║")
    print("║   (Compiled test programs that use the DLLs internally)           ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    tests = [
        ("Eigenvalue Computation", bin_dir / "test_eigenvalues.exe"),
        ("Parallel Array Operations", bin_dir / "test_parallel_ops.exe"),
        ("SVD Decomposition", bin_dir / "test_svd_comprehensive.exe"),
        ("Comprehensive Test Suite", bin_dir / "test_all.exe")
    ]
    
    results = []
    
    for test_name, exe_path in tests:
        if not exe_path.exists():
            print(f"\n⚠ Skipping {test_name}: Executable not found at {exe_path}")
            results.append((test_name, False, "Executable not found"))
            continue
        
        success, output = run_test(test_name, exe_path)
        results.append((test_name, success, output))
    
    # Summary
    print("\n" + "╔"+ "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "TEST SUMMARY".center(68) + "║")
    print("║" + " "*68 + "║")
    
    passed = 0
    for test_name, success, _ in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"║  {test_name:.<50} {status} ║")
        if success:
            passed += 1
    
    print("║" + " "*68 + "║")
    print(f"║  Total: {passed}/{len(results)} tests passed" + " "*(68-len(f"  Total: {passed}/{len(results)} tests passed")-1) + "║")
    print("║" + " "*68 + "║")
    print("╚"+ "═"*68 + "╝\n")
    
    return 0 if passed == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
