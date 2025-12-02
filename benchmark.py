#!/usr/bin/env python3
"""
Benchmark script to compare Serial vs Parallel PNRJ execution times
"""

import subprocess
import sys
import os
import re

def extract_time(output):
    """Extract execution time from output"""
    match = re.search(r'Completed in ([\d.]+) seconds', output)
    if match:
        return float(match.group(1))
    return None

def run_test(exe_name):
    """Run executable and return output"""
    try:
        result = subprocess.run([exe_name], capture_output=True, text=True, timeout=60)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"

def main():
    os.chdir("e:\\subjects\\parallel_computing\\final_project")
    
    print("="*70)
    print("BENCHMARK: Serial vs Parallel NORM-REDUCING JACOBI (PNRJ)")
    print("="*70)
    print()
    
    # Run serial version
    print("[1] Running SERIAL version (norm_serial.exe)...")
    output_serial = run_test(".\\bin\\norm_serial.exe")
    time_serial = extract_time(output_serial)
    
    if time_serial:
        print(f"    Serial time: {time_serial:.6f} seconds")
    else:
        print("    WARNING: Could not extract time from serial version")
        print(output_serial[-500:])
    
    print()
    
    # Run parallel version
    print("[2] Running PARALLEL version (norm_parallel.exe)...")
    output_parallel = run_test(".\\bin\\norm_parallel.exe")
    time_parallel = extract_time(output_parallel)
    
    if time_parallel:
        print(f"    Parallel time: {time_parallel:.6f} seconds")
    else:
        print("    WARNING: Could not extract time from parallel version")
        print(output_parallel[-500:])
    
    print()
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    if time_serial and time_parallel:
        speedup = time_serial / time_parallel
        efficiency = (speedup / 8.0) * 100.0  # Assuming 8 threads
        
        print(f"Serial execution time:     {time_serial:.6f} seconds")
        print(f"Parallel execution time:   {time_parallel:.6f} seconds")
        print()
        print(f"SPEEDUP (S/P):             {speedup:.2f}x")
        print(f"PARALLEL EFFICIENCY:       {efficiency:.1f}%")
        print()
        
        if speedup > 1.0:
            print(f"✓ Parallel version is {speedup:.2f}x FASTER")
        else:
            print(f"✗ Parallel version is {1/speedup:.2f}x SLOWER")
    else:
        print("ERROR: Could not extract timing information")
        sys.exit(1)

if __name__ == "__main__":
    main()
