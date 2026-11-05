"""
Novel Visualiser — Full Test Suite Regression Runner (Phases 0 through 5)
"""

import subprocess
import sys
from pathlib import Path

tests = [
    "test_phase0_verification.py",
    "test_phase1_verification.py",
    "test_phase2_verification.py",
    "test_phase3_verification.py",
    "test_phase4_verification.py",
    "test_phase5_verification.py",
]

base_dir = Path(__file__).resolve().parent.parent / "tests"
results = {}

print("=======================================================")
print(" NOVEL VISUALISER — FULL REGRESSION TEST SUITE (0 - 5)")
print("=======================================================\n")

for test_file in tests:
    test_path = base_dir / test_file
    print(f"Executing: {test_file} ...", end=" ", flush=True)
    res = subprocess.run([sys.executable, str(test_path)], capture_output=True, text=True, errors="ignore")
    if res.returncode == 0:
        results[test_file] = (True, "")
        print("[PASS]")
    else:
        results[test_file] = (False, res.stderr or res.stdout)
        print("[FAIL]")

print("\n==================== TEST RESULTS ====================")
for t, (passed, err) in results.items():
    status_str = "PASS [OK]" if passed else "FAIL [ERROR]"
    print(f"  {t:<32} : {status_str}")
    if not passed:
        print(f"    Details: {err[:300]}")

total_passed = sum(1 for p, _ in results.values() if p)
total_tests = len(tests)
print("=======================================================")
print(f" Final Score: {total_passed}/{total_tests} test suites passed ({total_passed/total_tests*100:.1f}%)")
print("=======================================================\n")

if total_passed < total_tests:
    sys.exit(1)

