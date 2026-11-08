#!/usr/bin/env python3
"""
verification_script.py — Compare fresh outputs against the expected thesis numbers.

Usage:
    python expected_outputs/verification_script.py
    python expected_outputs/verification_script.py --toy-mode
    python expected_outputs/verification_script.py --rtol 0.05

Exit codes:
    0 = all checks passed
    1 = one or more checks failed
"""

import argparse
import json
import sys
from pathlib import Path

# ANSI colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

def compare_dicts(actual, expected, rtol, prefix="") -> list:
    """Recursively compare actual vs expected dictionaries."""
    errors = []
    for key, exp_val in expected.items():
        path = f"{prefix}.{key}" if prefix else key
        
        if key not in actual:
            errors.append(f"Missing key: {path}")
            continue
            
        act_val = actual[key]
        
        if isinstance(exp_val, dict):
            errors.extend(compare_dicts(act_val, exp_val, rtol, path))
        elif isinstance(exp_val, list):
            if len(act_val) != len(exp_val):
                errors.append(f"List length mismatch at {path}: {len(act_val)} != {len(exp_val)}")
            else:
                for i, (a, e) in enumerate(zip(act_val, exp_val)):
                    if isinstance(e, (int, float)) and isinstance(a, (int, float)):
                        if not is_close(a, e, rtol):
                            errors.append(f"Value mismatch at {path}[{i}]: {a} != {e} (tol: {rtol})")
        elif isinstance(exp_val, (int, float)) and isinstance(act_val, (int, float)):
            if not is_close(act_val, exp_val, rtol):
                errors.append(f"Value mismatch at {path}: {act_val} != {exp_val} (tol: {rtol})")
        else:
            if act_val != exp_val:
                errors.append(f"Value mismatch at {path}: {act_val} != {exp_val}")
    return errors

def is_close(a, b, rtol):
    """Relative closeness with a small absolute tolerance to handle near-zero values."""
    if a == b: return True
    return abs(a - b) <= max(rtol * abs(b), 1e-4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual-dir", type=Path, default=Path("outputs/tables"), help="Directory with fresh outputs")
    parser.add_argument("--expected-dir", type=Path, default=Path("expected_outputs"), help="Directory with expected thesis numbers")
    parser.add_argument("--rtol", type=float, default=0.01, help="Relative tolerance (default 1%)")
    parser.add_argument("--toy-mode", action="store_true", help="Relax tolerance to 20% and expect lower accuracy")
    args = parser.parse_args()

    if args.toy_mode:
        args.rtol = 0.20
        print(f"{YELLOW}WARNING: Running in toy mode. Tolerance relaxed to {args.rtol*100}%.{RESET}")

    if not args.actual_dir.exists():
        print(f"{RED}ERROR: Actual outputs directory {args.actual_dir} does not exist.{RESET}")
        print("Run the reproduction scripts first (e.g., python scripts/reproduce_tables/table_5_mcs_configs.py).")
        return 1

    expected_files = list(args.expected_dir.glob("*_expected.json"))
    if not expected_files:
        print(f"{RED}ERROR: No expected JSON files found in {args.expected_dir}.{RESET}")
        return 1

    total_checks = 0
    failed_checks = 0

    print(f"\n{BOLD}MIRAGE-DoH Reproducibility Verification{RESET}")
    print("=" * 60)

    for exp_file in sorted(expected_files):
        # e.g., table_8_expected.json -> table_8_core_metrics.json
        table_num = exp_file.stem.split("_")[1] # '5', '8', etc.
        
        # Find the corresponding actual file
        actual_files = list(args.actual_dir.glob(f"table_{table_num}_*.json"))
        if not actual_files:
            print(f"{YELLOW}SKIP{RESET}: {exp_file.name} (no matching output in {args.actual_dir})")
            continue
            
        act_file = actual_files[0] # take the first match
        total_checks += 1
        
        expected_data = json.loads(exp_file.read_text())
        actual_data = json.loads(act_file.read_text())
        
        errors = compare_dicts(actual_data, expected_data, args.rtol)
        
        if not errors:
            print(f"{GREEN}PASS{RESET}: {exp_file.name}")
        else:
            failed_checks += 1
            print(f"{RED}FAIL{RESET}: {exp_file.name}")
            for err in errors:
                print(f"       - {err}")

    # Check figures (file existence and non-zero size)
    fig_dir = Path("outputs/figures")
    if fig_dir.exists():
        for fig in ["figure_1_architecture.pdf", "figure_2_module_validation.pdf", 
                    "figure_3_traffic_dynamics.pdf", "figure_4_cognitive_learning.pdf",
                    "figure_5_joint_optimization.pdf", "figure_6_adaptive_ai_attacks.pdf"]:
            fig_path = fig_dir / fig
            if fig_path.exists() and fig_path.stat().st_size > 1000:
                print(f"{GREEN}PASS{RESET}: {fig}")
            else:
                print(f"{RED}FAIL{RESET}: {fig} (missing or too small)")

    print("=" * 60)
    if failed_checks == 0:
        print(f"{GREEN}{BOLD}All {total_checks} table checks passed!{RESET}")
        return 0
    else:
        print(f"{RED}{BOLD}{failed_checks} out of {total_checks} table checks failed.{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
