#!/usr/bin/env python3
"""Preprocess Gen-C-DDD-2022 into the canonical 46-feature parquet.

Identical logic to preprocess_cic_doh2023.py, with different default paths.

Usage:
    python data/preprocess_gen_c_ddd2022.py
    python data/preprocess_gen_c_ddd2022.py --input data/raw/gen_c_ddd2022/
"""

import sys
from pathlib import Path

# Reuse the CIC preprocessor's logic; only the default paths differ.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_cic_doh2023 import main as _main_cic  # noqa: E402


def main():
    # Rewrite sys.argv's defaults if the caller didn't override them.
    import argparse
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    log = logging.getLogger("preprocess_gen_c_ddd2022")
    log.info("Delegating to preprocess_cic_doh2023.py with Gen-C defaults.")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/gen_c_ddd2022"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/gen_c_ddd2022.parquet"))
    parser.add_argument("--report", type=Path, default=Path("data/processed/gen_c_ddd2022_report.json"))
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    # Re-enter the CIC main with the Gen-C paths.
    sys.argv = [
        "preprocess_cic_doh2023.py",
        "--input", str(args.input),
        "--output", str(args.output),
        "--report", str(args.report),
        "--chunk-size", str(args.chunk_size),
    ]
    return _main_cic()


if __name__ == "__main__":
    sys.exit(main())
