#!/usr/bin/env python3
"""Generate the 1 MB toy dataset for the MIRAGE-DoH artifact.

Two modes:

    --synthetic  (default)  Generate 1000 benign + 1000 malicious rows with
                            the canonical 46-feature schema, drawn from
                            distributions that are NOT real attack
                            distributions. Classifier accuracy on this data
                            is ~60-70% because there is no real signal.
                            Use this to exercise the code path end-to-end.

    --from-full             Sample 1000 benign + 1000 malicious rows from
                            the preprocessed full data (requires
                            data/processed/*.parquet, which requires the
                            32 GB download and preprocessing first).
                            Classifier accuracy on this data is ~85-90%
                            because it is real data.

Usage:
    python data/make_toy_dataset.py --synthetic
    python data/make_toy_dataset.py --from-full
    python data/make_toy_dataset.py --synthetic --seed 42 --n-per-class 1000
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# sys.path trick so we can import data._data_utils from a script in data/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_utils import (  # noqa: E402
    generate_synthetic,
    sample_from_processed,
    write_checksums,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger("make_toy_dataset")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true", default=True,
                      help="Generate synthetic data (default).")
    mode.add_argument("--from-full", action="store_true",
                      help="Sample from preprocessed full data.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-class", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("data/toy_dataset"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"),
                        help="Where to find processed full data for --from-full.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_full:
        log.info("Mode: sample from full data in %s", args.processed_dir)
        benign_df, mal_df = sample_from_processed(
            processed_dir=args.processed_dir,
            n_per_class=args.n_per_class,
            seed=args.seed,
        )
        accuracy_note = (
            "Real 2000-row subset of the full data. "
            "Classifier accuracy on this data should be ~85-90%."
        )
    else:
        log.info("Mode: synthetic")
        benign_df, mal_df = generate_synthetic(
            n_per_class=args.n_per_class,
            seed=args.seed,
        )
        accuracy_note = (
            "Synthetic stand-in. Classifier accuracy on this data is "
            "~60-70% because the synthetic distributions are not real "
            "attack distributions. Use --from-full for realistic accuracy."
        )

    benign_path = args.output_dir / "benign_sample.csv"
    mal_path = args.output_dir / "malicious_sample.csv"
    benign_df.to_csv(benign_path, index=False)
    mal_df.to_csv(mal_path, index=False)
    log.info("Wrote %s (%d rows)", benign_path, len(benign_df))
    log.info("Wrote %s (%d rows)", mal_path, len(mal_df))

    # Update checksums.
    write_checksums([benign_path, mal_path], args.output_dir / "CHECKSUMS.txt")
    log.info("Updated %s", args.output_dir / "CHECKSUMS.txt")

    # Write a META.json that documents how the toy dataset was generated.
    import json
    meta = {
        "mode": "from_full" if args.from_full else "synthetic",
        "seed": args.seed,
        "n_per_class": args.n_per_class,
        "schema_columns": 47,  # 1 label + 46 features
        "accuracy_note": accuracy_note,
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    (args.output_dir / "META.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    log.info("Wrote %s", args.output_dir / "META.json")
    log.info("Done. %s", accuracy_note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
