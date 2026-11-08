#!/usr/bin/env python3
"""Split preprocessed data into train/val/test with stratification and temporal holdout.

Thesis reference: Section 3.11.1 — "The splits were stratified according to
class and for the temporal evaluation capture date to avoid overlap with the
training period."

Usage:
    python data/split_data.py --seed 42
    python data/split_data.py --input data/processed --output data/processed
    python data/split_data.py --toy  # split the toy CSVs instead

Outputs (in --output):
    split_train.parquet
    split_val.parquet
    split_test.parquet
    split_report.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_utils import ALL_COLUMNS, LABEL_COLUMN  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger("split_data")


def load_data(input_dir: Path, toy: bool) -> pd.DataFrame:
    """Load full parquets (or toy CSVs)."""
    if toy:
        toy_dir = input_dir / "toy_dataset"
        benign_path = toy_dir / "benign_sample.csv"
        mal_path = toy_dir / "malicious_sample.csv"
        if not benign_path.is_file() or not mal_path.is_file():
            raise FileNotFoundError(
                f"Toy CSVs not found at {toy_dir}. "
                f"Run `python data/make_toy_dataset.py --synthetic` first."
            )
        benign = pd.read_csv(benign_path)
        mal = pd.read_csv(mal_path)
        combined = pd.concat([benign, mal], ignore_index=True)
        log.info("Loaded toy data: %d rows", len(combined))
    else:
        parquets = list(input_dir.glob("*.parquet"))
        if not parquets:
            raise FileNotFoundError(
                f"No .parquet files in {input_dir}. "
                f"Run data/preprocess_*.py first."
            )
        combined = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
        log.info("Loaded full data: %d rows from %d parquet files",
                 len(combined), len(parquets))

    # Sanity
    missing = [c for c in ALL_COLUMNS if c not in combined.columns]
    if missing:
        raise ValueError(
            f"Data is missing columns: {missing}. "
            f"Expected schema: {ALL_COLUMNS}"
        )
    return combined[ALL_COLUMNS]


def temporal_holdout(df: pd.DataFrame, timestamp_col: str | None) -> tuple:
    """If a timestamp column exists, split by capture time; else return None."""
    if timestamp_col is None or timestamp_col not in df.columns:
        return None
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return (
        df.iloc[:train_end].drop(columns=[timestamp_col]),
        df.iloc[train_end:val_end].drop(columns=[timestamp_col]),
        df.iloc[val_end:].drop(columns=[timestamp_col]),
    )


def stratified_random_split(df: pd.DataFrame, seed: int) -> tuple:
    """Random stratified 70/15/15 split."""
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df[LABEL_COLUMN], random_state=seed
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp[LABEL_COLUMN], random_state=seed
    )
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--toy", action="store_true",
                        help="Split the toy CSVs instead of the full parquets.")
    parser.add_argument("--timestamp-col", type=str, default=None,
                        help="Optional timestamp column for temporal holdout "
                             "(Section 3.11.1). If absent, uses random stratified split.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input, toy=args.toy)
    log.info("Total rows: %d  label counts: %s",
             len(df), df[LABEL_COLUMN].value_counts().to_dict())

    temporal = temporal_holdout(df, args.timestamp_col)
    if temporal is not None:
        train, val, test = temporal
        split_method = "temporal_holdout"
        log.info("Split method: temporal holdout (thesis Section 3.11.1)")
    else:
        train, val, test = stratified_random_split(df, seed=args.seed)
        split_method = "stratified_random"
        log.info("Split method: stratified random (seed=%d); no timestamp "
                 "column available for temporal holdout.", args.seed)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = args.output / f"split_{name}.parquet"
        split.reset_index(drop=True).to_parquet(path, index=False)
        log.info("Wrote %s (%d rows, %.1f%%)",
                 path, len(split), 100 * len(split) / len(df))

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "split_method": split_method,
        "seed": args.seed,
        "toy_mode": args.toy,
        "total_rows": int(len(df)),
        "splits": {
            "train": {"rows": int(len(train)),
                      "label_counts": {int(k): int(v) for k, v in
                                       train[LABEL_COLUMN].value_counts().items()}},
            "val":   {"rows": int(len(val)),
                      "label_counts": {int(k): int(v) for k, v in
                                       val[LABEL_COLUMN].value_counts().items()}},
            "test":  {"rows": int(len(test)),
                      "label_counts": {int(k): int(v) for k, v in
                                       test[LABEL_COLUMN].value_counts().items()}},
        },
        "proportions": {
            "train": len(train) / len(df),
            "val": len(val) / len(df),
            "test": len(test) / len(df),
        },
    }
    (args.output / "split_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    log.info("Wrote split_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
