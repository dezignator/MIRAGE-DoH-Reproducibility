#!/usr/bin/env python3
"""
train_baselines.py — Train CNN-LSTM and Random Forest baselines.

Runs under the exact same fairness protocol as MIRAGE-DoH
(configs/baselines_config.yaml §1: same data, same preprocessing, same
tuning budget).

Usage:
    python scripts/train_baselines.py --baseline cnn_lstm --full-data
    python scripts/train_baselines.py --baseline random_forest
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirage.utils.seed import seed_everything
from mirage.utils.logger import setup_logger
from mirage.baselines import CNNLSTMBaseline, RandomForestBaseline

logger = logging.getLogger("train_baselines")

def load_splits(config_path: Path, full_data: bool, toy_dir: Path):
    """Load splits, mirroring train_mirage.py exactly."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    if full_data:
        processed_dir = Path(config["paths"]["processed_datasets"])
        train = pd.read_parquet(processed_dir / "split_train.parquet")
        val = pd.read_parquet(processed_dir / "split_val.parquet")
        test = pd.read_parquet(processed_dir / "split_test.parquet")
    else:
        # Toy mode: ensure data exists
        benign = pd.read_csv(toy_dir / "benign_sample.csv")
        mal = pd.read_csv(toy_dir / "malicious_sample.csv")
        combined = pd.concat([benign, mal], ignore_index=True)
        
        from sklearn.model_selection import train_test_split
        train, temp = train_test_split(combined, test_size=0.2, stratify=combined["label"], random_state=42)
        val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

    feature_cols = [c for c in train.columns if c != "label"]
    return (
        train[feature_cols].values, train["label"].values,
        val[feature_cols].values, val["label"].values,
        test[feature_cols].values, test["label"].values,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default_mirage_config.yaml"))
    parser.add_argument("--baseline", choices=["cnn_lstm", "random_forest"], required=True)
    parser.add_argument("--full-data", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    seed_everything(config.get("seed", 42))
    
    run_dir = Path("outputs/logs") / f"baseline_{args.baseline}_{'full' if args.full_data else 'toy'}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(f"baseline_{args.baseline}", log_dir=run_dir, config_path=args.config)

    toy_dir = Path(config["paths"]["toy_dataset"])
    if not args.full_data and not (toy_dir / "benign_sample.csv").exists():
        log.error("Toy data missing. Run `python data/make_toy_dataset.py --synthetic` first.")
        return 1

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(args.config, args.full_data, toy_dir)
    log.info("Training %s on %d samples...", args.baseline, len(X_train))

    if args.baseline == "cnn_lstm":
        model = CNNLSTMBaseline()
        model.train(X_train, y_train, X_val, y_val)
    else:
        model = RandomForestBaseline()
        model.train(X_train, y_train)

    # Save checkpoint
    ckpt_dir = Path("outputs/checkpoints") / ("full" if args.full_data else "toy")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log.info("Checkpoint saving not fully implemented for baselines; skipping.")
    log.info("Training complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
