#!/usr/bin/env python3
"""Table 5: Metamorphic Cryptographic Shards across threat levels."""
import json
import sys
import yaml
from pathlib import Path

def main():
    config_path = Path("configs/mcs_crypto_params.yaml")
    if not config_path.exists():
        print("Config not found. Run from repo root.")
        return 1
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    shards = config["shard_configurations"]
    table_5 = {}
    for level, data in shards.items():
        table_5[level] = {
            "name": data["name"],
            "lattice_kem": data["lattice_kem"],
            "second_kem": data["second_kem"],
            "nist_category": data["nist_category"],
            "lambda_pq_min_bits": data["lambda_pq_min_bits"],
            "morph_time_ms": data["morph_time_target_ms"],
            "cli_level": data["cli_level"],
        }

    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_5_mcs_configs.json"
    out_path.write_text(json.dumps(table_5, indent=2))
    print(f"Wrote {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
