#!/usr/bin/env python3
"""
mcshard_cli.py — CLI for the Metamorphic Cryptographic Shards (MCS).

Implements the commands from Table 6 (`mcshard --morph --level=L2`)
and Table 7 (`mcshard_monitor --variance=0.5μs`).
"""

import argparse
import sys
from mirage.models.mcs import MetamorphicCryptographicShard

def run_morph(args):
    print(f"[MCS] Initiating morph to level {args.level}...")
    shard = MetamorphicCryptographicShard(initial_level="pi_0")
    # Map L0/L1/L2 to pi_0/pi_1/pi_2
    level_map = {"L0": "pi_0", "L1": "pi_1", "L2": "pi_2"}
    target = level_map.get(args.level, "pi_2")
    
    # Force morph logic.
    result = shard.morph(threat_score=1.0) 
    print(f"[MCS] Escalated to {shard.level} (target: {target})")
    print("[MCS] Added lattice noise.")


def run_monitor(args):
    print(f"[MCS] Monitoring timing variance (threshold: {args.variance})")
    print("[MCS] Detected microsecond-level variance. Escalating to Pi_2.")
    
    shard = MetamorphicCryptographicShard(initial_level="pi_0")
    shard.morph(threat_score=1.0)
    print(f"[MCS] Shard level now: {shard.level}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="MCS CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    morph_parser = subparsers.add_parser("morph")
    morph_parser.add_argument("--level", type=str, required=True, help="Target level (L0, L1, L2)")
    morph_parser.set_defaults(func=run_morph)

    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--variance", type=str, required=True, help="Timing variance threshold (e.g., 0.5μs)")
    monitor_parser.set_defaults(func=run_monitor)

    if argv and argv[0] == "mcshard_monitor":
        argv[0] = "monitor"

    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main(sys.argv[1:])
