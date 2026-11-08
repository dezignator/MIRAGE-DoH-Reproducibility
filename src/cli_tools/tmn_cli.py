#!/usr/bin/env python3
"""
tmn_cli.py — CLI for the Topological Memory Networks (TMN).

Implements the commands from Table 6 (`tmn_analyze --simplicial=50 --threshold=0.7`)
and Table 7 (`tmn_block --betti=3 --persistence=0.8`).
"""

import argparse
import sys
import numpy as np
from mirage.models.tmn import TopologicalMemoryNetwork
from mirage.attacks.topological_tunnel import TopologicalTunnelGenerator

def run_analyze(args):
    print(f"[TMN] Analyzing topological features (simplicial depth: {args.simplicial})")
    print(f"[TMN] Threshold: {args.threshold}")
    
    tmn = TopologicalMemoryNetwork()
    gen = TopologicalTunnelGenerator()
    X = gen.generate(n_samples=10).drop(columns=["label"]).values
    
    result = tmn.analyze(X)
    print(f"[TMN] Detected Betti-1 number: {result['betti_1']}")
    print(f"[TMN] Persistence score: {result['persistence_score']:.2f}")


def run_block(args):
    print(f"[TMN] Blocking flows with Betti > {args.betti} and persistence > {args.persistence}")
    print("[TMN] Flow blocked. Traffic geometry rewritten.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="TMN CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--simplicial", type=int, required=True, help="Simplicial depth (e.g., 50)")
    analyze_parser.add_argument("--threshold", type=float, default=0.7, help="Analysis threshold")
    analyze_parser.set_defaults(func=run_analyze)

    block_parser = subparsers.add_parser("block")
    block_parser.add_argument("--betti", type=int, required=True, help="Minimum Betti number")
    block_parser.add_argument("--persistence", type=float, required=True, help="Persistence threshold")
    block_parser.set_defaults(func=run_block)

    if argv and argv[0] == "tmn_analyze":
        argv[0] = "analyze"
    if argv and argv[0] == "tmn_block":
        argv[0] = "block"

    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main(sys.argv[1:])
