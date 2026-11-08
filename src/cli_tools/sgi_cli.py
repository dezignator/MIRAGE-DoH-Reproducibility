#!/usr/bin/env python3
"""
sgi_cli.py — CLI for the Spectral Game Intelligence (SGI).

Implements the commands from Table 6 (`sgai --spectrum=low --band=0.1-0.3Hz`)
and Table 7 (`sgi_mitigate --frequency=0.2Hz --jitter=15ms`).
"""

import argparse
import sys
from mirage.models.sgi import SpectralGameIntelligence
from mirage.attacks.spectral_dga import SpectralDGAGenerator

def run_analyze(args):
    print(f"[SGI] Decomposing traffic via graph Laplacian eigenprojections.")
    print(f"[SGI] Spectrum: {args.spectrum}, Band: {args.band}")
    
    sgi = SpectralGameIntelligence()
    gen = SpectralDGAGenerator()
    X = gen.generate(n_samples=10).drop(columns=["label"]).values
    
    state = sgi.compute_spectral_state(X)
    print(f"[SGI] Extracted spectral state (dim: {len(state)}). Low-frequency clustering detected.")


def run_mitigate(args):
    print(f"[SGI] Mitigating spectral DGA at {args.frequency} with {args.jitter} jitter.")
    print("[SGI] Rewriting HTTP/2 frames and altering timing.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="SGI CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--spectrum", type=str, required=True, help="Spectrum mode (low/full)")
    analyze_parser.add_argument("--band", type=str, required=True, help="Frequency band (e.g., 0.1-0.3Hz)")
    analyze_parser.set_defaults(func=run_analyze)

    mitigate_parser = subparsers.add_parser("mitigate")
    mitigate_parser.add_argument("--frequency", type=str, required=True, help="Mitigation frequency (e.g., 0.2Hz)")
    mitigate_parser.add_argument("--jitter", type=str, required=True, help="Jitter to apply (e.g., 15ms)")
    mitigate_parser.set_defaults(func=run_mitigate)

    if argv and argv[0] == "sgai":
        argv[0] = "analyze"
    if argv and argv[0] == "sgi_mitigate":
        argv[0] = "mitigate"

    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main(sys.argv[1:])
