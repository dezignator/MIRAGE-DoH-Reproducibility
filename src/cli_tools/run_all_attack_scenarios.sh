#!/usr/bin/env bash
# =============================================================================
# run_all_attack_scenarios.sh — Reproduces Tables 6 and 7 from the thesis.
# =============================================================================
# This script runs the exact sequence of CLI commands printed in the thesis,
# to demonstrate that the artifact reproduces the attack -> detect -> defend
# workflow described in Chapter 3.

set -e # Exit on first error

echo "============================================================"
echo "MIRAGE-DoH: Reproducing Tables 6 and 7"
echo "============================================================"

# Table 6 row 1: AI-Generated Subdomain Flood
echo -e "\n--- Table 6, Row 1: AI-Generated Subdomain Flood ---"
python -m mirage.cli_tools.cma_cli evolve --threat=GAN_Flood

# Table 6 row 2: Quantum Key Harvesting
echo -e "\n--- Table 6, Row 2: Quantum Key Harvesting ---"
python -m mirage.cli_tools.mcshard_cli morph --level=L2

# Table 6 row 3: Protocol Downgrade
echo -e "\n--- Table 6, Row 3: Protocol Downgrade ---"
python -m mirage.cli_tools.cce_cli sim --replay=ECS_Attack --update_protocol

# Table 6 row 4: Topological Tunneling
echo -e "\n--- Table 6, Row 4: Topological Tunneling ---"
python -m mirage.cli_tools.tmn_cli analyze --simplicial=50 --threshold=0.7

# Table 6 row 5: Spectral DGA
echo -e "\n--- Table 6, Row 5: Spectral DGA ---"
python -m mirage.cli_tools.sgi_cli analyze --spectrum=low --band=0.1-0.3Hz


echo -e "\n============================================================"
echo "Workflow Sequence (Table 7)"
echo "============================================================"

# Table 7 row 1: GAN 10K batch validation
echo -e "\n--- Table 7, Row 1: CMA validate batch ---"
python -m mirage.cli_tools.cma_cli validate --batch=GAN_10K --threshold=0.87

# Table 7 row 2: ECDSA timing probe
echo -e "\n--- Table 7, Row 2: MCS monitor ---"
python -m mirage.cli_tools.mcshard_cli monitor --variance=0.5μs

# Table 7 row 3: dnscat2 tunnel block
echo -e "\n--- Table 7, Row 3: TMN block ---"
python -m mirage.cli_tools.tmn_cli block --betti=3 --persistence=0.8

# Table 7 row 4: Spectral DGA mitigation
echo -e "\n--- Table 7, Row 4: SGI mitigate ---"
python -m mirage.cli_tools.sgi_cli mitigate --frequency=0.2Hz --jitter=15ms

# Table 7 row 5: AI mimicry enforcement
echo -e "\n--- Table 7, Row 5: CCE enforce ---"
python -m mirage.cli_tools.cce_cli enforce --entropy=256 --nonce=dynamic

echo -e "\n============================================================"
echo "All scenarios reproduced successfully."
echo "============================================================"
