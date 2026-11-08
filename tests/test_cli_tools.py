"""
test_cli_tools.py — Integration tests for the CLI tools (Tables 6 and 7).

Verifies that the exact commands from the thesis execute successfully.

Run with:
    pytest tests/test_cli_tools.py -v
"""

import sys
import pytest
from pathlib import Path

# Adjust path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirage.cli_tools import cma_cli, mcshard_cli, cce_cli, tmn_cli, sgi_cli


class TestTable6Commands:
    """Commands from Table 6 (Attack scenarios)."""

    def test_cma_evolve(self):
        """Table 6: cma --evolve --threat=GAN_Flood"""
        # argparse expects the subcommand as the first arg
        args = ["evolve", "--threat=GAN_Flood"]
        cma_cli.main(args)  # Should not raise

    def test_mcshard_morph(self):
        """Table 6: mcshard --morph --level=L2"""
        args = ["morph", "--level=L2"]
        mcshard_cli.main(args)

    def test_cce_sim(self):
        """Table 6: cce_sim --replay=ECS_Attack --update_protocol"""
        args = ["sim", "--replay=ECS_Attack", "--update_protocol"]
        cce_cli.main(args)

    def test_tmn_analyze(self):
        """Table 6: tmn_analyze --simplicial=50 --threshold=0.7"""
        args = ["analyze", "--simplicial=50", "--threshold=0.7"]
        tmn_cli.main(args)

    def test_sgi_analyze(self):
        """Table 6: sgai --spectrum=low --band=0.1-0.3Hz"""
        args = ["analyze", "--spectrum=low", "--band=0.1-0.3Hz"]
        sgi_cli.main(args)


class TestTable7Commands:
    """Commands from Table 7 (Defense workflow)."""

    def test_cma_validate(self):
        """Table 7: cmda_validate --batch=GAN_10K --threshold=0.87"""
        args = ["validate", "--batch=GAN_10K", "--threshold=0.87"]
        cma_cli.main(args)

    def test_mcshard_monitor(self):
        """Table 7: mcshard_monitor --variance=0.5μs"""
        args = ["monitor", "--variance=0.5μs"]
        mcshard_cli.main(args)

    def test_tmn_block(self):
        """Table 7: tmn_block --betti=3 --persistence=0.8"""
        args = ["block", "--betti=3", "--persistence=0.8"]
        tmn_cli.main(args)

    def test_sgi_mitigate(self):
        """Table 7: sgi_mitigate --frequency=0.2Hz --jitter=15ms"""
        args = ["mitigate", "--frequency=0.2Hz", "--jitter=15ms"]
        sgi_cli.main(args)

    def test_cce_enforce(self):
        """Table 7: cce_enforce --entropy=256 --nonce=dynamic"""
        args = ["enforce", "--entropy=256", "--nonce=dynamic"]
        cce_cli.main(args)
