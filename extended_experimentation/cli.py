#!/usr/bin/env python3
"""
extended_experimentation/cli.py

Unified command-line interface for the MIRAGE-DoH extended experimentation
suite.

We implemented this CLI as a thin dispatcher over the individual module
entry points. Each subcommand forwarded its arguments unchanged to the
corresponding module's ``main`` and returned the exit code it produced.
That kept the individual modules independently runnable while giving the
suite a single discoverable entry point.

The subcommands mirror the pipeline order:

  generate-campaign     polymorphic_shaping_simulator.main
  generate-tunneling    staged_tunneling_orchestrator.main
  load-dataset          public_dataset_loaders.main
  train-tmn             topological_memory_trainer.main
  train-mirage          full_mirage_training_loop.main
  benchmark-mcs         mcs_morph_benchmark.main
  calibrate-mcs         cryptographic_shard_calibrator.main
  extract-tda           persistent_homology_extractor.main
  train-spectral        spectral_embedding_predictor.main
  evaluate-core         core_metrics_evaluator.main
  significance          statistical_significance_harness.main
  ablation              ablation_study_runner.main
  tables                extended_result_table_generator.main
  figures               thesis_figure_reproducer.main

We did not add an ``all`` subcommand. The pipeline stages had real
dependencies on each other's outputs, and a monolithic runner would have
made it too easy to start the whole pipeline with the wrong config. Each
subcommand accepted ``--out`` and the required inputs, so a shell script
or Makefile was the right place to sequence them.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from typing import Callable, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOGGER = logging.getLogger("mirage.extended.cli")

# --------------------------------------------------------------------------- #
# Subcommand registry
# --------------------------------------------------------------------------- #
# Each entry maps a CLI verb to the module path whose ``main`` should be
# invoked. We resolved the modules lazily so that optional dependencies
# (torch for the training commands, cryptography for the MCS commands,
# matplotlib for the figure command) were only imported when the
# corresponding subcommand was actually invoked.

SUBCOMMANDS: Dict[str, str] = {
    "generate-campaign": "extended_experimentation.attack_scenarios.polymorphic_shaping_simulator",
    "generate-tunneling": "extended_experimentation.attack_scenarios.staged_tunneling_orchestrator",
    "load-dataset": "extended_experimentation.datasets.public_dataset_loaders",
    "train-tmn": "extended_experimentation.training.topological_memory_trainer",
    "train-mirage": "extended_experimentation.training.full_mirage_training_loop",
    "benchmark-mcs": "extended_experimentation.cryptographic_agility.mcs_morph_benchmark",
    "calibrate-mcs": "extended_experimentation.cryptographic_agility.cryptographic_shard_calibrator",
    "extract-tda": "extended_experimentation.topological_analysis.persistent_homology_extractor",
    "train-spectral": "extended_experimentation.spectral_analysis.spectral_embedding_predictor",
    "evaluate-core": "extended_experimentation.evaluation.core_metrics_evaluator",
    "significance": "extended_experimentation.evaluation.statistical_significance_harness",
    "ablation": "extended_experimentation.evaluation.ablation_study_runner",
    "tables": "extended_experimentation.reporting.extended_result_table_generator",
    "figures": "extended_experimentation.reporting.thesis_figure_reproducer",
    "select-shard": "extended_experimentation.deployment.runtime_shard_selector",
}


def _resolve_main(module_path: str) -> Callable[[Optional[Sequence[str]]], int]:
    module = importlib.import_module(module_path)
    main = getattr(module, "main", None)
    if main is None:
        raise AttributeError(f"{module_path} has no main()")
    return main


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #


def _split_at_verb(argv: Sequence[str]) -> tuple[str, List[str]]:
    """We split ``argv`` at the first recognised verb.

    Everything before the verb (typically just global flags like
    ``--log-level``) was discarded; everything after was forwarded to the
    subcommand. This mirrors how most multi-command CLIs handle
    pass-through arguments.
    """
    for i, token in enumerate(argv):
        if token in SUBCOMMANDS:
            return token, list(argv[i + 1 :])
    raise SystemExit(
        f"expected one of {sorted(SUBCOMMANDS)} as the first non-flag argument"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("MIRAGE-DoH extended experimentation suite\n")
        print("Subcommands:")
        for verb in sorted(SUBCOMMANDS):
            print(f"  {verb:22s} {SUBCOMMANDS[verb]}")
        print("\nRun a subcommand with -h for its arguments.")
        return 0

    verb, forwarded = _split_at_verb(argv)
    module_path = SUBCOMMANDS[verb]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    LOGGER.debug("dispatching %s -> %s", verb, module_path)
    sub_main = _resolve_main(module_path)
    return int(sub_main(forwarded))


if __name__ == "__main__":
    raise SystemExit(main())
