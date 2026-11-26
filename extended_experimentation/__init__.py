"""
extended_experimentation — MIRAGE-DoH extended experimentation suite.

This package contains the codebase that produces the extended experimental
artefacts referenced from Chapter 3 of the thesis. Its constituent modules
are documented individually; this __init__ exposes only the version and
the top-level subpackages.
"""

__version__ = "0.3.1"

__all__ = [
    "attack_scenarios",
    "training",
    "evaluation",
    "cryptographic_agility",
    "topological_analysis",
    "spectral_analysis",
    "reporting",
    "deployment",
    "datasets",
]
