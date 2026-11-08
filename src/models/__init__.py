"""
MIRAGE-DoH models package.

Exports the five model components:

    CognitiveMetaAgent      (CMA, Eq. 1-5)
    TopologicalMemoryNetwork (TMN, Eq. 6-10)
    MetamorphicCryptographicShard (MCS, Eq. 11-13)
    CausalCounterfactualEnvironment (CCE, Eq. 14-16)
    SpectralGameIntelligence (SGI, Eq. 17-21)

Plus supporting classes:
    ThreatSeverityScorer    (Eq. 13a)
"""

from mirage.models.cma import CMAPolicyNetwork, CognitiveMetaAgent
from mirage.models.tmn import TMNAttentionLSTM, TopologicalMemoryNetwork
from mirage.models.mcs import MetamorphicCryptographicShard, ThreatSeverityScorer
from mirage.models.cce import CausalCounterfactualEnvironment
from mirage.models.sgi import DQNetwork, SpectralGameIntelligence

__all__ = [
    # CMA
    "CMAPolicyNetwork",
    "CognitiveMetaAgent",
    # TMN
    "TMNAttentionLSTM",
    "TopologicalMemoryNetwork",
    # MCS
    "MetamorphicCryptographicShard",
    "ThreatSeverityScorer",
    # CCE
    "CausalCounterfactualEnvironment",
    # SGI
    "DQNetwork",
    "SpectralGameIntelligence",
]
