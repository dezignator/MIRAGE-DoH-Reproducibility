"""
cce.py — Causal Counterfactual Environments (CCE).

Thesis reference: Chapter 3, Section 3.8.1 and 3.8.2, Equations 14-16,
and Section 3.13 (zero-day response time).

Implements:
- The structural causal model (SCM) with observed, latent, and exogenous variables.
- do-calculus interventions (Eq. 15).
- The counterfactual replay protocol (Section 3.8.2).
- The protocol hardening update rule (Eq. 16).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from mirage.utils.math_utils import (
    do_calculus_intervention,
    protocol_update_rule,
    scm_joint_distribution,
)

logger = logging.getLogger(__name__)


class CausalCounterfactualEnvironment:
    """Full CCE orchestrator.

    Manages the SCM, the counterfactual replay protocol, and the protocol
    hardening update rule.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        # The SCM's conditional probability tables (CPTs) are learned
        # from data in the full artifact; for the toy demo, we initialize
        # them to plausible defaults from Section 3.8.1.
        self.scm: Dict[str, float] = {
            "protocol_negotiation": 1.0,  # default: TLS 1.3
            "attack_stage": 0.0,          # default: benign
            "attacker_intent": 0.0,       # default: none
        }
        self.counterfactual_replays: List[Dict] = []

    def replay_attack(
        self,
        attack_name: str,
        intervention: Optional[Dict[str, float]] = None,
        num_samples: int = 1000,
    ) -> Dict[str, float]:
        """Replay an attack in a counterfactual environment (Section 3.8.2).

        Args:
            attack_name: which attack to replay (e.g., "ecs_attack").
            intervention: optional do-calculus intervention (Eq. 15).
            num_samples: number of Monte Carlo samples.

        Returns:
            Dict with keys: attack_prob, intervention_effects.
        """
        start_scm = dict(self.scm)
        if intervention:
            start_scm = do_calculus_intervention(start_scm, intervention)

        # Monte Carlo simulation of the SCM.
        successes = 0
        for _ in range(num_samples):
            sample = dict(start_scm)
            # Perturb observed variables with exogenous noise.
            sample["inter_arrival_mean"] = self.rng.normal(10.0, 5.0)
            sample["query_size_mean"] = self.rng.normal(60.0, 20.0)
            # Attack success predicate: protocol downgraded AND attack stage >= 3.
            if sample["protocol_negotiation"] < 1.0 and sample["attack_stage"] >= 3:
                successes += 1

        attack_prob = successes / num_samples

        result = {
            "attack_name": attack_name,
            "intervention": intervention,
            "attack_prob": attack_prob,
            "num_samples": num_samples,
        }
        self.counterfactual_replays.append(result)
        logger.info("Replayed %s: attack_prob=%.4f (intervention=%s)",
                    attack_name, attack_prob, intervention)
        return result

    def harden_protocol(
        self,
        attack_name: str,
        candidate_actions: List[str],
        num_samples: int = 1000,
    ) -> str:
        """Harden the protocol using Eq. 16.

        Runs counterfactual replays for each candidate action, then selects
        the action that minimizes (P_attack + cost).

        Args:
            attack_name: the attack to harden against.
            candidate_actions: list of actions to consider.
            num_samples: Monte Carlo samples per action.

        Returns:
            The selected action name.
        """
        probabilities: Dict[str, float] = {}
        costs: Dict[str, float] = {
            "allow": 0.0,
            "rate_limit": 0.1,
            "sandbox": 0.2,
            "block": 0.5,
        }
        interventions = {
            "allow": {},
            "rate_limit": {"protocol_negotiation": 1.0},
            "sandbox": {"protocol_negotiation": 1.0, "attack_stage": 0.0},
            "block": {"protocol_negotiation": 1.0, "attack_stage": 0.0},
        }

        for action in candidate_actions:
            intervention = interventions.get(action, {})
            result = self.replay_attack(attack_name, intervention, num_samples)
            probabilities[action] = result["attack_prob"]

        best_action = protocol_update_rule(candidate_actions, probabilities, costs)
        logger.info("Selected action %s for %s (probabilities=%s, costs=%s)",
                    best_action, attack_name, probabilities, costs)
        return best_action
