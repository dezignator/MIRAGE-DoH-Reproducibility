# MIRAGE-DoH: Reproducibility Artifact

**Metamorphic Intelligence and Resilient AI Grid for Autonomous Governance of Encrypted DNS**

This repository is the formal reproducibility artifact for **Chapter 3** of the thesis
*Next-Generation Secure Encrypted DNS Architectures: Four Complementary Frameworks
for Adaptive, Privacy-Preserving, and Quantum-Resilient Defense* (Section 3.1–3.18).

It contains the complete implementation of the five MIRAGE-DoH components
(Cognitive MetaAgents, Topological Memory Networks, Metamorphic Cryptographic Shards,
Causal Counterfactual Environments, and Spectral Game Intelligence), the experimental
harness, the adversarial traffic generators, the CLI tools referenced in Table 6 and
Table 7 of the thesis, and the scripts that regenerate every table and figure in
Chapter 3 from raw inputs.

> **Status of the artifact** — This is an *executable appendix*, not a software
> release. It is intended for reviewers and future researchers who need to verify
> the quantitative claims of Chapter 3. The artifact is self-contained: it ships
> with a 1 MB toy dataset for a 2-minute smoke test, download scripts for the full
> public datasets (CIC-DoH2023 and Gen-C-DDD-2022), and software simulations of the
> Intel Loihi 2 and NVIDIA A100 measurements.

---

## 1. What this artifact demonstrates

| Claim in Chapter 3 | Reproduced by |
|---|---|
| Detection accuracy 99.98 %, FPR 0.0008 %, latency 78 µs, throughput 15.3 M QPS, morph time 8.9 ms (Table 8) | `scripts/reproduce_tables/table_8_core_metrics.py` |
| Advanced threat mitigation: 6.9× detection speedup, 42× FPR reduction, 1.44× topological recall, 1.56× spectral precision (Table 9) | `scripts/reproduce_tables/table_9_threat_mitigation.py` |
| CMA adaptive learning: F1 92.4 → 99.7, AUC 0.934 → 0.999, forgetting 8.3 → 0.0 (Table 10) | `scripts/reproduce_tables/table_10_cma_adaptive_learning.py` |
| Comparative defense vs. legacy DoH and vs. CNN-LSTM / Random Forest (Tables 11–12) | `scripts/reproduce_tables/table_11_core_metrics_repeated.py`, `table_12_comparative_defense.py` |
| Shard configurations Π₀, Π₁, Π₂ with ML-KEM-768/1024, X25519, Classic McEliece (Table 5, Eq. 13) | `scripts/reproduce_tables/table_5_mcs_configs.py` |
| Five attack scenarios and MIRAGE-DoH responses (Table 6) | `scripts/reproduce_tables/table_6_attack_scenarios.py` |
| Attack injection → detection → defense workflow (Table 7) | `scripts/reproduce_tables/table_7_defense_workflow.py` |
| MIRAGE-DoH metamorphic defense grid architecture (Figure 1) | `scripts/reproduce_figures/figure_1_architecture.py` |
| Three-module verification and validation under morph triggers (Figure 2) | `scripts/reproduce_figures/figure_2_module_validation.py` |
| Encrypted traffic dynamics under AI attacks, per-phase detection accuracy (Figure 3) | `scripts/reproduce_figures/figure_3_traffic_dynamics.py` |
| Cognitive learning across 100 autonomous training epochs (Figure 4) | `scripts/reproduce_figures/figure_4_cognitive_learning.py` |
| Joint optimization: confidence, latency, hyperparameter sensitivity, comparison (Figure 5) | `scripts/reproduce_figures/figure_5_joint_optimization.py` |
| Timeout resilience and cognitive security scalability under adaptive AI attacks (Figure 6) | `scripts/reproduce_figures/figure_6_adaptive_ai_attacks.py` |

Every row above has a dedicated script. Every script writes a deterministic
output file (`.json` or `.csv`) to `expected_outputs/` when run with `--emit-expected`.
The verification harness `verification_script.py` compares a fresh run against the
expected file within a tolerance bound (`np.allclose(..., rtol=0.05)` unless the
table requires a tighter bound).
