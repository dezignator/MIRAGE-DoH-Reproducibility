# Upload guide: MIRAGE-DoH-Reproducibility, extended_v2

## Method A: one file at a time

1. Repo > **Add file > Create new file**.
2. Type the exact path below (typing `/` creates folders).
3. Paste the file content from the zip, commit with the message in the last column.

## Method B: whole zip (faster)

1. Unzip `MIRAGE-DoH_extended_v2.zip`.
2. Repo > **Add file > Upload files**, drag in the `extended_v2` and `.github` folders (enable hidden files to see `.github`).
3. Commit: `feat: add extended_v2 artifact (implementations, experiments, results)`.

PNG files and `.github` need Method B.

## Files

| # | Exact path | Commit message |
|---|---|---|
| 1 | `extended_v2/README_v2.md` | docs: README for the v2 artifact |
| 2 | `extended_v2/Makefile` | build: v2 make targets |
| 3 | `extended_v2/requirements_v2.txt` | build: v2 requirements |
| 4 | `extended_v2/.gitignore` | chore: ignore v2 build outputs |
| 5 | `extended_v2/mirage_v2/__init__.py` | feat: package init |
| 6 | `extended_v2/mirage_v2/traffic_v2.py` | feat(data): synthetic encrypted-DNS scenarios and metadata features |
| 7 | `extended_v2/mirage_v2/topology_v2.py` | feat(tmn): Vietoris-Rips and flag-complex persistent homology |
| 8 | `extended_v2/mirage_v2/tmn_v2.py` | feat(tmn): topological memory network, relay-loop memory |
| 9 | `extended_v2/mirage_v2/cma_v2.py` | feat(cma): neuroevolution with Fisher penalty (Eqs. 1-3) |
| 10 | `extended_v2/mirage_v2/mlkem_v2.py` | feat(mcs): educational ML-KEM-512/768/1024 |
| 11 | `extended_v2/mirage_v2/mcs_v2.py` | feat(mcs): hybrid combiner, policy, handshake (Eqs. 11-13) |
| 12 | `extended_v2/mirage_v2/cce_v2.py` | feat(cce): exact causal inference (Eqs. 14-16) |
| 13 | `extended_v2/mirage_v2/sgi_v2.py` | feat(sgi): spectral state and DQN defender |
| 14 | `extended_v2/mirage_v2/levy_se3_v2.py` | feat: Levy jump test and SE(3) re-randomisation (Eqs. 22, 26) |
| 15 | `extended_v2/mirage_v2/metrics_v2.py` | feat: CTAI, intervals, ROC |
| 16 | `extended_v2/tests/__init__.py` | test: package init |
| 17 | `extended_v2/tests/test_mirage_core_v2.py` | test: traffic, topology, TMN, CMA, CCE, SE(3), SGI, metrics |
| 18 | `extended_v2/tests/test_mirage_crypto_v2.py` | test: ML-KEM, combiner, policy, handshake |
| 19 | `extended_v2/experiments/mirage_detection_experiments_v2.py` | feat(exp): detection suite D1-D6 |
| 20 | `extended_v2/experiments/mirage_crypto_causal_experiments_v2.py` | feat(exp): crypto and causal suite C1-C7 |
| 21 | `extended_v2/experiments/mirage_system_report_v2.py` | feat(exp): system scenarios S1-S6 and report |
| 22 | `extended_v2/docs/EQUATION_TO_CODE_MAP_v2.md` | docs: equation to code map |
| 23 | `extended_v2/docs/HARDWARE_DEPENDENCY_MATRIX_v2.md` | docs: hardware dependency matrix |
| 24 | `.github/workflows/mirage_v2_ci.yml` | ci: v2 unit tests |
| 25 | `extended_v2/results_v2/detect/d1_all_protocol.csv` | data: experiment output d1_all_protocol.csv |
| 26 | `extended_v2/results_v2/detect/d1_cost.csv` | data: experiment output d1_cost.csv |
| 27 | `extended_v2/results_v2/detect/d1_ctai.csv` | data: experiment output d1_ctai.csv |
| 28 | `extended_v2/results_v2/detect/d1_seen_protocol.csv` | data: experiment output d1_seen_protocol.csv |
| 29 | `extended_v2/results_v2/detect/d1_throughput.csv` | data: experiment output d1_throughput.csv |
| 30 | `extended_v2/results_v2/detect/d2_curriculum.csv` | data: experiment output d2_curriculum.csv |
| 31 | `extended_v2/results_v2/detect/d2_d4_curves.png` | data: experiment output d2_d4_curves.png (binary: use Upload files) |
| 32 | `extended_v2/results_v2/detect/d3_permutation.csv` | data: experiment output d3_permutation.csv |
| 33 | `extended_v2/results_v2/detect/d3_tmn_hops.csv` | data: experiment output d3_tmn_hops.csv |
| 34 | `extended_v2/results_v2/detect/d3_tmn_hops.png` | data: experiment output d3_tmn_hops.png (binary: use Upload files) |
| 35 | `extended_v2/results_v2/detect/d4_sgi_dqn.csv` | data: experiment output d4_sgi_dqn.csv |
| 36 | `extended_v2/results_v2/detect/d5_ablation.csv` | data: experiment output d5_ablation.csv |
| 37 | `extended_v2/results_v2/detect/d6_levy_jumps.csv` | data: experiment output d6_levy_jumps.csv |
| 38 | `extended_v2/results_v2/detect/detection_config_v2.json` | data: experiment output detection_config_v2.json |
| 39 | `extended_v2/results_v2/detect/detection_report_v2.md` | data: experiment output detection_report_v2.md |
| 40 | `extended_v2/results_v2/detect/d2_runs/lam0.0_seed0.json` | data: experiment output lam0.0_seed0.json |
| 41 | `extended_v2/results_v2/detect/d2_runs/lam0.0_seed1.json` | data: experiment output lam0.0_seed1.json |
| 42 | `extended_v2/results_v2/detect/d2_runs/lam0.0_seed2.json` | data: experiment output lam0.0_seed2.json |
| 43 | `extended_v2/results_v2/detect/d2_runs/lam10.0_seed0.json` | data: experiment output lam10.0_seed0.json |
| 44 | `extended_v2/results_v2/detect/d2_runs/lam10.0_seed1.json` | data: experiment output lam10.0_seed1.json |
| 45 | `extended_v2/results_v2/detect/d2_runs/lam10.0_seed2.json` | data: experiment output lam10.0_seed2.json |
| 46 | `extended_v2/results_v2/detect/d2_runs/lam100.0_seed0.json` | data: experiment output lam100.0_seed0.json |
| 47 | `extended_v2/results_v2/detect/d2_runs/lam100.0_seed1.json` | data: experiment output lam100.0_seed1.json |
| 48 | `extended_v2/results_v2/detect/d2_runs/lam100.0_seed2.json` | data: experiment output lam100.0_seed2.json |
| 49 | `extended_v2/results_v2/detect/d2_runs/lam1000.0_seed0.json` | data: experiment output lam1000.0_seed0.json |
| 50 | `extended_v2/results_v2/detect/d2_runs/lam1000.0_seed1.json` | data: experiment output lam1000.0_seed1.json |
| 51 | `extended_v2/results_v2/detect/d2_runs/lam1000.0_seed2.json` | data: experiment output lam1000.0_seed2.json |
| 52 | `extended_v2/results_v2/crypto/c1_mlkem.csv` | data: experiment output c1_mlkem.csv |
| 53 | `extended_v2/results_v2/crypto/c2_combiner_avalanche.csv` | data: experiment output c2_combiner_avalanche.csv |
| 54 | `extended_v2/results_v2/crypto/c2_combiner_broken_component.csv` | data: experiment output c2_combiner_broken_component.csv |
| 55 | `extended_v2/results_v2/crypto/c3_mceliece_pk.csv` | data: experiment output c3_mceliece_pk.csv |
| 56 | `extended_v2/results_v2/crypto/c3_morph_budget.csv` | data: experiment output c3_morph_budget.csv |
| 57 | `extended_v2/results_v2/crypto/c3_morph_compute.csv` | data: experiment output c3_morph_compute.csv |
| 58 | `extended_v2/results_v2/crypto/c4_policy.csv` | data: experiment output c4_policy.csv |
| 59 | `extended_v2/results_v2/crypto/c5_actions.csv` | data: experiment output c5_actions.csv |
| 60 | `extended_v2/results_v2/crypto/c5_causal.csv` | data: experiment output c5_causal.csv |
| 61 | `extended_v2/results_v2/crypto/c5_confounding_sweep.csv` | data: experiment output c5_confounding_sweep.csv |
| 62 | `extended_v2/results_v2/crypto/c6_se3_checks.csv` | data: experiment output c6_se3_checks.csv |
| 63 | `extended_v2/results_v2/crypto/c6_se3_predictability.csv` | data: experiment output c6_se3_predictability.csv |
| 64 | `extended_v2/results_v2/crypto/c7_audit.csv` | data: experiment output c7_audit.csv |
| 65 | `extended_v2/results_v2/crypto/crypto_config_v2.json` | data: experiment output crypto_config_v2.json |
| 66 | `extended_v2/results_v2/crypto/crypto_report_v2.md` | data: experiment output crypto_report_v2.md |
| 67 | `extended_v2/results_v2/system/MIRAGE_Chapter3_Experiment_Report_v2.html` | data: experiment output MIRAGE_Chapter3_Experiment_Report_v2.html |
| 68 | `extended_v2/results_v2/system/MIRAGE_Chapter3_Experiment_Report_v2.md` | data: experiment output MIRAGE_Chapter3_Experiment_Report_v2.md |
| 69 | `extended_v2/results_v2/system/claims_vs_simulation_v2.csv` | data: experiment output claims_vs_simulation_v2.csv |
| 70 | `extended_v2/results_v2/system/s1_catalogue.csv` | data: experiment output s1_catalogue.csv |
| 71 | `extended_v2/results_v2/system/s2_pipeline.csv` | data: experiment output s2_pipeline.csv |
| 72 | `extended_v2/results_v2/system/s2_pipeline.png` | data: experiment output s2_pipeline.png (binary: use Upload files) |
| 73 | `extended_v2/results_v2/system/s3_resilience.csv` | data: experiment output s3_resilience.csv |
| 74 | `extended_v2/results_v2/system/s3_resilience.png` | data: experiment output s3_resilience.png (binary: use Upload files) |
| 75 | `extended_v2/results_v2/system/s4_evasion.csv` | data: experiment output s4_evasion.csv |
| 76 | `extended_v2/results_v2/system/s4_evasion.png` | data: experiment output s4_evasion.png (binary: use Upload files) |
| 77 | `extended_v2/results_v2/system/s5_raw_f1.csv` | data: experiment output s5_raw_f1.csv |
| 78 | `extended_v2/results_v2/system/s5_spread.csv` | data: experiment output s5_spread.csv |
| 79 | `extended_v2/results_v2/system/s6_arithmetic_audit.csv` | data: experiment output s6_arithmetic_audit.csv |
