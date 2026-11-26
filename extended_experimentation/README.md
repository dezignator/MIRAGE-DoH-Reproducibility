# MIRAGE-DoH extended experimentation suite

This package contains the codebase that produces the extended experimental
artefacts for **Chapter 3** of the thesis *Next-Generation Secure Encrypted
DNS Architectures*. It is structured as a sequence of independently
runnable stages whose outputs feed one another, plus a unified CLI at
`extended_experimentation/cli.py` for discovery.

## Scope

The suite covers:

- campaign generation (polymorphic attack families, staged tunneling),
- topological memory network training,
- cognitive metaagent training with HMC-guided neuroevolution,
- metamorphic cryptographic shard benchmarking and calibration,
- spectral game intelligence (graph Laplacian + next-hop predictor),
- core metrics evaluation with tolerance assertions,
- statistical significance (paired bootstrap, Wilcoxon, MMD, Friedman),
- partial ablation study,
- result table and figure generation,
- runtime shard selection.

The suite deliberately does **not** contain a full factorial ablation or
a new cryptographic primitive. Both omissions are stated in the thesis
text and are reflected in the module docstrings.

## Dependencies

The pipeline stages have different optional dependencies. The package
imports cleanly without any of them; a stage that requires a missing
dependency raises `ImportError` when invoked.

| Stage | Requires |
|-------|----------|
| `generate-campaign`, `generate-tunneling`, `load-dataset` | `numpy` |
| `train-tmn`, `train-mirage`, `train-spectral`, `evaluate-core` | `numpy`, `torch` |
| `benchmark-mcs`, `calibrate-mcs` | `numpy`, optional `pqcrypto` / `cryptography` |
| `extract-tda` | `numpy` |
| `significance`, `ablation` | `numpy` |
| `tables` | `numpy` |
| `figures` | `numpy`, optional `matplotlib` |

If `pqcrypto` or `cryptography >= 42` are unavailable, the MCS benchmark
falls back to a calibrated timing model derived from the X25519 baseline
and the published operational ratios between KEM families. The fallback
is recorded in the benchmark manifest as `"backend": "calibrated_model"`.

## Pipeline

A typical end-to-end run:

```bash
# 1. Generate a campaign.
python -m extended_experimentation.cli generate-campaign \
    --out runs/campaign --seed 20231130

# 2. Train the TMN, MetaAgent and joint model.
python -m extended_experimentation.cli train-mirage \
    --campaign-root runs/campaign \
    --checkpoint-root runs/checkpoints \
    --seed 20231130 --strict

# 3. Evaluate core metrics against the checkpoint.
python -m extended_experimentation.cli evaluate-core \
    --checkpoint runs/checkpoints/joint/mirage_full_model.pt \
    --campaign-root runs/campaign \
    --out runs/eval --strict

# 4. Benchmark and calibrate the MCS policy.
python -m extended_experimentation.cli benchmark-mcs --out runs/mcs
python -m extended_experimentation.cli calibrate-mcs --out runs/mcs

# 5. Extract TDA artefacts and train the spectral predictor.
python -m extended_experimentation.cli extract-tda \
    --campaign-root runs/campaign --out runs/tda
python -m extended_experimentation.cli train-spectral \
    --campaign-root runs/campaign --out runs/spectral

# 6. Statistical significance and ablation.
python -m extended_experimentation.cli significance \
    --mirage runs/eval/mirage_predictions.jsonl \
    --baseline runs/eval/cnn_lstm.jsonl runs/eval/rf.jsonl \
    --out runs/significance
python -m extended_experimentation.cli ablation \
    --checkpoint-root runs/checkpoints --out runs/ablation

# 7. Reporting.
python -m extended_experimentation.cli tables \
    --core-metrics runs/eval/core_metrics.json --out runs/tables
python -m extended_experimentation.cli figures \
    --core-metrics runs/eval/core_metrics.json \
    --tables-manifest runs/tables/tables_manifest.json \
    --mcs-manifest runs/mcs/benchmark_manifest.json \
    --extraction-manifest runs/tda/extraction_manifest.json \
    --out runs/figures
