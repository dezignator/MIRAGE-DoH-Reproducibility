# Dataset and preprocessing documentation

This document explains where the data used by the MIRAGE-DoH artifact comes
from, how it is processed, and how the artifact handles the two situations
where the original data cannot be redistributed or reproduced.

## 1. Datasets used in Chapter 3

| Dataset | Source | Size | Licence |
|---|---|---|---|
| **CIC-DoHBRW-2023** | https://www.unb.ca/cic/datasets/dohbrw-2023.html (Canadian Institute for Cybersecurity, UNB) | ≈ 22 GB | CC-BY 4.0 |
| **Gen-C-DDD-2022** | https://www.unb.ca/cic/datasets/dns-2022.html (CIC, UNB) | ≈ 10 GB | CC-BY 4.0 |
| **Zenodo mirror** | https://doi.org/10.5281/zenodo.5957677 | — | CC-BY 4.0 |

None of these datasets is redistributed inside this repository. The download
script `data/download_datasets.py` fetches them from their official sources
and verifies SHA-256 checksums against the TOFU manifest in
`data/checksums.json` (see §5 below).

## 2. Canonical 46-feature schema

Every row of processed data has exactly 47 columns: one `label` column
(0 = benign, 1 = malicious) plus the 46 metadata features listed in
`configs/cma_hyperparameters.yaml §6`. The list is the single source of
truth and is reproduced in `data/_data_utils.py::FEATURE_GROUPS`; a
runtime assertion checks that the two match.

The 46 features are grouped into five categories, one per feature
category in thesis Section 3.11.1:

| Group | Count | Examples |
|---|---|---|
| `temporal` | 8 | inter_arrival_mean, burst_count |
| `size` | 9 | query_size_mean, query_response_ratio_std |
| `tls` | 8 | cipher_suite_count, sni_present, tls_version_code |
| `http3_frames` | 8 | frame_count, control_frame_ratio |
| `flow` | 13 | packets_in, direction_ratio, jitter_mean_ms |

Total: 8 + 9 + 8 + 8 + 13 = **46**.

## 3. Preprocessing pipeline
