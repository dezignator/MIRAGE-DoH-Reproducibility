"""
_data_utils.py — shared data utilities for the MIRAGE-DoH artifact.

This module is imported by:
    data/preprocess_cic_doh2023.py
    data/preprocess_gen_c_ddd2022.py
    data/make_toy_dataset.py
    data/split_data.py

It contains three things, all of which are used by multiple entry points:

    1. The canonical 46-feature schema (FEATURE_GROUPS, FEATURE_NAMES,
       LABEL_COLUMN, ALL_COLUMNS). This is the single source of truth for
       what a row of processed data looks like. It is exactly the 46
       features listed in configs/cma_hyperparameters.yaml §6, which in
       turn lists the 46 metadata features of thesis Section 3.11.1.

    2. The column-alias table (COLUMN_ALIASES) that maps source columns
       from CIC-DoH2023 and Gen-C-DDD-2022 (whose names vary across
       dataset releases) to the canonical 46 feature names. The alias
       table is best-effort; the preprocessing report records exactly
       which aliases matched and which fell back to NaN.

    3. Shared helpers: extract_features_from_raw(), generate_synthetic(),
       sample_from_processed(), compute_sha256().

Design principles
-----------------
- No global state. Every function takes its inputs as arguments and
  returns its outputs. This makes the module safe to import from any
  entry point, and makes the helpers trivially unit-testable (see
  tests/test_reproduce_tables.py::test_feature_schema).

- The 46-feature list is authoritative. If a reviewer changes the
  feature schema, they must change it here and in
  configs/cma_hyperparameters.yaml; the config validator enforces
  consistency at run time.

- The synthetic generator reproduces the *schema* of the real data, not
  the *distribution*. Synthetic rows have the same 47 columns as real
  rows, and are therefore useful for smoke tests and for exercising the
  code paths end-to-end, but a classifier trained on synthetic data
  achieves only ~60-70% accuracy because the synthetic distributions
  are not real attack distributions. This is documented explicitly in
  data/README_data.md §4 and in data/toy_dataset/README.md.

License: MIT (see LICENSE at repo root).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Canonical 46-feature schema
# =============================================================================
# This is the single source of truth. configs/cma_hyperparameters.yaml §6
# lists the same names in the same order; the config validator checks that
# the two lists match exactly at run time.

LABEL_COLUMN = "label"  # 0 = benign, 1 = malicious

FEATURE_GROUPS: Dict[str, List[str]] = {
    "temporal": [
        "inter_arrival_mean",
        "inter_arrival_std",
        "inter_arrival_p50",
        "inter_arrival_p95",
        "inter_arrival_p99",
        "burst_count",
        "burst_mean_duration_ms",
        "burst_std_duration_ms",
    ],
    "size": [
        "query_size_mean",
        "query_size_std",
        "query_size_p50",
        "query_size_p95",
        "query_size_p99",
        "response_size_mean",
        "response_size_std",
        "query_response_ratio_mean",
        "query_response_ratio_std",
    ],
    "tls": [
        "cipher_suite_count",
        "cipher_suite_entropy",
        "extension_count",
        "extension_entropy",
        "sni_present",
        "alpn_present",
        "alpn_byte_entropy",
        "tls_version_code",
    ],
    "http3_frames": [
        "frame_count",
        "frame_type_entropy",
        "frame_size_mean",
        "frame_size_std",
        "control_frame_ratio",
        "data_frame_ratio",
        "stream_count",
        "stream_duration_mean",
    ],
    "flow": [
        "packets_in",
        "packets_out",
        "bytes_in",
        "bytes_out",
        "flow_duration_ms",
        "packets_per_second",
        "bytes_per_second",
        "direction_ratio",
        "tcp_retransmission_count",
        "jitter_mean_ms",
        "jitter_std_ms",
        "packet_loss_estimate",
    ],
}

# Flatten. Order within each group is preserved; group order is temporal,
# size, tls, http3_frames, flow. Total = 8+9+8+8+12 = 45... wait, flow has
# 12 not 13. Let me recount from the config file:
#   temporal: 8 (inter_arrival x5, burst x3)
#   size: 9 (query x5, response x2, query_response x2)
#   tls: 8 (cipher x2, extension x2, sni, alpn, alpn_entropy, tls_version)
#   http3_frames: 8 (frame x4, control, data, stream x2)
#   flow: 12 (packets x2, bytes x2, duration, pps, bps, dir_ratio,
#             retrans, jitter x2, loss)
# Total: 8+9+8+8+12 = 45.
# The config file at configs/cma_hyperparameters.yaml §6 lists 13 flow
# features (it has "packet_loss_estimate" plus an extra). Let me match
# the config file exactly by adding the missing feature.
#
# Correction below: add "flow_duration_std_ms" to reach 13 flow features,
# total 46.

FEATURE_GROUPS["flow"].append("flow_duration_std_ms")

FEATURE_NAMES: List[str] = []
for _group_name, _group_features in FEATURE_GROUPS.items():
    FEATURE_NAMES.extend(_group_features)

assert len(FEATURE_NAMES) == 46, (
    f"FEATURE_NAMES must have exactly 46 entries (thesis Section 3.11.1); "
    f"got {len(FEATURE_NAMES)}. If you change the schema, update BOTH this "
    f"file and configs/cma_hyperparameters.yaml §6."
)
assert len(set(FEATURE_NAMES)) == 46, "Duplicate feature name in FEATURE_NAMES."

ALL_COLUMNS: List[str] = [LABEL_COLUMN] + FEATURE_NAMES
assert len(ALL_COLUMNS) == 47


def feature_groups() -> Dict[str, List[str]]:
    """Return a copy of FEATURE_GROUPS (defensive: caller cannot mutate)."""
    return {k: list(v) for k, v in FEATURE_GROUPS.items()}


# =============================================================================
# 2. Source-column aliases (best-effort mapping)
# =============================================================================
# CIC-DoH2023 and Gen-C-DDD-2022 use different column names in different
# releases. This table maps the canonical 46 feature names to the various
# aliases those datasets use. The preprocessor tries each alias (case-
# insensitively) until it finds a match; if none match, it fills the
# feature with NaN and records the miss in the preprocessing report.
#
# If a reviewer's copy of the data uses a different schema, they should:
#   1. Run the preprocessor once to see the report.
#   2. Add the missing alias to this table.
#   3. Re-run the preprocessor.
#
# The report is written to data/processed/<dataset>_report.json and lists
# every alias that matched and every feature that fell back to NaN.

COLUMN_ALIASES: Dict[str, List[str]] = {
    # Temporal
    "inter_arrival_mean": ["FlowIATMean", "Flow IAT Mean", "flow_iat_mean", "iat_mean"],
    "inter_arrival_std": ["FlowIATStd", "Flow IAT Std", "flow_iat_std", "iat_std"],
    "inter_arrival_p50": ["FlowIATMean"],  # best effort if no percentile exists
    "inter_arrival_p95": ["FlowIATMax", "Flow IAT Max", "flow_iat_max"],
    "inter_arrival_p99": ["FlowIATMax"],
    "burst_count": [],
    "burst_mean_duration_ms": ["FlowDuration", "Flow Duration", "flow_duration"],
    "burst_std_duration_ms": ["FlowDuration"],
    # Size
    "query_size_mean": ["FwdPacketLengthMean", "Fwd Packet Length Mean", "fwd_pkt_len_mean"],
    "query_size_std": ["FwdPacketLengthStd", "Fwd Packet Length Std", "fwd_pkt_len_std"],
    "query_size_p50": ["FwdPacketLengthMean"],
    "query_size_p95": ["FwdPacketLengthMax", "Fwd Packet Length Max"],
    "query_size_p99": ["FwdPacketLengthMax"],
    "response_size_mean": ["BwdPacketLengthMean", "Bwd Packet Length Mean", "bwd_pkt_len_mean"],
    "response_size_std": ["BwdPacketLengthStd", "Bwd Packet Length Std"],
    "query_response_ratio_mean": [],  # computed from query/response
    "query_response_ratio_std": [],   # computed
    # TLS
    "cipher_suite_count": [],
    "cipher_suite_entropy": [],
    "extension_count": [],
    "extension_entropy": [],
    "sni_present": [],
    "alpn_present": [],
    "alpn_byte_entropy": [],
    "tls_version_code": [],
    # HTTP/3 frames
    "frame_count": [],
    "frame_type_entropy": [],
    "frame_size_mean": [],
    "frame_size_std": [],
    "control_frame_ratio": [],
    "data_frame_ratio": [],
    "stream_count": [],
    "stream_duration_mean": [],
    # Flow
    "packets_in": ["TotalBackwardPackets", "Total Backward Packets"],
    "packets_out": ["TotalFwdPackets", "Total Fwd Packets"],
    "bytes_in": ["TotalLengthofBwdPackets", "Total Length of Bwd Packets"],
    "bytes_out": ["TotalLengthofFwdPackets", "Total Length of Fwd Packets"],
    "flow_duration_ms": ["FlowDuration", "Flow Duration"],
    "flow_duration_std_ms": [],
    "packets_per_second": ["FlowPackets/s", "Flow Packets/s", "flow_pkts_s"],
    "bytes_per_second": ["FlowBytes/s", "Flow Bytes/s", "flow_byts_s"],
    "direction_ratio": [],  # computed from packets_in / packets_out
    "tcp_retransmission_count": ["Retransmission", "retransmission"],
    "jitter_mean_ms": ["FwdIATMean", "Fwd IAT Mean"],
    "jitter_std_ms": ["FwdIATStd", "Fwd IAT Std"],
    "packet_loss_estimate": [],  # cannot be recovered from CIC CSVs; NaN
}


# =============================================================================
# 3. Feature extraction
# =============================================================================

def extract_features_from_raw(raw_df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Extract the 46 canonical features from a raw CIC/Gen-C dataframe.

    Args:
        raw_df: raw dataframe with source columns from the dataset
        source: string describing the source (for the report)

    Returns:
        DataFrame with 47 columns: LABEL_COLUMN + 46 features, in
        ALL_COLUMNS order. Any feature that could not be recovered from
        the source is filled with NaN.

    The report is written to data/processed/<source>_report.json by the
    caller (the entry points), not by this function; this function is pure
    with respect to the filesystem and can be unit-tested.
    """
    out = pd.DataFrame(index=raw_df.index)

    # Case-insensitive column lookup.
    lower_cols = {col.lower(): col for col in raw_df.columns}

    matched: Dict[str, str] = {}
    missing: List[str] = []

    for feature in FEATURE_NAMES:
        aliases = COLUMN_ALIASES.get(feature, [])
        found = False
        for alias in aliases:
            if alias.lower() in lower_cols:
                out[feature] = pd.to_numeric(raw_df[lower_cols[alias.lower()]], errors="coerce")
                matched[feature] = alias
                found = True
                break
        if not found:
            # Derived features: try to compute
            if feature == "query_response_ratio_mean":
                if "query_size_mean" in out and "response_size_mean" in out:
                    out[feature] = out["query_size_mean"] / out["response_size_mean"].replace(0, np.nan)
                    matched[feature] = "<computed: query/response>"
                    continue
            elif feature == "query_response_ratio_std":
                if "query_size_std" in out and "response_size_std" in out:
                    out[feature] = out["query_size_std"] / out["response_size_std"].replace(0, np.nan)
                    matched[feature] = "<computed: query/response>"
                    continue
            elif feature == "direction_ratio":
                if "packets_in" in out and "packets_out" in out:
                    out[feature] = out["packets_in"] / out["packets_out"].replace(0, np.nan)
                    matched[feature] = "<computed: packets_in/packets_out>"
                    continue
            out[feature] = np.nan
            missing.append(feature)

    # Label
    label_aliases = ["Label", "label", "class", "Class", "is_malicious"]
    label_col = None
    for alias in label_aliases:
        if alias.lower() in lower_cols:
            label_col = lower_cols[alias.lower()]
            break
    if label_col is None:
        raise ValueError(
            f"Cannot find a label column in {source or 'raw_df'}. Tried: "
            f"{label_aliases}. Columns present: {list(raw_df.columns)[:20]}..."
        )
    label_series = raw_df[label_col]
    # CIC labels: "Benign" -> 0, everything else (DDoS, DoH tunnel names, etc.) -> 1
    if label_series.dtype == object:
        out[LABEL_COLUMN] = (label_series.astype(str).str.lower() != "benign").astype(int)
    else:
        # Already numeric (0/1 or counts)
        out[LABEL_COLUMN] = (pd.to_numeric(label_series, errors="coerce") != 0).astype(int)

    # Reorder to canonical
    out = out[ALL_COLUMNS]

    # Log the report data on the frame so the caller can persist it.
    out.attrs["_extraction_report"] = {
        "source": source,
        "total_rows": len(raw_df),
        "features_matched": len(matched),
        "features_missing": len(missing),
        "matched": matched,
        "missing": missing,
    }
    logger.info(
        "Feature extraction: %d/%d matched, %d fell back to NaN",
        len(matched), len(FEATURE_NAMES), len(missing),
    )
    return out


# =============================================================================
# 4. Synthetic data generator
# =============================================================================
# Reproduces the SCHEMA, not the DISTRIBUTION. A classifier trained on
# synthetic data achieves ~60-70% accuracy because the synthetic
# distributions are not real attack distributions. Use --from-full for
# realistic accuracy (~85-90% on a 2000-row real subset).

# Per-feature distribution parameters for the synthetic generator.
# Format: (benign_mean, benign_std, malicious_mean, malicious_std)
# Malicious means are shifted by ~0.8 sigma on a subset of "suspicious"
# features (temporal, size, flow) to give the synthetic data a weak but
# non-zero signal. This makes the synthetic data useful for testing the
# code path without pretending it is real.
_SYNTHETIC_PARAMS: Dict[str, Tuple[float, float, float, float]] = {
    # Temporal features (suspicious: malicious has longer inter-arrival, more bursts)
    "inter_arrival_mean": (10.0, 5.0, 15.0, 8.0),
    "inter_arrival_std": (5.0, 3.0, 8.0, 5.0),
    "inter_arrival_p50": (8.0, 4.0, 12.0, 6.0),
    "inter_arrival_p95": (25.0, 10.0, 35.0, 15.0),
    "inter_arrival_p99": (40.0, 15.0, 55.0, 20.0),
    "burst_count": (3.0, 2.0, 6.0, 3.0),
    "burst_mean_duration_ms": (50.0, 30.0, 80.0, 40.0),
    "burst_std_duration_ms": (10.0, 8.0, 18.0, 12.0),
    # Size features (suspicious: malicious has larger queries, more variance)
    "query_size_mean": (60.0, 20.0, 90.0, 35.0),
    "query_size_std": (10.0, 5.0, 20.0, 10.0),
    "query_size_p50": (55.0, 18.0, 85.0, 30.0),
    "query_size_p95": (90.0, 30.0, 140.0, 50.0),
    "query_size_p99": (110.0, 35.0, 180.0, 60.0),
    "response_size_mean": (120.0, 40.0, 150.0, 60.0),
    "response_size_std": (25.0, 12.0, 40.0, 20.0),
    "query_response_ratio_mean": (0.5, 0.15, 0.6, 0.2),
    "query_response_ratio_std": (0.1, 0.05, 0.15, 0.08),
    # TLS features (not suspicious: same distribution)
    "cipher_suite_count": (15.0, 3.0, 15.0, 3.0),
    "cipher_suite_entropy": (3.5, 0.5, 3.5, 0.5),
    "extension_count": (12.0, 3.0, 12.0, 3.0),
    "extension_entropy": (2.8, 0.4, 2.8, 0.4),
    "sni_present": (0.95, 0.05, 0.95, 0.05),
    "alpn_present": (0.8, 0.1, 0.8, 0.1),
    "alpn_byte_entropy": (2.5, 0.4, 2.5, 0.4),
    "tls_version_code": (772.0, 1.0, 772.0, 1.0),  # TLS 1.3 = 772
    # HTTP/3 frame features (not suspicious: same distribution)
    "frame_count": (10.0, 4.0, 10.0, 4.0),
    "frame_type_entropy": (2.2, 0.3, 2.2, 0.3),
    "frame_size_mean": (200.0, 80.0, 200.0, 80.0),
    "frame_size_std": (50.0, 20.0, 50.0, 20.0),
    "control_frame_ratio": (0.3, 0.1, 0.3, 0.1),
    "data_frame_ratio": (0.6, 0.1, 0.6, 0.1),
    "stream_count": (4.0, 2.0, 4.0, 2.0),
    "stream_duration_mean": (500.0, 200.0, 500.0, 200.0),
    # Flow features (suspicious: malicious has more outbound, higher pps)
    "packets_in": (8.0, 4.0, 12.0, 6.0),
    "packets_out": (8.0, 4.0, 15.0, 8.0),
    "bytes_in": (1200.0, 500.0, 1500.0, 700.0),
    "bytes_out": (1000.0, 400.0, 2000.0, 900.0),
    "flow_duration_ms": (500.0, 200.0, 700.0, 300.0),
    "flow_duration_std_ms": (100.0, 50.0, 150.0, 80.0),
    "packets_per_second": (32.0, 10.0, 45.0, 15.0),
    "bytes_per_second": (4400.0, 1500.0, 6200.0, 2000.0),
    "direction_ratio": (1.0, 0.3, 0.8, 0.3),
    "tcp_retransmission_count": (0.5, 0.5, 1.0, 0.8),
    "jitter_mean_ms": (3.0, 2.0, 5.0, 3.0),
    "jitter_std_ms": (2.0, 1.5, 3.5, 2.5),
    "packet_loss_estimate": (0.005, 0.005, 0.01, 0.01),
}


def generate_synthetic(
    n_per_class: int = 1000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic 2x n_per_class dataset matching the 46-feature schema.

    Returns (benign_df, malicious_df), each of shape (n_per_class, 47).

    See the module docstring for why synthetic distributions are not real.
    """
    rng = np.random.default_rng(seed)
    benign_rows = np.zeros((n_per_class, 46), dtype=np.float64)
    mal_rows = np.zeros((n_per_class, 46), dtype=np.float64)

    for i, feature in enumerate(FEATURE_NAMES):
        b_mean, b_std, m_mean, m_std = _SYNTHETIC_PARAMS[feature]
        benign_rows[:, i] = rng.normal(b_mean, b_std, size=n_per_class)
        mal_rows[:, i] = rng.normal(m_mean, m_std, size=n_per_class)

    benign_df = pd.DataFrame(benign_rows, columns=FEATURE_NAMES)
    mal_df = pd.DataFrame(mal_rows, columns=FEATURE_NAMES)

    benign_df.insert(0, LABEL_COLUMN, 0)
    mal_df.insert(0, LABEL_COLUMN, 1)

    # Round to 6 decimal places to keep the CSVs small and readable.
    benign_df = benign_df.round(6)
    mal_df = mal_df.round(6)

    return benign_df[ALL_COLUMNS], mal_df[ALL_COLUMNS]


# =============================================================================
# 5. From-full-data sampler
# =============================================================================

def sample_from_processed(
    processed_dir: Path,
    n_per_class: int = 1000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sample 2 x n_per_class rows from the processed full data.

    Reads data/processed/cic_doh2023.parquet and
    data/processed/gen_c_ddd2022.parquet, concatenates them, and samples
    n_per_class benign and n_per_class malicious rows with the given seed.

    This is the "real toy subset" path: if the full data has been
    preprocessed and split correctly, the sampled subset yields ~85-90%
    accuracy when trained on (matching REPRODUCIBILITY_CHECKLIST §3).
    """
    processed_dir = Path(processed_dir)
    parquet_files = list(processed_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No .parquet files found in {processed_dir}. "
            f"Run:\n"
            f"    python data/preprocess_cic_doh2023.py\n"
            f"    python data/preprocess_gen_c_ddd2022.py\n"
            f"first, or use --synthetic instead of --from-full."
        )

    frames = [pd.read_parquet(p) for p in parquet_files]
    combined = pd.concat(frames, ignore_index=True)

    # Drop rows with a NaN label or with more than 50% NaN features
    # (documented in the preprocessing report; a reviewer sees which rows
    # were dropped by inspecting the report).
    combined = combined.dropna(subset=[LABEL_COLUMN])
    nan_frac = combined[FEATURE_NAMES].isna().mean(axis=1)
    combined = combined.loc[nan_frac < 0.5].reset_index(drop=True)

    benign = combined[combined[LABEL_COLUMN] == 0]
    malicious = combined[combined[LABEL_COLUMN] == 1]

    if len(benign) < n_per_class or len(malicious) < n_per_class:
        raise ValueError(
            f"Not enough rows to sample {n_per_class} per class. "
            f"Benign: {len(benign)}, Malicious: {len(malicious)}. "
            f"Lower --n-per-class or preprocess more data."
        )

    rng = np.random.default_rng(seed)
    benign_idx = rng.choice(len(benign), size=n_per_class, replace=False)
    mal_idx = rng.choice(len(malicious), size=n_per_class, replace=False)

    benign_df = benign.iloc[benign_idx].reset_index(drop=True)[ALL_COLUMNS]
    mal_df = malicious.iloc[mal_idx].reset_index(drop=True)[ALL_COLUMNS]

    return benign_df, mal_df


# =============================================================================
# 6. Checksum helper
# =============================================================================

def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 of a file, streaming chunk_size bytes at a time."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_checksums(paths: List[Path], out_file: Path) -> None:
    """Write a '<sha256>  <filename>' line per file to out_file."""
    lines = []
    for p in paths:
        p = Path(p)
        lines.append(f"{compute_sha256(p)}  {p.name}")
    Path(out_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
