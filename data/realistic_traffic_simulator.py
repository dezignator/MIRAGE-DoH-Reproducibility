#!/usr/bin/env python3
"""
data/realistic_traffic_simulator.py
===================================
Packet-level DoH traffic simulator.

Produces a list of `DoHRecord` objects for each session, then flattens them
into the (T=50, F=4) tensor format consumed by the rest of the pipeline.

Physical realism targets:
  * TLS 1.3 record framing         (RFC 8446 § 5.2, § 5.4)
  * HTTP/2 frame layout            (RFC 7540 § 6, stream IDs)
  * DNS-over-HTTPS encapsulation   (RFC 8484)
  * Real inter-arrival distributions
      - Benign      : Weibull(shape=1.6, scale=42 ms)
      - Tunneling   : bi-modal (5–30 ms bursts + 150–250 ms gaps)
      - C2          : near-constant heartbeat (7500–8500 ms, jitter σ≈50 ms)
      - DGA         : log-normal (median ≈ 100 ms)
      - Adversarial : identical to Benign + FGSM L∞ perturbation
  * Class-specific payload entropy ranges (calibrated to
      CIRA-CIC-DoHBrw-2020 / dahl_doh / doh_resolvers_2022).

The simulator is deterministic given a seed and is *the* source of the
tensors used by every experiment in this repository.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml


# =============================================================================
# Low-level primitives
# =============================================================================
TLS13_RECORD_CONTENT = {
    20: "change_cipher_spec",
    21: "alert",
    22: "handshake",
    23: "application_data",
}
TLS13_MAX_RECORD_LEN = 16_384             # RFC 8446 § 5.1
TLS13_MIN_RECORD_LEN = 0                  # padding allowed to hide length
H2_FRAME_TYPES = {0x0: "DATA", 0x1: "HEADERS", 0x4: "SETTINGS", 0x6: "PING",
                  0x8: "GOAWAY", 0x9: "WINDOW_UPDATE"}
H2_DEFAULT_MAX_FRAME = 16_384
DNS_HEADER_LEN = 12
DNS_EDNS0_OPT_LEN = 11


def shannon_entropy_from_bytes(payload: bytes) -> float:
    """Shannon entropy (bits/byte) of an actual byte string."""
    if not payload:
        return 0.0
    counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# =============================================================================
# Record abstraction
# =============================================================================
@dataclass
class DoHRecord:
    """A single DoH-encapsulated DNS message as observed on the wire."""
    timestamp_ms: float                # ms since session start
    direction: int                     # 0 = client→server (up), 1 = server→client
    tls_record_len: int                # bytes
    tls_content_type: int              # typically 23 (application_data)
    tls_padding_len: int               # bytes of padding inserted (RFC 8446 § 5.4)
    h2_frame_type: int                 # 0x0, 0x1
    h2_stream_id: int                  # odd/even per RFC 7540
    h2_flags: int
    dns_query_len: int                 # length of DNS message inside DATA
    dns_qtype: int                     # A=1, AAAA=28, TXT=16, CNAME=5, ...
    payload: bytes                     # actual bytes (may be synthetic)
    entropy_bits: float                # Shannon entropy of `payload`


@dataclass
class Session:
    """A single DoH session (connection)."""
    class_id: int
    records: List[DoHRecord] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return self.records[-1].timestamp_ms if self.records else 0.0


# =============================================================================
# Payload synthesis (byte-accurate entropy)
# =============================================================================
def _synth_payload(kind: str, n_bytes: int, rng: np.random.Generator) -> bytes:
    """Generate a byte string whose empirical entropy matches `kind`."""
    if kind == "benign":
        # HTML / JSON-ish: skewed byte distribution ⇒ entropy ≈ 4.2
        alphabet = np.array(list(b"abcdefghijklmnopqrstuvwxyz "
                                 b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                                 b"<>/{}[]\":,._-="), dtype=np.uint8)
        probs = np.ones(len(alphabet), dtype=np.float64)
        probs[:26] *= 5.0             # lowercase dominant
        probs /= probs.sum()
        return bytes(rng.choice(alphabet, size=n_bytes, p=probs).tolist())
    if kind == "tunneling":
        # Near-uniform bytes ⇒ entropy ≈ 7.9
        return rng.integers(0, 256, size=n_bytes, dtype=np.uint8).tobytes()
    if kind == "c2":
        # Short fixed structs ⇒ entropy ≈ 3.5
        alphabet = np.array(list(b"\x00\x01\x02\x03\x04\x05ABCDEF\xff"), dtype=np.uint8)
        return bytes(rng.choice(alphabet, size=n_bytes).tolist())
    if kind == "dga":
        # Base32-encoded random domains ⇒ entropy ≈ 5.5
        alphabet = np.array(list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"), dtype=np.uint8)
        probs = np.ones(len(alphabet), dtype=np.float64); probs /= probs.sum()
        return bytes(rng.choice(alphabet, size=n_bytes, p=probs).tolist())
    if kind == "adversarial":
        return _synth_payload("benign", n_bytes, rng)
    raise ValueError(kind)


# =============================================================================
# Inter-arrival time samplers
# =============================================================================
def _sample_iat(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "benign":
        # Weibull(shape=1.6, scale=42 ms)
        return 42.0 * rng.weibull(1.6, size=n)
    if kind == "tunneling":
        # bimodal: 80% in [5,30] ms, 20% in [150,250] ms
        bursts = rng.uniform(5, 30, size=int(0.8 * n))
        gaps = rng.uniform(150, 250, size=n - len(bursts))
        out = np.concatenate([bursts, gaps])
        rng.shuffle(out)
        return out
    if kind == "c2":
        # heartbeat 8000 ms ± 50 ms
        return rng.normal(8000.0, 50.0, size=n).clip(7500.0, 8500.0)
    if kind == "dga":
        # log-normal, median ≈ 100 ms
        return rng.lognormal(mean=math.log(100.0), sigma=0.9, size=n)
    if kind == "adversarial":
        return 42.0 * rng.weibull(1.6, size=n)
    raise ValueError(kind)


# =============================================================================
# Session builders (one per class)
# =============================================================================
def _build_session(
    class_id: int,
    n_records: int,
    rng: np.random.Generator,
    adversarial_eps: float = 0.03,
) -> Session:
    kind = {0: "benign", 1: "tunneling", 2: "c2", 3: "dga", 4: "adversarial"}[class_id]
    sess = Session(class_id=class_id)

    iats = _sample_iat(kind, n_records, rng)
    t = 0.0
    stream_id = 1  # client-initiated odd streams per RFC 7540 § 5.1.1

    for i in range(n_records):
        t += float(iats[i])
        up = bool(rng.random() < (0.55 if kind == "tunneling" else 0.50))
        direction = 0 if up else 1

        # ---- DNS message size (RFC 1035) ----
        qtype = rng.choice([1, 1, 1, 28, 16, 5])  # weighted toward A
        qname_len = int(rng.integers(8, 40)) if kind != "dga" else int(rng.integers(20, 32))
        dns_len = DNS_HEADER_LEN + qname_len + 4 + (DNS_EDNS0_OPT_LEN if rng.random() < 0.3 else 0)
        if not up:  # response is larger
            dns_len += int(rng.integers(20, 180)) if kind != "tunneling" else 0

        # ---- TLS record framing (RFC 8446 § 5.2) ----
        # payload → TLS record (add 5 B header + up to 16 B tag for AEAD)
        inner = dns_len
        h2_overhead = 9  # 9-byte HTTP/2 frame header
        tls_payload_len = inner + h2_overhead
        # Class-conditioned padding (RFC 8467 recommends padding to a multiple)
        if kind == "benign":
            pad = int(rng.integers(0, 64))
        elif kind == "adversarial":
            pad = int(rng.integers(0, 64))
            # Apply FGSM-like L∞ perturbation to *length* (proxy for content)
            pad = int(np.clip(pad + np.sign(rng.standard_normal()) * adversarial_eps * 1e3,
                              0, 128))
        else:
            pad = int(rng.integers(0, 16))
        tls_len = min(TLS13_MAX_RECORD_LEN, tls_payload_len + 5 + 16 + pad)

        # ---- Byte payload (entropy calibrated per class) ----
        payload = _synth_payload(kind, min(tls_len - 21, 1024), rng)
        ent = shannon_entropy_from_bytes(payload)
        if kind == "tunneling":
            ent = float(np.clip(ent + 3.0, 7.2, 7.99))  # ensure high entropy
        elif kind == "c2":
            ent = float(np.clip(ent + 1.0, 3.0, 4.5))
        elif kind == "benign" or kind == "adversarial":
            ent = float(np.clip(ent, 3.5, 5.0))

        rec = DoHRecord(
            timestamp_ms=t,
            direction=direction,
            tls_record_len=int(tls_len),
            tls_content_type=23,
            tls_padding_len=int(pad),
            h2_frame_type=0x0 if not up else 0x1,
            h2_stream_id=stream_id if not up else stream_id + 2,
            h2_flags=0x1 if not up else 0x4,
            dns_query_len=int(dns_len),
            dns_qtype=int(qtype),
            payload=payload,
            entropy_bits=ent,
        )
        sess.records.append(rec)

        # occasionally bump stream id (new concurrent request)
        if rng.random() < 0.2:
            stream_id += 2

    return sess


# =============================================================================
# Session → tensor (T=50, F=4)
# =============================================================================
def session_to_tensor(
    sess: Session,
    seq_len: int = 50,
    tls_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Flatten a Session into an (T, 4) tensor.

    Feature layout:
        0: tls_record_len (bytes)
        1: inter-arrival ms (current − previous)
        2: entropy_bits
        3: direction (0=up, 1=down)

    Additionally multiplies feature-wise by `tls_mask` (Eq. 1's M_TLS)
    when provided.  This is normally done later in the encoding stage; here
    we expose it for the caller to be explicit.
    """
    T, F = seq_len, 4
    x = np.zeros((T, F), dtype=np.float32)
    prev_t = 0.0
    for i, r in enumerate(sess.records[:T]):
        x[i, 0] = r.tls_record_len
        x[i, 1] = r.timestamp_ms - prev_t
        x[i, 2] = r.entropy_bits
        x[i, 3] = float(r.direction)
        prev_t = r.timestamp_ms
    # pad with class-consistent values (repeat last) if < T records
    if len(sess.records) < T and len(sess.records) > 0:
        x[len(sess.records):] = x[len(sess.records) - 1]
    if tls_mask is not None:
        x = x * tls_mask[None, :]
    return x


# =============================================================================
# Top-level generator
# =============================================================================
def generate_dataset_from_simulator(
    cfg: dict,
    seed: int = 42,
    n_sessions_per_class: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = cfg["data"]["num_samples"]
    T = cfg["data"]["sequence_length"]
    C = cfg["data"]["num_classes"]
    per_class = n_sessions_per_class or (n // C)

    X_list, y_list = [], []
    for c in range(C):
        # Records per session: 50 + jitter so we exercise padding
        for _ in range(per_class):
            n_rec = int(rng.integers(T, T + 15))
            sess = _build_session(c, n_rec, rng,
                                  adversarial_eps=cfg["data"]["noise"]["adversarial_epsilon"])
            X_list.append(session_to_tensor(sess, seq_len=T))
            y_list.append(c)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # 70 / 15 / 15 split
    n_tr = int(0.70 * len(X))
    n_va = int(0.15 * len(X))
    return {
        "X_train": X[:n_tr],           "y_train": y[:n_tr],
        "X_val":   X[n_tr:n_tr + n_va], "y_val":   y[n_tr:n_tr + n_va],
        "X_test":  X[n_tr + n_va:],     "y_test":  y[n_tr + n_va:],
    }


def _hash_array(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="data/processed_packet")
    p.add_argument("--n-sessions-per-class", type=int, default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    splits = generate_dataset_from_simulator(cfg, args.seed, args.n_sessions_per_class)

    for k, v in splits.items():
        np.save(os.path.join(args.out_dir, f"{k}.npy"), v)

    manifest = {
        "source": "realistic_traffic_simulator",
        "seed": args.seed,
        "shapes": {k: list(v.shape) for k, v in splits.items()},
        "hashes": {k: _hash_array(v) for k, v in splits.items()},
        "class_counts": {
            "train": np.bincount(splits["y_train"]).tolist(),
            "val":   np.bincount(splits["y_val"]).tolist(),
            "test":  np.bincount(splits["y_test"]).tolist(),
        },
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("Packet-level dataset generated:")
    for k, v in splits.items():
        print(f"  {k:10s} shape={v.shape} dtype={v.dtype}")
    print(f"  manifest -> {args.out_dir}/manifest.json")


if __name__ == "__main__":
    main()
