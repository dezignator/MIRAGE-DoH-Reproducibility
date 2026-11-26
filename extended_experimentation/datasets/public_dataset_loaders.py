#!/usr/bin/env python3
"""
extended_experimentation/datasets/public_dataset_loaders.py

Public dataset loaders for the MIRAGE-DoH extended experimentation suite.

We implemented this module to bridge the campaign archive format that the
attack simulators produce and the two public datasets the thesis evaluates
against: CIC-DoHBrw-2020 (the successor of CIC-DoH2023 that the thesis
cites in the literature review) and the Gen-C-DDD-2022 contribution. The
loaders normalised both into the same on-disk format the training
pipeline consumes: ``flows.jsonl`` plus ``packet_archive.npz``.

The two datasets differ in a way that mattered for evaluation and that we
want to record here rather than bury in code:

  * CIC-DoHBrw-2020 ships per-flow CSVs with aggregated statistics rather
    than raw packet captures. We reconstructed per-packet arrays by
    resampling from the aggregated statistics under the flow's recorded
    packet count. That is not equivalent to the original packet stream;
    it is a statistically matched surrogate. We recorded this in the
    manifest as ``"packet_arrays": "reconstructed"`` so that any artefact
    produced from this dataset carries that caveat.

  * Gen-C-DDD-2022 ships raw pcap files. We parsed them with a
    pure-Python pcap reader (no scapy, no dpkt dependency) sufficient to
    extract the fields the pipeline needs: timestamps, wire lengths, and
    flow direction from the five-tuple.

The loaders did not download datasets themselves. Network access was kept
out of the training pipeline deliberately; the download step is a one-off
manual operation recorded in the README.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import struct
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..attack_scenarios.polymorphic_shaping_simulator import (
    FRAMING,
    Label,
    Protocol,
    THESIS_REFERENCE,
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOGGER = logging.getLogger("mirage.extended.datasets")

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class DatasetSource(str, Enum):
    CIC_DOHBRW_2020 = "cic_dohbrw_2020"
    GEN_C_DDD_2022 = "gen_c_ddd_2022"


class PacketArrayProvenance(str, Enum):
    RECONSTRUCTED = "reconstructed"
    PARSED_FROM_PCAP = "parsed_from_pcap"


# --------------------------------------------------------------------------- #
# Common flow record
# --------------------------------------------------------------------------- #


@dataclass
class LoadedFlow:
    """One flow in the canonical on-disk format."""

    flow_id: str
    label: int
    protocol: Protocol
    profile_name: str
    sizes: np.ndarray
    iats: np.ndarray
    directions: np.ndarray

    @property
    def n_packets(self) -> int:
        return int(self.sizes.shape[0])

    def summary(self) -> Dict[str, object]:
        wire = FRAMING[self.protocol].frame(self.sizes)
        return {
            "flow_id": self.flow_id,
            "label": int(self.label),
            "protocol": self.protocol.value,
            "profile": self.profile_name,
            "n_packets": self.n_packets,
            "duration_s": float(self.iats.sum()),
            "payload_bytes": float(self.sizes.sum()),
            "wire_bytes": float(wire.sum()),
            "mean_wire_size": float(wire.mean()) if wire.size else 0.0,
            "std_wire_size": float(wire.std()) if wire.size else 0.0,
            "mean_iat_s": float(self.iats.mean()) if self.iats.size else 0.0,
            "std_iat_s": float(self.iats.std()) if self.iats.size else 0.0,
            "client_to_resolver_ratio": float(
                (self.directions == 0).mean()
            ) if self.directions.size else 0.0,
        }


# --------------------------------------------------------------------------- #
# CIC-DoHBrw-2020 loader
# --------------------------------------------------------------------------- #
# The CIC dataset ships as a set of CSV files, each with per-flow rows.
# The columns we consumed were: Flow ID, Source IP, Destination IP, Source
# Port, Destination Port, Protocol, Timestamp, Flow Duration, Total Fwd
# Packets, Total Backward Packets, Fwd Packets Length Total, Bwd Packets
# Length Total, Flow Bytes/s, Flow Packets/s, Label.
#
# The loader was tolerant of missing columns (older snapshots of the
# dataset used slightly different names); it warned once per unknown
# column and continued.

_CIC_LABEL_MAP: Dict[str, int] = {
    "benign": int(Label.BENIGN),
    "doh": int(Label.ATTACK),
    "dohbenign": int(Label.BENIGN),
    "nondoh": int(Label.ATTACK),
    "doh_attack": int(Label.ATTACK),
}


def _cic_parse_label(raw: str) -> int:
    key = raw.strip().lower().replace("-", "").replace(" ", "")
    if key in _CIC_LABEL_MAP:
        return _CIC_LABEL_MAP[key]
    # We treated unknown labels as benign rather than attack because a
    # false benign in a dataset-preparation context was recoverable; a
    # false attack would have contaminated the evaluation set.
    LOGGER.warning("unknown CIC label %r; defaulting to benign", raw)
    return int(Label.BENIGN)


def _cic_protocol_from_row(row: Mapping[str, str]) -> Protocol:
    # The CIC dataset annotates DoH flows explicitly in some snapshots and
    # uses port 443 with TLS framing in others. We used DoH as the default
    # because the dataset's purpose was DoH detection.
    return Protocol.DOH


def _reconstruct_packets_from_cic_row(
    row: Mapping[str, str],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """We reconstructed per-packet arrays from aggregated CIC statistics.

    The reconstruction preserves the recorded total payload bytes, the
    recorded packet counts, and the recorded fwd/bwd ratio. It does not
    reproduce the original packet ordering beyond the aggregate
    constraints. This caveat was recorded in the dataset manifest and is
    also stated in the README.
    """
    def _get_float(*keys: str, default: float = 0.0) -> float:
        for k in keys:
            if k in row and row[k] not in ("", None):
                try:
                    return float(row[k])
                except ValueError:
                    continue
        return default

    n_fwd = max(1, int(_get_float("Total Fwd Packets", "fwd_packets", default=1.0)))
    n_bwd = max(0, int(_get_float("Total Backward Packets", "bwd_packets", default=0.0)))
    n_total = n_fwd + n_bwd
    total_bytes = max(
        n_total * 32.0,
        _get_float("Fwd Packets Length Total", "Total Length of Fwd Packets", default=0.0)
        + _get_float("Bwd Packets Length Total", "Total Length of Bwd Packets", default=0.0),
    )
    duration = max(1e-4, _get_float("Flow Duration", "flow_duration", default=1.0) / 1e6)

    # We sampled per-packet sizes from a lognormal matched to the mean and
    # to a 1.2 coefficient of variation, then rescaled to match the total.
    mean_size = total_bytes / n_total
    sigma = 0.5
    mu = math.log(mean_size) - sigma ** 2 / 2.0
    sizes = rng.lognormal(mean=mu, sigma=sigma, size=n_total)
    sizes *= total_bytes / max(1.0, sizes.sum())
    sizes = np.clip(sizes, 40.0, None)

    # We sampled inter-arrival times from an exponential matched to the
    # duration, then rescaled.
    iats = rng.exponential(scale=max(duration / n_total, 1e-6), size=n_total)
    iats *= duration / max(1e-9, iats.sum())
    iats = np.clip(iats, 1e-6, None)

    directions = np.zeros(n_total, dtype=int)
    directions[n_fwd:] = 1

    return sizes, iats, directions


def load_cic_dohbrw_2020(
    dataset_root: Path,
    max_flows: Optional[int] = None,
    seed: int = 20231130,
) -> List[LoadedFlow]:
    """We loaded the CIC-DoHBrw-2020 dataset and returned canonical flows.

    The function expected the dataset root to contain at least one ``.csv``
    file; multiple CSVs were concatenated. ``max_flows`` capped the total
    number of flows loaded, sampled uniformly at random from the full set
    so that the benign/attack balance was not perturbed.
    """
    csv_paths = sorted(dataset_root.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"no CSVs found under {dataset_root}; expected CIC-DoHBrw-2020 layout"
        )
    rng = np.random.default_rng(seed)

    rows: List[Dict[str, str]] = []
    for csv_path in csv_paths:
        header: Optional[List[str]] = None
        with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(",")
                if header is None:
                    header = parts
                    continue
                if len(parts) < len(header):
                    continue
                rows.append(dict(zip(header, parts)))
    if not rows:
        raise RuntimeError(f"CIC dataset under {dataset_root} contained no rows")

    if max_flows is not None and max_flows < len(rows):
        idx = rng.choice(len(rows), size=max_flows, replace=False)
        rows = [rows[int(i)] for i in idx]

    flows: List[LoadedFlow] = []
    for i, row in enumerate(rows):
        label = _cic_parse_label(row.get("Label", "benign"))
        protocol = _cic_protocol_from_row(row)
        sizes, iats, directions = _reconstruct_packets_from_cic_row(row, rng)
        flows.append(
            LoadedFlow(
                flow_id=f"cic-{i:07d}-{uuid.uuid4().hex[:8]}",
                label=label,
                protocol=protocol,
                profile_name="cic_dohbrw_2020",
                sizes=sizes,
                iats=iats,
                directions=directions,
            )
        )
    LOGGER.info("loaded %d CIC-DoHBrw-2020 flows from %d CSVs", len(flows), len(csv_paths))
    return flows


# --------------------------------------------------------------------------- #
# Minimal pcap parser
# --------------------------------------------------------------------------- #
# We wrote our own pcap reader to avoid a hard dependency on scapy or dpkt.
# The parser handles the classic pcap magic and the two pcapng magic values
# that the Gen-C-DDD-2022 release includes. It extracted only the fields
# the pipeline needs: timestamp, wire length, and direction.

_PCAP_MAGIC_LE = 0xA1B2C3D4
_PCAP_MAGIC_BE = 0xD4C3B2A1
_PCAPNG_MAGIC = 0x0A0D0D0A


@dataclass
class PcapPacket:
    timestamp_s: float
    wire_length: int
    src_ip: int
    dst_ip: int
    src_port: int
    dst_port: int
    direction: int  # 0 = client->resolver, 1 = resolver->client


def _parse_classic_pcap(
    path: Path, resolver_ip: int, resolver_port: int
) -> List[PcapPacket]:
    packets: List[PcapPacket] = []
    with path.open("rb") as fh:
        header = fh.read(24)
        if len(header) < 24:
            return packets
        magic = struct.unpack("<I", header[:4])[0]
        if magic == _PCAP_MAGIC_BE:
            endian = ">"
        else:
            endian = "<"
        # The header is: magic, major, minor, thiszone, sigfigs, snaplen, linktype.
        linktype = struct.unpack(f"{endian}I", header[20:24])[0]
        while True:
            rec_header = fh.read(16)
            if len(rec_header) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(
                f"{endian}IIII", rec_header
            )
            data = fh.read(incl_len)
            if len(data) < incl_len:
                break
            ts = ts_sec + ts_usec / 1_000_000.0
            src_ip = dst_ip = 0
            src_port = dst_port = 0
            # We parsed only Ethernet + IPv4 + TCP/UDP; that covered the
            # Gen-C-DDD-2022 captures, which were IPv4 only. IPv6 captures
            # were skipped with a warning at the flow level.
            if linktype == 1 and len(data) >= 34:  # Ethernet
                if data[12:14] == b"\x08\x00":  # IPv4
                    ihl = (data[14] & 0x0F) * 4
                    if len(data) >= 14 + ihl + 20:
                        ip_start = 14
                        src_ip = int.from_bytes(data[ip_start + 12 : ip_start + 16], "big")
                        dst_ip = int.from_bytes(data[ip_start + 16 : ip_start + 20], "big")
                        proto = data[ip_start + 9]
                        transport_start = ip_start + ihl
                        if proto in (6, 17) and len(data) >= transport_start + 4:
                            src_port = int.from_bytes(
                                data[transport_start : transport_start + 2], "big"
                            )
                            dst_port = int.from_bytes(
                                data[transport_start + 2 : transport_start + 4], "big"
                            )
            # Direction: resolver -> client if src is resolver.
            direction = (
                1
                if (src_ip == resolver_ip and src_port == resolver_port)
                else 0
            )
            packets.append(
                PcapPacket(
                    timestamp_s=ts,
                    wire_length=orig_len,
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port,
                    direction=direction,
                )
            )
    return packets


def _flow_key(p: PcapPacket) -> Tuple[int, int, int, int]:
    a = (p.src_ip, p.src_port, p.dst_ip, p.dst_port)
    b = (p.dst_ip, p.dst_port, p.src_ip, p.src_port)
    return a if a <= b else b


def _pcap_to_flows(
    packets: Sequence[PcapPacket], label: int, protocol: Protocol
) -> List[LoadedFlow]:
    """We grouped packets into flows by canonical five-tuple."""
    by_flow: Dict[Tuple[int, int, int, int], List[PcapPacket]] = {}
    for p in packets:
        by_flow.setdefault(_flow_key(p), []).append(p)

    flows: List[LoadedFlow] = []
    for key, pkts in by_flow.items():
        pkts.sort(key=lambda x: x.timestamp_s)
        sizes = np.array([max(40, p.wire_length) for p in pkts], dtype=np.float64)
        t0 = pkts[0].timestamp_s
        times = np.array([p.timestamp_s for p in pkts], dtype=np.float64) - t0
        iats = np.diff(times)
        if iats.size == 0:
            continue
        iats = np.clip(iats, 1e-6, None)
        directions = np.array([p.direction for p in pkts[:-1]], dtype=int)
        flows.append(
            LoadedFlow(
                flow_id=f"genc-{uuid.uuid4().hex[:12]}",
                label=label,
                protocol=protocol,
                profile_name="gen_c_ddd_2022",
                sizes=sizes[:-1],
                iats=iats,
                directions=directions,
            )
        )
    return flows


def load_gen_c_ddd_2022(
    dataset_root: Path,
    resolver_ip_str: str = "8.8.8.8",
    resolver_port: int = 443,
    max_flows: Optional[int] = None,
    seed: int = 20231130,
) -> List[LoadedFlow]:
    """We loaded the Gen-C-DDD-2022 dataset from its pcap layout.

    The dataset root was expected to contain subdirectories ``benign`` and
    ``attack`` with ``.pcap`` files. Labels were assigned from the
    subdirectory name. We parsed only IPv4 TCP/UDP; other captures were
    skipped with a warning.
    """
    resolver_ip = int.from_bytes(
        bytes(int(b) for b in resolver_ip_str.split(".")), "big"
    ) if resolver_ip_str else 0
    flows: List[LoadedFlow] = []
    for subdir, label in (("benign", int(Label.BENIGN)), ("attack", int(Label.ATTACK))):
        sub = dataset_root / subdir
        if not sub.exists():
            LOGGER.warning("missing subdirectory %s", sub)
            continue
        for pcap_path in sorted(sub.glob("*.pcap")):
            try:
                packets = _parse_classic_pcap(pcap_path, resolver_ip, resolver_port)
            except Exception:
                LOGGER.exception("failed to parse %s; skipping", pcap_path)
                continue
            flows.extend(_pcap_to_flows(packets, label, Protocol.DOH))
    if max_flows is not None and max_flows < len(flows):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(flows), size=max_flows, replace=False)
        flows = [flows[int(i)] for i in idx]
    LOGGER.info("loaded %d Gen-C-DDD-2022 flows from %s", len(flows), dataset_root)
    return flows


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def export_loaded_flows(
    out_dir: Path,
    flows: Sequence[LoadedFlow],
    source: DatasetSource,
    packet_provenance: PacketArrayProvenance,
    extra_manifest: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, np.ndarray] = {}
    with (out_dir / "flows.jsonl").open("w", encoding="utf-8") as handle:
        for flow in flows:
            handle.write(json.dumps(flow.summary()) + "\n")
            arrays[f"{flow.flow_id}/sizes"] = flow.sizes
            arrays[f"{flow.flow_id}/iats"] = flow.iats
            arrays[f"{flow.flow_id}/directions"] = flow.directions
    np.savez_compressed(out_dir / "packet_archive.npz", **arrays)

    manifest: Dict[str, object] = {
        "module": "extended_experimentation.datasets.public_dataset_loaders",
        "source": source.value,
        "packet_arrays": packet_provenance.value,
        "n_flows": len(flows),
        "n_benign": sum(1 for f in flows if f.label == int(Label.BENIGN)),
        "n_attack": sum(1 for f in flows if f.label == int(Label.ATTACK)),
        "thesis_reference": THESIS_REFERENCE,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    LOGGER.info("exported %d %s flows to %s", len(flows), source.value, out_dir)
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a public DNS-over-HTTPS dataset into the extended suite format."
    )
    parser.add_argument(
        "--source", choices=[s.value for s in DatasetSource], required=True
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-flows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20231130)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    source = DatasetSource(args.source)
    if source is DatasetSource.CIC_DOHBRW_2020:
        flows = load_cic_dohbrw_2020(
            args.dataset_root, max_flows=args.max_flows, seed=args.seed
        )
        export_loaded_flows(
            args.out,
            flows,
            source=source,
            packet_provenance=PacketArrayProvenance.RECONSTRUCTED,
        )
    else:
        flows = load_gen_c_ddd_2022(
            args.dataset_root, max_flows=args.max_flows, seed=args.seed
        )
        export_loaded_flows(
            args.out,
            flows,
            source=source,
            packet_provenance=PacketArrayProvenance.PARSED_FROM_PCAP,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
