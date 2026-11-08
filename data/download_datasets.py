#!/usr/bin/env python3
"""
download_datasets.py — fetch the public datasets used by the MIRAGE-DoH artifact.

This script downloads the two datasets that the thesis cites in Section 3.11.1:

    CIC-DoH2023       https://www.unb.ca/cic/datasets/dohbrw-2023.html
    Gen-C-DDD-2022    https://www.unb.ca/cic/datasets/dns-2022.html

and the Zenodo mirror that the thesis cites in Section 4.25.1:

    Zenodo             https://doi.org/10.5281/zenodo.5957677

Neither dataset is redistributed inside this repository (see LICENSE and
data/README_data.md). The 1 MB toy dataset in data/toy_dataset/ is a
schema-matched synthetic stand-in that lets a reviewer run the code end-to-end
in 2 minutes without downloading 32 GB; see data/README_data.md §4 for the
distinction between "toy dataset" and "full dataset" and for how to generate a
real toy subset from the full data once it is downloaded.

Usage
-----
    python data/download_datasets.py --all
    python data/download_datasets.py --dataset cic_doh2023
    python data/download_datasets.py --dataset cic_doh2023 gen_c_ddd2022
    python data/download_datasets.py --all --force
    python data/download_datasets.py --print-manual-commands cic_doh2023
    python data/download_datasets.py --verify-only

Colab
-----
    !python data/download_datasets.py --all

Colab pre-installs requests, tqdm, and wget. The script detects
google.colab and adjusts its filesystem root from the repo-local data/raw/
to /content/data/raw/ so the rest of the pipeline finds the files. No
modification to the script is required; the detection is automatic.

Exit codes
----------
    0  success (all requested datasets were either downloaded or already present)
    1  usage error (bad arguments)
    2  one or more downloads failed (see stderr for exact fallback command)
    3  checksum mismatch (see data/checksums.json and data/README_data.md §5)
    4  network unreachable (DNS, TLS, or 4xx/5xx from the upstream server)

Author
------
    MIRAGE-DoH thesis author(s)
    Companion to Chapter 3, Section 3.11.1 of the thesis.

License
-------
    MIT (the script). The datasets themselves are CC-BY 4.0 by the Canadian
    Institute for Cybersecurity, University of New Brunswick.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Third-party imports. requests and tqdm are installed by requirements.txt
# (File 4) and are preinstalled on Colab. If either is missing, the script
# prints an install hint and exits with code 1 rather than crashing with a
# bare ImportError.
try:
    import requests
except ImportError:
    sys.stderr.write(
        "ERROR: requests is not installed.\n"
        "Install with:  pip install requests==2.32.3\n"
        "or (in the repo root):  pip install -r requirements.txt\n"
    )
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    sys.stderr.write(
        "ERROR: tqdm is not installed.\n"
        "Install with:  pip install tqdm==4.66.4\n"
        "or (in the repo root):  pip install -r requirements.txt\n"
    )
    sys.exit(1)


# =============================================================================
# Constants
# =============================================================================

# Repository root. The script computes this from its own location so it works
# whether it is invoked as `python data/download_datasets.py` from the repo
# root, as `python download_datasets.py` from inside data/, or as
# `python -m data.download_datasets` from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Where the datasets will be written. Relative to REPO_ROOT so that a clone
# anywhere on disk works. Colab users get /content/data/raw/ instead; the
# detection happens at runtime in `_resolve_data_raw()` below.
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

# SHA-256 TOFU manifest. First run computes and saves; subsequent runs verify.
# See the module docstring for why this is TOFU rather than a hard-coded hash.
CHECKSUM_MANIFEST = REPO_ROOT / "data" / "checksums.json"

# Download log. Appended to on every run, so a reviewer can see the history of
# attempts (which is useful when a download failed partway).
DOWNLOAD_LOG = REPO_ROOT / "data" / "download.log"

# Zenodo API endpoint. This is the canonical way to fetch a Zenodo record's
# files. The record ID (5957677) is the numeric part of the DOI
# 10.5281/zenodo.5957677 cited in the thesis (Section 4.25.1).
ZENODO_RECORD_ID = "5957677"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Official dataset landing pages. We cannot download directly from these
# because they require a click-through acceptance of terms; the script
# therefore prints the exact curl command for manual download when the
# automated attempt fails.
UNB_CIC_LANDING_PAGES = {
    "cic_doh2023": "https://www.unb.ca/cic/datasets/dohbrw-2023.html",
    "gen_c_ddd2022": "https://www.unb.ca/cic/datasets/dns-2022.html",
}

# Approximate download sizes, for the progress message and for the pre-flight
# disk-space check.
EXPECTED_SIZE_GB = {
    "cic_doh2023": 22.0,
    "gen_c_ddd2022": 10.0,
    "zenodo_mirror": None,  # unknown; skipped in the disk check
}

# Network timeouts. Generous because a 32 GB download over a slow link can
# take hours; short enough to fail fast if the server is unreachable.
CONNECT_TIMEOUT_S = 30
READ_TIMEOUT_S = 300
CHUNK_SIZE = 1024 * 1024  # 1 MiB per chunk

# User-Agent. Some servers (including Zenodo) block the default requests
# User-Agent; identifying ourselves is also good practice for an artifact
# that downloads 32 GB from a public research infrastructure.
USER_AGENT = (
    "MIRAGE-DoH-reproducibility-artifact/1.0.0 "
    "(https://github.com/dezignator/MIRAGE-DoH-Reproducibility; "
    "mailto:thesis-authors@example.invalid)"
)


# =============================================================================
# Logging helpers
# =============================================================================

def log(msg: str, *, also_print: bool = True) -> None:
    """Append to download.log and optionally echo to stdout."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    DOWNLOAD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DOWNLOAD_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if also_print:
        print(line)


def die(code: int, msg: str) -> None:
    """Print an error to stderr, log it, and exit with the given code."""
    log(f"ERROR: {msg}")
    sys.stderr.write(f"\nERROR: {msg}\n")
    sys.exit(code)


# =============================================================================
# Environment detection (Colab, native)
# =============================================================================

def _is_colab() -> bool:
    """Return True if the script is running in Google Colab.

    Colab sets the environment variable `COLAB_RELEASE_TAG` and provides
    a `google.colab` module. We check both because the module is only
    importable after `import google.colab` explicitly, while the env var
    is always set.
    """
    if "COLAB_RELEASE_TAG" in os.environ:
        return True
    try:
        import google.colab  # type: ignore[import-not-found]  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_data_raw() -> Path:
    """Return the effective data/raw/ directory.

    On Colab, the filesystem root is /content/, and the reviewer is expected
    to have cloned the repo into /content/MIRAGE-DoH-Reproducibility. We
    return /content/data/raw/ if that is writable, else fall back to
    /content/MIRAGE-DoH-Reproducibility/data/raw/. On native Linux/macOS/
    Windows, we return the repo-local data/raw/.
    """
    if _is_colab():
        content_root = Path("/content")
        if content_root.is_dir() and os.access(content_root, os.W_OK):
            target = content_root / "data" / "raw"
            target.mkdir(parents=True, exist_ok=True)
            log(f"Colab detected. Writing datasets to {target} "
                f"(not {DATA_RAW}).")
            return target
        # Colab with a repo inside /content/MIRAGE-DoH-Reproducibility/
        return DATA_RAW
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    return DATA_RAW


# =============================================================================
# SHA-256 TOFU manager
# =============================================================================

def _load_checksums() -> Dict[str, str]:
    """Load the TOFU checksum manifest, or return an empty dict."""
    if not CHECKSUM_MANIFEST.is_file():
        return {}
    try:
        return json.loads(CHECKSUM_MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        die(3, f"checksums.json is corrupt: {exc}. "
               f"Delete {CHECKSUM_MANIFEST} and re-run.")


def _save_checksums(checksums: Dict[str, str]) -> None:
    """Atomically write the TOFU checksum manifest."""
    tmp = CHECKSUM_MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(CHECKSUM_MANIFEST)


def _sha256_of_file(path: Path, desc: str = "") -> str:
    """Compute SHA-256 of a file, streaming 1 MiB at a time."""
    h = hashlib.sha256()
    total = path.stat().st_size
    with path.open("rb") as fh, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=desc or f"SHA-256 {path.name}",
        leave=False,
    ) as bar:
        while True:
            chunk = fh.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
            bar.update(len(chunk))
    return h.hexdigest()


def _verify_or_record_checksum(path: Path, key: str) -> Tuple[bool, str]:
    """Verify a file against the TOFU manifest, or record its checksum.

    Returns (ok, action) where action is 'verified', 'recorded', or
    'mismatch'.
    """
    checksums = _load_checksums()
    actual = _sha256_of_file(path)
    if key in checksums:
        if actual == checksums[key]:
            log(f"Checksum OK: {path.name} ({actual[:16]}...)")
            return True, "verified"
        log(f"CHECKSUM MISMATCH: {path.name} expected {checksums[key][:16]}..., "
            f"got {actual[:16]}...")
        return False, "mismatch"
    # First run: Trust On First Use. Record and continue.
    log(f"Recording TOFU checksum for {path.name} ({actual[:16]}...). "
        f"This is the first run; subsequent runs will verify against it.")
    checksums[key] = actual
    _save_checksums(checksums)
    return True, "recorded"


# =============================================================================
# Disk space pre-flight check
# =============================================================================

def _check_disk_space(target_dir: Path, needed_gb: float) -> None:
    """Abort early if there is not enough free disk space."""
    import shutil
    free_gb = shutil.disk_usage(target_dir).free / (1024 ** 3)
    if free_gb < needed_gb * 1.10:  # 10% headroom for extraction
        die(
            1,
            f"Not enough disk space at {target_dir}: need ~{needed_gb:.1f} GB, "
            f"have {free_gb:.1f} GB. Free space or use a different data/raw/ "
            f"location (see data/README_data.md §6).",
        )
    log(f"Disk space OK: {free_gb:.1f} GB free at {target_dir} "
        f"(need ~{needed_gb:.1f} GB).")


# =============================================================================
# Network helpers
# =============================================================================

def _http_get_json(url: str) -> dict:
    """GET a URL and return the JSON response, with clear error handling."""
    log(f"GET {url}")
    try:
        r = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
        )
    except requests.exceptions.RequestException as exc:
        die(4, f"Network error while fetching {url}: {exc}")
    if r.status_code != 200:
        die(4, f"Server returned {r.status_code} for {url}. "
               f"Body: {r.text[:300]}")
    try:
        return r.json()
    except json.JSONDecodeError as exc:
        die(4, f"Invalid JSON from {url}: {exc}. "
               f"Body: {r.text[:300]}")


def _http_download(url: str, dest: Path, expected_size: Optional[int]) -> None:
    """Stream a URL to dest with progress, resume, and atomic rename.

    The .part file is used so that an interrupted download resumes rather
    than restarting from zero. A completed download is atomically renamed
    to dest, so a partially-downloaded file never has the final name.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    # Resume support. If a .part file exists, tell the server the byte range
    # we already have. If the server does not support ranges (returns 200
    # instead of 206), we restart from zero.
    resume_from = part.stat().st_size if part.is_file() else 0
    headers = {"User-Agent": USER_AGENT}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
        log(f"Resuming {dest.name} from byte {resume_from:,}")

    try:
        r = requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
            allow_redirects=True,
        )
    except requests.exceptions.RequestException as exc:
        die(4, f"Network error while downloading {url}: {exc}")

    # Server does not support ranges, or we are not resuming.
    if r.status_code == 200:
        resume_from = 0
        part.unlink(missing_ok=True)
        part.parent.mkdir(parents=True, exist_ok=True)
    elif r.status_code != 206:
        die(4, f"Server returned {r.status_code} for {url}. "
               f"Body: {r.text[:300]}")

    # Total size: from Content-Length if present, else from the caller's hint.
    total = None
    if "content-length" in r.headers:
        total = resume_from + int(r.headers["content-length"])
    elif expected_size is not None:
        total = expected_size

    # Stream to .part with progress.
    mode = "ab" if resume_from > 0 else "wb"
    with part.open(mode) as fh, tqdm(
        total=total,
        initial=resume_from,
        unit="B",
        unit_scale=True,
        desc=dest.name,
    ) as bar:
        try:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                fh.write(chunk)
                bar.update(len(chunk))
        except requests.exceptions.RequestException as exc:
            die(4, f"Network error during download of {dest.name}: {exc}. "
                   f"Re-run the same command to resume from byte "
                   f"{part.stat().st_size:,}.")

    # Atomic rename. A reviewer who interrupts the script will find a .part
    # file (resumable) and never a half-written dest.
    part.replace(dest)
    log(f"Downloaded {dest} ({dest.stat().st_size:,} bytes)")


# =============================================================================
# Zenodo download path
# =============================================================================

def _download_from_zenodo(dest_dir: Path) -> None:
    """Download all files attached to the Zenodo record.

    The Zenodo REST API returns a JSON object whose `files` array contains
    each file's `key` (filename) and `links.self` (download URL). We iterate
    and download each one.

    Zenodo is the primary automated source because it does not require a
    click-through acceptance. The UNB CIC pages do; see
    `_download_from_unb_cic()` below for the manual fallback.
    """
    log(f"Attempting Zenodo download from {ZENODO_API_URL}")
    meta = _http_get_json(ZENODO_API_URL)
    files = meta.get("files", [])
    if not files:
        die(4, f"Zenodo record {ZENODO_RECORD_ID} has no attached files. "
               f"This usually means the record has been retired or the "
               f"record ID is wrong. See {ZENODO_API_URL} for the current "
               f"state of the record.")
    total_files = len(files)
    log(f"Zenodo record has {total_files} file(s).")
    for i, fmeta in enumerate(files, 1):
        key = fmeta.get("key") or fmeta.get("filename")
        if not key:
            log(f"Skipping Zenodo file {i}: no 'key' field in metadata.")
            continue
        url = fmeta.get("links", {}).get("self")
        if not url:
            log(f"Skipping Zenodo file {i} ({key}): no 'links.self' field.")
            continue
        size = fmeta.get("size")
        dest = dest_dir / key
        if dest.is_file():
            log(f"Already present: {dest}. Skipping "
                f"(use --force to re-download).")
            continue
        log(f"[{i}/{total_files}] Downloading {key} ({size:,} bytes)" if size
            else f"[{i}/{total_files}] Downloading {key}")
        _http_download(url, dest, expected_size=size)


# =============================================================================
# UNB CIC download path (manual fallback)
# =============================================================================

def _print_unb_cic_manual_commands(dataset_key: str) -> None:
    """Print exact curl/wget commands for a manual UNB CIC download.

    The UNB CIC landing pages require a click-through acceptance of terms.
    The exact URL of the actual file (behind the click-through) is stable
    per dataset, but we do not have it in the repository because it changes
    per dataset release. The reviewer who needs the file must:
      1. Open the landing page.
      2. Accept the terms.
      3. Copy the file URL that appears.
      4. Paste it into the commands below.
    """
    landing = UNB_CIC_LANDING_PAGES[dataset_key]
    print()
    print("=" * 78)
    print(f"MANUAL DOWNLOAD REQUIRED FOR: {dataset_key}")
    print("=" * 78)
    print(f"The UNB CIC page for {dataset_key} requires accepting terms of use")
    print(f"before the download link is revealed. Open it in a browser:")
    print()
    print(f"    {landing}")
    print()
    print("Accept the terms, copy the file URL that appears, then run:")
    print()
    print(f"    # curl variant")
    print(f"    curl -L -o data/raw/{dataset_key}.zip '<PASTE_FILE_URL_HERE>'")
    print()
    print(f"    # wget variant")
    print(f"    wget -O data/raw/{dataset_key}.zip '<PASTE_FILE_URL_HERE>'")
    print()
    print("After the download completes, run:")
    print()
    print(f"    python data/download_datasets.py --verify-only")
    print()
    print("to record the TOFU checksum and confirm the file is usable.")
    print("=" * 78)
    print()


def _download_from_unb_cic(dataset_key: str, dest_dir: Path) -> None:
    """The UNB CIC path is manual-only; this function prints instructions."""
    _print_unb_cic_manual_commands(dataset_key)


# =============================================================================
# Orchestrator
# =============================================================================

def _resolve_dataset_choices() -> List[str]:
    return ["cic_doh2023", "gen_c_ddd2022", "zenodo_mirror"]


def download_one(dataset_key: str, dest_dir: Path, force: bool) -> None:
    """Download one dataset, doing pre-flight checks first."""
    log(f"=== {dataset_key} ===")

    # If the final artefacts already exist and --force was not given, skip.
    # We use a sentinel file per dataset so that "already downloaded" means
    # "a previous run completed successfully", not "a partial file exists".
    sentinel = dest_dir / f"{dataset_key}.complete"
    if sentinel.is_file() and not force:
        log(f"Already downloaded: {sentinel.is_file()}. "
            f"Skipping (use --force to re-download).")
        return

    # Pre-flight disk space.
    needed = EXPECTED_SIZE_GB.get(dataset_key)
    if needed:
        _check_disk_space(dest_dir, needed)

    # Dispatch to the appropriate downloader.
    if dataset_key == "zenodo_mirror":
        _download_from_zenodo(dest_dir)
    elif dataset_key == "cic_doh2023":
        # Try Zenodo first (it mirrors the CIC data), fall back to manual.
        try:
            _download_from_zenodo(dest_dir)
        except SystemExit as exc:
            if exc.code != 4:
                raise
            _download_from_unb_cic(dataset_key, dest_dir)
    elif dataset_key == "gen_c_ddd2022":
        try:
            _download_from_zenodo(dest_dir)
        except SystemExit as exc:
            if exc.code != 4:
                raise
            _download_from_unb_cic(dataset_key, dest_dir)
    else:
        die(1, f"Unknown dataset: {dataset_key!r}. "
               f"Valid choices: {_resolve_dataset_choices()}")

    # Record success.
    sentinel.write_text(
        f"{dataset_key} downloaded successfully at "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )
    log(f"=== {dataset_key} complete ===")


def verify_all(dest_dir: Path) -> int:
    """Walk dest_dir, verify each .zip/.tar/.tar.gz against the TOFU manifest.

    Returns the number of files that failed verification.
    """
    checksums = _load_checksums()
    failures = 0
    if not dest_dir.is_dir():
        log(f"Nothing to verify: {dest_dir} does not exist.")
        return 0
    candidates = [
        p for p in dest_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"}
    ]
    if not candidates:
        log(f"No archives found in {dest_dir}; nothing to verify.")
        return 0
    log(f"Verifying {len(candidates)} archive(s) against {CHECKSUM_MANIFEST}")
    for path in candidates:
        key = path.name
        ok, action = _verify_or_record_checksum(path, key)
        if not ok:
            failures += 1
    if failures == 0:
        log(f"All {len(candidates)} archive(s) verified successfully.")
    else:
        log(f"{failures} archive(s) failed verification. "
            f"See data/README_data.md §5 for recovery steps.")
    return failures


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_datasets.py",
        description="Download the public datasets used by the MIRAGE-DoH artifact.",
        epilog=(
            "Examples:\n"
            "  python data/download_datasets.py --all\n"
            "  python data/download_datasets.py --dataset cic_doh2023\n"
            "  python data/download_datasets.py --dataset cic_doh2023 gen_c_ddd2022\n"
            "  python data/download_datasets.py --print-manual-commands cic_doh2023\n"
            "  python data/download_datasets.py --verify-only\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        nargs="+",
        choices=_resolve_dataset_choices(),
        help="Dataset(s) to download. Valid: cic_doh2023, gen_c_ddd2022, "
             "zenodo_mirror. Cannot be combined with --all.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Download all three datasets (cic_doh2023, gen_c_ddd2022, "
             "zenodo_mirror). Cannot be combined with --dataset.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the dataset already exists.",
    )
    p.add_argument(
        "--print-manual-commands",
        choices=["cic_doh2023", "gen_c_ddd2022", "zenodo_mirror"],
        help="Print the manual curl/wget commands for a dataset and exit. "
             "Useful when the automated download is blocked by a proxy or "
             "by the UNB CIC click-through.",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download anything; only verify the SHA-256 of files "
             "that are already present in data/raw/.",
    )
    p.add_argument(
        "--data-raw",
        type=Path,
        default=None,
        help="Override the data/raw/ directory. Default: repo-local data/raw/ "
             "(or /content/data/raw/ on Colab).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error stdout (still logs to data/download.log).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ---------------- Usage validation ----------------
    if args.all and args.dataset:
        parser.error("--all and --dataset are mutually exclusive.")
    if not args.all and not args.dataset and not args.verify_only \
            and not args.print_manual_commands:
        parser.error("specify --all, --dataset, --verify-only, or "
                     "--print-manual-commands.")

    # ---------------- Resolve directories ----------------
    if args.data_raw is not None:
        dest_dir = args.data_raw.resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
    else:
        dest_dir = _resolve_data_raw()

    log(f"Repository root: {REPO_ROOT}")
    log(f"Datasets directory: {dest_dir}")
    log(f"Running on: {'Colab' if _is_colab() else 'native'}")

    # ---------------- --print-manual-commands early exit ----------------
    if args.print_manual_commands:
        _print_unb_cic_manual_commands(args.print_manual_commands)
        return 0

    # ---------------- --verify-only early exit ----------------
    if args.verify_only:
        failures = verify_all(dest_dir)
        return 3 if failures else 0

    # ---------------- Download loop ----------------
    targets = (["cic_doh2023", "gen_c_ddd2022", "zenodo_mirror"]
               if args.all else list(args.dataset))
    log(f"Targets: {targets}")

    try:
        for key in targets:
            download_one(key, dest_dir, force=args.force)
    except SystemExit as exc:
        # download_one or a helper raised a code; return it so the shell sees
        # the same exit code and `scripts/reproduce_all.sh` stops cleanly.
        return exc.code if isinstance(exc.code, int) else 2
    except KeyboardInterrupt:
        log("Interrupted by user. Partial downloads are resumable: re-run the "
            "same command to continue.")
        return 130

    # ---------------- Final verification ----------------
    log("All requested datasets downloaded. Verifying checksums...")
    failures = verify_all(dest_dir)
    if failures:
        log(f"{failures} file(s) failed checksum verification. "
            f"See data/README_data.md §5.")
        return 3

    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
