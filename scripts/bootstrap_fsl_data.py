#!/usr/bin/env python3
"""
bootstrap_fsl_data.py — Download a few-shot benchmark dataset and freeze it
immutably in Google Cloud Storage with full provenance.

This is rung 1 of the MLOps ladder: data treated as a versioned, checksummed,
reproducible artifact — not a side effect of `download=True` in a training
script.

What it does (idempotently — safe to re-run):
  1. Creates the GCS bucket if it does not exist (regional, uniform access).
  2. Skips all work if the frozen artifact already exists (unless --force).
  3. Downloads the dataset via learn2learn into a local scratch dir.
  4. Computes a SHA256 for every file (the integrity anchor).
  5. Packs a *deterministic* tar.gz, so the archive checksum depends only on
     file contents + paths (sorted entries, zeroed mtime/uid/gid).
  6. Writes MANIFEST.json with full provenance: source, library versions,
     timestamp, per-file checksums, environment.
  7. Uploads {archive, manifest, checksums} to gs://<bucket>/raw/<dataset>/.

Designed for Vertex AI Workbench, where Application Default Credentials are
already configured. No secrets are handled here.

Dependencies:
    pip install learn2learn google-cloud-storage

Usage:
    python bootstrap_fsl_data.py --project my-proj --dataset omniglot
    python bootstrap_fsl_data.py --project my-proj --dataset miniimagenet --force
    python bootstrap_fsl_data.py --project my-proj --location europe-west4
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import platform
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NoReturn

LOG = logging.getLogger("bootstrap_fsl_data")


# --------------------------------------------------------------------------- #
# Dataset registry — add new few-shot datasets here; the rest is generic.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    download: Callable[[str], None]  # given a local root dir, fetch into it
    source: str                      # human-readable provenance note


def _download_omniglot(root: str) -> None:
    import learn2learn as l2l
    # FullOmniglot pulls the raw Omniglot data + builds its on-disk layout.
    # We freeze whatever it produces under `root` (raw freeze, intentionally).
    l2l.vision.datasets.FullOmniglot(root=root, download=True)


def _download_miniimagenet(root: str) -> None:
    import learn2learn as l2l
    # Pre-processed 84x84 caches (Ravi & Larochelle splits) — no ImageNet
    # licence / sign-up hurdle. One cache per meta-split.
    for mode in ("train", "validation", "test"):
        l2l.vision.datasets.MiniImagenet(root=root, mode=mode, download=True)


DATASETS: dict[str, DatasetSpec] = {
    "omniglot": DatasetSpec(
        name="omniglot",
        download=_download_omniglot,
        source=(
            "learn2learn FullOmniglot (Lake et al. 2015): 1623 character classes "
            "x 20 examples, handwritten characters from 50 alphabets."
        ),
    ),
    "miniimagenet": DatasetSpec(
        name="miniimagenet",
        download=_download_miniimagenet,
        source=(
            "learn2learn MiniImagenet (Ravi & Larochelle 2017 splits): "
            "64/16/20 train/val/test classes x 600 examples, 84x84 ImageNet subset cache."
        ),
    ),
}


# --------------------------------------------------------------------------- #
# Integrity & packaging helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def checksum_tree(root: Path) -> dict[str, str]:
    """SHA256 of every file under `root`, keyed by sorted POSIX relative path."""
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = sha256_file(p)
    return out


def make_deterministic_tar(root: Path, archive_path: Path) -> str:
    """Pack `root` into a reproducible tar.gz; return the archive SHA256.

    Reproducible means: entries sorted by path, and mtime/uid/gid/owner/mode
    normalized, plus a gzip header with mtime=0. The resulting bytes therefore
    depend only on file contents and relative paths — so the same data yields
    the same archive checksum on any machine, any day.
    """
    files = sorted(p for p in root.rglob("*") if p.is_file())
    with open(archive_path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:  # uncompressed tar into gzip
                for p in files:
                    info = tarfile.TarInfo(name=p.relative_to(root).as_posix())
                    info.size = p.stat().st_size
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with p.open("rb") as fh:
                        tar.addfile(info, fh)
    return sha256_file(archive_path)


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# GCS helpers (Application Default Credentials on Workbench)
# --------------------------------------------------------------------------- #
def get_storage_client(project: str):
    try:
        from google.cloud import storage
    except ImportError as e:
        die("google-cloud-storage not installed. Run: pip install google-cloud-storage", e)
    return storage.Client(project=project)


def ensure_bucket(client, bucket_name: str, location: str):
    from google.cloud.exceptions import NotFound
    try:
        bucket = client.get_bucket(bucket_name)
        LOG.info("Bucket gs://%s already exists.", bucket_name)
        return bucket
    except NotFound:
        LOG.info("Creating bucket gs://%s in %s ...", bucket_name, location)
        bucket = client.bucket(bucket_name)
        # Professional defaults: uniform bucket-level access (no legacy ACLs).
        bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        client.create_bucket(bucket, location=location)
        LOG.info("Created bucket gs://%s.", bucket_name)
        return client.get_bucket(bucket_name)


def blob_exists(bucket, name: str) -> bool:
    return bucket.blob(name).exists()


def upload(bucket, local: Path, name: str) -> None:
    LOG.info("Uploading %s -> gs://%s/%s", local.name, bucket.name, name)
    bucket.blob(name).upload_from_filename(str(local))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def die(msg: str, exc: Exception | None = None) -> NoReturn:
    LOG.error(msg)
    if exc is not None:
        LOG.debug("cause: %r", exc)
    sys.exit(1)


def resolve_project(arg: str | None) -> str:
    if arg:
        return arg
    env = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if env:
        LOG.info("Using project from environment: %s", env)
        return env
    die("No --project given and GOOGLE_CLOUD_PROJECT is not set.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Freeze a few-shot dataset into GCS with provenance (MLOps rung 1).",
    )
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="omniglot")
    ap.add_argument("--project", default=None, help="GCP project (else $GOOGLE_CLOUD_PROJECT)")
    ap.add_argument("--bucket", default=None, help="Bucket name (default: <project>-fsl-data)")
    ap.add_argument("--location", default="europe-west4",
                    help="Bucket region — co-locate with your Workbench/compute.")
    ap.add_argument("--scratch-dir", default=None,
                    help="Local download dir (default: an auto-cleaned temp dir).")
    ap.add_argument("--force", action="store_true",
                    help="Re-download and overwrite even if the frozen artifact exists.")
    ap.add_argument("--keep-scratch", action="store_true",
                    help="Do not delete the local scratch dir on success.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    project = resolve_project(args.project)
    bucket_name = args.bucket or f"{project}-fsl-data"
    spec = DATASETS[args.dataset]
    prefix = f"raw/{spec.name}"
    archive_blob = f"{prefix}/{spec.name}.tar.gz"
    manifest_blob = f"{prefix}/MANIFEST.json"
    checksums_blob = f"{prefix}/checksums.sha256"

    client = get_storage_client(project)
    bucket = ensure_bucket(client, bucket_name, args.location)

    # Idempotency: if already frozen, stop (unless forced).
    if blob_exists(bucket, archive_blob) and not args.force:
        LOG.info("Frozen artifact already present: gs://%s/%s", bucket_name, archive_blob)
        LOG.info("Nothing to do — use --force to refreeze. OK")
        return 0

    # Local scratch (auto-cleaned unless --keep-scratch / --scratch-dir).
    scratch_ctx = None
    if args.scratch_dir:
        scratch = Path(args.scratch_dir).expanduser().resolve()
        scratch.mkdir(parents=True, exist_ok=True)
    else:
        scratch_ctx = tempfile.TemporaryDirectory(prefix="fsl-")
        scratch = Path(scratch_ctx.name)

    data_root = scratch / spec.name
    data_root.mkdir(parents=True, exist_ok=True)

    # 1. Download.
    LOG.info("Downloading '%s' via learn2learn into %s ...", spec.name, data_root)
    try:
        spec.download(str(data_root))
    except ImportError as e:
        die("learn2learn not installed. Run: pip install learn2learn", e)
    except Exception as e:  # noqa: BLE001 - surface any download failure clearly
        die(f"Download failed for '{spec.name}': {e}", e)

    files = [p for p in data_root.rglob("*") if p.is_file()]
    if not files:
        die(f"No files were downloaded under {data_root}; aborting.")
    total_bytes = sum(p.stat().st_size for p in files)
    LOG.info("Downloaded %d files, %.1f MB.", len(files), total_bytes / 1e6)

    # 2. Per-file checksums (integrity anchor).
    LOG.info("Computing SHA256 for %d files ...", len(files))
    checksums = checksum_tree(data_root)

    # 3. Deterministic archive (stable checksum).
    archive_local = scratch / f"{spec.name}.tar.gz"
    LOG.info("Packing deterministic archive ...")
    archive_sha = make_deterministic_tar(data_root, archive_local)
    LOG.info("Archive SHA256: %s (%.1f MB)", archive_sha, archive_local.stat().st_size / 1e6)

    # 4. Manifest (provenance).
    manifest = {
        "dataset": spec.name,
        "source": spec.source,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "gcs": {
            "bucket": bucket_name,
            "archive": archive_blob,
            "manifest": manifest_blob,
            "checksums": checksums_blob,
        },
        "archive_sha256": archive_sha,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "learn2learn": _pkg_version("learn2learn"),
            "google_cloud_storage": _pkg_version("google-cloud-storage"),
        },
        "files_sha256": checksums,
    }
    manifest_local = scratch / "MANIFEST.json"
    manifest_local.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    checksums_local = scratch / "checksums.sha256"
    checksums_local.write_text(
        "".join(f"{sha}  {rel}\n" for rel, sha in sorted(checksums.items()))
    )

    # 5. Upload {archive, checksums, manifest}.
    upload(bucket, archive_local, archive_blob)
    upload(bucket, checksums_local, checksums_blob)
    upload(bucket, manifest_local, manifest_blob)

    LOG.info("Frozen gs://%s/%s  OK", bucket_name, prefix)
    LOG.info("  archive   : gs://%s/%s", bucket_name, archive_blob)
    LOG.info("  manifest  : gs://%s/%s", bucket_name, manifest_blob)
    LOG.info("  checksums : gs://%s/%s", bucket_name, checksums_blob)

    if scratch_ctx and not args.keep_scratch:
        scratch_ctx.cleanup()
    elif args.keep_scratch:
        LOG.info("Scratch kept at %s", scratch)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())