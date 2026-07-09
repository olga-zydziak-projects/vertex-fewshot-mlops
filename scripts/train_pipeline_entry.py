#!/usr/bin/env python3
"""Pipeline entry point for training inside a container component.

This is the CONTAINER-mode counterpart to scripts/train.py. Both import the
SAME core (`fsl.training.loop.train`) - zero duplicated model logic. The
difference is how results leave:
  - scripts/train.py (inner loop): train() logs to Vertex Experiments, done.
  - this script (pipeline): train() returns a dict; we save the model to GCS
    and write metric values to the KFP output paths so the gate/register
    components downstream can consume them.

`train()` already returns everything (metrics + the model object). The only
thing this wrapper adds is persistence: the returned model lives in memory, and
the pipeline needs it in GCS - so we save state_dict + architecture.json there,
exactly like the lightweight component did, and hand back the GCS dir.

Usage (KFP fills the --*-output-path args via OutputPath):
    python scripts/train_pipeline_entry.py \
        --project P --bucket B --dataset omniglot \
        --n-way 5 --k-shot 5 --query 15 --seed 0 --train-iters 300 \
        --embedding-hid 64 \
        --accuracy-mean-output-path /path --accuracy-ci95-output-path /path \
        --accuracy-std-output-path /path --train-time-output-path /path \
        --model-dir-output-path /path --data-sha-output-path /path
"""
import argparse
import json
import time
from pathlib import Path


def _write(path: str, value) -> None:
    """Write a single output value to a KFP output path (creates parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(value))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ProtoNet inside a pipeline container.")
    # training params (mirror scripts/train.py, forwarded into TrainConfig)
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--dataset", default="omniglot")
    ap.add_argument("--n-way", type=int, default=5)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--query", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-iters", type=int, default=300)
    ap.add_argument("--embedding-hid", type=int, default=64)
    # in-pipeline runs don't log to Experiments from here (a dedicated log
    # component does that downstream); keep training pure.
    ap.add_argument("--no-vertex", action="store_true", default=True)
    # KFP output paths (filled by OutputPath in the container component)
    ap.add_argument("--accuracy-mean-output-path", required=True)
    ap.add_argument("--accuracy-ci95-output-path", required=True)
    ap.add_argument("--accuracy-std-output-path", required=True)
    ap.add_argument("--train-time-output-path", required=True)
    ap.add_argument("--model-dir-output-path", required=True)
    ap.add_argument("--data-sha-output-path", required=True)
    args = ap.parse_args()

    # import the SAME core the inner loop uses - no duplicated model logic
    import torch
    from google.cloud import storage

    from fsl.config import TrainConfig
    from fsl.data.omniglot import load_frozen_omniglot
    from fsl.training.loop import train

    bucket = args.bucket or f"{args.project}-fsl-data"

    cfg = TrainConfig(
        project=args.project, region=args.region, bucket=bucket, dataset=args.dataset,
        n_way=args.n_way, k_shot=args.k_shot, query=args.query, seed=args.seed,
        train_iters=args.train_iters, embedding_hid=args.embedding_hid,
        log_to_vertex=False,   # pipeline logs via a separate component, not here
    )

    # train() returns metrics + the model object (see fsl/training/loop.py)
    result = train(cfg)

    # --- persist the model to GCS (the piece loop.py doesn't do) ---
    # We need the archive sha for provenance; load_frozen_omniglot returns it,
    # and train() already loaded the data, but doesn't pass the sha back in the
    # dict - so we re-read just the manifest (cheap) to get it.
    client = storage.Client(project=args.project)
    gcs = client.bucket(bucket)
    manifest = json.loads(
        gcs.blob(f"raw/{args.dataset}/MANIFEST.json").download_as_bytes().decode("utf-8")
    )
    archive_sha = manifest["archive_sha256"]

    import tempfile
    local_model = Path(tempfile.mkdtemp()) / "model"
    local_model.mkdir(parents=True, exist_ok=True)
    torch.save(result["model"].state_dict(), local_model / "model.pt")
    (local_model / "architecture.json").write_text(json.dumps({
        "architecture": "Conv4",
        "embedding_hid": cfg.embedding_hid,
        "in_channels": 1,
        "n_way": cfg.n_way,
        "k_shot": cfg.k_shot,
        "seed": cfg.seed,       # so evaluation can reconstruct the same class split
    }))
    model_prefix = f"models/{args.dataset}/protonet-seed{cfg.seed}-{int(time.time())}"
    for fpath in local_model.iterdir():
        gcs.blob(f"{model_prefix}/{fpath.name}").upload_from_filename(str(fpath))
    model_gcs_dir = f"gs://{bucket}/{model_prefix}"
    print(f"Model saved to {model_gcs_dir}")

    # --- write outputs for downstream KFP components ---
    _write(args.accuracy_mean_output_path, result["test_accuracy_mean"])
    _write(args.accuracy_ci95_output_path, result["test_accuracy_ci95"])
    _write(args.accuracy_std_output_path, result["test_accuracy_std"])
    _write(args.train_time_output_path, result["train_time_seconds"])
    _write(args.model_dir_output_path, model_gcs_dir)
    _write(args.data_sha_output_path, archive_sha)
    print("Wrote all KFP outputs.")


if __name__ == "__main__":
    main()
