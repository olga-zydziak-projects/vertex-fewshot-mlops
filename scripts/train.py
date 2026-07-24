#!/usr/bin/env python3
"""Entry point: train ProtoNet on a frozen dataset, logging to Vertex Experiments.

This is the LOCAL loop — the fastest way to find out whether a change works,
without paying for a pipeline run. The pipeline uses `train_pipeline_entry.py`
instead; both call the same `fsl.training.loop.train`.

Usage:
    python scripts/train.py --project MY-PROJECT
    python scripts/train.py --project MY-PROJECT --dataset resisc45 --no-vertex
    python scripts/train.py --project MY-PROJECT --k-shot 1 --train-iters 1000
    python scripts/train.py --project MY-PROJECT --no-vertex      # local, no logging

Input geometry (`--in-channels`, `--image-size`) defaults to the domain's
registered values — 1x28 for Omniglot, 3x84 for RESISC45 — so a domain swap is
just `--dataset`. Pass them explicitly only when experimenting with a different
resolution; the values used are printed, so a run is never ambiguous.
"""
import argparse

from fsl.config import TrainConfig
from fsl.data.registry import get_input_geometry
from fsl.training.loop import train


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ProtoNet on a frozen dataset.")
    ap.add_argument("--project", required=True, help="GCP project ID")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--dataset", default="omniglot",
                    help="which frozen dataset to episode over (omniglot, resisc45)")
    ap.add_argument("--n-way", type=int, default=5)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--query", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate (default: TrainConfig's)")
    ap.add_argument("--embedding-hid", type=int, default=None,
                    help="encoder width (default: TrainConfig's)")
    ap.add_argument("--in-channels", type=int, default=None,
                    help="input channels (default: the dataset's registered geometry)")
    ap.add_argument("--image-size", type=int, default=None,
                    help="input size after resize (default: the dataset's registered geometry)")
    ap.add_argument("--no-vertex", action="store_true",
                    help="Disable Vertex Experiments logging (local run).")
    args = ap.parse_args()

    # domain defaults, overridable per flag
    default_channels, default_size = get_input_geometry(args.dataset)
    in_channels = args.in_channels if args.in_channels is not None else default_channels
    image_size = args.image_size if args.image_size is not None else default_size

    overrides = {}
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.embedding_hid is not None:
        overrides["embedding_hid"] = args.embedding_hid

    cfg = TrainConfig(
        project=args.project,
        region=args.region,
        dataset=args.dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        query=args.query,
        seed=args.seed,
        train_iters=args.train_iters,
        in_channels=in_channels,
        image_size=image_size,
        log_to_vertex=not args.no_vertex,
        **overrides,
    )
    print(f"dataset={cfg.dataset} | input {cfg.in_channels}x{cfg.image_size}x{cfg.image_size} "
          f"| {cfg.n_way}-way {cfg.k_shot}-shot | lr={cfg.lr} hid={cfg.embedding_hid} "
          f"| iters={cfg.train_iters}")
    train(cfg)


if __name__ == "__main__":
    main()
