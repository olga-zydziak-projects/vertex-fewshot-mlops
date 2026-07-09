#!/usr/bin/env python3
"""Entry point: train ProtoNet on frozen Omniglot, logging to Vertex Experiments.

Usage:
    python scripts/train.py --project MY-PROJECT
    python scripts/train.py --project MY-PROJECT --k-shot 1 --train-iters 1000
    python scripts/train.py --project MY-PROJECT --no-vertex      # local, no logging
"""
import argparse

from fsl.config import TrainConfig
from fsl.training.loop import train


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ProtoNet on frozen Omniglot.")
    ap.add_argument("--project", required=True, help="GCP project ID")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--n-way", type=int, default=5)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--query", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-iters", type=int, default=300)
    ap.add_argument("--no-vertex", action="store_true",
                    help="Disable Vertex Experiments logging (local run).")
    args = ap.parse_args()

    cfg = TrainConfig(
        project=args.project,
        region=args.region,
        n_way=args.n_way,
        k_shot=args.k_shot,
        query=args.query,
        seed=args.seed,
        train_iters=args.train_iters,
        log_to_vertex=not args.no_vertex,
    )
    train(cfg)


if __name__ == "__main__":
    main()
