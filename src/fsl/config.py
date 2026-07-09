"""Training configuration as a single typed object.

Replaces the notebook's config cell: instead of editing module-level constants,
you construct a TrainConfig (or build one from CLI args). Every run is fully
described by its config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # --- GCP / data ---
    project: str = ""
    region: str = "us-central1"
    bucket: str = ""                       # derived from project if left empty
    dataset: str = "omniglot"
    cache_dir: str = os.path.expanduser("~/data_cache/omniglot")

    # --- episode (few-shot is defined by these) ---
    n_way: int = 5
    k_shot: int = 5
    query: int = 15

    # --- training ---
    seed: int = 0
    train_iters: int = 300
    eval_every: int = 50
    val_episodes: int = 200
    test_episodes: int = 1000
    lr: float = 1e-3
    embedding_hid: int = 64

    # --- experiment tracking ---
    experiment_name: str = "fsl-omniglot-protonet"
    log_to_vertex: bool = True

    def __post_init__(self) -> None:
        if not self.bucket and self.project:
            self.bucket = f"{self.project}-fsl-data"

    def as_params(self) -> dict:
        """Flat parameter dict for experiment logging (provenance added by caller)."""
        return {
            "model": "protonet",
            "encoder": "conv4",
            "embedding_hid": self.embedding_hid,
            "distance": "euclidean",
            "n_way": self.n_way,
            "k_shot": self.k_shot,
            "query": self.query,
            "seed": self.seed,
            "train_iters": self.train_iters,
            "lr": self.lr,
            "optimizer": "adam",
            "eval_episodes": self.test_episodes,
            "dataset": self.dataset,
        }
