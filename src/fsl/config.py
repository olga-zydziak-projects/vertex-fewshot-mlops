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
    cache_dir: str = ""                    # derived from dataset if left empty

    # --- input geometry (the encoder and the loader must agree) ---
    # Conv4 has four max-pools, so image_size decides the embedding width:
    # 28 -> 1x1 (Omniglot), 84 -> 5x5 (mini-ImageNet convention, used for
    # RESISC45), 256 -> 16x16. in_channels is 1 for grayscale, 3 for RGB.
    in_channels: int = 1
    image_size: int = 28

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
        if not self.cache_dir:
            self.cache_dir = os.path.expanduser(f"~/data_cache/{self.dataset}")

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
            "in_channels": self.in_channels,
            "image_size": self.image_size,
        }
