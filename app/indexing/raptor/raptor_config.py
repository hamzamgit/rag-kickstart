"""Configuration for the RAPTOR tree-building workflow."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RaptorSettings:
    """Settings supplied through environment variables or their safe defaults."""

    max_levels: int = 3
    clustering_algorithm: str = "gmm"
    min_clusters: int = 2
    max_clusters: int = 10
    stop_condition: str = "single_root"


def load_raptor_settings() -> RaptorSettings:
    """Load and validate the RAPTOR-specific environment settings."""
    settings = RaptorSettings(
        max_levels=int(os.getenv("RAPTOR_MAX_LEVELS", "3")),
        clustering_algorithm=os.getenv("RAPTOR_CLUSTERING_ALGO", "gmm").lower(),
        min_clusters=int(os.getenv("RAPTOR_MIN_CLUSTERS", "2")),
        max_clusters=int(os.getenv("RAPTOR_MAX_CLUSTERS", "10")),
        stop_condition=os.getenv("RAPTOR_STOP_CONDITION", "single_root").lower(),
    )

    if settings.max_levels < 1:
        raise ValueError("RAPTOR_MAX_LEVELS must be at least 1.")
    if settings.clustering_algorithm != "gmm":
        raise ValueError("Only RAPTOR_CLUSTERING_ALGO=gmm is currently supported.")
    if settings.min_clusters < 2:
        raise ValueError("RAPTOR_MIN_CLUSTERS must be at least 2.")
    if settings.max_clusters < settings.min_clusters:
        raise ValueError("RAPTOR_MAX_CLUSTERS must be >= RAPTOR_MIN_CLUSTERS.")
    if settings.stop_condition != "single_root":
        raise ValueError("Only RAPTOR_STOP_CONDITION=single_root is supported.")

    return settings
