from __future__ import annotations
from pathlib import Path
from aic2026.paths import default_data_root
from .manifest import build_manifest, load_manifest

__all__ = ["build_manifest", "load_manifest", "resolve_feature_sources"]

def resolve_feature_sources(manifest_path=None, features_path=None, *, prefer_official=False):
    """Select the SigLIP2 pair; explicit paths are never silently substituted."""
    if manifest_path is not None and features_path is not None:
        return Path(manifest_path), Path(features_path)
    if manifest_path is not None:
        manifest = Path(manifest_path)
        return manifest, manifest.parent / "siglip2_features.npy"
    if features_path is not None:
        features = Path(features_path)
        return features.parent / "siglip2_manifest.jsonl", features
    processed = default_data_root() / "processed"
    return processed / "siglip2_manifest.jsonl", processed / "siglip2_features.npy"
