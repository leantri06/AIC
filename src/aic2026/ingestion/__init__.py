from __future__ import annotations

from pathlib import Path

from .asr_manifest import enrich_manifest_with_asr
from .manifest import build_manifest, load_manifest

__all__ = ["build_manifest", "enrich_manifest_with_asr", "load_manifest", "resolve_feature_sources"]

LOCAL_SIGLIP2_MANIFEST = Path(r"D:\ai_challenge\data\processed\siglip2_manifest.jsonl")
LOCAL_SIGLIP2_FEATURES = Path(r"D:\ai_challenge\data\processed\siglip2_features.npy")

LOCAL_SIGLIP_MANIFEST = Path(r"D:\ai_challenge\data\processed\siglip_manifest.jsonl")
LOCAL_SIGLIP_FEATURES = Path(r"D:\ai_challenge\data\processed\siglip_features.npy")

LOCAL_DERIVED_MANIFEST = Path(r"D:\ai_challenge\data\processed\derived_manifest.jsonl")
LOCAL_DERIVED_FEATURES = Path(r"D:\ai_challenge\data\processed\derived_features.npy")

SIGLIP_MANIFEST = Path("data/processed/siglip_manifest.jsonl")
SIGLIP_FEATURES = Path("data/processed/siglip_features.npy")

OFFICIAL_MANIFEST = Path("data/processed/official_manifest.jsonl")
OFFICIAL_FEATURES = Path("data/processed/official_features.npy")

DERIVED_MANIFEST = Path("data/processed/derived_manifest.jsonl")
DERIVED_FEATURES = Path("data/processed/derived_features.npy")


def resolve_feature_sources(
    manifest_path: str | Path | None = None,
    features_path: str | Path | None = None,
    *,
    prefer_official: bool = False,
) -> tuple[Path, Path]:
    if manifest_path is not None and features_path is not None:
        m = Path(manifest_path)
        f = Path(features_path)
        if not m.exists():
            for cand_m in (LOCAL_SIGLIP2_MANIFEST, LOCAL_SIGLIP_MANIFEST, LOCAL_DERIVED_MANIFEST):
                if cand_m.exists() and m.name == cand_m.name:
                    m = cand_m
                    break
        if not f.exists():
            for cand_f in (LOCAL_SIGLIP2_FEATURES, LOCAL_SIGLIP_FEATURES, LOCAL_DERIVED_FEATURES):
                if cand_f.exists() and f.name == cand_f.name:
                    f = cand_f
                    break
        return m, f

    if LOCAL_SIGLIP2_MANIFEST.exists() and LOCAL_SIGLIP2_FEATURES.exists():
        return LOCAL_SIGLIP2_MANIFEST, LOCAL_SIGLIP2_FEATURES

    if LOCAL_SIGLIP_MANIFEST.exists() and LOCAL_SIGLIP_FEATURES.exists():
        return LOCAL_SIGLIP_MANIFEST, LOCAL_SIGLIP_FEATURES

    if LOCAL_DERIVED_MANIFEST.exists() and LOCAL_DERIVED_FEATURES.exists():
        return LOCAL_DERIVED_MANIFEST, LOCAL_DERIVED_FEATURES

    if SIGLIP_MANIFEST.exists() and SIGLIP_FEATURES.exists():
        return SIGLIP_MANIFEST, SIGLIP_FEATURES

    if prefer_official and OFFICIAL_MANIFEST.exists() and OFFICIAL_FEATURES.exists():
        return OFFICIAL_MANIFEST, OFFICIAL_FEATURES

    if DERIVED_MANIFEST.exists() and DERIVED_FEATURES.exists():
        return DERIVED_MANIFEST, DERIVED_FEATURES

    return LOCAL_DERIVED_MANIFEST if LOCAL_DERIVED_MANIFEST.exists() else DERIVED_MANIFEST, LOCAL_DERIVED_FEATURES if LOCAL_DERIVED_FEATURES.exists() else DERIVED_FEATURES
