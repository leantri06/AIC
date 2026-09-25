"""Portable paths for the precomputed SigLIP2 dataset."""
from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath


def default_data_root() -> Path:
    configured = os.environ.get("AIC_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    workspace_data = Path(__file__).resolve().parents[3] / "data"
    for candidate in (workspace_data, Path.cwd() / "data", Path.cwd().parent / "data"):
        if (candidate / "processed").is_dir():
            return candidate.resolve()
    return workspace_data


def resolve_data_root_and_paths(base_str: str | Path, model_choice: str = "siglip2") -> dict[str, str]:
    if model_choice not in ("siglip2", "siglip2-giant"):
        raise ValueError("This application uses the precomputed SigLIP2 index only.")
    s_val = str(base_str).strip(" '\"") if base_str else ""
    base = Path(s_val) if s_val else default_data_root()
    data = (base / "data" if (base / "data").is_dir() else base).resolve()

    def directory(*names: str) -> str:
        candidates = [data / "extracted" / name for name in names]
        return str(next((p for p in candidates if p.is_dir()), candidates[0]))

    prefix = "siglip2_giant" if model_choice == "siglip2-giant" else "siglip2"
    return {
        "data_dir": str(data),
        "manifest": str(data / "processed" / f"{prefix}_manifest.jsonl"),
        "features": str(data / "processed" / f"{prefix}_features.npy"),
        "keyframes": directory("keyframes", "Keyframes"),
        "map_keyframes": directory("map_keyframes", "map-keyframes"),
        "videos": directory("Videos", "videos"),
        "clip_model": model_choice,
        "clip_pretrained": "webli",
    }


def resolve_keyframe(stored_path: str, root: Path | None = None) -> Path:
    path = Path(stored_path)
    if path.is_file():
        return path
    root = root or Path(resolve_data_root_and_paths(str(default_data_root()))["keyframes"])
    # Windows manifests must also resolve when the app runs on Linux.
    parts = PureWindowsPath(stored_path).parts
    for i, part in enumerate(parts):
        if part.lower() == "keyframes":
            candidate = root.joinpath(*parts[i + 1:])
            if candidate.is_file():
                return candidate
    return root.joinpath(*parts[-2:])
