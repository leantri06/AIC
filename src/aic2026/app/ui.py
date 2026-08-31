from __future__ import annotations

import os

# Fix OpenMP duplicate lib runtime conflict & thread pool crashes on Windows (0xC0000005)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if os.path.exists(r"D:\bachkhoa\ai_challenge\data"):
    os.environ.setdefault("HF_HOME", r"D:\bachkhoa\ai_challenge\data\.cache\huggingface")
elif os.path.exists(r"D:\ai_challenge\data"):
    os.environ.setdefault("HF_HOME", r"D:\ai_challenge\data\.cache\huggingface")

import csv
import html
import json
import urllib.parse
from collections import defaultdict
from pathlib import Path

import httpx
import streamlit as st
import streamlit.components.v1 as components
from pydantic import ValidationError

from aic2026.ingestion import load_manifest
from aic2026.models import Candidate, Query
from aic2026.submission import competition_answer, csv_row

import re as _re

# Ký tự có dấu tiếng Việt → dùng để nhận diện query tiếng Việt (cần dịch).
_VI_MARK_RE = _re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵđèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]",
    _re.IGNORECASE,
)


def _is_english(text: str) -> bool:
    """True nếu text KHÔNG chứa ký tự tiếng Việt có dấu."""
    return not bool(_VI_MARK_RE.search(text or ""))


# Helper to avoid Streamlit 1.40+ deprecation warnings for width/use_container_width
_STREAMLIT_VERSION_PARTS = [int(p) for p in st.__version__.split(".")[:2] if p.isdigit()]
_USE_WIDTH_STRETCH = bool(
    _STREAMLIT_VERSION_PARTS
    and (
        _STREAMLIT_VERSION_PARTS[0] > 1
        or (_STREAMLIT_VERSION_PARTS[0] == 1 and _STREAMLIT_VERSION_PARTS[1] >= 40)
    )
)
_FULL_WIDTH = {"width": "stretch"} if _USE_WIDTH_STRETCH else {"use_container_width": True}


class BackendRequestError(RuntimeError):
    """Raised when the UI cannot obtain a valid result from the backend."""


def _detect_default_path(candidates: list[str]) -> str:
    for c in candidates:
        if Path(c).exists():
            return c
    return candidates[0]


DEFAULT_MASTER_ROOT = _detect_default_path([
    r"D:\bachkhoa\ai_challenge",
    r"D:\bachkhoa\ai_challenge\data",
    r"D:\ai_challenge",
    r"D:\ai_challenge\data",
    "data",
])


def resolve_data_root_and_paths(base_str: str, model_choice: str = "siglip2") -> dict[str, str]:
    """
    Tự động suy luận toàn bộ các đường dẫn con từ 1 đường dẫn gốc duy nhất và lựa chọn mô hình.
    """
    cleaned = (base_str or "").strip(" '\"")
    base = Path(cleaned) if cleaned else Path("data")
    
    if (base / "data").exists() and (base / "data").is_dir():
        data_dir = base / "data"
    else:
        data_dir = base

    model_configs = {
        "siglip2": {
            "clip_model": "siglip2",
            "clip_pretrained": "webli",
            "manifest_names": ["siglip2_manifest.jsonl", "siglip_manifest.jsonl", "derived_manifest.jsonl", "official_manifest.jsonl"],
            "feature_names": ["siglip2_features.npy", "siglip_features.npy", "derived_features.npy", "official_features.npy"],
        },
        "siglip": {
            "clip_model": "siglip",
            "clip_pretrained": "webli",
            "manifest_names": ["siglip_manifest.jsonl", "siglip2_manifest.jsonl", "derived_manifest.jsonl", "official_manifest.jsonl"],
            "feature_names": ["siglip_features.npy", "siglip2_features.npy", "derived_features.npy", "official_features.npy"],
        },
        "siglip2-base": {
            "clip_model": "siglip2-base",
            "clip_pretrained": "webli",
            "manifest_names": ["siglip2_base_manifest.jsonl", "siglip2_manifest.jsonl", "siglip_manifest.jsonl", "derived_manifest.jsonl"],
            "feature_names": ["siglip2_base_features.npy", "siglip2_features.npy", "siglip_features.npy", "derived_features.npy"],
        },
        "clip-l14": {
            "clip_model": "ViT-L-14",
            "clip_pretrained": "openai",
            "manifest_names": ["derived_manifest.jsonl", "official_manifest.jsonl", "clip_manifest.jsonl", "siglip2_manifest.jsonl"],
            "feature_names": ["derived_features.npy", "official_features.npy", "clip_features.npy", "siglip2_features.npy"],
        },
    }
    
    cfg = model_configs.get(model_choice, model_configs["siglip2"])

    manifest_candidates = [data_dir / "processed" / name for name in cfg["manifest_names"]]
    manifest = next((c for c in manifest_candidates if c.exists()), manifest_candidates[0])

    feature_candidates = [data_dir / "processed" / name for name in cfg["feature_names"]]
    features = next((c for c in feature_candidates if c.exists()), feature_candidates[0])

    asr_dense = data_dir / "processed" / "asr_dense_features.npy"

    kf_candidates = [
        data_dir / "extracted" / "Keyframes",
        data_dir / "extracted",
        data_dir / "raw" / "Keyframes",
    ]
    keyframes = next((c for c in kf_candidates if c.exists()), kf_candidates[0])

    vid_candidates = [
        data_dir / "extracted" / "Videos",
        data_dir / "Videos",
        data_dir / "raw" / "Videos",
    ]
    videos = next((c for c in vid_candidates if c.exists()), vid_candidates[0])

    map_candidates = [
        data_dir / "extracted" / "map-keyframes",
        data_dir / "map-keyframes",
        data_dir / "raw" / "map-keyframes",
    ]
    map_keyframes = next((c for c in map_candidates if c.exists()), map_candidates[0])

    return {
        "data_dir": str(data_dir),
        "manifest": str(manifest),
        "features": str(features),
        "asr_features": str(asr_dense),
        "keyframes": str(keyframes),
        "videos": str(videos),
        "map_keyframes": str(map_keyframes),
        "clip_model": cfg["clip_model"],
        "clip_pretrained": cfg["clip_pretrained"],
    }


# Initial default paths
_INIT_PATHS = resolve_data_root_and_paths(DEFAULT_MASTER_ROOT, "siglip2")
DEFAULT_MANIFEST = _INIT_PATHS["manifest"]
DEFAULT_FEATURES = _INIT_PATHS["features"]
DEFAULT_ROOT = _INIT_PATHS["keyframes"]
DEFAULT_VIDEOS_ROOT = _INIT_PATHS["videos"]
DEFAULT_MAP_KEYFRAMES = _INIT_PATHS["map_keyframes"]
DEFAULT_ASR_FEATURES = _INIT_PATHS["asr_features"]
DEFAULT_CLIP_MODEL = _INIT_PATHS["clip_model"]
DEFAULT_CLIP_PRETRAINED = _INIT_PATHS["clip_pretrained"]


st.set_page_config(page_title="PEGASUS AI Challenge 2026", layout="wide", page_icon="🦄")

# Khởi tạo an toàn toàn bộ session state trước khi widget khởi tạo
if "query_type" not in st.session_state:
    st.session_state["query_type"] = "kis"
if "search_mode" not in st.session_state:
    st.session_state["search_mode"] = "hybrid"
if "selected_model_choice" not in st.session_state:
    st.session_state["selected_model_choice"] = "siglip2"
if "app_theme" not in st.session_state:
    st.session_state["app_theme"] = "midnight"
if "theme_selector" not in st.session_state:
    st.session_state["theme_selector"] = "midnight"

current_theme = st.session_state["app_theme"]

# CSS Themes: Midnight Dark (Dịu mắt, hiện đại) vs Soft Light (Sáng dịu, không chói)
if current_theme == "midnight":
    CSS_VARIABLES = """
    :root {
      --bg-main: #0b0f17;
      --bg-gradient: radial-gradient(1200px 700px at 15% -10%, rgba(59, 130, 246, 0.12), transparent 70%),
                     radial-gradient(1000px 600px at 85% 0%, rgba(99, 102, 241, 0.12), transparent 70%),
                     linear-gradient(180deg, #0b0f17 0%, #0f172a 100%);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --card-bg: rgba(21, 28, 44, 0.75);
      --card-border: rgba(51, 65, 85, 0.6);
      --card-hover-border: #6366f1;
      --card-selected-border: #38bdf8;
      --card-selected-bg: rgba(14, 116, 144, 0.2);
      --topbar-bg: rgba(15, 23, 42, 0.82);
      --topbar-border: rgba(51, 65, 85, 0.7);
      --header-bg: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #042f2e 100%);
      --header-border: 1px solid rgba(99, 102, 241, 0.35);
      --input-bg: rgba(15, 23, 42, 0.85);
      --input-border: #334155;
      --input-text: #f8fafc;
      --meta-bg: rgba(11, 15, 23, 0.75);
      --meta-border: rgba(51, 65, 85, 0.5);
      --badge-score-high: rgba(16, 185, 129, 0.9);
      --badge-score-mid: rgba(245, 158, 11, 0.9);
      --badge-score-low: rgba(71, 85, 105, 0.85);
      --btn-copy-bg: #1e293b;
      --btn-copy-color: #cbd5e1;
      --btn-copy-border: #475569;
      --btn-copy-hover: #334155;
    }
    """
else:
    # Soft Minimal Light (Dịu nhẹ, không chói lóa)
    CSS_VARIABLES = """
    :root {
      --bg-main: #f1f5f9;
      --bg-gradient: radial-gradient(1200px 700px at 15% -10%, rgba(224, 231, 255, 0.6), transparent 70%),
                     radial-gradient(1000px 600px at 85% 0%, rgba(207, 250, 254, 0.6), transparent 70%),
                     linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
      --text-main: #0f172a;
      --text-muted: #64748b;
      --card-bg: rgba(255, 255, 255, 0.88);
      --card-border: rgba(203, 213, 225, 0.85);
      --card-hover-border: #818cf8;
      --card-selected-border: #0284c7;
      --card-selected-bg: rgba(240, 249, 255, 0.95);
      --topbar-bg: rgba(255, 255, 255, 0.82);
      --topbar-border: rgba(203, 213, 225, 0.9);
      --header-bg: linear-gradient(135deg, #312e81 0%, #1e293b 60%, #0f172a 100%);
      --header-border: 1px solid rgba(255, 255, 255, 0.2);
      --input-bg: #ffffff;
      --input-border: #cbd5e1;
      --input-text: #0f172a;
      --meta-bg: rgba(248, 250, 252, 0.9);
      --meta-border: rgba(226, 232, 240, 0.9);
      --badge-score-high: rgba(16, 185, 129, 0.9);
      --badge-score-mid: rgba(245, 158, 11, 0.9);
      --badge-score-low: rgba(148, 163, 184, 0.9);
      --btn-copy-bg: #f8fafc;
      --btn-copy-color: #334155;
      --btn-copy-border: #cbd5e1;
      --btn-copy-hover: #e2e8f0;
    }
    """

st.markdown(
    f"""
<style>
{CSS_VARIABLES}

/* Nền mượt mà dịu mắt */
html, body, [data-testid="stAppViewContainer"] {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: var(--text-main);
}}
[data-testid="stAppViewContainer"] {{
  background: var(--bg-gradient);
  background-attachment: fixed;
}}

/* Typography fixes */
h1, h2, h3, h4, h5, p, span, label {{
  color: var(--text-main);
}}

/* ---- Header (Sleek Matte Titanium Bar) ---- */
.peg-header {{
  position: relative; display: flex; align-items: center; justify-content: space-between;
  background: var(--header-bg);
  padding: .65rem 1.2rem; border-radius: .85rem;
  box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.4); margin-bottom: .75rem;
  border: var(--header-border);
}}
.peg-header-left {{ display: flex; align-items: center; gap: .85rem; }}
.peg-logo {{
  display: flex; align-items: center; justify-content: center;
  width: 44px; height: 44px; flex: 0 0 auto; background: rgba(255, 255, 255, 0.1);
  border-radius: .65rem; border: 1px solid rgba(255, 255, 255, 0.25);
  backdrop-filter: blur(8px);
}}
.peg-horse {{ width: 30px; height: 30px; display: block; }}
.peg-titles {{ display: flex; flex-direction: column; line-height: 1.15; }}
.peg-word {{ font-size: 1.35rem; font-weight: 900; color: #fff; letter-spacing: .1em; margin: 0; }}
.peg-sub {{ font-size: .72rem; color: #cbd5e1; font-weight: 600; letter-spacing: .04em; margin-top: .1rem; }}
.peg-badge {{
  background: rgba(255, 255, 255, 0.12); color: #38bdf8; font-size: .72rem; font-weight: 800;
  padding: .35rem .75rem; border-radius: 999px; border: 1px solid rgba(56, 189, 248, 0.35);
  white-space: nowrap; backdrop-filter: blur(6px);
}}

/* ---- Topbar Search Panel (Frosted Matte Card) ---- */
.topbar {{
  background: var(--topbar-bg); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--topbar-border); border-radius: .85rem;
  padding: .75rem .95rem; margin-bottom: .75rem; box-shadow: 0 8px 24px -4px rgba(0,0,0,0.25);
}}
.mini-label {{
  font-size: .68rem; text-transform: uppercase; letter-spacing: .08em;
  color: #818cf8; margin-bottom: .25rem; font-weight: 800; display: flex; align-items: center; gap: 4px;
}}

/* ---- Path Status Cards ---- */
.path-card {{
  background: var(--card-bg); border-radius: 6px; padding: 6px 10px;
  border: 1px solid var(--card-border); font-size: 0.76rem; margin-bottom: 4px; display: flex; align-items: center; justify-content: space-between;
}}
.path-label {{ font-weight: 700; color: var(--text-main); display: flex; align-items: center; gap: 6px; }}
.path-val {{ font-family: monospace; color: var(--text-muted); font-size: 0.72rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 320px; }}

/* ---- Gallery Cards (Modern Matte Cards) ---- */
.gallery-card-container {{
  background: var(--card-bg); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border: 1.5px solid var(--card-border); border-radius: 10px;
  padding: 7px; margin-bottom: 12px;
  box-shadow: 0 4px 14px -2px rgba(0,0,0,0.25);
  transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s ease, border-color 0.2s ease;
  position: relative;
}}
.gallery-card-container:hover {{
  transform: translateY(-4px) scale(1.015);
  box-shadow: 0 12px 24px -4px rgba(0,0,0,0.4);
  border-color: var(--card-hover-border);
}}
.gallery-card-container.is-selected {{
  border: 2.5px solid var(--card-selected-border);
  background: var(--card-selected-bg);
  box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25), 0 12px 24px -4px rgba(0,0,0,0.4);
}}

/* Hero Image Container */
.hero-wrap {{
  position: relative; border-radius: 6px; overflow: hidden;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3); background: #000;
}}
.hero-wrap img {{ display: block; width: 100%; height: auto; transition: transform 0.25s ease; }}
.hero-wrap:hover img {{ transform: scale(1.03); }}

.badge-time {{
  position: absolute; top: 5px; left: 5px; z-index: 3;
  background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
  color: #f8fafc; font-size: 0.65rem; font-weight: 800; font-family: monospace;
  padding: 2px 5px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.2);
}}
.badge-score {{
  position: absolute; top: 5px; right: 5px; z-index: 3;
  backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
  font-size: 0.65rem; font-weight: 800;
  padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.25);
}}
.badge-score-high {{ background: var(--badge-score-high); color: #fff; }}
.badge-score-mid  {{ background: var(--badge-score-mid); color: #fff; }}
.badge-score-low  {{ background: var(--badge-score-low); color: #fff; }}

.badge-selected-order {{
  position: absolute; bottom: 5px; right: 5px; z-index: 3;
  background: linear-gradient(135deg, #0284c7, #0369a1); color: #fff;
  font-size: 0.7rem; font-weight: 900; padding: 2px 7px; border-radius: 999px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.4); border: 1.5px solid #fff;
}}

/* Card metadata info */
.card-meta {{
  font-size: 0.72rem; color: var(--text-main); line-height: 1.3; margin-top: 5px;
  background: var(--meta-bg); border-radius: 6px; padding: 4px 6px;
  border: 1px solid var(--meta-border);
}}
.card-rank {{ font-weight: 900; color: #818cf8; font-size: 0.78rem; }}
.card-video {{ font-weight: 800; color: var(--text-main); font-family: monospace; font-size: 0.78rem; }}

/* 1-Click Copy Button */
.btn-copy-fast {{
  width: 100%; border: 1px solid var(--btn-copy-border); background: var(--btn-copy-bg); color: var(--btn-copy-color);
  padding: 4px 0; border-radius: 5px; font-size: 0.72rem; font-weight: 700;
  cursor: pointer; transition: all 0.15s ease; display: flex; align-items: center; justify-content: center; gap: 4px;
}}
.btn-copy-fast:hover {{ background: var(--btn-copy-hover); color: var(--text-main); }}
.btn-copy-fast:active {{ transform: scale(0.96); }}

/* Inputs & Form Submit Button */
.stButton > button[data-testid="baseButton-primary"] {{
  background: linear-gradient(135deg, #4f46e5 0%, #0284c7 100%) !important;
  color: #fff !important; font-weight: 800 !important; letter-spacing: .03em;
  border: none !important; border-radius: .55rem !important; padding: .55rem .75rem !important;
  box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
  transition: transform .15s ease, box-shadow .15s ease !important;
}}
.stButton > button[data-testid="baseButton-primary"]:hover {{
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 18px rgba(79, 70, 229, 0.45) !important;
}}

.stTextInput > div > div > input {{
  border: 1.5px solid var(--input-border) !important; border-radius: .55rem !important;
  background: var(--input-bg) !important; color: var(--input-text) !important;
  padding: .5rem .75rem !important; font-size: .9rem !important;
  transition: border-color .15s ease !important;
}}
.stTextInput > div > div > input:focus {{
  border-color: #6366f1 !important;
  box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

# Logo PEGASUS: đầu KỲ LÂN (unicorn) — SVG line-art trắng
PEGASUS_UNICORN_SVG = """
<svg class="peg-horse" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"
     stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"
     aria-label="PEGASUS unicorn">
  <path d="m15.6 4.8 2.7 2.3M15.5 10S19 7 22 2c-6 2-10 5-10 5m-.5 5H11"/>
  <path d="M5 15a4 4 0 0 0 4 4h7.8l.3.3a3 3 0 0 0 4-4.46L12 7c0-3-1-5-1-5S8 3 8 7c-4 1-6 3-6 3"/>
  <path d="M2 4.5C4 3 6 3 6 3l2 4M6.14 17.8S4 19 2 22"/>
</svg>
"""


# ---------------------------------------------------------------------------
# Backend call
# ---------------------------------------------------------------------------
def run_backend(task_type: str, query: Query, runtime: dict, backend_url: str):
    from aic2026.agent.types import AgentResult

    try:
        response = httpx.post(
            f"{backend_url.rstrip('/')}/tasks/{task_type}/run",
            json={
                "query_id": query.query_id,
                "text": query.text,
                "question": query.question,
                "events": query.events,
                "runtime": runtime,
            },
            timeout=300,
        )
        response.raise_for_status()
        return AgentResult.model_validate(response.json())
    except httpx.TimeoutException as exc:
        raise BackendRequestError("Backend xử lý quá thời gian cho phép.") from exc
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json()
        except json.JSONDecodeError:
            detail = exc.response.text
        raise BackendRequestError(f"Backend trả HTTP {exc.response.status_code}: {detail}") from exc
    except httpx.RequestError:
        try:
            from aic2026.app.api import RuntimeConfig, TaskRequest, run_task
            task_req = TaskRequest(
                query_id=query.query_id,
                text=query.text,
                question=query.question,
                events=query.events,
                runtime=RuntimeConfig(**runtime) if isinstance(runtime, dict) else runtime,
            )
            return run_task(task_type, task_req)
        except Exception as local_exc:
            raise BackendRequestError(f"Lỗi khi chạy Agent trực tiếp: {local_exc}") from local_exc
    except (json.JSONDecodeError, ValidationError) as exc:
        raise BackendRequestError("Backend không trả về dữ liệu hợp lệ.") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_events(events_text: str, text: str = "") -> list[str]:
    raw_lines = [line.strip() for line in (events_text or "").splitlines() if line.strip()]
    if not raw_lines and text:
        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(raw_lines) == 1:
            parts = _re.split(r"(?:^|\s+)\d+[\s:.)-]+\s*", raw_lines[0])
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 2:
                raw_lines = parts
    if len(raw_lines) == 1:
        if ";" in raw_lines[0]:
            raw_lines = [part.strip() for part in raw_lines[0].split(";") if part.strip()]
        elif "->" in raw_lines[0]:
            raw_lines = [part.strip() for part in raw_lines[0].split("->") if part.strip()]
    cleaned = []
    for line in raw_lines:
        l = _re.sub(r"^(?:event\s*\d+[\s:.-]*|\d+[\s:.)-]+\s*|[-*•]\s*)", "", line, flags=_re.IGNORECASE).strip()
        if l:
            cleaned.append(l)
    return cleaned


def build_query(query_type: str, text: str, question: str, events_text: str) -> Query:
    evs = parse_events(events_text, text=text)
    return Query(
        query_id="live-query",
        type=query_type or "kis",
        text=text.strip() or (evs[0] if evs else ""),
        question=question.strip() or None,
        events=evs,
    )


@st.cache_data(show_spinner=False)
def load_keyframe_lookup(manifest_path: str) -> dict[tuple[str, int], str]:
    records = load_manifest(Path(manifest_path))
    return {(record.video_id, record.frame_id): record.keyframe_path for record in records}


@st.cache_data(show_spinner=False)
def load_asr_lookup(manifest_path: str) -> dict[tuple[str, int], list[str]]:
    try:
        records = load_manifest(Path(manifest_path))
        return {
            (record.video_id, record.frame_id): record.asr_text
            for record in records
            if getattr(record, "asr_text", None)
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_video_asr_timeline(manifest_path: str, video_id: str) -> list[tuple[int, list[str]]]:
    try:
        records = load_manifest(Path(manifest_path))
        timeline = []
        for r in records:
            if r.video_id == video_id and getattr(r, "asr_text", None):
                timeline.append((r.frame_id, r.asr_text))
        timeline.sort(key=lambda x: x[0])
        return timeline
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def load_ocr_lookup(manifest_path: str) -> dict[tuple[str, int], list[str]]:
    try:
        records = load_manifest(Path(manifest_path))
        return {
            (record.video_id, record.frame_id): record.object_labels
            for record in records
            if getattr(record, "object_labels", None)
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_video_keyframes(manifest_path: str | None) -> dict[str, list[int]]:
    by_video: dict[str, list[int]] = defaultdict(list)
    if not manifest_path or not Path(manifest_path).exists():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                by_video[rec["video_id"]].append(int(rec["frame_id"]))
        for vid in by_video:
            by_video[vid] = sorted(set(by_video[vid]))
    except Exception:
        pass
    return dict(by_video)


def resolve_keyframe_path(stored_path: str, root: Path) -> Path:
    path = Path(stored_path)
    if path.exists():
        return path

    parts = path.parts
    rel_candidates: list[Path] = []
    if "Keyframes" in parts:
        kf_idx = parts.index("Keyframes")
        rel_candidates.append(Path(*parts[kf_idx + 1 :]))
        rel_candidates.append(Path(*parts[kf_idx:]))
    elif len(parts) >= 2:
        rel_candidates.append(Path(parts[-2]) / parts[-1])

    rel_candidates.append(path)
    if len(parts) >= 1:
        rel_candidates.append(Path(parts[-1]))

    for rel in rel_candidates:
        for candidate in (
            root / rel,
            root / "Keyframes" / rel,
            Path(r"D:\bachkhoa\ai_challenge\data\extracted\Keyframes") / rel,
            Path(r"D:\bachkhoa\ai_challenge\data\extracted") / rel,
            Path(r"D:\ai_challenge\data\extracted\Keyframes") / rel,
            Path(r"D:\ai_challenge\data\extracted") / rel,
            Path.cwd() / rel,
            Path.cwd() / "data" / "extracted" / "Keyframes" / rel,
            Path.cwd() / "data" / "processed" / rel,
        ):
            if candidate.exists():
                return candidate
    return root / path.name


def keyframe_file(candidate: Candidate, root: Path) -> Path | None:
    if not candidate.keyframe_path:
        return None
    return resolve_keyframe_path(candidate.keyframe_path, root)


def event_keyframe_file(video_id: str, frame_id: int, lookup: dict[tuple[str, int], str], root: Path) -> Path | None:
    stored_path = lookup.get((video_id, frame_id))
    if stored_path is not None:
        p = resolve_keyframe_path(stored_path, root)
        if p and p.exists():
            return p
    for base in (
        root,
        root / "Keyframes",
        Path(r"D:\bachkhoa\ai_challenge\data\extracted\Keyframes"),
        Path(r"D:\bachkhoa\ai_challenge\data\extracted"),
        Path(r"D:\ai_challenge\data\extracted\Keyframes"),
        Path(r"D:\ai_challenge\data\extracted"),
        Path.cwd() / "data" / "raw" / "Keyframes",
    ):
        for pattern in (f"{frame_id:03d}.jpg", f"{frame_id:04d}.jpg", f"{frame_id}.jpg"):
            candidate = base / video_id / pattern
            if candidate.exists():
                return candidate
    return None


def build_btc_csv(query: Query, candidates: list[Candidate], selected_idx: list[int]) -> str:
    picked = [candidates[i] for i in selected_idx if 0 <= i < len(candidates)]
    return "\n".join(csv_row(query, c) for c in picked)


_VIDEO_FILE_CACHE: dict[str, Path] = {}
_PTS_CACHE: dict[str, dict[int, float]] = {}


def resolve_video_file(video_id: str, custom_root: str | Path | None = None) -> Path | None:
    clean_vid = video_id.strip()
    cache_key = f"{clean_vid}_{custom_root}"
    if cache_key in _VIDEO_FILE_CACHE:
        cached = _VIDEO_FILE_CACHE[cache_key]
        if cached.exists():
            return cached

    roots = []
    if custom_root:
        roots.append(Path(custom_root))
    roots.extend([
        Path(DEFAULT_VIDEOS_ROOT),
        Path(r"D:\bachkhoa\ai_challenge\data\extracted\Videos"),
        Path(r"D:\ai_challenge\data\extracted\Videos"),
        Path("data/extracted/Videos"),
        Path("data/raw/Videos"),
    ])

    for r in roots:
        for ext in (".mp4", ".MP4", ".mkv", ".avi", ".mov"):
            c = r / f"{clean_vid}{ext}"
            if c.exists() and c.is_file():
                _VIDEO_FILE_CACHE[cache_key] = c
                return c
            c_sub = r / clean_vid / f"{clean_vid}{ext}"
            if c_sub.exists() and c_sub.is_file():
                _VIDEO_FILE_CACHE[cache_key] = c_sub
                return c_sub

    return None


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 100))
    return f"{m:02d}:{s:02d}.{ms:02d}"


def get_single_video_pts_map(video_id: str, map_dir: str | Path | None = None) -> dict[int, float]:
    clean_vid = video_id.strip()
    if clean_vid in _PTS_CACHE:
        return _PTS_CACHE[clean_vid]

    roots = [
        Path(map_dir) if map_dir else Path(DEFAULT_MAP_KEYFRAMES),
        Path(DEFAULT_MAP_KEYFRAMES),
        Path(r"D:\bachkhoa\ai_challenge\data\extracted\map-keyframes"),
        Path(r"D:\ai_challenge\data\extracted\map-keyframes"),
        Path("data/extracted/map-keyframes"),
    ]

    for r in roots:
        candidates = [
            r / f"{clean_vid}.csv",
            r / "map-keyframes" / f"{clean_vid}.csv",
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                pts_map: dict[int, float] = {}
                try:
                    with c.open(encoding="utf-8") as fh:
                        reader = csv.DictReader(fh)
                        for row in reader:
                            f_idx = row.get("frame_idx")
                            pts = row.get("pts_time")
                            if f_idx and pts:
                                try:
                                    pts_map[int(f_idx)] = float(pts)
                                except ValueError:
                                    pass
                    if pts_map:
                        _PTS_CACHE[clean_vid] = pts_map
                        return pts_map
                except Exception:
                    pass
    return {}


def get_frame_timestamp(
    video_id: str,
    frame_id: int,
    map_dir: str | Path | None = None,
    default_fps: float = 25.0,
) -> float:
    pts_map = get_single_video_pts_map(video_id, map_dir)
    if frame_id in pts_map:
        return pts_map[frame_id]
    return float(frame_id) / default_fps


@st.dialog("🎬 Trình Xem Video & Keyframe Inspector Pro", width="large")
def show_video_dialog(
    video_id: str,
    initial_frame_id: int,
    manifest_path: str | None = None,
    keyframes_root: Path | None = None,
    videos_root: str | None = None,
    map_dir: str | None = None,
) -> None:
    state_key = f"_cur_fid_{video_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = initial_frame_id
    current_fid = st.session_state[state_key]

    v_path = resolve_video_file(video_id, videos_root)
    ts = get_frame_timestamp(video_id, current_fid, map_dir)
    lookup = load_keyframe_lookup(manifest_path) if manifest_path else {}
    video_kf_map = load_video_keyframes(manifest_path)
    all_frames = sorted(set(video_kf_map.get(video_id, [current_fid])))
    asr_lookup = load_asr_lookup(manifest_path) if manifest_path else {}

    st.markdown(
        f"### 📹 Video: `{video_id}` &nbsp;|&nbsp; Frame: `{current_fid}` &nbsp;|&nbsp; ⏱️ **{format_timestamp(ts)}** ({ts:.2f}s)"
    )

    col_player, col_side = st.columns([3, 2], gap="medium")

    with col_player:
        if v_path and v_path.exists():
            st.video(str(v_path), start_time=int(ts), autoplay=True)
            st.caption(f"📁 Video: `{v_path}` (mốc: {format_timestamp(ts)})")
        else:
            st.warning(
                f"⚠️ Không tìm thấy file video `{video_id}.mp4` trong `{videos_root or DEFAULT_VIDEOS_ROOT}`.\n\n"
                f"Hãy kiểm tra lại đường dẫn video trong 'Cấu hình đường dẫn dữ liệu'."
            )

        # Tracker Frame HTML5
        tracker_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
body {{ background: transparent; overflow: hidden; }}
.card {{
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    border: 1.5px solid #4f46e5; border-radius: 8px;
    padding: 8px 14px; display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 4px 12px rgba(0,0,0,0.35);
}}
.title {{ font-size: 0.68rem; color: #a5b4fc; font-weight: 800; text-transform: uppercase; }}
.frame-row {{ display: flex; align-items: baseline; gap: 6px; margin-top: 2px; }}
.frame-label {{ font-size: 0.85rem; color: #cbd5e1; font-weight: 600; }}
.frame-val {{ font-size: 1.35rem; font-weight: 900; color: #38bdf8; font-family: monospace; background: rgba(0,0,0,0.55); padding: 1px 7px; border-radius: 5px; }}
.time-val {{ font-size: 0.8rem; color: #94a3b8; font-weight: 500; }}
.btn-copy {{
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    color: white; border: none; padding: 8px 16px; border-radius: 7px;
    font-weight: 800; font-size: 0.88rem; cursor: pointer; display: flex; align-items: center; gap: 6px;
    box-shadow: 0 4px 10px rgba(2, 132, 199, 0.4);
}}
.btn-copy:active {{ transform: scale(0.96); }}
</style>
</head>
<body>
<div class="card">
    <div>
        <div class="title">⚡ FRAME LIVE ĐANG PHÁT</div>
        <div class="frame-row">
            <span class="frame-label">Frame:</span>
            <span id="live-frame" class="frame-val">{current_fid}</span>
            <span id="live-time" class="time-val">({format_timestamp(ts)})</span>
        </div>
    </div>
    <button class="btn-copy" id="btn-copy" onclick="doCopy()">
        <span>📋</span>
        <span id="btn-text">BẤM COPY FRAME NÀY</span>
    </button>
</div>

<script>
const vid = {json.dumps(video_id)};
const fallbackTs = {ts};

function getVideo() {{
    try {{
        const pDoc = window.parent.document;
        const dlg = pDoc.querySelector('div[data-testid="stDialog"]') || pDoc.body;
        return dlg.querySelector('video') || pDoc.querySelector('video');
    }} catch(e) {{ return null; }}
}}

function pad(n) {{ return String(n).padStart(2, '0'); }}
function fmt(sec) {{
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 100);
    return pad(m) + ':' + pad(s) + '.' + pad(ms);
}}

function update() {{
    const v = getVideo();
    if (!v) return;
    const cur = v.currentTime || 0;
    const fid = Math.round(cur * 25.0);
    const fEl = document.getElementById('live-frame');
    const tEl = document.getElementById('live-time');
    if (fEl) fEl.innerText = String(fid);
    if (tEl) tEl.innerText = '(' + fmt(cur) + ')';
}}

function doCopy() {{
    const v = getVideo();
    const cur = v ? (v.currentTime || 0) : fallbackTs;
    const fid = Math.round(cur * 25.0);
    const copyTxt = vid ? (vid + ', ' + fid) : String(fid);

    let copied = false;
    try {{
        const ta = window.parent.document.createElement('textarea');
        ta.value = copyTxt;
        window.parent.document.body.appendChild(ta);
        ta.select();
        copied = window.parent.document.execCommand('copy');
        ta.remove();
    }} catch(e) {{}}

    if (!copied && navigator.clipboard && navigator.clipboard.writeText) {{
        navigator.clipboard.writeText(copyTxt).catch(() => {{}});
    }}

    const btn = document.getElementById('btn-copy');
    const btnTxt = document.getElementById('btn-text');
    if (btnTxt) btnTxt.innerText = '✅ ĐÃ COPY: ' + fid;
    if (btn) btn.style.background = '#059669';

    setTimeout(() => {{
        if (btnTxt) btnTxt.innerText = 'BẤM COPY FRAME NÀY';
        if (btn) btn.style.background = 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)';
    }}, 2500);
}}

function setup() {{
    const v = getVideo();
    if (v) {{
        v.addEventListener('timeupdate', update);
        v.addEventListener('seeked', update);
        v.addEventListener('play', update);
        v.addEventListener('pause', update);
    }}
}}
setTimeout(setup, 200);
setInterval(update, 200);
</script>
</body>
</html>"""
        components.html(tracker_html, height=72)

        st.text_input("📋 Chuỗi nộp bài chuẩn BTC (Video, Frame):", value=f"{video_id}, {current_fid}", key=f"txt_cur_val_{video_id}_{current_fid}")

    with col_side:
        kf_path = event_keyframe_file(video_id, current_fid, lookup, keyframes_root or Path("data/processed"))
        if kf_path and kf_path.exists():
            st.image(str(kf_path), **_FULL_WIDTH, caption=f"Keyframe: {current_fid} ({format_timestamp(ts)})")
        else:
            st.caption("Không có ảnh keyframe")

        if len(all_frames) > 1 and current_fid in all_frames:
            curr_idx = all_frames.index(current_fid)
            c_prev, c_next = st.columns(2)
            with c_prev:
                if curr_idx > 0:
                    prev_fid = all_frames[curr_idx - 1]
                    if st.button(f"⏮️ Frame trước ({prev_fid})", key=f"btn_prev_{video_id}", **_FULL_WIDTH):
                        st.session_state[state_key] = prev_fid
                        st.rerun()
            with c_next:
                if curr_idx < len(all_frames) - 1:
                    next_fid = all_frames[curr_idx + 1]
                    if st.button(f"⏭️ Frame sau ({next_fid})", key=f"btn_next_{video_id}", **_FULL_WIDTH):
                        st.session_state[state_key] = next_fid
                        st.rerun()

    # Section Tabs: Keyframes & Lời thoại ASR toàn video
    st.markdown("---")
    tab_kfs, tab_asr = st.tabs([f"🎞️ Keyframes ({len(all_frames)} frames)", "🗣️ Kịch bản Lời thoại ASR Toàn Video"])

    with tab_kfs:
        cols_per_row = 6
        curr_idx = all_frames.index(current_fid) if current_fid in all_frames else 0
        start_idx = max(0, curr_idx - 5)
        end_idx = min(len(all_frames), start_idx + 12)
        slice_frames = all_frames[start_idx:end_idx]

        def _render_kf_row(frames_chunk: list[int], prefix: str = "adj", offset: int = 0):
            cols = st.columns(len(frames_chunk))
            for c_idx, fid in enumerate(frames_chunk):
                idx_key = offset + c_idx
                with cols[c_idx]:
                    f_ts = get_frame_timestamp(video_id, fid, map_dir)
                    f_path = event_keyframe_file(video_id, fid, lookup, keyframes_root or Path("data/processed"))
                    is_active = (fid == current_fid)
                    if f_path and f_path.exists():
                        st.image(str(f_path), **_FULL_WIDTH)
                    st.caption(f"{'👉 ' if is_active else ''}**`{fid}`** ({format_timestamp(f_ts)})")
                    
                    c_k1, c_k2 = st.columns(2)
                    with c_k1:
                        if not is_active:
                            if st.button("▶️", key=f"kf_jump_{prefix}_{video_id}_{fid}_{idx_key}", **_FULL_WIDTH, help=f"Nhảy video tới frame {fid}"):
                                st.session_state[state_key] = fid
                                st.rerun()
                    with c_k2:
                        st.markdown(
                            f"""<button onclick="navigator.clipboard.writeText('{html.escape(video_id)}, {fid}'); this.innerText='✅';"
                                       style="width:100%; font-size:0.75rem; padding:4px 0; background:#334155; color:white; border:none; border-radius:4px; cursor:pointer;"
                                       title="Copy '{video_id}, {fid}'">📋</button>""",
                            unsafe_allow_html=True,
                        )

        for row_start in range(0, len(slice_frames), cols_per_row):
            _render_kf_row(slice_frames[row_start : row_start + cols_per_row], prefix="adj", offset=row_start)

        if len(all_frames) > len(slice_frames):
            with st.expander(f"📂 Xem toàn bộ {len(all_frames)} keyframes của video `{video_id}`", expanded=False):
                for row_start in range(0, len(all_frames), cols_per_row):
                    _render_kf_row(all_frames[row_start : row_start + cols_per_row], prefix="all", offset=row_start)

    with tab_asr:
        video_timeline = load_video_asr_timeline(manifest_path, video_id)
        if not video_timeline:
            st.caption("Video này không có dữ liệu lời thoại ASR.")
        else:
            for fid_asr, texts in video_timeline:
                asr_ts = get_frame_timestamp(video_id, fid_asr, map_dir)
                c_t1, c_t2 = st.columns([1, 4])
                with c_t1:
                    if st.button(f"⏱️ {format_timestamp(asr_ts)} (f{fid_asr})", key=f"btn_jump_asr_{video_id}_{fid_asr}", **_FULL_WIDTH):
                        st.session_state[state_key] = fid_asr
                        st.rerun()
                with c_t2:
                    st.markdown(f"🗣️ *{' '.join(texts)}*")


# ---------------------------------------------------------------------------
# Gallery renderers
# ---------------------------------------------------------------------------
def _toggle(idx: int, nonce: int):
    checked = st.session_state.get(f"sel_{nonce}_{idx}", False)
    s = st.session_state["selected"]
    if checked:
        if idx not in s:
            if not s:
                s.insert(0, idx)
            else:
                s.append(idx)
    else:
        if idx in s:
            s.remove(idx)


def _render_frame_gallery(query: Query, visible: list[tuple[int, Candidate]], root: Path, selected: list[int]) -> None:
    manifest_path = st.session_state.get("manifest_path")
    asr_lookup = load_asr_lookup(manifest_path) if manifest_path else {}
    ocr_lookup = load_ocr_lookup(manifest_path) if manifest_path else {}
    map_dir = st.session_state.get("cfg_map_keyframes", DEFAULT_MAP_KEYFRAMES)
    videos_root = st.session_state.get("cfg_videos_root", DEFAULT_VIDEOS_ROOT)
    nonce = int(st.session_state.get("_query_nonce", 0))

    st.subheader(f"🖼️ Gallery Kết quả ({len(visible)} ảnh keyframe)")
    grid = st.columns(5)
    for pos, (idx, item) in enumerate(visible):
        path = keyframe_file(item, root)
        ts = get_frame_timestamp(item.video_id, item.frame_id, map_dir)
        is_sel = idx in selected
        sel_order = (selected.index(idx) + 1) if is_sel else None

        score_class = "badge-score-high" if item.score >= 0.7 else ("badge-score-mid" if item.score >= 0.45 else "badge-score-low")
        order_html = f'<span class="badge-selected-order">#{sel_order}</span>' if is_sel else ''

        with grid[pos % 5]:
            card_extra_class = " is-selected" if is_sel else ""
            st.markdown(f'<div class="gallery-card-container{card_extra_class}" data-video-id="{html.escape(item.video_id)}">', unsafe_allow_html=True)
            
            if path and path.exists():
                st.markdown(
                    f'<div class="hero-wrap">'
                    f'<span class="badge-time">⏱️ {format_timestamp(ts)}</span>'
                    f'<span class="badge-score {score_class}">🎯 {item.score:.3f}</span>'
                    f'{order_html}</div>',
                    unsafe_allow_html=True,
                )
                st.image(str(path), **_FULL_WIDTH)
            else:
                st.caption("Không có ảnh")

            asr_snippets = asr_lookup.get((item.video_id, item.frame_id), [])
            asr_html = ""
            if asr_snippets:
                joined_asr = " ".join(asr_snippets)
                short_asr = (joined_asr[:55] + "...") if len(joined_asr) > 55 else joined_asr
                asr_html = (
                    f'<div style="font-size:0.72rem; color:#818cf8; background:rgba(99, 102, 241, 0.15); '
                    f'border-radius:4px; padding:2px 5px; margin-top:3px; line-height:1.2;" '
                    f'title="{html.escape(joined_asr)}">🗣️ {html.escape(short_asr)}</div>'
                )

            ocr_labels = ocr_lookup.get((item.video_id, item.frame_id), [])
            ocr_html = ""
            if ocr_labels:
                joined_ocr = ", ".join(ocr_labels[:4])
                short_ocr = (joined_ocr[:50] + "...") if len(joined_ocr) > 50 else joined_ocr
                ocr_html = (
                    f'<div style="font-size:0.72rem; color:#34d399; background:rgba(16, 185, 129, 0.15); '
                    f'border-radius:4px; padding:2px 5px; margin-top:2px; line-height:1.2;" '
                    f'title="{html.escape(", ".join(ocr_labels))}">🔤 {html.escape(short_ocr)}</div>'
                )

            st.markdown(
                f'<div class="card-meta">'
                f'<span class="card-rank">#{idx + 1}</span> &nbsp;'
                f'<span class="card-video">{html.escape(item.video_id)}</span> · <b>{item.frame_id}</b>'
                f"{ocr_html}"
                f"{asr_html}</div>",
                unsafe_allow_html=True,
            )

            # Action bar: Xem Video, 1-Click Copy, Checkbox chọn
            c_vid, c_cp, c_chk = st.columns([2.2, 1.3, 1.1], gap="small")
            with c_vid:
                if st.button(f"🎬 Xem ({format_timestamp(ts)})", key=f"btn_v_{nonce}_{idx}", **_FULL_WIDTH, help=f"Mở video {item.video_id} tại {format_timestamp(ts)}"):
                    show_video_dialog(
                        video_id=item.video_id,
                        initial_frame_id=item.frame_id,
                        manifest_path=manifest_path,
                        keyframes_root=root,
                        videos_root=videos_root,
                        map_dir=map_dir,
                    )
            with c_cp:
                st.markdown(
                    f"""<button class="btn-copy-fast" onclick="navigator.clipboard.writeText('{html.escape(item.video_id)}, {item.frame_id}'); this.innerText='✅ Copied'; setTimeout(() => this.innerText='📋 Copy', 1500);" title="Copy '{item.video_id}, {item.frame_id}'">📋 Copy</button>""",
                    unsafe_allow_html=True,
                )
            with c_chk:
                st.checkbox("Chọn", key=f"sel_{nonce}_{idx}", value=is_sel, on_change=_toggle, args=(idx, nonce), label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)


def _render_trake_gallery(query: Query, visible: list[tuple[int, Candidate]], root: Path, selected: list[int]) -> None:
    manifest_path = st.session_state.get("manifest_path")
    lookup = load_keyframe_lookup(manifest_path) if manifest_path else {}
    map_dir = st.session_state.get("cfg_map_keyframes", DEFAULT_MAP_KEYFRAMES)
    videos_root = st.session_state.get("cfg_videos_root", DEFAULT_VIDEOS_ROOT)
    nonce = int(st.session_state.get("_query_nonce", 0))

    n_events = len(query.events)
    st.subheader(f"🎬 Kết quả TRAKE: {len(visible)} Cụm Video ({n_events} hành động tuần tự)")

    for idx, item in visible:
        is_sel = idx in selected
        sel_order = (selected.index(idx) + 1) if is_sel else None
        event_frames = item.event_frames or []

        st.markdown(
            f'<div style="background: var(--card-bg); border: 2px solid {"var(--card-selected-border)" if is_sel else "var(--card-border)"}; '
            f'border-radius: 12px; padding: 12px 16px; margin-bottom: 16px; box-shadow: 0 4px 14px rgba(0,0,0,0.25);">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">'
            f'  <div><span style="font-weight:900; color:#818cf8; font-size:1.15rem;">#{idx + 1}</span> '
            f'  <span style="font-weight:800; font-size:1.05rem; margin-left:6px;">📹 Video: <code>{html.escape(item.video_id)}</code></span> · '
            f'  <span style="font-size:0.88rem; color:var(--text-muted);">Độ khớp chuỗi: <b>{item.score:.3f}</b></span></div>'
            f'  <span style="background: rgba(99, 102, 241, 0.2); color:#a5b4fc; padding: 3px 10px; border-radius: 999px; font-weight:700; font-size:0.78rem;">'
            f'  Chuỗi {len(event_frames)} hành động {f"(Chọn #{sel_order})" if is_sel else ""}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if not event_frames:
            st.caption("Candidate này không có event_frames.")
        else:
            cols = st.columns(len(event_frames))
            for ev_idx, (col, frame_id) in enumerate(zip(cols, event_frames)):
                path = event_keyframe_file(item.video_id, frame_id, lookup, root)
                ev_text = query.events[ev_idx] if ev_idx < len(query.events) else f"Hành động {ev_idx + 1}"
                ev_ts = get_frame_timestamp(item.video_id, frame_id, map_dir)
                with col:
                    st.markdown(
                        f'<div style="font-size:0.78rem; font-weight:800; color:var(--text-main); background:var(--meta-bg); '
                        f'padding:3px 6px; border-radius:6px; margin-bottom:4px; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;" '
                        f'title="{html.escape(ev_text)}">'
                        f'#{ev_idx + 1}: {html.escape(ev_text)}</div>',
                        unsafe_allow_html=True,
                    )
                    if path and path.exists():
                        st.markdown(
                            f'<div class="hero-wrap">'
                            f'<span class="badge-time">⏱️ {format_timestamp(ev_ts)}</span></div>',
                            unsafe_allow_html=True,
                        )
                        st.image(str(path), **_FULL_WIDTH)
                    else:
                        st.caption("Không có ảnh")
                    st.caption(f"**`frame {frame_id}`** ({format_timestamp(ev_ts)})")
                    if st.button("🎬 Xem", key=f"trake_v_{nonce}_{idx}_{frame_id}", **_FULL_WIDTH, help=f"Phát video tại {format_timestamp(ev_ts)}"):
                        show_video_dialog(
                            video_id=item.video_id,
                            initial_frame_id=frame_id,
                            manifest_path=manifest_path,
                            keyframes_root=root,
                            videos_root=videos_root,
                            map_dir=map_dir,
                        )

        c_sel_col1, c_sel_col2 = st.columns([3, 1])
        with c_sel_col1:
            st.caption(f"Dòng CSV nộp bài: `{item.video_id}, {', '.join(str(f) for f in event_frames)}`")
        with c_sel_col2:
            st.checkbox(
                "✅ Chọn cụm này",
                key=f"sel_{nonce}_{idx}",
                value=is_sel,
                on_change=_toggle,
                args=(idx, nonce),
            )
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Callback Function for Preset Chips (Executes BEFORE widgets instantiate)
# ---------------------------------------------------------------------------
def apply_preset_query(q_type: str, s_mode: str, q_text: str, q_question: str = "", q_events: str = "", translate: bool = False, model: str = "siglip2"):
    st.session_state["query_type"] = q_type
    st.session_state["search_mode"] = s_mode
    st.session_state["query_text"] = q_text
    st.session_state["query_question"] = q_question
    st.session_state["query_events"] = q_events
    st.session_state["cfg_translate"] = translate
    st.session_state["selected_model_choice"] = model
    st.session_state["_run_agent"] = True


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="peg-header">'
    '<div class="peg-header-left">'
    '<div class="peg-logo">'
    f"{PEGASUS_UNICORN_SVG}"
    "</div>"
    '<div class="peg-titles">'
    '<h1 class="peg-word">PEGASUS</h1>'
    '<span class="peg-sub">Video Retrieval AI Challenge 2026</span>'
    "</div>"
    "</div>"
    '<span class="peg-badge">🏆 KHÔNG THẮNG THÌ THÔI</span>'
    "</div>",
    unsafe_allow_html=True,
)


# ===========================================================================
# TOPBAR CONTROL PANEL (4 Cột Tinh Gọn)
# ===========================================================================
st.markdown('<div class="topbar">', unsafe_allow_html=True)

top_c1, top_c2, top_c3, top_c4 = st.columns([1.1, 1.8, 1.8, 0.9], gap="small")
with top_c1:
    st.markdown('<div class="mini-label">🎯 Luồng bài toán</div>', unsafe_allow_html=True)
    query_type = st.segmented_control(
        "Luồng",
        ["kis", "qa", "trake"],
        key="query_type",
        label_visibility="collapsed",
    )
with top_c2:
    st.markdown('<div class="mini-label">⚡ Chế độ tìm kiếm</div>', unsafe_allow_html=True)
    search_mode = st.segmented_control(
        "Chế độ",
        ["hybrid", "siglip", "asr", "ocr"],
        format_func=lambda x: {
            "hybrid": "🌈 Đa kênh",
            "siglip": "👁️ Thị giác",
            "asr": "🗣️ Lời thoại",
            "ocr": "🔤 OCR",
        }.get(x, x),
        key="search_mode",
        label_visibility="collapsed",
    )
with top_c3:
    st.markdown('<div class="mini-label">🤖 Mô hình Thị giác</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Mô hình",
        ["siglip2", "siglip", "clip-l14", "siglip2-base"],
        format_func=lambda x: {
            "siglip2": "🚀 SigLIP 2 (SO400M - SOTA)",
            "siglip": "👁️ SigLIP (SO400M - Chuẩn)",
            "clip-l14": "🎨 OpenAI CLIP (Derived)",
            "siglip2-base": "⚡ SigLIP 2 Base (ViT-B-16)",
        }.get(x, x),
        key="selected_model_choice",
        label_visibility="collapsed",
    )
with top_c4:
    st.markdown('<div class="mini-label">🌓 Giao diện</div>', unsafe_allow_html=True)
    theme_choice = st.selectbox(
        "Theme",
        ["midnight", "light"],
        format_func=lambda x: "🌙 Tối Dịu" if x == "midnight" else "☁️ Sáng Dịu",
        key="theme_selector",
        label_visibility="collapsed",
    )
    if theme_choice != st.session_state.get("app_theme", "midnight"):
        st.session_state["app_theme"] = theme_choice
        st.rerun()

question = ""
events_text = ""
if query_type == "qa":
    question = st.text_input("Câu hỏi Q&A", placeholder="Ví dụ: Xe màu gì? Có bao nhiêu người?", label_visibility="collapsed",
                             key="query_question")
if query_type == "trake":
    events_text = st.text_area(
        "Event TRAKE (mỗi event 1 dòng, theo thứ tự thời gian trong video)",
        placeholder=(
            "A person enters the room\n"
            "The person sits down\n"
            "The person starts speaking"
        ),
        height=120,
        label_visibility="collapsed",
        key="query_events",
        help="TRAKE tìm chuỗi sự kiện theo thứ tự thời gian. Nhập mỗi hành động trên 1 dòng từ sớm đến muộn.",
    )

# Form nhập Query & Nút Chạy
with st.form("query_form", border=False):
    t_col, run_col = st.columns([3.8, 1], gap="small")
    with t_col:
        st.markdown('<div class="mini-label">🔍 Nội dung truy vấn (Enter để tìm kiếm)</div>', unsafe_allow_html=True)
        text = st.text_input("Query", placeholder="Nhập câu mô tả tiếng Việt hoặc tiếng Anh...",
                             label_visibility="collapsed", key="query_text")
    with run_col:
        st.markdown('<div class="mini-label">&nbsp;</div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Chạy Tìm Kiếm", type="primary", **_FULL_WIDTH)
        if submitted:
            st.session_state["_run_agent"] = True

# Gợi ý đề mẫu 1-Click (Sử dụng on_click callback an toàn) & Tùy chọn Dịch
chip_col1, chip_col2, chip_col3, chip_col4, tr_col = st.columns([1, 1, 1.2, 1, 1], gap="small")
with chip_col1:
    st.button(
        "🚗 Xe tự hành (QA)",
        key="chip_qa_car",
        on_click=apply_preset_query,
        kwargs={
            "q_type": "qa",
            "s_mode": "siglip",
            "q_text": "The clip was filmed from inside a self-driving car; the steering wheel can be clearly seen turning to the right. The camera angle then shifts to the outside, capturing the white car turning left, with a red sign bearing six Chinese characters visible in the upper part of the frame.",
            "q_question": "What number is displayed on the side of the white car?",
            "model": "siglip2",
        },
        **_FULL_WIDTH,
    )
with chip_col2:
    st.button(
        "🍳 Đầu bếp (KIS)",
        key="chip_kis_cook",
        on_click=apply_preset_query,
        kwargs={
            "q_type": "kis",
            "s_mode": "hybrid",
            "q_text": "Một đầu bếp chế biến món ăn trong chảo, với các miếng dồi trường màu trắng và rau xanh. Đầu bếp cho bông hẹ vào chảo.",
            "translate": True,
            "model": "siglip2",
        },
        **_FULL_WIDTH,
    )
with chip_col3:
    st.button(
        "🚶‍♂️ Lấy nước (TRAKE)",
        key="chip_trake_fridge",
        on_click=apply_preset_query,
        kwargs={
            "q_type": "trake",
            "s_mode": "siglip",
            "q_text": "",
            "q_events": "A person opens the refrigerator\nThe person takes out a bottle of water\nThe person drinks water from the bottle",
            "model": "siglip2",
        },
        **_FULL_WIDTH,
    )
with chip_col4:
    st.button(
        "🗣️ Lời thoại ASR",
        key="chip_asr_news",
        on_click=apply_preset_query,
        kwargs={
            "q_type": "kis",
            "s_mode": "asr",
            "q_text": "bản tin thời sự thông báo sạt lở bờ sông",
            "model": "siglip2",
        },
        **_FULL_WIDTH,
    )
with tr_col:
    translate_query = st.checkbox(
        "🌐 Dịch VI→EN",
        value=st.session_state.get("cfg_translate", False),
        key="cfg_translate",
        help="Dịch truy vấn tiếng Việt sang tiếng Anh trước khi đưa vào SigLIP. Bật khi nhập tiếng Việt.",
    )

# ---------------------------------------------------------------------------
# Cấu hình 1-Click Master Root: Tự động ánh xạ toàn bộ đường dẫn con theo mô hình đã chọn
# ---------------------------------------------------------------------------
with st.expander("📂 Cấu hình Thư mục Dữ liệu Gốc (Chỉ cần điền 1 lần)", expanded=False):
    master_root_input = st.text_input(
        "📂 Thư mục Gốc Dự án / Data (Master Root):",
        value=st.session_state.get("cfg_master_root", DEFAULT_MASTER_ROOT),
        key="cfg_master_root",
        help="Ví dụ: D:\\bachkhoa\\ai_challenge hoặc D:\\bachkhoa\\ai_challenge\\data. Hệ thống sẽ tự động tìm thấy toàn bộ manifest, features, keyframes, videos, map-keyframes!",
    )
    
    # Auto-resolve all paths from single root input and active model choice
    auto_paths = resolve_data_root_and_paths(master_root_input, model_choice)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown(f'<div class="path-card"><span class="path-label">{"🟢" if Path(auto_paths["manifest"]).exists() else "🔴"} Manifest ({model_choice})</span><span class="path-val" title="{auto_paths["manifest"]}">{Path(auto_paths["manifest"]).name}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="path-card"><span class="path-label">{"🟢" if Path(auto_paths["features"]).exists() else "🔴"} Features ({model_choice})</span><span class="path-val" title="{auto_paths["features"]}">{Path(auto_paths["features"]).name}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="path-card"><span class="path-label">{"🟢" if Path(auto_paths["asr_features"]).exists() else "⚪"} Dense ASR Features</span><span class="path-val" title="{auto_paths["asr_features"]}">{Path(auto_paths["asr_features"]).name}</span></div>', unsafe_allow_html=True)
    with col_p2:
        st.markdown(f'<div class="path-card"><span class="path-label">{"🟢" if Path(auto_paths["keyframes"]).exists() else "🔴"} Keyframes Dir</span><span class="path-val" title="{auto_paths["keyframes"]}">{auto_paths["keyframes"]}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="path-card"><span class="path-label">{"🟢" if Path(auto_paths["videos"]).exists() else "⚪"} Videos (.mp4) Dir</span><span class="path-val" title="{auto_paths["videos"]}">{auto_paths["videos"]}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="path-card"><span class="path-label">{"🟢" if Path(auto_paths["map_keyframes"]).exists() else "⚪"} Map Keyframes (.csv)</span><span class="path-val" title="{auto_paths["map_keyframes"]}">{auto_paths["map_keyframes"]}</span></div>', unsafe_allow_html=True)

    with st.expander("🔧 Tùy chỉnh chi tiết từng file riêng lẻ (Tùy chọn nâng cao)", expanded=False):
        st.text_input("Backend API", "http://127.0.0.1:8000", key="cfg_backend")
        st.text_input("Manifest JSONL (Tự động theo model)", auto_paths["manifest"], disabled=True)
        st.text_input("Feature .npy (Tự động theo model)", auto_paths["features"], disabled=True)
        st.text_input("Dense ASR Feature .npy (BGE-M3)", auto_paths["asr_features"], disabled=True)
        st.text_input("Root Keyframes", auto_paths["keyframes"], key="cfg_root")
        st.text_input("Root Videos (.mp4)", auto_paths["videos"], key="cfg_videos_root")
        st.text_input("Thư mục map-keyframes (.csv)", auto_paths["map_keyframes"], key="cfg_map_keyframes")
        st.text_input("CLIP / SigLIP model", auto_paths["clip_model"], disabled=True)
        st.text_input("CLIP / SigLIP pretrained", auto_paths["clip_pretrained"], disabled=True)

st.markdown("</div>", unsafe_allow_html=True)

# Lấy các đường dẫn thực thi tự động theo model đang chọn
active_manifest = auto_paths["manifest"]
active_features = auto_paths["features"]
active_asr_features = auto_paths["asr_features"]
active_root = st.session_state.get("cfg_root") or auto_paths["keyframes"]
active_videos_root = st.session_state.get("cfg_videos_root") or auto_paths["videos"]
active_map_keyframes = st.session_state.get("cfg_map_keyframes") or auto_paths["map_keyframes"]
active_clip_model = auto_paths["clip_model"]
active_clip_pretrained = auto_paths["clip_pretrained"]

if st.session_state.get("_run_agent"):
    st.session_state["_run_agent"] = False
    if not text.strip() and not (query_type == "trake" and events_text.strip()):
        st.warning("Hãy nhập nội dung truy vấn trước khi chạy.")
    else:
        try:
            query = build_query(query_type, text, question, events_text)
            if query_type == "trake" and not query.events:
                st.error("TRAKE cần ít nhất một event.")
            else:
                runtime = {
                    "manifest_path": active_manifest,
                    "features_path": active_features,
                    "clip_model": active_clip_model,
                    "clip_pretrained": active_clip_pretrained,
                    "llm_model": st.session_state.get("cfg_llm", "qwen3.5:4b"),
                    "ollama_url": st.session_state.get("cfg_ollama", "http://127.0.0.1:11434"),
                    "metadata_filter": st.session_state.get("cfg_filter", ""),
                    "translate_query": bool(st.session_state.get("cfg_translate", False)),
                    "search_mode": str(st.session_state.get("search_mode", "hybrid") or "hybrid"),
                    "coarse_top_k": int(st.session_state.get("cfg_coarse_top_k", 200)),
                    "vlm_backend": "none",
                }
                backend_url = st.session_state.get("cfg_backend", "http://127.0.0.1:8000")

                with st.spinner(f"🔍 PEGASUS đang tìm kiếm frame với mô hình {model_choice}..."):
                    result = run_backend(query_type, query, runtime, backend_url)

                st.session_state.update(
                    query=query,
                    candidates=result.candidates,
                    trace=result.trace,
                    raw_root=active_root,
                    manifest_path=runtime["manifest_path"],
                    selected=[],
                    _query_nonce=st.session_state.get("_query_nonce", 0) + 1,
                )

                translated = None
                translation_source = None
                for step in result.trace:
                    if step.step == "translate":
                        try:
                            _td = json.loads(step.detail)
                            translation_source = _td.get("source")
                            _changes = _td.get("changes") or {}
                            _text_change = _changes.get("text") or {}
                            translated = _text_change.get("to") or _td.get("to")
                        except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
                            translated = None
                st.session_state["translated_query"] = translated
                if translated:
                    st.toast(f"✅ Đã dịch VI→EN: {translated}", icon="🌐")
                elif text.strip() and _is_english(text):
                    st.toast("ℹ️ Query đã là tiếng Anh — không cần dịch", icon="🌐")
        except (BackendRequestError, ValidationError) as error:
            st.error(f"Không chạy được agent: {error}")


# ===========================================================================
# GALLERY TRUNG TÂM
# ===========================================================================
if "candidates" not in st.session_state:
    st.info("💡 Nhập nội dung truy vấn ở trên hoặc bấm vào các nút **Đề mẫu 1-Click** để bắt đầu tìm kiếm.")
else:
    query: Query = st.session_state["query"]
    candidates: list[Candidate] = st.session_state["candidates"]
    root = Path(st.session_state.get("raw_root", active_root))
    if "selected" not in st.session_state or not isinstance(st.session_state["selected"], list):
        st.session_state["selected"] = []
    selected: list[int] = st.session_state["selected"]

    if not candidates:
        st.warning("Không tìm thấy candidate phù hợp. Hãy thử đổi query hoặc chế độ tìm kiếm.")
    else:
        total = len(candidates)
        candidates.sort(key=lambda c: c.score, reverse=True)

        col_filter, col_vid_quick, col_limit, col_all = st.columns([2, 1.2, 1, 1])
        with col_filter:
            if query.type == "TRAKE":
                video_options = ["Tất cả"] + sorted({c.video_id for c in candidates})
                selected_video = st.selectbox("Lọc theo video", video_options, index=0)
            else:
                selected_video = "Tất cả"
                video_options = sorted({c.video_id for c in candidates})
                opts = "".join([f'<option value="{html.escape(vid)}">{html.escape(vid)}</option>' for vid in video_options])
                filter_html = f"""
                <div>
                    <label style="font-weight: 800; font-size: 0.8rem; display: block; margin-bottom: 3px;">🎬 Lọc nhanh video</label>
                    <select id="client-video-filter" onchange="window.filterKeyframes(this.value)" style="
                        width: 100%; padding: 5px 8px; border-radius: 6px; border: 1.5px solid var(--input-border);
                        background: var(--input-bg); color: var(--input-text); font-size: 0.85rem; height: 36px; font-weight: 600;
                    ">
                        <option value="Tất cả">Tất cả ({len(video_options)} videos)</option>
                        {opts}
                    </select>
                </div>
                <script>
                window.filterKeyframes = function(selectedVideo) {{
                    const containers = document.querySelectorAll('.gallery-card-container');
                    containers.forEach(container => {{
                        const column = container.closest('[data-testid="column"]');
                        if (!column) return;
                        if (selectedVideo === 'Tất cả' || container.getAttribute('data-video-id') === selectedVideo) {{
                            column.style.display = 'block';
                        }} else {{
                            column.style.display = 'none';
                        }}
                    }});
                }};
                document.addEventListener('keydown', function(e) {{
                    if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
                    if (e.key === 'v' || e.key === 'V' || e.key === 'k' || e.key === 'K') {{
                        const btn = document.querySelector('button[id*="btn_quick_vid_top1"]');
                        if (btn) btn.click();
                    }}
                }});
                </script>
                """
                st.markdown(filter_html, unsafe_allow_html=True)
        with col_vid_quick:
            st.markdown('<label style="font-weight: 800; font-size: 0.8rem; display: block; margin-bottom: 3px;">⚡ Xem Video Top #1 (Phím: V/K)</label>', unsafe_allow_html=True)
            if candidates and st.button("🎬 Mở Video #1", key="btn_quick_vid_top1", **_FULL_WIDTH, help="Mở video & duyệt keyframes của Candidate #1"):
                top_cand = candidates[0]
                show_video_dialog(
                    video_id=top_cand.video_id,
                    initial_frame_id=top_cand.frame_id,
                    manifest_path=st.session_state.get("manifest_path", active_manifest),
                    keyframes_root=root,
                    videos_root=st.session_state.get("cfg_videos_root", active_videos_root),
                    map_dir=st.session_state.get("cfg_map_keyframes", active_map_keyframes),
                )
        with col_limit:
            max_show = st.number_input("Số ảnh hiển thị", min_value=1, max_value=total, value=min(20, total), step=5, key="max_show_input")
        with col_all:
            st.markdown('<label style="font-weight: 800; font-size: 0.8rem; display: block; margin-bottom: 3px;">&nbsp;</label>', unsafe_allow_html=True)
            if st.button("Chọn tất cả", **_FULL_WIDTH):
                vis_idx = [i for i, c in enumerate(candidates) if (selected_video == "Tất cả" or c.video_id == selected_video)][:max_show]
                for i in vis_idx:
                    if i not in selected:
                        selected.append(i)
                st.rerun()

        filtered = [(i, c) for i, c in enumerate(candidates) if (selected_video == "Tất cả" or c.video_id == selected_video)]
        sel_set = set(selected)
        chosen = [(i, c) for (i, c) in filtered if i in sel_set]
        rest = [(i, c) for (i, c) in filtered if i not in sel_set]
        order = {idx: pos for pos, idx in enumerate(selected)}
        chosen.sort(key=lambda ic: order.get(ic[0], len(selected)))
        visible = (chosen + rest)[:max_show]

        if query.type.lower() == "trake":
            _render_trake_gallery(query, visible, root, selected)
        else:
            _render_frame_gallery(query, visible, root, selected)

        # Thanh xuất CSV chuẩn nộp bài BTC
        st.divider()
        n_sel = len(selected)
        st.markdown(f"**📌 Đã chọn {n_sel} ảnh** *(thứ tự xuất bài = thứ tự bạn đã click chọn)*")
        if n_sel > 0:
            btc_csv = build_btc_csv(query, candidates, selected)
            col_dl, col_cl = st.columns([1, 2])
            with col_dl:
                st.download_button(
                    "📥 Tải CSV Chuẩn Nộp Bài",
                    btc_csv,
                    file_name=f"query-{query.query_id}-{query.type}.csv",
                    mime="text/csv",
                    **_FULL_WIDTH,
                )
            with col_cl:
                st.code(btc_csv, language="text")
        else:
            st.caption("Chưa chọn ảnh nào. Hãy tick chọn các ảnh kết quả đúng để xuất file nộp bài.")
