"""Floodify — Satellite Flood Intelligence. Streamlit frontend.

Serves the checkpoint produced by `train_pipeline.py`. Every value shared with
training (class order, input size, normalization, architecture, OOD features)
is imported from `flood_core`; Grad-CAM++ rendering is imported from
`grad_cam`, so the dashboard and the offline CLI produce identical evidence maps.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from flood_core import (
    ARCHITECTURE,
    CAM_TARGET_LAYER,
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    FLOOD_IDX,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    OOD_FEATURE_LAYER,
    CheckpointError,
    LoadedCheckpoint,
    OodDetector,
    load_checkpoint,
)
from grad_cam import explain_image, render_evidence

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

MODEL_PATH = DEFAULT_MODEL_PATH
ASSETS_DIR = Path("assets")
LOGO_PATH = ASSETS_DIR / "floodify_logo.svg"
ICON_PATH = ASSETS_DIR / "floodify_icon.svg"
REPO_URL = "https://github.com/meeasadamin/floodify"
IMAGES_DIR = Path("images")
SAMPLES_DIR = Path("samples")
METRICS_PATH = IMAGES_DIR / "final_metrics.json"
DATASET_SUMMARY_PATH = IMAGES_DIR / "dataset_summary.json"

UPLOAD_TYPES = [ext.lstrip(".") for ext in IMAGE_EXTENSIONS]
MAX_BATCH_TILES = 50
# Everything PIL raises for undecodable, truncated, unsupported-mode, or oversized input (audit #13).
IMAGE_DECODE_ERRORS = (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError)

FLOODED = CLASS_NAMES[FLOOD_IDX]
NOT_FLOODED = next(name for name in CLASS_NAMES if name != FLOODED)
CLASS_DISPLAY = {FLOODED: "Flooded", NOT_FLOODED: "Not flooded"}
assert set(CLASS_DISPLAY) == set(CLASS_NAMES), "CLASS_DISPLAY out of sync with CLASS_NAMES"

# Reference tiles: Sentinel-2 tiles from the held-out Pakistan event (never seen
# in training), plus one synthetic tile that demonstrates the OOD guard.
# Neutral button labels: naming the class would reveal the answer before the
# model does. Ground truth is shown after inference instead.
SAMPLE_TILES = [
    {"path": SAMPLES_DIR / "pakistan_flooded_1.png", "id": "TILE-01", "truth": FLOODED},
    {"path": SAMPLES_DIR / "pakistan_clear_1.png", "id": "TILE-02", "truth": NOT_FLOODED},
    {"path": SAMPLES_DIR / "pakistan_flooded_2.png", "id": "TILE-03", "truth": FLOODED},
    {"path": SAMPLES_DIR / "pakistan_clear_2.png", "id": "TILE-04", "truth": NOT_FLOODED},
    {"path": SAMPLES_DIR / "synthetic_ood_demo.jpg", "id": "SYNTH-05", "truth": None},
]

# Single source of truth for the KPI verdict, the risk bar, the triage queue, and
# the exported report (audit #9). Bands describe model confidence, not flood
# severity. (lower %, upper %, label, colour), ordered high to low.
RISK_BANDS = (
    (80, 100, "FLOOD DETECTED · HIGH CONFIDENCE", "#F85149"),
    (50, 80, "PROBABLE FLOOD · ANALYST REVIEW", "#D29922"),
    (20, 50, "INCONCLUSIVE · ANALYST REVIEW", "#8B98A5"),
    (0, 20, "NO FLOOD SIGNATURE", "#3FB950"),
)
OOD_LABEL = "UNFAMILIAR INPUT · RESULT UNRELIABLE"

# Brand mark: an orbit ring over three water waves (inline SVG, no external asset).
BRAND_MARK_SVG = (
    '<svg viewBox="0 0 48 48" fill="none" aria-hidden="true">'
    '<circle cx="24" cy="24" r="21" stroke="#4FB3D9" stroke-opacity=".35" stroke-width="2"/>'
    '<circle cx="38.8" cy="9.2" r="3" fill="#E6EDF3"/>'
    '<path d="M9 22c3 0 3-3 6-3s3 3 6 3 3-3 6-3 3 3 6 3 3-3 6-3" stroke="#4FB3D9" stroke-width="2.6" stroke-linecap="round"/>'
    '<path d="M9 29c3 0 3-3 6-3s3 3 6 3 3-3 6-3 3 3 6 3 3-3 6-3" stroke="#4FB3D9" stroke-opacity=".7" stroke-width="2.6" stroke-linecap="round"/>'
    '<path d="M12 36c3 0 3-3 6-3s3 3 6 3 3-3 6-3 3 3 6 3" stroke="#4FB3D9" stroke-opacity=".4" stroke-width="2.6" stroke-linecap="round"/>'
    "</svg>"
)

COLOR_TEXT = "#E6EDF3"
COLOR_MUTED = "#8B98A5"
COLOR_ACCENT = "#4FB3D9"
COLOR_SURFACE = "#11161D"
COLOR_WARN = "#E3B341"
COLOR_BAD = "#F85149"
COLOR_OK = "#3FB950"
PLOT_FONT = dict(family="IBM Plex Mono, Consolas, monospace", color=COLOR_MUTED)

st.set_page_config(
    page_title="Floodify · Satellite Flood Intelligence",
    page_icon=":material/satellite_alt:",
    layout="wide",
    initial_sidebar_state="auto",  # open on desktop, closed on phones so content is visible first
)


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

def inject_css() -> None:
    """Inject the dashboard stylesheet.

    Phase 1 styled `div[data-testid="stVerticalBlockBorderWrapper"]`, which
    Streamlit applies to every vertical block (columns, tabs, the main body),
    nesting cards inside cards (audit #14). All card styling is now scoped to
    this app's own class names; no Streamlit layout test-ids are restyled.

    Only the toolbar's right-hand items are hidden. `stToolbar` itself must stay
    visible: it contains `stExpandSidebarButton`, the only way to reopen a
    collapsed sidebar.
    """
    st.markdown(
        """<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{--bg:#0B0F14;--surface:#11161D;--surface2:#161C24;--border:#222B36;--text:#E6EDF3;--muted:#8B98A5;--accent:#4FB3D9;
--mono:'IBM Plex Mono',ui-monospace,Consolas,monospace;--sans:'IBM Plex Sans',system-ui,'Segoe UI',sans-serif}
.stApp,.stApp p,.stApp label,.stApp li,.stApp button,.stApp input,.stApp textarea{font-family:var(--sans)}
#MainMenu,footer,[data-testid="stMainMenu"],[data-testid="stToolbarActions"],[data-testid="stAppDeployButton"],[data-testid="stDecoration"]{display:none}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.2rem;max-width:1400px}
[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--border)}
[data-testid="stFileUploaderDropzone"]{background:var(--surface2);border:1px dashed var(--border)}
[data-testid="stImage"] img{border-radius:4px;border:1px solid var(--border)}
.poc-banner{background:#1F1705;color:#E3B341;border:1px solid #4A3A10;border-radius:4px;padding:6px 10px;margin-bottom:14px;
font:500 .7rem/1.3 var(--mono);letter-spacing:.14em;text-transform:uppercase;text-align:center}
.app-header{display:flex;flex-direction:column;align-items:center;text-align:center;gap:6px;
padding:10px 0 18px;margin-bottom:6px;border-bottom:1px solid var(--border);
background:radial-gradient(ellipse 60% 90% at 50% 0%,rgba(79,179,217,.10),transparent 70%)}
.app-brand{display:flex;align-items:center;justify-content:center;gap:14px}
.app-brand svg{width:clamp(34px,4vw,48px);height:auto;flex:none}
.app-wordmark{font:700 clamp(2.4rem,5vw,3.6rem)/1 var(--sans);letter-spacing:-.03em;
background:linear-gradient(100deg,#E6EDF3 20%,#4FB3D9 95%);-webkit-background-clip:text;background-clip:text;color:transparent}
.app-tagline{font:500 .8rem var(--mono);letter-spacing:.32em;text-transform:uppercase;color:var(--muted);padding-left:.32em}
.app-meta{font:.72rem var(--mono);color:var(--muted);margin-top:4px}
.app-meta b{font-weight:600}
/* Navigation: segmented control. Streamlit 1.63 tabs are react-aria (stTab / role=tablist), not baseweb. */
[data-testid="stTabs"] [role="tablist"]{display:flex;gap:6px;padding:6px;margin:6px 0 4px;background:var(--surface);
border:1px solid var(--border);border-radius:14px;overflow-x:auto}
[data-testid="stTab"]{flex:1 1 0;min-width:max-content;justify-content:center;padding:11px 18px;border-radius:10px;
color:var(--muted);transition:background .15s ease,color .15s ease,box-shadow .15s ease}
[data-testid="stTab"] p{font:600 .92rem/1 var(--sans);letter-spacing:.01em;white-space:nowrap}
[data-testid="stTab"]:hover{background:var(--surface2);color:var(--text)}
[data-testid="stTab"][aria-selected="true"]{color:var(--text);
background:linear-gradient(180deg,rgba(79,179,217,.20),rgba(79,179,217,.06));box-shadow:inset 0 0 0 1px rgba(79,179,217,.55)}
[data-testid="stTab"][aria-selected="true"] [role="img"]{color:var(--accent)}
[data-testid="stTabs"] .react-aria-SelectionIndicator{display:none}
/* Page headings inside each tab */
.page-head{display:flex;align-items:flex-start;gap:14px;margin:18px 0 6px}
.page-head .bar{width:4px;align-self:stretch;border-radius:4px;background:linear-gradient(180deg,var(--accent),rgba(79,179,217,.15))}
.page-title{font:700 1.35rem/1.2 var(--sans);color:var(--text);letter-spacing:-.01em}
.page-sub{font-size:.9rem;color:var(--muted);margin-top:4px;line-height:1.5}
.sub-head{font:600 .72rem var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--accent);
margin:22px 0 10px;display:flex;align-items:center;gap:10px}
.sub-head::after{content:"";flex:1;height:1px;background:var(--border)}
/* Sidebar */
[data-testid="stSidebarHeader"]{padding-bottom:4px}
[data-testid="stSidebarHeader"] img{height:2.1rem !important;max-width:none}
[data-testid="stSidebarUserContent"]{padding-top:0}
.sb-status{display:flex;flex-direction:column;gap:3px;padding:10px 12px;margin:2px 0 6px;border-radius:10px;
background:var(--surface2);border:1px solid var(--border)}
.sb-status .pill{display:inline-flex;align-items:center;gap:8px;font:600 .74rem var(--mono);letter-spacing:.08em;text-transform:uppercase}
.sb-status .dot{width:8px;height:8px;border-radius:50%;background:currentColor;box-shadow:0 0 0 3px color-mix(in srgb,currentColor 25%,transparent)}
.sb-status .meta{font:.72rem var(--mono);color:var(--muted)}
[class*="st-key-sb_card"]{background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:14px 14px 16px}
.sb-head{display:flex;gap:10px;align-items:flex-start;margin-bottom:2px}
.sb-num{flex:none;width:24px;height:24px;border-radius:7px;display:grid;place-items:center;font:600 .72rem var(--mono);
color:var(--accent);background:rgba(79,179,217,.12);border:1px solid rgba(79,179,217,.35)}
.sb-title{font:600 .95rem/1.2 var(--sans);color:var(--text)}
.sb-sub{font-size:.78rem;line-height:1.45;color:var(--muted);margin-top:3px}
[data-testid="stSidebar"] [data-testid="stImage"] img{border-radius:8px}
[data-testid="stSidebar"] button p{font:600 .78rem var(--mono);letter-spacing:.04em}
.sb-foot{font:.72rem/1.6 var(--mono);color:var(--muted);text-align:center;padding:6px 0 10px}
.sb-foot a{color:var(--accent);text-decoration:none}
/* Keep the sidebar's 2-up tile grid on narrow screens (Streamlit stacks columns below 640px). */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]{flex-wrap:nowrap;gap:.6rem}
[data-testid="stSidebar"] [data-testid="stColumn"]{min-width:0 !important;flex:1 1 0 !important;width:auto !important}
/* Phones */
@media (max-width:640px){
.stApp .block-container{padding-left:1rem;padding-right:1rem;padding-top:3.6rem}
.poc-banner{font-size:.6rem;letter-spacing:.08em;padding:6px 8px}
.app-header{padding:4px 0 12px}
.app-tagline{font-size:.62rem;letter-spacing:.22em}
.app-meta{font-size:.62rem;line-height:1.6}
[data-testid="stTabs"] [role="tablist"]{display:grid;grid-template-columns:1fr 1fr;overflow:visible}
[data-testid="stTab"]{min-width:0;padding:10px 8px}
[data-testid="stTab"] p{font-size:.8rem;white-space:normal;text-align:center;line-height:1.2}
.stApp .kpi-row,.stApp .kpi-row.kpi-4{grid-template-columns:1fr}
.page-title{font-size:1.15rem}
}
.kpi-row{display:grid;grid-template-columns:2.2fr repeat(4,1fr);gap:1px;background:var(--border);
border:1px solid var(--border);border-radius:6px;overflow:hidden;margin:16px 0 10px}
.kpi-row.kpi-4{grid-template-columns:repeat(4,1fr)}
@media (max-width:1000px){.kpi-row,.kpi-row.kpi-4{grid-template-columns:1fr 1fr}}
.kpi{background:var(--surface);padding:14px 16px;min-width:0}
.kpi-label,.panel-label{font:500 .68rem var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
.kpi-value{font:600 1.45rem var(--mono);color:var(--text);font-variant-numeric:tabular-nums;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kpi-verdict{border-left:4px solid var(--band)}
.kpi-verdict .kpi-value{color:var(--band);font:600 1rem var(--sans);letter-spacing:.05em;padding-top:.3rem;white-space:normal}
.kpi-small{font-size:1rem;padding-top:.35rem}
.kpi-sub{font:.7rem var(--mono);color:var(--muted);margin-top:4px}
.kpi-muted{color:var(--muted);text-decoration:line-through}
.truth-line{font:.76rem var(--mono);color:var(--muted);margin:0 0 6px}
.ood-note{background:#1F1705;border:1px solid #4A3A10;color:#E3B341;border-radius:4px;padding:8px 12px;font-size:.85rem;margin:0 0 8px}
.spec-heading{color:var(--text);font-size:1rem;font-weight:600;margin:1.4rem 0 .5rem}
.spec-body{color:#C4CDD5;font-size:.94rem;line-height:1.6}
.elevated-card{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:16px 18px;margin-bottom:1rem}
.elevated-card ul{margin:0;padding-left:1.1rem}
.tech-badge{display:inline-block;border:1px solid var(--border);color:var(--muted);border-radius:3px;
padding:3px 9px;margin:3px 6px 3px 0;font:.76rem var(--mono)}
.prov-table{width:100%;border-collapse:collapse;font:.8rem var(--mono)}
.prov-table td{padding:6px 4px;border-bottom:1px solid var(--border);color:#C4CDD5;word-break:break-all}
.prov-table td:first-child{color:var(--muted);width:34%;word-break:normal}
</style>""",
        unsafe_allow_html=True,
    )


def image_to_base64(img: Image.Image | np.ndarray) -> str:
    """Encode a PIL image or uint8 array as a base64 PNG data URI.

    Used for triage-queue thumbnails, which `st.column_config.ImageColumn`
    accepts as data URIs.

    Args:
        img: A PIL Image, or a [H,W,3] uint8 numpy array.

    Returns:
        A data URI string.
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def risk_band(p_pct: float) -> tuple[int, int, str, str]:
    """Return the RISK_BANDS row containing a flood probability percentage."""
    return next(band for band in RISK_BANDS if p_pct >= band[0])


def assessment(r: dict) -> tuple[str, str]:
    """Final (label, colour) for an inference result: the OOD guard overrides the risk band."""
    if r["ood_flag"]:
        return OOD_LABEL, COLOR_WARN
    _, _, label, color = risk_band(r["flood_prob"] * 100)
    return label, color


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to an rgba() string for Plotly."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _ci(interval: list[float] | None) -> str:
    return "" if not interval else f"95% CI {interval[0] * 100:.0f}–{interval[1] * 100:.0f}%"


# --------------------------------------------------------------------------
# Model loading and inference
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model weights…")
def load_model() -> LoadedCheckpoint:
    """Load and verify the checkpoint once per server process.

    Failures raise instead of returning None (audit #8): Streamlit does not
    cache exceptions, so a replaced or repaired checkpoint is picked up on the
    next rerun without restarting the server, and the caller can show the
    actual reason instead of a generic "not found".

    Returns:
        LoadedCheckpoint from flood_core.
    """
    return load_checkpoint(MODEL_PATH)


@st.cache_data(max_entries=128, show_spinner=False)
def run_inference(_model: torch.nn.Module, _ood: OodDetector | None, model_sha: str, image_bytes: bytes) -> dict:
    """Run inference, Grad-CAM++, and the OOD check once per (checkpoint, image) pair (audit #11).

    `_model` and `_ood` are excluded from Streamlit's cache key; `model_sha`
    (which covers both, since the detector lives in the checkpoint) is included
    so results are invalidated when the checkpoint changes.

    Args:
        _model: Eval-mode model.
        _ood: OOD detector from the checkpoint, or None.
        model_sha: SHA-256 of the loaded checkpoint (cache key component).
        image_bytes: Raw uploaded or reference file bytes.

    Returns:
        Dict with probs, flood_prob, pred_class, latency_ms, sha256, native_size,
        source_rgb, overlay_rgb, ood_distance, ood_threshold, ood_flag.

    Raises:
        Any of IMAGE_DECODE_ERRORS for undecodable input (not cached).
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        img.load()
        native_size = img.size
        explanation = explain_image(_model, img)
        source_rgb, overlay_rgb = render_evidence(img, explanation)

    ood_distance = float(_ood.distance(explanation.features)[0]) if _ood is not None else None
    ood_threshold = _ood.threshold if _ood is not None else None
    return {
        "probs": explanation.probs,
        "flood_prob": explanation.flood_prob,
        "pred_class": explanation.pred_class,
        "latency_ms": explanation.latency_ms,
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "native_size": native_size,
        "source_rgb": source_rgb,
        "overlay_rgb": overlay_rgb,
        "ood_distance": ood_distance,
        "ood_threshold": ood_threshold,
        "ood_flag": bool(ood_distance is not None and ood_distance > ood_threshold),
    }


def infer(loaded: LoadedCheckpoint, image_bytes: bytes) -> dict:
    return run_inference(loaded.model, loaded.ood, loaded.sha256, image_bytes)


# --------------------------------------------------------------------------
# Header, input selection
# --------------------------------------------------------------------------

def render_header(loaded: LoadedCheckpoint | None) -> None:
    """Render the prototype banner, title, and live model status line."""
    if loaded is None:
        status = f"<b style='color:{COLOR_BAD}'>OFFLINE</b>"
    elif loaded.format == "legacy":
        status = f"<b style='color:{COLOR_WARN}'>LEGACY CHECKPOINT · UNVERIFIED</b> · weights {loaded.sha256[:10]}"
    elif not loaded.integrity_verified:
        status = f"<b style='color:{COLOR_WARN}'>ONLINE · NO SHA-256 SIDECAR</b> · weights {loaded.sha256[:10]}"
    else:
        status = f"<b style='color:{COLOR_OK}'>ONLINE · VERIFIED</b> · weights {loaded.sha256[:10]}"
    guard = ""
    if loaded is not None:
        guard = " · OOD guard " + ("on" if loaded.ood is not None else f"<b style='color:{COLOR_WARN}'>off</b>")
    st.markdown(
        '<div class="poc-banner">Research prototype · Sentinel-2 true colour (Sen1Floods11) · '
        "not for operational use</div>"
        '<div class="app-header">'
        f'<div class="app-brand">{BRAND_MARK_SVG}<span class="app-wordmark">Floodify</span></div>'
        '<div class="app-tagline">Satellite Flood Intelligence</div>'
        f'<div class="app-meta">EfficientNet-B0 · Grad-CAM++ · CPU &nbsp;|&nbsp; {status}{guard}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _sidebar_head(number: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="sb-head"><span class="sb-num">{number}</span><div>'
        f'<div class="sb-title">{title}</div><div class="sb-sub">{subtitle}</div></div></div>',
        unsafe_allow_html=True,
    )


def _sidebar_status(loaded: LoadedCheckpoint | None) -> None:
    if loaded is None:
        pill, color, meta = "Model offline", COLOR_BAD, "See the error in the main panel"
    elif loaded.format == "legacy" or not loaded.integrity_verified:
        pill, color, meta = "Model online · unverified", COLOR_WARN, f"weights {loaded.sha256[:10]}"
    else:
        guard = "OOD guard on" if loaded.ood is not None else "OOD guard off"
        pill, color, meta = "Model online · verified", COLOR_OK, f"EfficientNet-B0 · {guard}"
    st.markdown(
        f'<div class="sb-status"><span class="pill" style="color:{color}"><span class="dot"></span>{pill}</span>'
        f'<span class="meta">{meta}</span></div>',
        unsafe_allow_html=True,
    )


def select_input(loaded: LoadedCheckpoint | None) -> dict | None:
    """Render the sidebar (logo, status, input cards) and return the active image.

    Phase 1 assigned the uploader's value on every rerun, so a persistent upload
    overwrote a reference-tile selection as soon as any other widget was used
    (audit #10). The upload is now adopted only when its file_id changes.

    Args:
        loaded: Loaded checkpoint, or None (controls are disabled).

    Returns:
        Dict with name, bytes, source ("upload" | "reference"), truth (class name
        or None), or None if nothing is selected.
    """
    enabled = loaded is not None
    if enabled and "active" not in st.session_state:
        # First visit: show a real result immediately instead of an empty page.
        first = next(t for t in SAMPLE_TILES if t["truth"] is not None and t["path"].exists())
        st.session_state.active = {
            "name": f"{first['id']} ({first['path'].name})", "bytes": first["path"].read_bytes(),
            "source": "reference", "truth": first["truth"],
        }
    st.logo(str(LOGO_PATH), size="large", icon_image=str(ICON_PATH), link=REPO_URL)
    with st.sidebar:
        _sidebar_status(loaded)

        with st.container(key="sb_card_upload"):
            _sidebar_head("01", "Analyze your tile", "Sentinel-2 true-colour PNG or JPG, ideally 2–3 km across.")
            uploaded = st.file_uploader(
                "Upload satellite tile", type=UPLOAD_TYPES, label_visibility="collapsed",
                key="single_upload", disabled=not enabled,
            )
            if uploaded is not None and uploaded.file_id != st.session_state.get("adopted_upload_id"):
                st.session_state.adopted_upload_id = uploaded.file_id
                st.session_state.active = {
                    "name": uploaded.name, "bytes": uploaded.getvalue(), "source": "upload", "truth": None,
                }

        with st.container(key="sb_card_reference"):
            _sidebar_head("02", "Pakistan reference set",
                          "Real tiles from a flood event held out of training. Hand labels appear after analysis.")
            cols = st.columns(2)
            for i, tile in enumerate(t for t in SAMPLE_TILES if t["truth"] is not None):
                with cols[i % 2]:
                    _reference_tile(tile, enabled)

        with st.container(key="sb_card_guard"):
            _sidebar_head("03", "Guard test",
                          "Not satellite imagery. Floodify should flag it as unfamiliar instead of trusting its score.")
            for tile in (t for t in SAMPLE_TILES if t["truth"] is None):
                thumb_col, _ = st.columns([1, 1])
                with thumb_col:
                    _reference_tile(tile, enabled)

        st.markdown(
            f'<div class="sb-foot">Research prototype · not for operational use<br>'
            f'<a href="{REPO_URL}" target="_blank">GitHub</a> · '
            f'<a href="https://github.com/cloudtostreet/Sen1Floods11" target="_blank">Sen1Floods11</a></div>',
            unsafe_allow_html=True,
        )
    return st.session_state.get("active")


def _reference_tile(tile: dict, enabled: bool) -> None:
    """Thumbnail + select button; the active tile's button is highlighted."""
    if not tile["path"].exists():
        st.caption(f"Missing: {tile['path'].name}")
        return
    active = st.session_state.get("active") or {}
    is_active = active.get("source") == "reference" and active.get("name", "").startswith(tile["id"])
    st.image(str(tile["path"]), width="stretch")
    if st.button(tile["id"], key=f"tile_{tile['id']}", width="stretch", disabled=not enabled,
                 type="primary" if is_active else "secondary"):
        st.session_state.active = {
            "name": f"{tile['id']} ({tile['path'].name})",
            "bytes": tile["path"].read_bytes(),
            "source": "reference",
            "truth": tile["truth"],
        }
        st.rerun()  # re-render so the highlight moves to the tile just selected


# --------------------------------------------------------------------------
# Threat analysis components
# --------------------------------------------------------------------------

def render_kpi_strip(r: dict) -> None:
    """Verdict-first KPI row: assessment, P(flood), input familiarity, latency, input hash (audit #15)."""
    p_pct = r["flood_prob"] * 100
    label, color = assessment(r)
    p_class = "kpi-value kpi-muted" if r["ood_flag"] else "kpi-value"
    if r["ood_distance"] is None:
        familiarity, fam_sub, fam_color = "n/a", "no OOD detector in checkpoint", COLOR_MUTED
    else:
        ratio = r["ood_distance"] / r["ood_threshold"]
        familiarity = f"{ratio:.2f}×"
        fam_sub = "of limit · unfamiliar" if r["ood_flag"] else "of limit · in distribution"
        fam_color = COLOR_WARN if r["ood_flag"] else COLOR_OK
    # Single-line HTML on purpose: indented lines inside st.markdown render as a code block.
    st.markdown(
        '<div class="kpi-row">'
        f'<div class="kpi kpi-verdict" style="--band:{color}"><div class="kpi-label">Assessment</div>'
        f'<div class="kpi-value">{label}</div></div>'
        f'<div class="kpi"><div class="kpi-label">P({FLOODED})</div><div class="{p_class}">{p_pct:.1f}%</div></div>'
        '<div class="kpi"><div class="kpi-label">Input familiarity</div>'
        f'<div class="kpi-value" style="color:{fam_color}">{familiarity}</div><div class="kpi-sub">{fam_sub}</div></div>'
        '<div class="kpi"><div class="kpi-label">Latency · CPU · measured</div>'
        f'<div class="kpi-value">{r["latency_ms"]:.0f} ms</div></div>'
        '<div class="kpi"><div class="kpi-label">Input SHA-256</div>'
        f'<div class="kpi-value kpi-small">{r["sha256"][:12]}</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    if r["ood_flag"]:
        st.markdown(
            '<div class="ood-note">This input is far from the imagery the model was trained on '
            "(Sentinel-2 true-colour tiles). The flood probability is shown for reference only and "
            "must not be used for triage.</div>",
            unsafe_allow_html=True,
        )


def render_risk_bar(p_pct: float) -> go.Figure:
    """Banded bullet bar whose bands are generated from RISK_BANDS (audit #9).

    Replaces the Phase 1 gauge, whose 33/66 sectors contradicted the badge's
    50/80 thresholds.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=p_pct,
            gauge={
                "shape": "bullet",
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickfont": {"color": COLOR_MUTED, "size": 10}},
                "bar": {"color": COLOR_TEXT, "thickness": 0.3},
                "steps": [{"range": [lo, hi], "color": _rgba(color, 0.35)} for lo, hi, _, color in RISK_BANDS],
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
            },
        )
    )
    fig.update_layout(height=64, margin=dict(l=8, r=8, t=6, b=22), paper_bgcolor="rgba(0,0,0,0)", font=PLOT_FONT)
    return fig


def render_analyst_review(r: dict, name: str) -> None:
    """Human-in-the-loop confirm / override log, keyed by input SHA-256."""
    log = st.session_state.setdefault("review_log", {})
    key = r["sha256"]
    sub_head("Analyst sign-off")
    c1, c2, c3 = st.columns([1, 1, 3])
    note = c3.text_input(
        "Rationale", key=f"note_{key}", placeholder="Rationale (optional)", label_visibility="collapsed"
    )
    for verdict, col, icon in (("CONFIRMED", c1, ":material/check:"), ("OVERRIDDEN", c2, ":material/flag:")):
        if col.button(verdict.title(), key=f"{verdict}_{key}", icon=icon, width="stretch"):
            log[key] = {
                "utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
                "file": name,
                "input_sha256": key[:12],
                "model_assessment": assessment(r)[0],
                f"p_{FLOODED}": f"{r['flood_prob']:.1%}",
                "analyst": verdict,
                "note": note,
            }
    if log:
        st.dataframe(list(log.values()), hide_index=True, width="stretch")


def build_assessment_report(r: dict, name: str, loaded: LoadedCheckpoint) -> dict:
    """Assemble the auditable JSON record for one assessment."""
    lo, hi, band_label, _ = risk_band(r["flood_prob"] * 100)
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {"filename": name, "sha256": r["sha256"], "native_resolution": list(r["native_size"])},
        "model": {
            "architecture": ARCHITECTURE,
            "weights_file": loaded.path.as_posix(),
            "weights_sha256": loaded.sha256,
            "checkpoint_format": loaded.format,
            "integrity_verified": loaded.integrity_verified,
            "schema_verified": loaded.schema_verified,
            "training_data": loaded.metadata.get("data_dir", "unknown"),
            "test_set": loaded.metadata.get("test_set"),
        },
        "assessment": {
            "final": assessment(r)[0],
            "class_names": list(CLASS_NAMES),
            "probabilities": [round(p, 6) for p in r["probs"]],
            "predicted_class": CLASS_NAMES[r["pred_class"]],
            f"p_{FLOODED}": round(r["flood_prob"], 6),
            "band": band_label,
            "band_range_pct": [lo, hi],
        },
        "input_check": {
            "method": "kNN cosine distance on penultimate features" if r["ood_distance"] is not None else None,
            "distance": r["ood_distance"],
            "threshold": r["ood_threshold"],
            "out_of_distribution": r["ood_flag"],
        },
        "explainability": {
            "method": "Grad-CAM++",
            "target_layer": CAM_TARGET_LAYER,
            "target_class": FLOODED,
            "overlay_opacity_scaled_by": f"p_{FLOODED}",
        },
        "analyst_review": st.session_state.get("review_log", {}).get(r["sha256"]),
        "runtime": {"latency_ms": round(r["latency_ms"], 1), "device": "cpu", "torch": str(torch.__version__)},
        "disclaimer": "Decision-support output only. Requires analyst verification.",
    }


def render_evidence_download(r: dict, name: str, loaded: LoadedCheckpoint) -> None:
    """Export a chain-of-custody ZIP: assessment.json + source.png + evidence_map.png."""
    report = build_assessment_report(r, name, loaded)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("assessment.json", json.dumps(report, indent=2))
        for fname, arr in (("source.png", r["source_rgb"]), ("evidence_map.png", r["overlay_rgb"])):
            png = io.BytesIO()
            Image.fromarray(arr).save(png, format="PNG")
            archive.writestr(fname, png.getvalue())
    st.download_button(
        "Export evidence package",
        data=buf.getvalue(),
        file_name=f"assessment_{r['sha256'][:8]}.zip",
        mime="application/zip",
        icon=":material/download:",
        width="stretch",
        on_click="ignore",  # exporting must not rerun the app or reset review state
    )


def page_head(title: str, subtitle: str) -> None:
    """Title block at the top of each tab."""
    st.markdown(
        f'<div class="page-head"><span class="bar"></span><div><div class="page-title">{title}</div>'
        f'<div class="page-sub">{subtitle}</div></div></div>',
        unsafe_allow_html=True,
    )


def sub_head(text: str) -> None:
    """Section divider inside a tab."""
    st.markdown(f'<div class="sub-head">{text}</div>', unsafe_allow_html=True)


def render_threat_analysis_tab(loaded: LoadedCheckpoint, active: dict | None) -> None:
    """Render Tab 1: verdict first, then evidence, then review and export.

    Args:
        loaded: Verified checkpoint.
        active: Output of `select_input`.
    """
    page_head("Flood Assessment", "Analyze a single satellite tile: verdict, evidence map, and analyst sign-off.")
    if active is None:
        st.info("Select a reference tile or upload imagery from the sidebar.", icon=":material/satellite_alt:")
        return

    try:
        r = infer(loaded, active["bytes"])
    except IMAGE_DECODE_ERRORS as exc:
        st.error(
            f"`{active['name']}` could not be decoded as an image ({type(exc).__name__}).",
            icon=":material/broken_image:",
        )
        return

    sub_head("Verdict")
    render_kpi_strip(r)
    if active["truth"] is not None:
        predicted = CLASS_NAMES[r["pred_class"]]
        agreement = "model agrees" if predicted == active["truth"] else "MODEL DISAGREES"
        st.markdown(
            f'<div class="truth-line">Hand-labeled ground truth: {CLASS_DISPLAY[active["truth"]]} · '
            f"model: {CLASS_DISPLAY[predicted]} · {agreement}</div>",
            unsafe_allow_html=True,
        )
    st.plotly_chart(render_risk_bar(r["flood_prob"] * 100), width="stretch", config={"displayModeBar": False})

    sub_head("Evidence")
    width, height = r["native_size"]
    left, right = st.columns(2, gap="medium")
    left.markdown(
        f'<div class="panel-label">Source · {active["name"]} · {width}×{height}</div>', unsafe_allow_html=True
    )
    left.image(r["source_rgb"], width="stretch")
    right.markdown(
        f'<div class="panel-label">Flood-evidence map · Grad-CAM++ on P({FLOODED})</div>', unsafe_allow_html=True
    )
    right.image(r["overlay_rgb"], width="stretch")

    render_analyst_review(r, active["name"])
    render_evidence_download(r, active["name"], loaded)


def render_batch_triage_tab(loaded: LoadedCheckpoint) -> None:
    """Rank many tiles by flood probability so analysts review the riskiest first."""
    page_head("Priority Triage", f"Drop up to {MAX_BATCH_TILES} tiles; Floodify ranks them so the likeliest floods are "
              "reviewed first.")
    files = st.file_uploader(
        f"Drop up to {MAX_BATCH_TILES} tiles for ranked triage",
        type=UPLOAD_TYPES, accept_multiple_files=True, key="batch_upload", label_visibility="collapsed",
    )
    if not files:
        st.caption(
            "Tiles are scored and ranked by flood probability so analysts review the highest-risk areas first. "
            "Unfamiliar inputs are flagged and should not be triaged on the model score."
        )
        return
    if len(files) > MAX_BATCH_TILES:
        st.warning(f"Only the first {MAX_BATCH_TILES} of {len(files)} tiles are triaged.", icon=":material/warning:")
    files = files[:MAX_BATCH_TILES]

    rows, skipped = [], []
    progress = st.progress(0.0, text="Triaging…")
    for i, f in enumerate(files, 1):
        try:
            r = infer(loaded, f.getvalue())
        except IMAGE_DECODE_ERRORS:
            skipped.append(f.name)
        else:
            thumb = Image.fromarray(r["source_rgb"])
            thumb.thumbnail((64, 64))
            rows.append({
                "Tile": image_to_base64(thumb),
                "File": f.name,
                f"P({FLOODED})": r["flood_prob"] * 100,
                "Assessment": assessment(r)[0],
                "Input check": "n/a" if r["ood_distance"] is None else ("UNFAMILIAR" if r["ood_flag"] else "OK"),
                "Latency (ms)": round(r["latency_ms"]),
                "SHA-256": r["sha256"][:12],
            })
        progress.progress(i / len(files), text=f"Triaged {i}/{len(files)}")
    progress.empty()

    if skipped:
        st.warning(f"Skipped undecodable files: {', '.join(skipped)}", icon=":material/broken_image:")
    rows.sort(key=lambda row: row[f"P({FLOODED})"], reverse=True)
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "Tile": st.column_config.ImageColumn(width="small"),
            f"P({FLOODED})": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
        },
    )


# --------------------------------------------------------------------------
# Diagnostics and overview
# --------------------------------------------------------------------------

def render_confusion(cm: list[list[int]]) -> go.Figure:
    """Dark Plotly confusion matrix (rows = actual, columns = predicted, CLASS_NAMES order)."""
    fig = go.Figure(
        go.Heatmap(
            z=cm, x=list(CLASS_NAMES), y=list(CLASS_NAMES), text=cm, texttemplate="%{text}",
            textfont=dict(color=COLOR_TEXT, size=18, family=PLOT_FONT["family"]),
            # Darkest-to-mid accent so the light count text stays readable on every cell.
            colorscale=[[0, COLOR_SURFACE], [1, "#1F6F8B"]], showscale=False, xgap=2, ygap=2,
            hovertemplate="actual %{y}<br>predicted %{x}<br>n=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=PLOT_FONT, xaxis_title="Predicted",
        yaxis=dict(title="Actual", autorange="reversed"),
    )
    return fig


def load_json(path: Path) -> tuple[dict | None, str | None]:
    """Read a JSON artifact.

    Returns:
        (data, problem): data is None when missing or unreadable, with `problem`
        describing why.
    """
    if not path.exists():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable ({type(exc).__name__}: {exc})"


def _kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value kpi-small">{value}</div>{sub_html}</div>'


def render_model_diagnostics_tab() -> None:
    """Render Tab 3: test-set metrics with intervals, OOD calibration, curves (audit #16 consumer)."""
    page_head("Model Performance", "How Floodify performs on a flood event it never saw, and how the input guard is "
              "calibrated.")
    metrics, problem = load_json(METRICS_PATH)

    if metrics is None:
        st.info(
            f"`{METRICS_PATH.as_posix()}` is {problem}. It is produced by `python train_pipeline.py` together "
            "with the checkpoint.",
            icon=":material/monitoring:",
        )
    elif metrics.get("class_names", list(CLASS_NAMES)) != list(CLASS_NAMES):
        st.error(
            f"Metrics class order {metrics.get('class_names')} does not match serving order {CLASS_NAMES}.",
            icon=":material/error:",
        )
        metrics = None
    elif metrics.get("schema_version") not in (2, 3):
        st.warning(
            "Legacy metrics file: checkpoint selection and reporting used the same validation split, "
            "so these numbers are optimistic. Retrain with `train_pipeline.py`.",
            icon=":material/warning:",
        )

    if metrics is not None:
        version = metrics.get("schema_version")
        is_current = version in (2, 3)
        test = metrics["test"] if is_current else {"classification_report": metrics.get("classification_report", {})}
        report = test["classification_report"]
        test_name = test.get("name", "Held-out test set") if is_current else "Validation set (legacy)"

        if is_current:
            data, sel, ci = metrics["data"], metrics["selection"], test.get("ci95", {})
            counts = data.get("class_counts", {}).get("test")
            count_note = " · ".join(f"{CLASS_DISPLAY[k]} {v}" for k, v in counts.items()) if counts else ""
            flood_recall = report.get(FLOODED, {}).get("recall")
            st.markdown(f'<div class="spec-heading">{test_name}</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="kpi-row kpi-4">'
                + _kpi("Balanced accuracy", _pct(test.get("balanced_accuracy", test["accuracy"])),
                       _ci(ci.get("balanced_accuracy")))
                + _kpi(f"Recall · {CLASS_DISPLAY[FLOODED]}", _pct(flood_recall), _ci(ci.get("flooded_recall")))
                + _kpi("ROC-AUC", "n/a" if test.get("roc_auc") is None else f"{test['roc_auc']:.3f}",
                       _ci(ci.get("roc_auc")))
                + _kpi("Test tiles", str(test.get("n_samples", data.get("n_test"))), count_note)
                + "</div>",
                unsafe_allow_html=True,
            )
            n_flooded = (counts or {}).get(FLOODED)
            if n_flooded is not None and n_flooded < 30:
                st.caption(
                    f"Only {n_flooded} flooded tiles in this test set: treat point estimates with caution and read "
                    "the confidence intervals."
                )

        cm_col, table_col = st.columns([1, 1.3], gap="large")
        with cm_col:
            st.markdown('<div class="spec-heading">Confusion matrix</div>', unsafe_allow_html=True)
            if is_current:
                st.plotly_chart(render_confusion(test["confusion_matrix"]), width="stretch",
                                config={"displayModeBar": False})
            elif (IMAGES_DIR / "confusion_matrix.png").exists():
                st.image(str(IMAGES_DIR / "confusion_matrix.png"), width="stretch")
        with table_col:
            st.markdown('<div class="spec-heading">Per-class metrics</div>', unsafe_allow_html=True)
            rows = [
                {
                    "Class": name,
                    "Precision": round(report[name]["precision"], 3),
                    "Recall": round(report[name]["recall"], 3),
                    "F1-Score": round(report[name]["f1-score"], 3),
                    "Support": int(report[name]["support"]),
                }
                for name in CLASS_NAMES
                if name in report
            ]
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
            if is_current:
                st.caption(
                    f"Selected checkpoint: stage {sel['stage']}, epoch {sel['epoch']} ({sel['rule']}). "
                    f"Train/val/test tiles: {data['n_train']}/{data['n_val']}/{data['n_test']}, "
                    f"grouped by {data.get('grouping', 'file')}. The test set was evaluated once, after selection."
                )

        ood = metrics.get("ood") if version == 3 else None
        if ood:
            probe = ood.get("probe") or {}
            st.markdown('<div class="spec-heading">Input familiarity guard</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="kpi-row kpi-4">'
                + _kpi("Method", "kNN cosine", ood["method"])
                + _kpi("Threshold", f"{ood['threshold']:.3f}", ood["calibration"])
                + _kpi("Test tiles flagged", _pct(ood["test_flagged_frac"]), "real imagery, unseen event")
                + _kpi("Synthetic probe flagged", _pct(probe.get("flagged_frac")),
                       f"n={probe['n']} procedural tiles" if probe.get("n") else "no probe run")
                + "</div>",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="spec-heading">Training convergence</div>', unsafe_allow_html=True)
    curves_path = IMAGES_DIR / "training_curves.png"
    if curves_path.exists():
        st.image(str(curves_path), width="stretch")
    else:
        st.info("Training curves not found. Run `python train_pipeline.py`.", icon=":material/monitoring:")


def _card(heading: str, body_html: str) -> None:
    st.markdown(f'<div class="spec-heading">{heading}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="spec-body elevated-card">{body_html}</div>', unsafe_allow_html=True)


def render_system_overview_tab(loaded: LoadedCheckpoint | None) -> None:
    """Render Tab 4: data, architecture, explainability, limitations, provenance, stack."""
    page_head("System & Provenance", "Training data, method, limitations, and a verifiable record of the model in use.")
    summary, _ = load_json(DATASET_SUMMARY_PATH)
    if summary:
        counts = summary["counts"]
        rule = summary["rule"]
        _card(
            "Training data",
            f"Sen1Floods11 hand-labeled Sentinel-2 chips ({summary['n_chips']} chips, 11 flood events), rendered as "
            f"true colour and cut into {summary['tile_size']}&times;{summary['tile_size']} px tiles (10 m/px). "
            f"A tile is <b>flooded</b> when &ge;{rule['flood_min']:.0%} of valid pixels are flood water "
            f"(hand-labeled water that is not JRC permanent water), <b>not flooded</b> when &le;{rule['clear_max']:.0%}; "
            f"tiles in between or with &lt;{rule['valid_min']:.0%} valid pixels are discarded. "
            f"The <b>{summary['holdout_event']}</b> event is held out entirely for testing: "
            f"{counts.get('test_pakistan/flooded', 0)} flooded and {counts.get('test_pakistan/not_flooded', 0)} "
            f"not-flooded tiles. Training pool: {counts.get('train/flooded', 0)} flooded, "
            f"{counts.get('train/not_flooded', 0)} not flooded, split into train/validation by source chip.",
        )
    _card(
        "Model architecture",
        "EfficientNet-B0 backbone with two-stage transfer learning. Stage one trains a replaced linear classifier "
        "head (1280 &rarr; 2) against a frozen ImageNet-pretrained backbone with BatchNorm statistics held fixed. "
        "Stage two unfreezes the last MBConv stage and the 1&times;1 head convolution and fine-tunes at a reduced "
        "learning rate. Class-weighted loss; the checkpoint is selected on validation balanced accuracy.",
    )
    _card(
        "Explainability",
        f"Grad-CAM++ is computed on the <b>{FLOODED}</b> logit at <code>{CAM_TARGET_LAYER}</code> for every tile, so "
        "highlighted regions always mean evidence for flooding. Overlay opacity is scaled by the flood probability, "
        "so confidently clear tiles show almost no heat, and the map is upsampled to the source's native aspect ratio.",
    )
    _card(
        "Input familiarity guard",
        f"Every input's {OOD_FEATURE_LAYER} are compared with the training tiles: the score is one minus the mean "
        "cosine similarity to the 10 most similar training tiles. The threshold is the 99th percentile of "
        "validation scores, so about 1% of genuine validation tiles are flagged; tiles from an unseen region are "
        "flagged more often (see Model Performance). Flagged inputs keep their score for reference but are labeled "
        "unreliable and must not be triaged on it. The check catches clearly different imagery (photos, documents, "
        "synthetic renders); it is not a guarantee against subtle shifts. A Gaussian Mahalanobis score was evaluated "
        "first and rejected: it could not separate synthetic inputs from real tiles.",
    )
    _card(
        "Intended use",
        "Rapid post-disaster triage of Sentinel-2 true-colour tiles: ranking large tile volumes so analysts review "
        "the highest-probability areas first. Outputs are decision support and require analyst verification.",
    )
    _card(
        "Known limitations",
        "<ul>"
        "<li><b>Small, single-event test set.</b> The Pakistan hold-out has few flooded tiles after cloud "
        "filtering; confidence intervals are wide.</li>"
        "<li><b>Optical RGB only.</b> Monsoon floods occur under cloud cover; operational systems rely on SAR "
        "(e.g. Sentinel-1, also in Sen1Floods11), which this model does not ingest.</li>"
        "<li><b>Tile-level labels from thresholds.</b> Labels are derived from pixel masks with fixed cut-offs; "
        "tiles with small flood fractions are excluded from training and evaluation.</li>"
        "<li><b>Uncalibrated probabilities.</b> Softmax output is not calibrated; 80% does not mean an 80% "
        "empirical flood rate.</li>"
        f"<li><b>Coarse maps.</b> <code>{CAM_TARGET_LAYER}</code> is a 7&times;7 grid ({IMG_SIZE}px input), so each "
        "cell covers roughly 32px. Grad-CAM++'s closed-form weights assume ReLU networks; EfficientNet uses SiLU.</li>"
        "<li><b>Whole-tile resize.</b> Inputs are resized to a square model input; large scenes should be tiled "
        "upstream at 10 m/px.</li>"
        "</ul>",
    )

    st.markdown('<div class="spec-heading">Checkpoint provenance</div>', unsafe_allow_html=True)
    if loaded is None:
        st.markdown('<div class="spec-body elevated-card">No checkpoint loaded.</div>', unsafe_allow_html=True)
    else:
        facts = {
            "file": loaded.path.as_posix(),
            "sha256": loaded.sha256,
            "format": loaded.format,
            "integrity (sidecar)": "verified" if loaded.integrity_verified else "not verified — no .sha256 sidecar",
            "schema (class order, size, normalization)": "verified" if loaded.schema_verified
            else "not verified — legacy checkpoint",
            "class order": " · ".join(f"{i}={name}" for i, name in enumerate(CLASS_NAMES)),
            "OOD detector": f"threshold {loaded.ood.threshold:.2f} (p{loaded.ood.percentile:g})" if loaded.ood
            else "not present",
        }
        facts.update({f"trained · {k}": str(v) for k, v in loaded.metadata.items()})
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in facts.items())
        st.markdown(f'<div class="elevated-card"><table class="prov-table">{rows}</table></div>', unsafe_allow_html=True)

    st.markdown('<div class="spec-heading">Technology stack</div>', unsafe_allow_html=True)
    badges = ["PyTorch", "Torchvision", "EfficientNet-B0", "Grad-CAM++", "scikit-learn", "Streamlit", "Plotly",
              "Sentinel-2", "Sen1Floods11"]
    badge_html = "".join(f'<span class="tech-badge">{b}</span>' for b in badges)
    st.markdown(f'<div class="elevated-card">{badge_html}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    """Application entrypoint."""
    inject_css()

    loaded: LoadedCheckpoint | None = None
    load_error: str | None = None
    try:
        loaded = load_model()
    except FileNotFoundError:
        load_error = f"No checkpoint at `{MODEL_PATH.as_posix()}`. Run `python train_pipeline.py` (see README)."
    except CheckpointError as exc:
        load_error = f"Checkpoint rejected — {exc}"
    except Exception as exc:  # corrupt pickle, architecture mismatch: surface the real reason (audit #8)
        load_error = f"Checkpoint failed to load — {type(exc).__name__}: {exc}"

    render_header(loaded)
    active = select_input(loaded)

    tab_threat, tab_triage, tab_diag, tab_overview = st.tabs(
        [
            ":material/radar: Flood Assessment",
            ":material/format_list_numbered: Priority Triage",
            ":material/query_stats: Model Performance",
            ":material/verified_user: System & Provenance",
        ]
    )
    with tab_threat:
        if loaded is None:
            st.error(load_error, icon=":material/error:")
        else:
            render_threat_analysis_tab(loaded, active)
    with tab_triage:
        if loaded is None:
            st.error(load_error, icon=":material/error:")
        else:
            render_batch_triage_tab(loaded)
    with tab_diag:
        render_model_diagnostics_tab()
    with tab_overview:
        render_system_overview_tab(loaded)


if __name__ == "__main__":
    main()
