"""Grad-CAM++ explainability for the EfficientNet-B0 flood detector.

Shared by `app.py` (live inference) and this file's CLI (offline sample grids),
so both render identical evidence maps.

Design decisions (audit #7):
    * The CAM always explains the FLOOD logit (CLASS_NAMES[FLOOD_IDX]), never
      the predicted class. A hotspot therefore always means "evidence for
      flooding", regardless of the verdict.
    * pytorch-grad-cam min-max normalises every map, so every tile would show a
      hotspot. Overlay opacity is scaled by P(flood): a confidently clear tile
      renders a near-invisible overlay.
    * Heat is alpha-blended with the perceptually uniform "inferno" colormap,
      where low activation is transparent. JET renders low activation blue —
      the colour of water — which misleads on a flood product.
    * The CAM is upsampled to the source image's own size, so the overlay keeps
      the native aspect ratio instead of a squashed 224 x 224 copy.

Method caveats (surfaced in the dashboard's System Overview):
    * Grad-CAM++'s closed-form weights assume a piecewise-linear (ReLU) network;
      EfficientNet uses SiLU, so the weights are an approximation.
    * The target layer (features[-1]) is 7 x 7, so each cell covers ~32 px.

Usage:
    python grad_cam.py --model_path model/flood_detector.pth --data_dir data \
        --out_path images/gradcam_sample.png --n_samples 4
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from flood_core import (
    CAM_TARGET_LAYER,
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    FLOOD_IDX,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    cam_target_layer,
    load_checkpoint,
    preprocess,
)

HEATMAP_CMAP = "inferno"
MAX_OVERLAY_ALPHA = 0.7
MAX_DISPLAY_SIDE = 1024

# The serving model is a single object shared by every Streamlit session.
# GradCAMPlusPlus registers forward/backward hooks on it, so two concurrent
# calls would read each other's activations and gradients (audit #12).
_CAM_LOCK = threading.Lock()


@dataclass(frozen=True)
class FloodExplanation:
    """Model output plus Grad-CAM++ map for one image.

    Attributes:
        probs: Softmax probabilities indexed by CLASS_NAMES.
        flood_prob: probs[FLOOD_IDX].
        pred_class: argmax class index.
        cam: [IMG_SIZE, IMG_SIZE] float32 map in [0, 1] for the FLOOD logit.
        latency_ms: Wall-clock time for the forward + backward CAM pass
            (excludes lock wait and image decoding).
        features: [1280] penultimate features from the same forward pass, for OOD scoring.
    """

    probs: list[float]
    flood_prob: float
    pred_class: int
    cam: np.ndarray
    latency_ms: float
    features: np.ndarray


def explain_image(model: torch.nn.Module, img: Image.Image) -> FloodExplanation:
    """Run one forward + backward pass producing probabilities and a flood CAM.

    Phase 1 ran a separate no_grad forward and then Grad-CAM's own forward;
    the logits captured by Grad-CAM are reused instead (audit #11).

    Args:
        model: Eval-mode EfficientNet-B0 from flood_core.load_checkpoint.
        img: Source image, any mode or size.

    Returns:
        FloodExplanation for the image.
    """
    input_tensor = preprocess(img)
    captured: list[torch.Tensor] = []
    with _CAM_LOCK:
        # avgpool output == flood_core.extract_features; captured from the CAM forward pass.
        hook = model.avgpool.register_forward_hook(lambda _m, _i, out: captured.append(out.detach()))
        try:
            t0 = time.perf_counter()
            with GradCAMPlusPlus(model=model, target_layers=[cam_target_layer(model)]) as cam:
                cam_map = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(FLOOD_IDX)])[0]
                logits = cam.outputs.detach()
            latency_ms = (time.perf_counter() - t0) * 1000
        finally:
            hook.remove()
            # Parameter .grad buffers from the CAM backward pass would otherwise stay allocated.
            model.zero_grad(set_to_none=True)

    probs = torch.softmax(logits, dim=1)[0].tolist()
    return FloodExplanation(
        probs=probs,
        flood_prob=float(probs[FLOOD_IDX]),
        pred_class=int(np.argmax(probs)),
        cam=np.asarray(cam_map, dtype=np.float32),
        latency_ms=latency_ms,
        features=torch.flatten(captured[0], 1)[0].numpy(),
    )


def overlay_heatmap(rgb: np.ndarray, cam: np.ndarray, strength: float) -> np.ndarray:
    """Alpha-blend a CAM onto an RGB image at the image's own resolution.

    Args:
        rgb: [H, W, 3] uint8 image.
        cam: [h, w] float map in [0, 1], any resolution (resized bilinearly to H x W).
        strength: Global opacity multiplier in [0, 1] (the flood probability).

    Returns:
        [H, W, 3] uint8 blended image.
    """
    height, width = rgb.shape[:2]
    cam_img = Image.fromarray(np.uint8(np.clip(cam, 0.0, 1.0) * 255))
    cam_resized = np.asarray(cam_img.resize((width, height), Image.BILINEAR), dtype=np.float32) / 255.0
    heat = matplotlib.colormaps[HEATMAP_CMAP](cam_resized)[..., :3]
    alpha = (cam_resized * float(np.clip(strength, 0.0, 1.0)) * MAX_OVERLAY_ALPHA)[..., None]
    base = rgb.astype(np.float32) / 255.0
    return np.uint8(np.clip(base * (1.0 - alpha) + heat * alpha, 0.0, 1.0) * 255)


def render_evidence(img: Image.Image, explanation: FloodExplanation) -> tuple[np.ndarray, np.ndarray]:
    """Produce display-ready source and overlay arrays at native aspect ratio.

    Args:
        img: Source image (not modified).
        explanation: Result of `explain_image` for the same image.

    Returns:
        Tuple of ([H, W, 3] uint8 source, [H, W, 3] uint8 overlay), longest side
        capped at MAX_DISPLAY_SIDE to bound memory.
    """
    display = img.convert("RGB")
    display.thumbnail((MAX_DISPLAY_SIDE, MAX_DISPLAY_SIDE))
    rgb = np.asarray(display, dtype=np.uint8)
    return rgb, overlay_heatmap(rgb, explanation.cam, strength=explanation.flood_prob)


def generate_gradcam(
    image_path: Path, model: torch.nn.Module, img_size: int = IMG_SIZE
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Generate a flood-evidence overlay for one image file (Phase 1 signature preserved).

    Args:
        image_path: Path to the source image.
        model: Trained model in eval mode.
        img_size: Must equal flood_core.IMG_SIZE; kept for signature compatibility.

    Returns:
        Tuple of (original_rgb uint8, cam_overlay uint8, pred_class, confidence of pred_class).
    """
    if img_size != IMG_SIZE:
        raise ValueError(f"img_size must be {IMG_SIZE} (flood_core.IMG_SIZE); got {img_size}")
    with Image.open(image_path) as img:
        img.load()
        explanation = explain_image(model, img)
        original_rgb, cam_overlay = render_evidence(img, explanation)
    return original_rgb, cam_overlay, explanation.pred_class, explanation.probs[explanation.pred_class]


def collect_sample_paths(data_dir: Path, n_samples: int) -> list[Path]:
    """Pick up to n_samples images, alternating across classes, any supported extension.

    Phase 1 globbed only *.png, so JPEG datasets raised "No sample images
    found" (audit #19).

    Args:
        data_dir: Root with one subfolder per CLASS_NAMES entry.
        n_samples: Total images to return.

    Returns:
        Sorted-per-class, round-robin-interleaved image paths.
    """
    per_class = [
        sorted(p for p in (data_dir / name).glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if (data_dir / name).is_dir() else []
        for name in reversed(CLASS_NAMES)  # flooded first, matching Phase 1 grid order
    ]
    picked: list[Path] = []
    for i in range(max((len(paths) for paths in per_class), default=0)):
        for paths in per_class:
            if i < len(paths) and len(picked) < n_samples:
                picked.append(paths[i])
    return picked


def save_sample_grid(
    model: torch.nn.Module,
    data_dir: Path,
    out_path: Path,
    n_samples: int = 4,
    img_size: int = IMG_SIZE,
) -> None:
    """Save a grid of source | flood-evidence pairs.

    Args:
        model: Trained model.
        data_dir: Root directory with one subfolder per class.
        out_path: PNG destination.
        n_samples: Total images to visualise.
        img_size: Must equal flood_core.IMG_SIZE.
    """
    sample_paths = collect_sample_paths(data_dir, n_samples)
    if not sample_paths:
        raise FileNotFoundError(
            f"No {'/'.join(IMAGE_EXTENSIONS)} images found under {data_dir}/{{{','.join(CLASS_NAMES)}}}."
        )

    bg, fg = "#0B0F14", "#C4CDD5"
    fig, axes = plt.subplots(len(sample_paths), 2, figsize=(7, 3.2 * len(sample_paths)), facecolor=bg, squeeze=False)

    for row, img_path in enumerate(sample_paths):
        original, overlay, pred_class, confidence = generate_gradcam(img_path, model, img_size)
        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"Source\n{img_path.parent.name}/{img_path.name}", fontsize=9, color=fg)
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(
            f"Flood evidence (Grad-CAM++)\nPred: {CLASS_NAMES[pred_class]} ({confidence * 100:.1f}%)",
            fontsize=9, color=fg,
        )
        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle(f"Grad-CAM++ on P({CLASS_NAMES[FLOOD_IDX]}) · target {CAM_TARGET_LAYER}", fontsize=12, color=fg)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=bg)
    plt.close(fig)
    print(f"Saved {len(sample_paths)} Grad-CAM++ visualizations -> {out_path}")


def load_trained_model(model_path: Path) -> torch.nn.Module:
    """Load a trained model via flood_core (Phase 1 signature preserved).

    Args:
        model_path: Checkpoint path.

    Returns:
        The model in eval mode on CPU.
    """
    loaded = load_checkpoint(model_path)
    if not loaded.schema_verified:
        print(f"WARNING: {model_path} is a legacy checkpoint; class order cannot be verified.")
    return loaded.model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM++ flood-evidence visualizations.")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_path", type=str, default="images/gradcam_sample.png")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    args = parser.parse_args()

    save_sample_grid(
        model=load_trained_model(Path(args.model_path)),
        data_dir=Path(args.data_dir),
        out_path=Path(args.out_path),
        n_samples=args.n_samples,
        img_size=args.img_size,
    )
