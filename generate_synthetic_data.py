"""Generate the procedural proof-of-concept dataset used to validate the pipeline.

The Phase 1 README referenced this script but it was never committed, so the
dataset behind the shipped artifacts could not be regenerated (audit #18).
This is a from-scratch reconstruction that matches the documented design and
the shipped reference tiles; it is not byte-identical to the lost original.

Classes (folder names come from flood_core.CLASS_NAMES):
    flooded      — blue/turbid water field with tan sediment islands and debris streaks
    not_flooded  — green vegetation field with grey built-up blocks and a road grid

These classes are deliberately RGB-separable. The dataset validates pipeline
mechanics only; it says nothing about accuracy on real satellite imagery.

Usage:
    python generate_synthetic_data.py --out_dir data --n_per_class 250 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from flood_core import CLASS_NAMES, FLOOD_IDX, IMG_SIZE

NOT_FLOODED_IDX = 1 - FLOOD_IDX


def _noise_field(rng: np.random.Generator, size: int, blur: float) -> np.ndarray:
    """Smooth noise in [0, 1] via blurred uniform noise."""
    raw = Image.fromarray(np.uint8(rng.random((size, size)) * 255))
    smooth = np.asarray(raw.filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32)
    lo, hi = smooth.min(), smooth.max()
    return (smooth - lo) / (hi - lo + 1e-6)


def _blob_mask(rng: np.random.Generator, size: int, n_blobs: int, blur: float) -> np.ndarray:
    """Soft irregular blobs in [0, 1]."""
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    for _ in range(n_blobs):
        cx, cy = rng.integers(0, size, 2)
        rx, ry = rng.integers(size // 12, size // 4, 2)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    mask = np.asarray(canvas.filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32) / 255.0
    return np.clip((mask - 0.35) * 3.0, 0.0, 1.0)


def _blend(base: np.ndarray, color: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return base * (1.0 - mask[..., None]) + color * mask[..., None]


def make_flooded_tile(rng: np.random.Generator, size: int = IMG_SIZE) -> Image.Image:
    """Water field + sediment islands + thin debris streaks."""
    water_rgb = np.array([70, 80, 190]) + rng.integers(-20, 20, 3)
    sediment_rgb = np.array([165, 140, 90]) + rng.integers(-20, 20, 3)
    texture = _noise_field(rng, size, blur=6)[..., None]
    img = water_rgb * (0.85 + 0.3 * texture)
    img = _blend(img, sediment_rgb * (0.9 + 0.2 * texture), _blob_mask(rng, size, int(rng.integers(3, 8)), blur=10))

    pil = Image.fromarray(np.uint8(np.clip(img, 0, 255)))
    draw = ImageDraw.Draw(pil)
    for _ in range(int(rng.integers(2, 6))):
        x0, y0, x1, y1 = rng.integers(0, size, 4)
        draw.line([x0, y0, x1, y1], fill=(50, 50, 55), width=int(rng.integers(1, 4)))
    return pil.filter(ImageFilter.GaussianBlur(0.6))


def make_not_flooded_tile(rng: np.random.Generator, size: int = IMG_SIZE) -> Image.Image:
    """Vegetation field + built-up blocks + straight road grid."""
    veg_rgb = np.array([70, 160, 70]) + rng.integers(-20, 20, 3)
    urban_rgb = np.array([185, 180, 170]) + rng.integers(-15, 15, 3)
    texture = _noise_field(rng, size, blur=4)[..., None]
    img = veg_rgb * (0.8 + 0.4 * texture)
    img = _blend(img, urban_rgb, _blob_mask(rng, size, int(rng.integers(2, 6)), blur=8))

    pil = Image.fromarray(np.uint8(np.clip(img, 0, 255)))
    draw = ImageDraw.Draw(pil)
    for _ in range(int(rng.integers(1, 3))):
        x = int(rng.integers(0, size))
        draw.line([x, 0, x, size], fill=(110, 110, 110), width=2)
    for _ in range(int(rng.integers(1, 3))):
        y = int(rng.integers(0, size))
        draw.line([0, y, size, y], fill=(110, 110, 110), width=2)
    return pil.filter(ImageFilter.GaussianBlur(0.6))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic flood PoC dataset.")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--n_per_class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    makers = {CLASS_NAMES[FLOOD_IDX]: make_flooded_tile, CLASS_NAMES[NOT_FLOODED_IDX]: make_not_flooded_tile}
    for class_name, maker in makers.items():
        class_dir = Path(args.out_dir) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.n_per_class):
            maker(rng).save(class_dir / f"{class_name}_{i:04d}.png")
        print(f"Wrote {args.n_per_class} tiles -> {class_dir}")


if __name__ == "__main__":
    main()
