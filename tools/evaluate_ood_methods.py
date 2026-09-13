"""Reproduce the OOD-method comparison that justified kNN cosine over Mahalanobis.

In-distribution: validation + Pakistan test tiles. Out-of-distribution: every
5th synthetic tile from `data/` plus 40 generated junk images (noise, flat
colour, text documents, gradients). Thresholds are p99 of validation scores.

Requires the trained checkpoint, images/split_manifest.json, data_real/, and data/:
    python prepare_sen1floods11.py
    python generate_synthetic_data.py --out_dir data
    python tools/evaluate_ood_methods.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import flood_core as fc  # noqa: E402


def features(model: torch.nn.Module, images: list[Image.Image]) -> np.ndarray:
    with torch.no_grad():
        return np.stack([fc.extract_features(model, fc.preprocess(im))[0].numpy() for im in images])


def junk_images(n: int = 40) -> list[Image.Image]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            out.append(Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)))
        elif kind == 1:
            out.append(Image.new("RGB", (256, 256), tuple(int(x) for x in rng.integers(0, 255, 3))))
        elif kind == 2:
            im = Image.new("RGB", (400, 300), "white")
            draw = ImageDraw.Draw(im)
            for y in range(10, 290, 20):
                draw.text((10, y), "Situation report flood district " * 2, fill="black")
            out.append(im)
        else:
            ramp = np.linspace(0, 255, 256, dtype=np.uint8)
            out.append(Image.fromarray(np.stack(
                [np.tile(ramp, (256, 1)), np.tile(ramp[:, None], (1, 256)), np.full((256, 256), 128, np.uint8)], -1)))
    return out


def main() -> None:
    model = fc.load_checkpoint(ROOT / fc.DEFAULT_MODEL_PATH).model
    manifest = json.loads((ROOT / "images/split_manifest.json").read_text(encoding="utf-8"))
    load = lambda root, recs: [Image.open(root / r["path"]).convert("RGB") for r in recs]  # noqa: E731

    train = features(model, load(ROOT / "data_real/train", manifest["train"]))
    val = features(model, load(ROOT / "data_real/train", manifest["val"]))
    test = features(model, load(ROOT / "data_real/test_pakistan", manifest["test"]))
    synth = features(model, [Image.open(p).convert("RGB") for p in sorted((ROOT / "data").rglob("*.png"))[::5]])
    junk = features(model, junk_images())

    def report(name: str, score) -> None:
        s_val, s_test, s_syn, s_junk = (score(x) for x in (val, test, synth, junk))
        thr = np.percentile(s_val, 99)
        ind, ood = np.r_[s_val, s_test], np.r_[s_syn, s_junk]
        auc = roc_auc_score(np.r_[np.zeros(len(ind)), np.ones(len(ood))], np.r_[ind, ood])
        print(f"| {name} | {auc:.2f} | {np.mean(s_test > thr):.0%} | {np.mean(s_syn > thr):.0%} | {np.mean(s_junk > thr):.0%} |")

    print("| Method | AUROC | Pakistan test flagged | Synthetic flagged | Junk flagged |")
    print("|---|---|---|---|---|")
    pca = PCA(n_components=64, random_state=0).fit(train)
    lw = LedoitWolf().fit(pca.transform(train))

    def mahalanobis_pca(x):
        z = pca.transform(x) - lw.location_
        return np.sqrt(np.einsum("ij,jk,ik->i", z, lw.precision_, z))

    report("PCA-64 Mahalanobis (rejected)", mahalanobis_pca)
    full = LedoitWolf().fit(train)
    report("Full 1280-d Mahalanobis", lambda x: np.sqrt(
        np.einsum("ij,jk,ik->i", x - full.location_, full.precision_, x - full.location_)))
    detector = fc.OodDetector.fit(train, val, k=10, percentile=99.0)
    report("kNN cosine, k=10 (shipped)", detector.distance)


if __name__ == "__main__":
    main()
