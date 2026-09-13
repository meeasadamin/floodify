"""Train the flood / not-flooded classifier (Phase 2 pipeline; replaces train_model.py).

Backbone: EfficientNet-B0 (torchvision, ImageNet-pretrained by default).
Strategy: two-stage fine-tuning.
    Stage 1 — backbone frozen (including BatchNorm statistics), train the head.
    Stage 2 — unfreeze the last N top-level `features` stages (default 2: the
              final MBConv stage features[7] and the 1x1 head conv features[8]),
              fine-tune at a lower learning rate.

Data protocol (audit #6):
    * Splits are stratified AND grouped: tiles cut from the same source chip
      (`<chip>_r<row>c<col>.png`) always land in the same split, so neighbouring
      tiles cannot leak between train, validation, and test.
    * `--test_dir` holds out an entire external set (e.g. an unseen flood event);
      otherwise a test split is carved from `--data_dir`.
    * Validation selects the checkpoint; the test set is evaluated once, after
      selection, and is the only set reported as final performance.
    * Class-weighted loss and balanced-accuracy selection keep an imbalanced
      real dataset from rewarding "always predict the majority class".

After selection, an out-of-distribution detector (flood_core.OodDetector) is
fitted on training features and calibrated on validation features, and saved
inside the checkpoint.

Outputs (audit #16 — always written together by one run):
    model/flood_detector.pth            v3 checkpoint (weights + schema + OOD), see flood_core
    model/flood_detector.pth.sha256     integrity sidecar
    images/final_metrics.json           metrics consumed by app.py (schema_version 3)
    images/split_manifest.json          exact file lists of each split
    images/training_curves.png          loss / accuracy curves
    images/confusion_matrix.png         test-set confusion matrix

One script serves CPU and Colab (`--device cuda`), so every shipped artifact is
reproducible from this repository (audit #18).

Usage (real data):
    python prepare_sen1floods11.py
    python train_pipeline.py --data_dir data_real/train --test_dir data_real/test_pakistan \
        --test_name "Pakistan flood event (unseen)" --ood_probe_dir data
Usage (synthetic smoke test):
    python generate_synthetic_data.py --out_dir data
    python train_pipeline.py --data_dir data --epochs_stage1 1 --epochs_stage2 1 --no-pretrained
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from flood_core import (
    ARCHITECTURE,
    CLASS_NAMES,
    EVAL_TF,
    FLOOD_IDX,
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    NUM_CLASSES,
    OOD_FEATURE_LAYER,
    OodDetector,
    build_model as build_core_model,
    extract_features,
    preprocess,
    save_checkpoint,
)

METRICS_SCHEMA_VERSION = 3
TILE_GROUP_RE = re.compile(r"^(?P<group>.+)_r\d+c\d+$")

# Dark chart palette so diagnostics match the dashboard.
PLOT_BG = "#0B0F14"
PLOT_FG = "#C4CDD5"
PLOT_GRID = "#222B36"
PLOT_SERIES = ("#4FB3D9", "#E3B341", "#3FB950")


@dataclass
class TrainConfig:
    """Hyperparameters and paths for the training run.

    Attributes:
        data_dir: Root folder containing one subfolder per entry in CLASS_NAMES.
        test_dir: Optional external held-out set with the same layout; if set,
            no test split is carved from data_dir.
        test_name: Human-readable description of the test set.
        ood_probe_dir: Optional folder of images (searched recursively) expected
            to be out of distribution; the flagged fraction is reported.
        model_out: Path to save the checkpoint.
        images_out: Directory for diagnostic plots and metrics.
        batch_size: Training/evaluation batch size.
        val_split: Fraction of data_dir held out for checkpoint selection.
        test_split: Fraction of data_dir held out for testing (ignored with test_dir).
        epochs_stage1: Epochs training only the classifier head.
        epochs_stage2: Epochs fine-tuning the unfrozen stages + head.
        lr_stage1: Learning rate for stage 1.
        lr_stage2: Learning rate for stage 2.
        unfreeze_blocks: Trailing top-level `features` stages unfrozen in stage 2.
        ood_percentile: Validation distance percentile used as the OOD threshold.
        seed: Seed for the split, torch, numpy, and Python RNGs.
        device: "cpu" (default), "cuda", or "auto".
        pretrained: Initialise the backbone from ImageNet-1K weights.
        num_workers: DataLoader worker processes.
    """

    data_dir: Path
    test_dir: Path | None = None
    test_name: str = "Held-out test split"
    ood_probe_dir: Path | None = None
    model_out: Path = Path("model/flood_detector.pth")
    images_out: Path = Path("images")
    batch_size: int = 16
    val_split: float = 0.15
    test_split: float = 0.15
    epochs_stage1: int = 5
    epochs_stage2: int = 5
    lr_stage1: float = 1e-3
    lr_stage2: float = 1e-4
    unfreeze_blocks: int = 2
    ood_percentile: float = 99.0
    seed: int = 42
    device: str = "cpu"
    pretrained: bool = True
    num_workers: int = 0


class SatelliteFloodDataset(Dataset):
    """Binary classification dataset for satellite tiles.

    Expects `data_dir/<class_name>/*.png|jpg|jpeg` for every name in
    CLASS_NAMES. Label indices come from CLASS_NAMES order (audit #4), never
    from directory listing order.
    """

    def __init__(self, data_dir: Path, transform: transforms.Compose) -> None:
        """Index all image paths and labels.

        Args:
            data_dir: Root directory with one subfolder per class.
            transform: torchvision transform pipeline applied to each image.
        """
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = data_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing class folder {class_dir}. Expected subfolders: {CLASS_NAMES}. "
                    "Run prepare_sen1floods11.py or generate_synthetic_data.py first."
                )
            for p in sorted(class_dir.iterdir()):
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((p, label))

        if not self.samples:
            raise FileNotFoundError(f"No images found under {data_dir} for classes {CLASS_NAMES}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def tile_group(path: Path) -> str:
    """Group key: the source chip for `<chip>_r<row>c<col>` tiles, else the file stem."""
    match = TILE_GROUP_RE.match(path.stem)
    return match.group("group") if match else path.stem


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Build the training (augmented) and evaluation (deterministic) pipelines.

    Augmentation (audit #3): overhead imagery has no canonical "up", so both
    flips and exact 90-degree rotations are label-preserving and introduce no
    fill pixels. The square crop ratio keeps tiles undistorted. Evaluation uses
    EVAL_TF from flood_core (audit #5).

    Returns:
        Tuple of (train_transform, eval_transform).
    """
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(1.0, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomChoice([transforms.RandomRotation((angle, angle)) for angle in (0, 90, 180, 270)]),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_tf, EVAL_TF


def build_model(pretrained: bool, device: str) -> nn.Module:
    """Build the shared architecture and freeze the backbone for stage 1.

    Args:
        pretrained: Load ImageNet-1K backbone weights.
        device: Torch device string.

    Returns:
        Model with only the classifier head trainable.
    """
    model = build_core_model(pretrained=pretrained)
    for param in model.features.parameters():
        param.requires_grad = False
    return model.to(device)


def unfreeze_last_n_blocks(model: nn.Module, n_blocks: int = 2) -> None:
    """Unfreeze the last N top-level stages of EfficientNet-B0's `features`.

    `model.features` has 9 stages (0-8): stem conv, seven MBConv stages (1-7),
    and the final 1x1 conv (8). n_blocks=2 unfreezes features[7] (last MBConv
    stage) and features[8] (1x1 head conv).

    Args:
        model: Model from `build_model`.
        n_blocks: Number of trailing stages to unfreeze.
    """
    cutoff = len(model.features) - n_blocks
    for idx, stage in enumerate(model.features):
        if idx >= cutoff:
            for param in stage.parameters():
                param.requires_grad = True


def freeze_bn_stats(model: nn.Module) -> None:
    """Put BatchNorm layers whose affine params are frozen into eval mode (audit #2).

    `model.train()` sets every BatchNorm to train mode, which updates running
    mean/var even when `requires_grad=False`. That silently drifts the
    "frozen" ImageNet statistics on small batches. Call after `model.train()`.

    Args:
        model: Model currently in train mode.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not module.weight.requires_grad:
            module.eval()


def count_params(model: nn.Module) -> tuple[int, int]:
    """Count trainable vs. total parameters.

    Args:
        model: Any torch module.

    Returns:
        Tuple of (trainable_params, total_params).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: str,
) -> tuple[float, float, float]:
    """Run one epoch of training (if optimizer given) or evaluation (if None).

    Args:
        model: The model to train/evaluate.
        loader: DataLoader yielding (images, labels) batches.
        criterion: Loss function.
        optimizer: Optimizer for training; None for evaluation.
        device: Torch device string.

    Returns:
        Tuple of (average_loss, accuracy, balanced_accuracy) over the epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)
    if is_train:
        freeze_bn_stats(model)

    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)  # shape: [batch, NUM_CLASSES]
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    accuracy = sum(p == y for p, y in zip(all_preds, all_labels)) / n
    return running_loss / n, accuracy, float(balanced_accuracy_score(all_labels, all_preds))


@dataclass
class TrainingHistory:
    """Per-epoch metrics across both training stages."""

    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    val_balanced_acc: list[float] = field(default_factory=list)
    stage: list[int] = field(default_factory=list)


def is_better(val_score: float, val_loss: float, best_state: dict) -> bool:
    """Checkpoint selection rule (audit #1): higher val score, ties broken by lower val loss.

    Phase 1 used a strict `val_acc > best`, so a run that reached 100% val
    accuracy at epoch 1 never saved any later epoch and silently discarded all
    of stage 2. The score is validation balanced accuracy.
    """
    return (val_score, -val_loss) > (best_state["score"], -best_state["loss"])


def train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    epochs: int,
    lr: float,
    stage_num: int,
    history: TrainingHistory,
    best_state: dict,
    device: str,
) -> None:
    """Train for a fixed number of epochs, tracking history and the best checkpoint.

    Args:
        model: Model being trained (mutated in place).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (selection only — never the test set).
        criterion: Class-weighted loss shared by training and validation.
        epochs: Number of epochs for this stage.
        lr: Learning rate for this stage's optimizer.
        stage_num: 1 or 2, recorded into history.
        history: TrainingHistory appended to in place.
        best_state: Dict with keys score, acc, loss, epoch, stage, state_dict; updated in place.
        device: Torch device string.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)

    trainable, total = count_params(model)
    print(f"[Stage {stage_num}] Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_bal = run_epoch(model, val_loader, criterion, None, device)
        dt = time.time() - t0

        history.train_loss.append(tr_loss)
        history.train_acc.append(tr_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.val_balanced_acc.append(val_bal)
        history.stage.append(stage_num)
        global_epoch = len(history.val_acc)

        print(
            f"[Stage {stage_num}] Epoch {epoch}/{epochs} "
            f"| train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_bal_acc={val_bal:.4f} | {dt:.1f}s",
            flush=True,
        )

        if is_better(val_bal, val_loss, best_state):
            best_state.update(
                score=val_bal,
                acc=val_acc,
                loss=val_loss,
                epoch=global_epoch,
                stage=stage_num,
                state_dict=copy.deepcopy(model.state_dict()),
            )
            print(f"  -> New best (val_bal_acc={val_bal:.4f}, val_loss={val_loss:.4f}), checkpoint updated.")


def grouped_stratified_split(
    labels: list[int], groups: list[str], val_split: float, test_split: float, seed: int
) -> dict[str, list[int]]:
    """Split indices into stratified, group-disjoint train / val / test sets (audit #6).

    Uses StratifiedGroupKFold and takes one fold per held-out set, so actual
    fractions are approximately 1 / round(1 / fraction).

    Args:
        labels: Class index for every sample.
        groups: Group key for every sample (source chip).
        val_split: Approximate fraction of all samples for validation.
        test_split: Approximate fraction of all samples for test (0 disables).
        seed: Random state for reproducibility.

    Returns:
        Dict with sorted index lists under "train", "val", "test".
    """
    y = np.asarray(labels)
    g = np.asarray(groups)

    def hold_out(pool: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
        n_splits = max(2, round(1 / fraction))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        keep, held = next(splitter.split(pool, y[pool], g[pool]))
        return pool[keep], pool[held]

    pool = np.arange(len(y))
    test_idx = np.array([], dtype=int)
    if test_split > 0:
        pool, test_idx = hold_out(pool, test_split)
    train_idx, val_idx = hold_out(pool, val_split / (1.0 - test_split))
    return {"train": sorted(train_idx.tolist()), "val": sorted(val_idx.tolist()), "test": sorted(test_idx.tolist())}


def _style_axes(ax: plt.Axes) -> None:
    """Apply the dashboard's dark palette to a matplotlib axis."""
    ax.set_facecolor(PLOT_BG)
    ax.tick_params(colors=PLOT_FG)
    for spine in ax.spines.values():
        spine.set_color(PLOT_GRID)
    ax.xaxis.label.set_color(PLOT_FG)
    ax.yaxis.label.set_color(PLOT_FG)
    ax.title.set_color(PLOT_FG)
    ax.grid(color=PLOT_GRID, alpha=0.8)


def plot_training_curves(history: TrainingHistory, out_path: Path, best_epoch: int) -> None:
    """Plot loss and accuracy curves, marking the stage boundary and selected epoch.

    Args:
        history: Populated TrainingHistory.
        out_path: PNG destination.
        best_epoch: 1-based global epoch of the selected checkpoint.
    """
    epochs_x = list(range(1, len(history.train_loss) + 1))
    stage2_start = next((i for i, s in enumerate(history.stage) if s == 2), None)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=PLOT_BG)
    panels = (
        (axes[0], [("Train", history.train_loss), ("Validation", history.val_loss)], "Loss (class-weighted)",
         "Training / Validation Loss"),
        (axes[1], [("Train", history.train_acc), ("Validation", history.val_acc),
                   ("Validation balanced", history.val_balanced_acc)], "Accuracy", "Training / Validation Accuracy"),
    )
    for ax, series, ylabel, title in panels:
        _style_axes(ax)
        for (label, values), color in zip(series, PLOT_SERIES):
            ax.plot(epochs_x, values, label=label, marker="o", color=color)
        if stage2_start is not None:
            ax.axvline(x=stage2_start + 0.5, color=PLOT_FG, linestyle="--", alpha=0.5, label="Stage 2 starts")
        ax.axvline(x=best_epoch, color="#F85149", linestyle=":", label=f"Selected epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        legend = ax.legend(facecolor=PLOT_BG, edgecolor=PLOT_GRID)
        for text in legend.get_texts():
            text.set_color(PLOT_FG)

    fig.suptitle("EfficientNet-B0 two-stage fine-tuning — satellite flood detector", color=PLOT_FG)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=PLOT_BG)
    plt.close(fig)
    print(f"Saved training curves -> {out_path}")


def evaluate_and_plot_confusion(
    model: nn.Module, loader: DataLoader, out_path: Path, device: str, split_name: str = "Test"
) -> dict:
    """Evaluate the selected checkpoint on one set and save its confusion matrix.

    Args:
        model: Trained model (selected checkpoint already loaded).
        loader: DataLoader for the set being reported (the test set).
        out_path: PNG destination.
        device: Torch device string.
        split_name: Label used in the plot title and console output.

    Returns:
        Dict with "classification_report" (sklearn dict), "confusion_matrix"
        (nested list, rows=actual, cols=predicted, CLASS_NAMES order), "accuracy",
        "balanced_accuracy", "roc_auc" (None if one class is absent), "n_samples".
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    flood_probs: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu()
            all_preds.extend(probs.argmax(dim=1).tolist())
            flood_probs.extend(probs[:, FLOOD_IDX].tolist())
            all_labels.extend(labels.tolist())

    label_ids = list(range(NUM_CLASSES))
    cm = confusion_matrix(all_labels, all_preds, labels=label_ids)
    report = classification_report(
        all_labels, all_preds, labels=label_ids, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    has_both = len(set(all_labels)) == NUM_CLASSES
    roc_auc = float(roc_auc_score([int(y == FLOOD_IDX) for y in all_labels], flood_probs)) if has_both else None
    balanced = float(balanced_accuracy_score(all_labels, all_preds))
    print(f"\n=== {split_name} classification report ===")
    print(classification_report(all_labels, all_preds, labels=label_ids, target_names=CLASS_NAMES, zero_division=0))
    print(f"balanced_accuracy={balanced:.4f} roc_auc={roc_auc}")

    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=PLOT_BG)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="mako", cbar=False,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, linewidths=2, linecolor=PLOT_BG,
    )
    _style_axes(ax)
    ax.grid(False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {split_name} (n={len(all_labels)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=PLOT_BG)
    plt.close(fig)
    print(f"Saved confusion matrix -> {out_path}")

    ci = bootstrap_ci(np.asarray(all_labels), np.asarray(all_preds), np.asarray(flood_probs))
    print(f"95% bootstrap CIs: {ci}")

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "accuracy": float(report["accuracy"]),
        "balanced_accuracy": balanced,
        "roc_auc": roc_auc,
        "ci95": ci,
        "n_samples": len(all_labels),
    }


def bootstrap_ci(
    labels: np.ndarray, preds: np.ndarray, flood_probs: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict[str, list[float] | None]:
    """Percentile-bootstrap 95% intervals, resampling tiles with replacement.

    Small held-out sets (e.g. a single flood event) make point estimates
    misleading; the dashboard shows these intervals next to each metric.

    Returns:
        {"balanced_accuracy", "flooded_recall", "flooded_precision", "roc_auc"} -> [low, high] or None.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    samples: dict[str, list[float]] = {"balanced_accuracy": [], "flooded_recall": [], "flooded_precision": [],
                                       "roc_auc": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y, p, s = labels[idx], preds[idx], flood_probs[idx]
        if len(np.unique(y)) < NUM_CLASSES:
            continue
        samples["balanced_accuracy"].append(balanced_accuracy_score(y, p))
        positives = y == FLOOD_IDX
        samples["flooded_recall"].append(float((p[positives] == FLOOD_IDX).mean()))
        predicted_pos = p == FLOOD_IDX
        if predicted_pos.any():
            samples["flooded_precision"].append(float((y[predicted_pos] == FLOOD_IDX).mean()))
        samples["roc_auc"].append(roc_auc_score(positives.astype(int), s))
    return {
        name: [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] if vals else None
        for name, vals in samples.items()
    }


def collect_features(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Penultimate features for every sample in a loader, [N, 1280]."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for images, _ in loader:
            chunks.append(extract_features(model, images.to(device)).cpu().numpy())
    return np.concatenate(chunks)


def probe_ood(model: nn.Module, detector: OodDetector, probe_dir: Path, device: str) -> dict:
    """Fraction of images under probe_dir (recursive) flagged as out of distribution."""
    paths = sorted(p for p in probe_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        return {"dir": probe_dir.as_posix(), "n": 0, "flagged_frac": None}
    model.eval()
    feats = []
    with torch.no_grad():
        for p in paths:
            with Image.open(p) as img:
                feats.append(extract_features(model, preprocess(img).to(device)).cpu().numpy()[0])
    flagged = detector.distance(np.stack(feats)) > detector.threshold
    return {"dir": probe_dir.as_posix(), "n": len(paths), "flagged_frac": float(flagged.mean())}


def resolve_device(requested: str) -> str:
    """Map "auto" to cuda when available; validate explicit requests."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    return requested


def _manifest(dataset: SatelliteFloodDataset, root: Path, indices: list[int]) -> list[dict]:
    return [
        {"path": dataset.samples[i][0].relative_to(root).as_posix(), "label": CLASS_NAMES[dataset.samples[i][1]]}
        for i in indices
    ]


def main(cfg: TrainConfig) -> None:
    """End-to-end training: split, two-stage fit, selection, test evaluation, OOD fit, export.

    Args:
        cfg: Fully populated TrainConfig.
    """
    device = resolve_device(cfg.device)
    random.seed(cfg.seed)  # torchvision RandomChoice uses Python's `random`
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    cfg.model_out.parent.mkdir(parents=True, exist_ok=True)
    cfg.images_out.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms()

    # Two views over the same deterministic (sorted) file list, so indices align.
    eval_dataset = SatelliteFloodDataset(cfg.data_dir, transform=eval_tf)
    train_dataset = SatelliteFloodDataset(cfg.data_dir, transform=train_tf)
    assert eval_dataset.samples == train_dataset.samples, "Dataset views disagree on file order"

    labels = [label for _, label in eval_dataset.samples]
    groups = [tile_group(path) for path, _ in eval_dataset.samples]
    test_split = 0.0 if cfg.test_dir else cfg.test_split
    splits = grouped_stratified_split(labels, groups, cfg.val_split, test_split, cfg.seed)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not {groups[i] for i in splits[a]} & {groups[i] for i in splits[b]}, f"group leak {a}/{b}"

    if cfg.test_dir:
        test_dataset = SatelliteFloodDataset(cfg.test_dir, transform=eval_tf)
        test_subset, test_root, test_indices = test_dataset, cfg.test_dir, list(range(len(test_dataset)))
        test_labels = [label for _, label in test_dataset.samples]
    else:
        test_dataset, test_root, test_indices = eval_dataset, cfg.data_dir, splits["test"]
        test_subset = Subset(eval_dataset, test_indices)
        test_labels = [labels[i] for i in test_indices]

    train_labels = [labels[i] for i in splits["train"]]
    class_counts = {
        split: {name: lbls.count(i) for i, name in enumerate(CLASS_NAMES)}
        for split, lbls in (("train", train_labels), ("val", [labels[i] for i in splits["val"]]), ("test", test_labels))
    }
    print(f"Class counts per split: {class_counts}")

    manifest_path = cfg.images_out / "split_manifest.json"
    manifest = {
        "train": _manifest(eval_dataset, cfg.data_dir, splits["train"]),
        "val": _manifest(eval_dataset, cfg.data_dir, splits["val"]),
        "test": _manifest(test_dataset, test_root, test_indices),
    }
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"Saved split manifest -> {manifest_path}")

    loader_kwargs = {"batch_size": cfg.batch_size, "num_workers": cfg.num_workers}
    train_loader = DataLoader(Subset(train_dataset, splits["train"]), shuffle=True, **loader_kwargs)
    train_eval_loader = DataLoader(Subset(eval_dataset, splits["train"]), shuffle=False, **loader_kwargs)
    val_loader = DataLoader(Subset(eval_dataset, splits["val"]), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_subset, shuffle=False, **loader_kwargs)

    sample_images, _ = next(iter(train_loader))
    expected_shape = (sample_images.shape[0], 3, IMG_SIZE, IMG_SIZE)
    assert tuple(sample_images.shape) == expected_shape, (
        f"Shape mismatch: got {tuple(sample_images.shape)}, expected {expected_shape}"
    )
    print(f"Batch tensor shape verified: {tuple(sample_images.shape)}")

    model = build_model(pretrained=cfg.pretrained, device=device)
    with torch.no_grad():
        model.eval()
        dry_run_out = model(sample_images.to(device))
    assert tuple(dry_run_out.shape) == (sample_images.shape[0], NUM_CLASSES), (
        f"Classifier output shape mismatch: got {tuple(dry_run_out.shape)}"
    )
    print(f"Forward-pass dry run verified: output shape {tuple(dry_run_out.shape)}")

    # Inverse-frequency class weights computed from the training split only.
    train_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float64)
    class_weights = (train_counts.sum() / (NUM_CLASSES * np.maximum(train_counts, 1))).tolist()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    print(f"Class weights: {dict(zip(CLASS_NAMES, [round(w, 3) for w in class_weights]))}")

    history = TrainingHistory()
    best_state = {"score": -1.0, "acc": None, "loss": float("inf"), "epoch": None, "stage": None, "state_dict": None}

    print("\n--- STAGE 1: classifier head only (backbone frozen, BN stats frozen) ---")
    train_stage(model, train_loader, val_loader, criterion, cfg.epochs_stage1, cfg.lr_stage1, 1, history,
                best_state, device)

    print(f"\n--- STAGE 2: unfreezing last {cfg.unfreeze_blocks} feature stages ---")
    unfreeze_last_n_blocks(model, n_blocks=cfg.unfreeze_blocks)
    train_stage(model, train_loader, val_loader, criterion, cfg.epochs_stage2, cfg.lr_stage2, 2, history,
                best_state, device)

    assert best_state["state_dict"] is not None, "No checkpoint was ever selected."
    model.load_state_dict(best_state["state_dict"])
    model.eval()
    print(
        f"\nSelected checkpoint: stage {best_state['stage']} epoch {best_state['epoch']} "
        f"(val_bal_acc={best_state['score']:.4f}, val_loss={best_state['loss']:.4f})"
    )

    # Test set is evaluated exactly once, after selection (audit #6).
    test_results = evaluate_and_plot_confusion(
        model, test_loader, cfg.images_out / "confusion_matrix.png", device, split_name=cfg.test_name
    )
    test_results["name"] = cfg.test_name

    print("\nFitting out-of-distribution detector on training features…")
    detector = OodDetector.fit(
        collect_features(model, train_eval_loader, device),
        collect_features(model, val_loader, device),
        percentile=cfg.ood_percentile,
    )
    test_flagged = float((detector.distance(collect_features(model, test_loader, device)) > detector.threshold).mean())
    ood_report = {
        "feature_layer": OOD_FEATURE_LAYER,
        "method": f"kNN cosine distance (k={detector.k}) to {detector.bank.shape[0]} training tiles",
        "threshold": detector.threshold,
        "calibration": f"p{cfg.ood_percentile:g} of validation distances",
        "test_flagged_frac": test_flagged,
        "probe": probe_ood(model, detector, cfg.ood_probe_dir, device) if cfg.ood_probe_dir else None,
    }
    print(f"OOD: {json.dumps(ood_report)}")

    generated_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    checkpoint_metadata = {
        "generated_utc": generated_utc,
        "pretrained_backbone": cfg.pretrained,
        "data_dir": cfg.data_dir.as_posix(),
        "test_set": cfg.test_name,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(test_indices),
        "selected_stage": best_state["stage"],
        "selected_epoch": best_state["epoch"],
        "val_balanced_accuracy": best_state["score"],
        "val_loss": best_state["loss"],
        "test_balanced_accuracy": test_results["balanced_accuracy"],
        "seed": cfg.seed,
        "torch": str(torch.__version__),
    }
    model_sha = save_checkpoint(model, cfg.model_out, checkpoint_metadata, ood=detector)
    size_mb = cfg.model_out.stat().st_size / (1024 * 1024)
    print(f"Saved model -> {cfg.model_out} ({size_mb:.2f} MB, sha256 {model_sha[:12]}…)")
    assert size_mb < 25.0, f"Model checkpoint is {size_mb:.2f} MB, exceeds 25MB constraint!"

    # Metrics are written before plotting curves so a plotting failure cannot
    # leave the dashboard without final_metrics.json (audit #16).
    metrics_out = cfg.images_out / "final_metrics.json"
    config_record = {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "generated_utc": generated_utc,
        "class_names": list(CLASS_NAMES),
        "data": {
            "data_dir": cfg.data_dir.as_posix(),
            "test_source": cfg.test_dir.as_posix() if cfg.test_dir else "split of data_dir",
            "grouping": "source chip (<chip>_r<row>c<col>)",
            "class_counts": class_counts,
            "class_weights": dict(zip(CLASS_NAMES, class_weights)),
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(test_indices),
            "split_manifest": manifest_path.as_posix(),
        },
        "selection": {
            "rule": "max validation balanced accuracy, ties broken by min validation loss",
            "stage": best_state["stage"],
            "epoch": best_state["epoch"],
            "val_balanced_accuracy": best_state["score"],
            "val_accuracy": best_state["acc"],
            "val_loss": best_state["loss"],
        },
        "test": test_results,
        "ood": ood_report,
        "history": asdict(history),
        "model": {
            "architecture": ARCHITECTURE,
            "path": cfg.model_out.as_posix(),
            "size_mb": round(size_mb, 3),
            "sha256": model_sha,
            "pretrained_backbone": cfg.pretrained,
        },
        "config": {**config_record, "device_resolved": device},
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
        },
    }
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics summary -> {metrics_out}")

    plot_training_curves(history, cfg.images_out / "training_curves.png", best_epoch=best_state["epoch"])


def parse_args() -> TrainConfig:
    """Parse CLI arguments into a TrainConfig."""
    parser = argparse.ArgumentParser(description="Train satellite flood detector (EfficientNet-B0).")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--test_name", type=str, default=None)
    parser.add_argument("--ood_probe_dir", type=str, default=None)
    parser.add_argument("--model_out", type=str, default="model/flood_detector.pth")
    parser.add_argument("--images_out", type=str, default="images")
    parser.add_argument(
        "--img_size", type=int, default=IMG_SIZE,
        help=f"Accepted for Phase 1 CLI compatibility; must equal flood_core.IMG_SIZE ({IMG_SIZE}).",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--epochs_stage1", type=int, default=5)
    parser.add_argument("--epochs_stage2", type=int, default=5)
    parser.add_argument("--lr_stage1", type=float, default=1e-3)
    parser.add_argument("--lr_stage2", type=float, default=1e-4)
    parser.add_argument("--unfreeze_blocks", type=int, default=2)
    parser.add_argument("--ood_percentile", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    if args.img_size != IMG_SIZE:
        parser.error(f"--img_size must be {IMG_SIZE}; change flood_core.IMG_SIZE to alter resolution everywhere.")
    test_split = 0.0 if args.test_dir else args.test_split
    if not 0 < args.val_split + test_split < 1:
        parser.error("--val_split + --test_split must be between 0 and 1.")

    return TrainConfig(
        data_dir=Path(args.data_dir),
        test_dir=Path(args.test_dir) if args.test_dir else None,
        test_name=args.test_name or ("External held-out set" if args.test_dir else "Held-out test split"),
        ood_probe_dir=Path(args.ood_probe_dir) if args.ood_probe_dir else None,
        model_out=Path(args.model_out),
        images_out=Path(args.images_out),
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2,
        unfreeze_blocks=args.unfreeze_blocks,
        ood_percentile=args.ood_percentile,
        seed=args.seed,
        device=args.device,
        pretrained=args.pretrained,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main(parse_args())
