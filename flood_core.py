"""Single source of truth shared by training, explainability, and serving.

Every value that must agree between `train_pipeline.py`, `grad_cam.py`, and
`app.py` is defined here exactly once and imported everywhere else:

    * class-index mapping (CLASS_NAMES / FLOOD_IDX)            — audit #4
    * input resolution + normalization + resize filter (EVAL_TF) — audit #5
    * model architecture (build_model) and penultimate features (extract_features)
    * out-of-distribution detector (OodDetector)
    * checkpoint schema, integrity sidecar, and loader

Checkpoints written by `save_checkpoint` embed the class order, input size, and
normalization they were trained with; `load_checkpoint` refuses to serve a
checkpoint whose embedded values disagree with this module, so train/serve
drift fails loudly instead of silently inverting predictions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

# --------------------------------------------------------------------------
# Shared constants
# --------------------------------------------------------------------------

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Explicit index mapping (audit #4). This is deliberately NOT alphabetical:
# torchvision's ImageFolder sorts folder names, which would give flooded=0.
# Never infer labels from directory order — always index through this list.
CLASS_NAMES = ["not_flooded", "flooded"]
FLOOD_IDX = CLASS_NAMES.index("flooded")
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

ARCHITECTURE = "efficientnet_b0"
BACKBONE_OUT_FEATURES = 1280
CLASSIFIER_DROPOUT = 0.3
CAM_TARGET_LAYER = "features[-1]"
OOD_FEATURE_LAYER = "avgpool (1280-d penultimate features)"

DEFAULT_MODEL_PATH = Path("model/flood_detector.pth")
CHECKPOINT_FORMAT_VERSION = 3
_BASE_KEYS = frozenset(
    {"format_version", "architecture", "state_dict", "class_names", "img_size", "normalization", "metadata"}
)
CHECKPOINT_KEYS_BY_VERSION = {2: _BASE_KEYS, 3: _BASE_KEYS | {"ood"}}
CHECKPOINT_KEYS = CHECKPOINT_KEYS_BY_VERSION[CHECKPOINT_FORMAT_VERSION]

# --------------------------------------------------------------------------
# Preprocessing (audit #5)
# --------------------------------------------------------------------------

# Used for validation, test, Grad-CAM, and serving. Interpolation and antialias
# are pinned explicitly so a torchvision default change cannot introduce skew,
# and so no caller falls back to PIL.Image.resize (which defaults to BICUBIC).
EVAL_TF = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def preprocess(img: Image.Image) -> torch.Tensor:
    """Convert any PIL image into the model's [1, 3, IMG_SIZE, IMG_SIZE] input tensor.

    Args:
        img: Source image in any mode or size.

    Returns:
        Normalized batch-of-one tensor produced by EVAL_TF.
    """
    return EVAL_TF(img.convert("RGB")).unsqueeze(0)


# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------

def build_model(pretrained: bool = False) -> nn.Module:
    """Build EfficientNet-B0 with the project's 2-class classifier head.

    Args:
        pretrained: Load ImageNet-1K weights into the backbone (training only;
            serving always loads a trained checkpoint on top of random init).

    Returns:
        The model on CPU, all parameters trainable.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    if in_features != BACKBONE_OUT_FEATURES:
        raise RuntimeError(f"Expected {BACKBONE_OUT_FEATURES} backbone features, got {in_features}")
    model.classifier = nn.Sequential(
        nn.Dropout(p=CLASSIFIER_DROPOUT, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def cam_target_layer(model: nn.Module) -> nn.Module:
    """Return the Grad-CAM target: the last spatial feature map (1280 x 7 x 7)."""
    return model.features[-1]


def extract_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Penultimate features: global-average-pooled `features` output, [N, 1280].

    Must match what `model.avgpool` emits inside `model.forward`, because the
    serving path captures features with a hook on that module.
    """
    return torch.flatten(model.avgpool(model.features(x)), 1)


# --------------------------------------------------------------------------
# Out-of-distribution detection
# --------------------------------------------------------------------------

OOD_METHOD = "knn_cosine"


@dataclass(frozen=True)
class OodDetector:
    """k-nearest-neighbour cosine distance to the training feature bank.

    Score = 1 - mean cosine similarity to the k most similar training tiles.
    The threshold is the given percentile of scores on the validation split
    (which is not in the bank), so about (100 - percentile)% of in-distribution
    tiles are expected to be flagged.

    Why not Mahalanobis: measured on this model's real features, a Gaussian
    (PCA + Ledoit-Wolf) Mahalanobis score separated synthetic and junk inputs
    from real tiles with AUROC 0.22 — worse than chance, because those inputs
    land near the feature mean. kNN cosine reached AUROC 0.99 on the same data.

    Attributes:
        bank: [N, F] L2-normalised training features.
        k: Neighbours averaged per query.
        threshold: Score above which an input is flagged out of distribution.
        percentile: Validation percentile the threshold was taken from.
    """

    bank: torch.Tensor
    k: int
    threshold: float
    percentile: float

    @classmethod
    def fit(
        cls, train_features: np.ndarray, calib_features: np.ndarray, k: int = 10, percentile: float = 99.0
    ) -> OodDetector:
        """Build the feature bank from training tiles and calibrate on held-out tiles.

        Args:
            train_features: [N, F] features of training tiles (no augmentation).
            calib_features: [M, F] features of in-distribution validation tiles (not in the bank).
            k: Neighbours averaged per query (capped by bank size).
            percentile: Validation score percentile used as the threshold.

        Returns:
            Fitted detector.
        """
        bank = torch.nn.functional.normalize(torch.as_tensor(train_features, dtype=torch.float32), dim=1)
        detector = cls(bank=bank, k=int(min(k, bank.shape[0])), threshold=float("inf"), percentile=float(percentile))
        threshold = float(np.percentile(detector.distance(calib_features), percentile))
        return cls(bank=bank, k=detector.k, threshold=threshold, percentile=float(percentile))

    def distance(self, features: np.ndarray | torch.Tensor) -> np.ndarray:
        """kNN cosine distance for [F] or [N, F] features; returns [N] in [0, 2]."""
        f = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        if f.ndim == 1:
            f = f.unsqueeze(0)
        sims = torch.nn.functional.normalize(f, dim=1) @ self.bank.T
        return (1.0 - sims.topk(self.k, dim=1).values.mean(dim=1)).numpy()

    def to_payload(self) -> dict:
        return {"method": OOD_METHOD, "bank": self.bank, "k": self.k, "threshold": self.threshold,
                "percentile": self.percentile}

    @classmethod
    def from_payload(cls, payload: dict) -> OodDetector:
        if payload.get("method") != OOD_METHOD:
            raise CheckpointSchemaError(f"Unsupported OOD method {payload.get('method')!r} (expected {OOD_METHOD!r})")
        return cls(bank=payload["bank"], k=int(payload["k"]), threshold=float(payload["threshold"]),
                   percentile=float(payload["percentile"]))


# --------------------------------------------------------------------------
# Checkpoint schema, integrity, loading
# --------------------------------------------------------------------------

class CheckpointError(RuntimeError):
    """Base class for checkpoint problems that must block serving."""


class CheckpointIntegrityError(CheckpointError):
    """The checkpoint bytes do not match their recorded SHA-256."""


class CheckpointSchemaError(CheckpointError):
    """The checkpoint was trained with a different class order, size, or normalization."""


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A model ready for inference plus the provenance facts the UI must display.

    Attributes:
        model: EfficientNet-B0 in eval mode on CPU.
        path: File the weights were read from.
        sha256: SHA-256 of the checkpoint file bytes.
        format: "v3"/"v2" for schema-carrying checkpoints, "legacy" for bare state_dicts.
        integrity_verified: True only if a .sha256 sidecar exists and matched.
        schema_verified: True only if embedded class order/size/normalization matched.
        metadata: Training metadata embedded in the checkpoint (empty for legacy).
        ood: Out-of-distribution detector (v3 checkpoints only).
    """

    model: nn.Module
    path: Path
    sha256: str
    format: str
    integrity_verified: bool
    schema_verified: bool
    metadata: dict = field(default_factory=dict)
    ood: OodDetector | None = None


def sha256_file(path: Path) -> str:
    """Stream a file through SHA-256 without loading it fully into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_path(checkpoint_path: Path) -> Path:
    """Path of the `sha256sum`-compatible integrity file next to a checkpoint."""
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")


def save_checkpoint(model: nn.Module, path: Path, metadata: dict, ood: OodDetector | None = None) -> str:
    """Save a v3 checkpoint (weights + schema + OOD detector) and its SHA-256 sidecar.

    Only tensors, str, int, float, bool, None, lists, and dicts are stored, so
    the file loads under `torch.load(weights_only=True)`.

    Args:
        model: Trained model (any device; tensors are moved to CPU).
        path: Destination .pth path.
        metadata: JSON-like training provenance (dataset size, epochs, etc.).
        ood: Fitted OodDetector, or None if not fitted.

    Returns:
        The SHA-256 hex digest of the written file.
    """
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": ARCHITECTURE,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "class_names": list(CLASS_NAMES),
        "img_size": IMG_SIZE,
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        # JSON round-trip strips str subclasses (e.g. torch.__version__ is a TorchVersion),
        # which weights_only=True refuses to unpickle. Non-JSON values fail here, at save time.
        "metadata": json.loads(json.dumps(metadata)),
        "ood": ood.to_payload() if ood is not None else None,
    }
    assert set(payload) == CHECKPOINT_KEYS, "save_checkpoint payload drifted from CHECKPOINT_KEYS"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    digest = sha256_file(path)
    sidecar_path(path).write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def _validate_schema(payload: dict) -> None:
    """Raise CheckpointSchemaError if a versioned checkpoint disagrees with this module."""
    version = payload.get("format_version")
    if version not in CHECKPOINT_KEYS_BY_VERSION:
        raise CheckpointSchemaError(
            f"Unsupported checkpoint format_version {version} (supported: {sorted(CHECKPOINT_KEYS_BY_VERSION)})"
        )
    missing = CHECKPOINT_KEYS_BY_VERSION[version] - set(payload)
    if missing:
        raise CheckpointSchemaError(f"Checkpoint is missing keys: {sorted(missing)}")
    expected = {
        "architecture": ARCHITECTURE,
        "class_names": list(CLASS_NAMES),
        "img_size": IMG_SIZE,
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
    }
    for key, value in expected.items():
        if payload[key] != value:
            raise CheckpointSchemaError(f"Checkpoint {key}={payload[key]!r} does not match flood_core {value!r}")


def load_checkpoint(path: Path = DEFAULT_MODEL_PATH) -> LoadedCheckpoint:
    """Load a checkpoint for CPU inference, verifying integrity and schema.

    Accepts v3 (with OOD detector), v2, and the legacy Phase 1 bare
    `state_dict`. Legacy files load, but are reported as `schema_verified=False`
    because their class order cannot be proven.

    Args:
        path: Checkpoint path.

    Returns:
        LoadedCheckpoint with the eval-mode model and provenance flags.

    Raises:
        FileNotFoundError: No file at `path`.
        CheckpointIntegrityError: A sidecar exists and its digest does not match.
        CheckpointSchemaError: Embedded metadata disagrees with this module.
        RuntimeError: state_dict keys/shapes do not fit the architecture.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")

    digest = sha256_file(path)
    integrity_verified = False
    sidecar = sidecar_path(path)
    if sidecar.exists():
        recorded = sidecar.read_text(encoding="utf-8").split()[0].lower()
        if recorded != digest:
            raise CheckpointIntegrityError(
                f"{path.name} SHA-256 {digest[:12]}… does not match sidecar {recorded[:12]}…"
            )
        integrity_verified = True

    payload = torch.load(path, map_location=torch.device("cpu"), weights_only=True)

    ood = None
    if isinstance(payload, dict) and "format_version" in payload:
        _validate_schema(payload)
        state_dict, schema_verified, metadata = payload["state_dict"], True, payload["metadata"]
        fmt = f"v{payload['format_version']}"
        if payload.get("ood") is not None:
            ood = OodDetector.from_payload(payload["ood"])
    else:
        state_dict, fmt, schema_verified, metadata = payload, "legacy", False, {}

    model = build_model(pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return LoadedCheckpoint(
        model=model,
        path=path,
        sha256=digest,
        format=fmt,
        integrity_verified=integrity_verified,
        schema_verified=schema_verified,
        metadata=metadata,
        ood=ood,
    )
