"""Cross-file consistency and regression tests for the Phase 2 audit fixes.

Each test names the audit finding it guards. Run: pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import InterpolationMode, Resize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import flood_core  # noqa: E402
import grad_cam  # noqa: E402
import train_pipeline  # noqa: E402


# ---------------------------------------------------------------- #4 class order

def test_class_order_is_explicit_not_alphabetical():
    assert flood_core.CLASS_NAMES == ["not_flooded", "flooded"]
    assert flood_core.CLASS_NAMES != sorted(flood_core.CLASS_NAMES)  # ImageFolder order would differ
    assert flood_core.FLOOD_IDX == 1


def test_dataset_labels_follow_class_names(tmp_path):
    for name in flood_core.CLASS_NAMES:
        (tmp_path / name).mkdir()
        Image.new("RGB", (32, 32)).save(tmp_path / name / f"{name}.jpg")
    ds = train_pipeline.SatelliteFloodDataset(tmp_path, transform=flood_core.EVAL_TF)
    assert {p.parent.name: label for p, label in ds.samples} == {
        name: i for i, name in enumerate(flood_core.CLASS_NAMES)
    }


# ---------------------------------------------------------------- #5 preprocessing

def test_train_pipeline_eval_transform_is_flood_core_eval_tf():
    _, eval_tf = train_pipeline.build_transforms()
    assert eval_tf is flood_core.EVAL_TF


def test_eval_tf_pins_bilinear_antialias():
    resize = next(t for t in flood_core.EVAL_TF.transforms if isinstance(t, Resize))
    assert resize.interpolation == InterpolationMode.BILINEAR
    assert resize.antialias is True
    assert tuple(resize.size) == (flood_core.IMG_SIZE, flood_core.IMG_SIZE)


def test_preprocess_matches_eval_tf_for_any_mode():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 500, 4), dtype=np.uint8), "RGBA")
    expected = flood_core.EVAL_TF(img.convert("RGB")).unsqueeze(0)
    assert torch.equal(flood_core.preprocess(img), expected)


# ---------------------------------------------------------------- checkpoint schema

@pytest.fixture()
def tiny_model():
    torch.manual_seed(0)
    return flood_core.build_model(pretrained=False).eval()


def test_checkpoint_roundtrip_current_format(tmp_path, tiny_model):
    path = tmp_path / "m.pth"
    digest = flood_core.save_checkpoint(tiny_model, path, {"note": "test"})
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw) == flood_core.CHECKPOINT_KEYS
    loaded = flood_core.load_checkpoint(path)
    assert loaded.format == f"v{flood_core.CHECKPOINT_FORMAT_VERSION}"
    assert loaded.schema_verified and loaded.integrity_verified and loaded.ood is None
    assert loaded.sha256 == digest and loaded.metadata == {"note": "test"}
    x = torch.randn(1, 3, flood_core.IMG_SIZE, flood_core.IMG_SIZE)
    with torch.no_grad():
        assert torch.allclose(loaded.model(x), tiny_model(x))


def test_checkpoint_metadata_with_torch_version_loads_weights_only(tmp_path, tiny_model):
    path = tmp_path / "m.pth"
    flood_core.save_checkpoint(tiny_model, path, {"torch": torch.__version__})  # TorchVersion, a str subclass
    assert flood_core.load_checkpoint(path).metadata == {"torch": str(torch.__version__)}


def test_checkpoint_tamper_detected(tmp_path, tiny_model):
    path = tmp_path / "m.pth"
    flood_core.save_checkpoint(tiny_model, path, {})
    flood_core.sidecar_path(path).write_text("0" * 64 + "  m.pth\n")
    with pytest.raises(flood_core.CheckpointIntegrityError):
        flood_core.load_checkpoint(path)


def test_checkpoint_class_order_drift_rejected(tmp_path, tiny_model):
    path = tmp_path / "m.pth"
    flood_core.save_checkpoint(tiny_model, path, {})
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["class_names"] = sorted(payload["class_names"])
    torch.save(payload, path)
    flood_core.sidecar_path(path).unlink()
    with pytest.raises(flood_core.CheckpointSchemaError):
        flood_core.load_checkpoint(path)


def test_legacy_state_dict_loads_unverified(tmp_path, tiny_model):
    path = tmp_path / "legacy.pth"
    torch.save(tiny_model.state_dict(), path)
    loaded = flood_core.load_checkpoint(path)
    assert loaded.format == "legacy" and not loaded.schema_verified and not loaded.integrity_verified


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        flood_core.load_checkpoint(tmp_path / "absent.pth")


# ---------------------------------------------------------------- #1 #2 #3 #6 training

def test_selection_breaks_ties_on_val_loss():
    best = {"score": 1.0, "loss": 0.30}
    assert train_pipeline.is_better(1.0, 0.10, best)
    assert not train_pipeline.is_better(1.0, 0.30, best)
    assert not train_pipeline.is_better(0.99, 0.01, best)


def test_frozen_batchnorm_stats_do_not_drift():
    model = train_pipeline.build_model(pretrained=False, device="cpu")
    frozen_bn = model.features[1][0].block[0][1]
    assert isinstance(frozen_bn, nn.BatchNorm2d) and not frozen_bn.weight.requires_grad
    before = frozen_bn.running_mean.clone()
    model.train()
    train_pipeline.freeze_bn_stats(model)
    with torch.no_grad():
        model(torch.randn(4, 3, flood_core.IMG_SIZE, flood_core.IMG_SIZE) * 5)
    assert torch.equal(frozen_bn.running_mean, before)
    assert model.classifier[1].training


def test_train_augmentation_has_no_fill_rotation():
    train_tf, _ = train_pipeline.build_transforms()
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    for _ in range(20):
        x = train_tf(img)
        assert x.shape == (3, flood_core.IMG_SIZE, flood_core.IMG_SIZE)
        # A white tile normalises to > 0 even after brightness jitter; black fill pixels normalise to ~ -2.
        corners = x[:, [0, 0, -1, -1], [0, -1, 0, -1]]
        assert (corners > 0).all()


def test_grouped_split_disjoint_balanced_and_chip_grouped():
    # 50 chips x 4 tiles; chip class alternates 60/40.
    groups = [f"Event_{c}" for c in range(50) for _ in range(4)]
    labels = [int(c % 5 < 2) for c in range(50) for _ in range(4)]
    splits = train_pipeline.grouped_stratified_split(labels, groups, 0.15, 0.15, seed=1)
    all_idx = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_idx) == list(range(200))
    chips = {name: {groups[i] for i in idx} for name, idx in splits.items()}
    assert not chips["train"] & chips["val"] and not chips["train"] & chips["test"] and not chips["val"] & chips["test"]
    for idx in (splits["val"], splits["test"]):
        share = sum(labels[i] for i in idx) / len(idx)
        assert 0.2 <= share <= 0.6


def test_tile_group_parses_chip_ids():
    assert train_pipeline.tile_group(Path("Pakistan_94095_r1c0.png")) == "Pakistan_94095"
    assert train_pipeline.tile_group(Path("flooded_0001.png")) == "flooded_0001"


# ---------------------------------------------------------------- OOD detector

def test_ood_detector_flags_far_inputs_and_roundtrips(tmp_path, tiny_model):
    rng = np.random.default_rng(0)
    # Two tight clusters of "real" tiles; OOD inputs point in unrelated directions.
    centers = rng.normal(0, 1, (2, 1280))
    train = (np.repeat(centers, 150, axis=0) + rng.normal(0, 0.3, (300, 1280))).astype(np.float32)
    calib = (np.repeat(centers, 50, axis=0) + rng.normal(0, 0.3, (100, 1280))).astype(np.float32)
    det = flood_core.OodDetector.fit(train, calib, k=10, percentile=99.0)
    assert (det.distance(calib) > det.threshold).mean() <= 0.02
    assert (det.distance(rng.normal(0, 1, (20, 1280))) > det.threshold).all()

    path = tmp_path / "m.pth"
    flood_core.save_checkpoint(tiny_model, path, {}, ood=det)
    loaded = flood_core.load_checkpoint(path)
    assert loaded.format == "v3" and loaded.ood is not None
    assert np.allclose(loaded.ood.distance(calib), det.distance(calib), atol=1e-4)


def test_v2_checkpoint_still_loads_without_ood(tmp_path, tiny_model):
    path = tmp_path / "v2.pth"
    flood_core.save_checkpoint(tiny_model, path, {})
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload.pop("ood")
    payload["format_version"] = 2
    torch.save(payload, path)
    flood_core.sidecar_path(path).unlink()
    loaded = flood_core.load_checkpoint(path)
    assert loaded.format == "v2" and loaded.schema_verified and loaded.ood is None


def test_serving_features_match_training_features(tiny_model):
    img = Image.fromarray(np.random.default_rng(3).integers(0, 255, (128, 128, 3), dtype=np.uint8))
    exp = grad_cam.explain_image(tiny_model, img)
    with torch.no_grad():
        direct = flood_core.extract_features(tiny_model, flood_core.preprocess(img))[0].numpy()
    assert exp.features.shape == (flood_core.BACKBONE_OUT_FEATURES,)
    assert np.allclose(exp.features, direct, atol=1e-5)


# ---------------------------------------------------------------- #7 #11 #19 explainability

def test_explain_image_targets_flood_and_keeps_aspect(tiny_model):
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (180, 320, 3), dtype=np.uint8))
    exp = grad_cam.explain_image(tiny_model, img)
    assert pytest.approx(sum(exp.probs), abs=1e-5) == 1.0
    assert exp.flood_prob == exp.probs[flood_core.FLOOD_IDX]
    assert exp.cam.shape == (flood_core.IMG_SIZE, flood_core.IMG_SIZE)
    with torch.no_grad():
        direct = torch.softmax(tiny_model(flood_core.preprocess(img)), dim=1)[0].tolist()
    assert np.allclose(exp.probs, direct, atol=1e-5)
    assert all(p.grad is None for p in tiny_model.parameters())
    source, overlay = grad_cam.render_evidence(img, exp)
    assert source.shape == overlay.shape == (180, 320, 3)


def test_overlay_is_invisible_at_zero_strength():
    rgb = np.full((50, 80, 3), 120, dtype=np.uint8)
    cam = np.ones((7, 7), dtype=np.float32)
    assert np.array_equal(grad_cam.overlay_heatmap(rgb, cam, strength=0.0), rgb)


def test_sample_paths_include_jpeg(tmp_path):
    for name in flood_core.CLASS_NAMES:
        (tmp_path / name).mkdir()
        Image.new("RGB", (16, 16)).save(tmp_path / name / "a.jpg")
        Image.new("RGB", (16, 16)).save(tmp_path / name / "b.jpeg")
    assert len(grad_cam.collect_sample_paths(tmp_path, 4)) == 4


# ---------------------------------------------------------------- app (#8 #9 #10 #13 smoke)

def _app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _markdown_text(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_app_renders_without_exceptions():
    at = _app()
    assert len(at.tabs) == 4


def test_css_keeps_sidebar_reopen_control_visible():
    import re

    source = (ROOT / "app.py").read_text(encoding="utf-8")
    hidden_rules = re.findall(r"([^{}]+)\{display:none\}", source)
    hidden = ",".join(hidden_rules)
    # stExpandSidebarButton lives inside stHeader > stToolbar; hiding either strands a collapsed sidebar.
    for testid in ("stToolbar", "stHeader", "stExpandSidebarButton", "stSidebarCollapseButton"):
        assert f'[data-testid="{testid}"]' not in hidden, testid


@pytest.mark.skipif(not (ROOT / "model" / "flood_detector.pth").exists(), reason="no checkpoint on disk")
def test_app_reference_tile_survives_other_widget_reruns():
    at = _app()
    at.button(key="tile_TILE-01").click().run()
    assert not at.exception, [e.value for e in at.exception]
    text = _markdown_text(at)
    assert "Assessment" in text and "Hand-labeled ground truth: Flooded" in text
    assert "Input familiarity" in text

    confirm = next(b for b in at.button if b.label == "Confirmed")
    confirm.click().run()  # a rerun from an unrelated widget must keep TILE-01 active (audit #10)
    assert not at.exception
    assert at.session_state["active"]["name"].startswith("TILE-01")
    assert at.session_state["review_log"]


def _shipped_checkpoint_has_ood() -> bool:
    path = ROOT / "model" / "flood_detector.pth"
    try:
        return path.exists() and flood_core.load_checkpoint(path).ood is not None
    except flood_core.CheckpointError:
        return False


@pytest.mark.skipif(not _shipped_checkpoint_has_ood(), reason="shipped checkpoint has no OOD detector")
def test_shipped_model_flags_synthetic_and_junk_inputs():
    loaded = flood_core.load_checkpoint(ROOT / "model" / "flood_detector.pth")

    def distance(img: Image.Image) -> float:
        return float(loaded.ood.distance(grad_cam.explain_image(loaded.model, img).features)[0])

    with Image.open(ROOT / "samples" / "synthetic_ood_demo.jpg") as img:
        assert distance(img) > loaded.ood.threshold
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))
    assert distance(noise) > loaded.ood.threshold


@pytest.mark.skipif(not (ROOT / "model" / "flood_detector.pth").exists(), reason="no checkpoint on disk")
def test_app_corrupt_image_shows_error_not_traceback():
    at = _app()
    at.session_state["active"] = {"name": "broken.png", "bytes": b"not an image", "source": "upload", "truth": None}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("could not be decoded" in e.value for e in at.error)
