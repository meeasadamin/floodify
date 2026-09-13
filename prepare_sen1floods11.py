"""Download Sen1Floods11 hand-labeled Sentinel-2 chips and build a tile-level flood dataset.

Source: Sen1Floods11 (Bonafilia et al., CVPR Workshops 2020), public bucket
gs://sen1floods11, v1.1 hand-labeled split: 446 Sentinel-2 chips (512 x 512,
13 bands) from 11 flood events, each with a hand-drawn water mask (LabelHand:
-1 no data, 0 dry, 1 water) and a JRC permanent-water mask (JRCWaterHand).

Labelling rule (tile level, 256 x 256 = each chip split into a 2 x 2 grid):
    valid_frac  = pixels with LabelHand != -1
    flood_frac  = pixels that are water in LabelHand AND not permanent water in JRC,
                  divided by valid pixels
    discard     if valid_frac < VALID_MIN            (cloud / no-data dominated)
    flooded     if flood_frac >= FLOOD_MIN
    not_flooded if flood_frac <= CLEAR_MAX            (rivers / lakes stay "not flooded")
    discard     otherwise                            (ambiguous, a few % flood water)

Rendering: true colour from B4/B3/B2, reflectance = DN / 10000, linear stretch
0 - REFLECTANCE_MAX to 0 - 255. This is the RGB input the model sees.

Split: the Pakistan event is held out entirely as the test set; the other 10
events form the training pool (train_pipeline.py splits it into train/val,
grouped by chip so tiles of one chip never straddle train and val).

Outputs:
    <out_dir>/train/<class>/<Event>_<chip>_r<row>c<col>.png
    <out_dir>/test_pakistan/<class>/...
    <out_dir>/tiles_manifest.json          per-tile statistics and decisions
    images/dataset_summary.json            counts + rule (tracked in git for provenance)

Usage:
    python prepare_sen1floods11.py --raw_dir data_raw/sen1floods11 --out_dir data_real
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from flood_core import CLASS_NAMES, FLOOD_IDX

BUCKET = "sen1floods11"
PREFIX = "v1.1/data/flood_events/HandLabeled"
LAYERS = {"s2": "S2Hand", "label": "LabelHand", "jrc": "JRCWaterHand"}
HOLDOUT_EVENT = "Pakistan"

TILE = 256
VALID_MIN = 0.80
FLOOD_MIN = 0.10
CLEAR_MAX = 0.01
REFLECTANCE_MAX = 0.30
RGB_BANDS = (3, 2, 1)  # S2Hand band order is B1..B12 incl. B8A: index 3=B4 red, 2=B3 green, 1=B2 blue

FLOODED = CLASS_NAMES[FLOOD_IDX]
NOT_FLOODED = next(name for name in CLASS_NAMES if name != FLOODED)


def list_chip_ids() -> list[str]:
    """List chip ids ("<Event>_<number>") from the public bucket's JSON API."""
    ids, token = [], None
    while True:
        query = {"prefix": f"{PREFIX}/{LAYERS['s2']}/", "fields": "items(name),nextPageToken", "maxResults": "1000"}
        if token:
            query["pageToken"] = token
        url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            page = json.load(resp)
        ids += [Path(item["name"]).stem.removesuffix("_S2Hand") for item in page.get("items", [])]
        token = page.get("nextPageToken")
        if not token:
            return sorted(ids)


def download(url: str, dest: Path, retries: int = 5) -> None:
    """Download to dest atomically, skipping files that already exist."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return
        except OSError:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def fetch_chip(chip_id: str, raw_dir: Path) -> None:
    for layer in LAYERS.values():
        name = f"{chip_id}_{layer}.tif"
        download(f"https://storage.googleapis.com/{BUCKET}/{PREFIX}/{layer}/{name}", raw_dir / layer / name)


def to_rgb(s2: np.ndarray) -> np.ndarray:
    """13-band DN array -> uint8 true-colour image."""
    rgb = s2[list(RGB_BANDS)].astype(np.float32) / 10000.0
    rgb = np.clip(rgb / REFLECTANCE_MAX, 0.0, 1.0)
    return np.uint8(np.round(np.moveaxis(rgb, 0, -1) * 255))


def process_chip(chip_id: str, raw_dir: Path, out_dir: Path) -> list[dict]:
    """Split one chip into tiles, label them, and write kept tiles as PNG."""
    s2 = tifffile.imread(raw_dir / LAYERS["s2"] / f"{chip_id}_S2Hand.tif")
    label = tifffile.imread(raw_dir / LAYERS["label"] / f"{chip_id}_LabelHand.tif")
    jrc = tifffile.imread(raw_dir / LAYERS["jrc"] / f"{chip_id}_JRCWaterHand.tif")
    event = chip_id.split("_")[0]
    split = "test_pakistan" if event == HOLDOUT_EVENT else "train"
    rgb = to_rgb(s2)

    records = []
    rows, cols = label.shape[0] // TILE, label.shape[1] // TILE
    for r in range(rows):
        for c in range(cols):
            window = (slice(r * TILE, (r + 1) * TILE), slice(c * TILE, (c + 1) * TILE))
            lab, perm = label[window], jrc[window]
            valid = lab != -1
            n_valid = int(valid.sum())
            valid_frac = n_valid / lab.size
            flood_frac = float(((lab == 1) & (perm == 0)).sum() / n_valid) if n_valid else 0.0
            perm_frac = float(((perm == 1) & valid).sum() / n_valid) if n_valid else 0.0

            if valid_frac < VALID_MIN:
                decision = "discard_nodata"
            elif flood_frac >= FLOOD_MIN:
                decision = FLOODED
            elif flood_frac <= CLEAR_MAX:
                decision = NOT_FLOODED
            else:
                decision = "discard_ambiguous"

            tile_name = f"{chip_id}_r{r}c{c}.png"
            if decision in CLASS_NAMES:
                dest = out_dir / split / decision / tile_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb[window]).save(dest)
            records.append({
                "tile": tile_name, "event": event, "chip": chip_id, "split": split, "decision": decision,
                "valid_frac": round(valid_frac, 4), "flood_frac": round(flood_frac, 4),
                "permanent_water_frac": round(perm_frac, 4),
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Sen1Floods11 tile dataset.")
    parser.add_argument("--raw_dir", type=str, default="data_raw/sen1floods11")
    parser.add_argument("--out_dir", type=str, default="data_real")
    parser.add_argument("--summary_out", type=str, default="images/dataset_summary.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)

    chip_ids = list_chip_ids()
    print(f"{len(chip_ids)} hand-labeled chips listed")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_chip, cid, raw_dir): cid for cid in chip_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if i % 50 == 0 or i == len(chip_ids):
                print(f"downloaded {i}/{len(chip_ids)}", flush=True)

    records = []
    for cid in chip_ids:
        records += process_chip(cid, raw_dir, out_dir)

    (out_dir / "tiles_manifest.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    counts = Counter((r["split"], r["decision"]) for r in records)
    per_event = Counter((r["event"], r["decision"]) for r in records if r["decision"] in CLASS_NAMES)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"gs://{BUCKET}/{PREFIX} (Sen1Floods11 v1.1 hand-labeled, Sentinel-2)",
        "n_chips": len(chip_ids),
        "tile_size": TILE,
        "rule": {
            "valid_min": VALID_MIN, "flood_min": FLOOD_MIN, "clear_max": CLEAR_MAX,
            "flood_pixels": "LabelHand == 1 and JRCWaterHand == 0",
            "rgb": f"B4/B3/B2 reflectance, linear stretch 0-{REFLECTANCE_MAX}",
        },
        "holdout_event": HOLDOUT_EVENT,
        "counts": {f"{split}/{decision}": n for (split, decision), n in sorted(counts.items())},
        "per_event": {f"{event}/{decision}": n for (event, decision), n in sorted(per_event.items())},
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
