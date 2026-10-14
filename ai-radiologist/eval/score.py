"""Score a labelled image set once and write the raw outputs to JSON.

Stage one of the evaluation. Kept separate from `evaluate.py` so the GPU pass runs once:
scoring 4,128 images takes about half an hour, while the metrics on top of it take under a
second and are worth re-running whenever a definition changes.

Ground truth comes from NIH ChestX-ray14's `Data_Entry_2017.csv`, joined on the image id
embedded in each filename (`..._00001989_001.png` -> `00001989_001.png`). Images whose id
is absent from the metadata are written out with an empty label list and are skipped by the
metrics, rather than being silently treated as negatives.

Usage
-----
    python score.py --images ../demo_images --labels /path/to/Data_Entry_2017.csv

`Data_Entry_2017.csv` is distributed by NIH with the dataset and is not redistributed here;
see the README for the link.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

ID_RE = re.compile(r"(\d{8}_\d{3})\.png$")


def load_labels(csv_path: Path) -> dict[str, list[str]]:
    """Map image id -> list of findings. 'No Finding' becomes an empty list."""
    out: dict[str, list[str]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["Image Index"]] = sorted(
                x for x in row["Finding Labels"].split("|") if x and x != "No Finding"
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", type=Path, default=HERE.parent / "demo_images")
    ap.add_argument("--labels", type=Path, required=True,
                    help="path to NIH Data_Entry_2017.csv")
    ap.add_argument("--out", type=Path, default=HERE / "scores.json")
    args = ap.parse_args()

    from image_processor import process_xray
    from model_handler import TARGET_DISEASES, XRayModel

    labels = load_labels(args.labels)
    files = sorted(args.images.rglob("*.png"))
    if not files:
        raise SystemExit(f"no images under {args.images}")
    print(f"{len(files):,} images, {len(labels):,} labelled ids in metadata")

    model = XRayModel()
    model.load_model()

    records, unmatched, t0 = [], 0, time.time()
    for k, p in enumerate(files, 1):
        m = ID_RE.search(p.name)
        if not m:
            unmatched += 1
            continue
        img_id = m.group(1) + ".png"
        if img_id not in labels:
            unmatched += 1
        res = model.predict_diseases(process_xray(str(p)))
        records.append({
            "file": p.name,
            "folder": p.parent.name,
            "image_id": img_id,
            "labels": labels.get(img_id, []),
            "scores": {d: round(float(v["probability"]), 6)
                       for d, v in res.items()},
        })
        if k % 250 == 0:
            el = time.time() - t0
            print(f"  {k:,}/{len(files):,}  {el/60:.1f} min elapsed, "
                  f"~{el/k*(len(files)-k)/60:.1f} min left", flush=True)

    args.out.write_text(json.dumps({
        "model": model.model_name,
        "targets": TARGET_DISEASES,
        "ground_truth": "NIH ChestX-ray14 Data_Entry_2017.csv",
        "n": len(records),
        "n_unmatched_ids": unmatched,
        "records": records,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}  ({len(records):,} records, "
          f"{unmatched} unmatched ids) in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
