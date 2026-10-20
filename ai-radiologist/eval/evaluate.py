"""Evaluate the screening model against NIH ChestX-ray14 ground truth.

Why this exists
---------------
The demo ships five hardcoded decision thresholds. Nothing in the repository said where
they came from, so there was no way to check them or to regenerate them for a different
operating point. This script closes that: it scores a labelled image set, joins the NIH
ground-truth labels, and reports what each threshold actually buys.

Ground truth is `Data_Entry_2017.csv` from NIH ChestX-ray14, joined on the twelve-character
image id in each filename (for example `00001989_001`). The demo folders were assigned from
those labels rather than from model output, which is what makes the join an evaluation
rather than a comparison of the model with itself.

Two stages, deliberately separate:

    score.py      one GPU pass over the images, writes raw per-image scores to JSON
    evaluate.py   reads that JSON and computes metrics

so the expensive pass runs once and the analysis can be re-run, audited, or changed
without a GPU.

Metrics
-------
AUROC         threshold-free ranking quality. Reported because the model's raw outputs sit
              in a narrow band around 0.5, which makes any single number read off them
              misleading; AUROC is invariant to that.
AUPRC         the imbalance-aware companion. With prevalences from 2.6% to 13.2% here,
              AUROC alone flatters a screening model, so the random-baseline AUPRC (equal
              to prevalence) is reported alongside.
Youden's J    J = sensitivity + specificity - 1, maximised over the observed score grid.
              One threshold per finding, because a shared 0.5 is wrong when prevalence
              varies by a factor of five across findings.

Usage
-----
    python score.py                 # writes eval/scores.json
    python evaluate.py              # writes eval/metrics.json and prints the table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SCORES = HERE / "scores.json"
DEFAULT_OUT = HERE / "metrics.json"


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC. Ties get averaged ranks, so they contribute 0.5 as they should."""
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float(
        (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    )


def auprc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Average precision by the step-interpolated definition sklearn uses."""
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    order = s.argsort()[::-1]
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / max(1, y.sum()))


def best_youden(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float, float, float]:
    """Threshold maximising sensitivity + specificity - 1, and its operating point.

    Evaluated on the observed score grid rather than a fixed sweep, so the chosen cut is
    always one the data actually supports. Ties break toward the lower threshold, which
    favours sensitivity - the right bias for screening.
    """
    grid = np.unique(np.concatenate([pos, neg]))
    sens = (pos[None, :] >= grid[:, None]).mean(axis=1)
    spec = (neg[None, :] < grid[:, None]).mean(axis=1)
    j = sens + spec - 1.0
    k = int(np.argmax(j))
    return float(grid[k]), float(j[k]), float(sens[k]), float(spec[k])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    blob = json.loads(args.scores.read_text(encoding="utf-8"))
    records, targets = blob["records"], blob["targets"]

    # The shipped thresholds, read from the module they are defined in rather than copied,
    # so this cannot drift from what the demo actually applies.
    import sys

    sys.path.insert(0, str(HERE.parent))
    from model_handler import THRESHOLDS

    rows = []
    for d in targets:
        y = np.array([1.0 if d in r["labels"] else 0.0 for r in records])
        s = np.array([r["scores"][d] for r in records])
        pos, neg = s[y == 1], s[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue

        shipped = float(THRESHOLDS[d])
        t_j, j, sens_j, spec_j = best_youden(pos, neg)
        rows.append({
            "finding": d,
            "n_positive": len(pos),
            "n_negative": len(neg),
            "prevalence": float(y.mean()),
            "auroc": auroc(pos, neg),
            "auprc": auprc(pos, neg),
            "auprc_random_baseline": float(y.mean()),
            "score_range_negative": [float(neg.min()), float(neg.max())],
            "score_range_positive": [float(pos.min()), float(pos.max())],
            "shipped_threshold": shipped,
            "shipped_sensitivity": float((pos >= shipped).mean()),
            "shipped_specificity": float((neg < shipped).mean()),
            "shipped_youden_j": float((pos >= shipped).mean() + (neg < shipped).mean() - 1),
            "youden_threshold": t_j,
            "youden_j": j,
            "youden_sensitivity": sens_j,
            "youden_specificity": spec_j,
        })

    out = {
        "model": blob.get("model"),
        "ground_truth": blob.get("ground_truth"),
        "n_images": len(records),
        "findings": rows,
    }
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"model        : {out['model']}")
    print(f"ground truth : {out['ground_truth']}")
    print(f"images       : {out['n_images']:,}\n")
    hdr = (f"{'finding':14s} {'prev':>6s} {'pos':>5s} {'AUROC':>6s} {'AUPRC':>6s} "
           f"{'(rand)':>6s} | {'shipped':>7s} {'sens':>6s} {'spec':>6s} {'J':>6s} "
           f"| {'Youden':>7s} {'sens':>6s} {'spec':>6s} {'J':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['finding']:14s} {r['prevalence']:6.2%} {r['n_positive']:5d} "
              f"{r['auroc']:6.3f} {r['auprc']:6.3f} {r['auprc_random_baseline']:6.3f} | "
              f"{r['shipped_threshold']:7.3f} {r['shipped_sensitivity']:6.1%} "
              f"{r['shipped_specificity']:6.1%} {r['shipped_youden_j']:6.3f} | "
              f"{r['youden_threshold']:7.3f} {r['youden_sensitivity']:6.1%} "
              f"{r['youden_specificity']:6.1%} {r['youden_j']:6.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
