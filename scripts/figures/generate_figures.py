"""Figures describing what this repository implements.

No trained weights, patient data or held-out metrics exist here, so there are no
performance figures. Drawing a score this repository cannot produce would be
worse than drawing nothing. What can be shown honestly is the label design,
which is the part of the implementation that carries the real work.

Everything is read from the training script's own constants, so a figure cannot
describe a mapping the code does not have.

    python scripts/figures/generate_figures.py

Output: docs/figures/
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import portfolio_style as ps  # noqa: E402

OUT = ROOT / "docs" / "figures"


def load_script():
    """Import the training script for its constants, without running main()."""
    spec = importlib.util.spec_from_file_location("t4", ROOT / "train_phase4_optimized.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["t4"] = module
    spec.loader.exec_module(module)
    return module


def fig_label_coverage(script):
    """Which dataset supervises which finding. The reason masking exists."""
    labels = list(script.SELECTED_LABELS)
    chex_supported = set(script.CHEXPERT_TO_NIH.values())

    # Rows are datasets, columns are findings. 1 supervises, 0 masked out.
    sources = [
        ("NIH ChestX-ray14", [1] * len(labels)),
        ("CheXpert", [1 if lab in chex_supported else 0 for lab in labels]),
        ("Kermany pneumonia", [1 if lab == "Pneumonia" else 0 for lab in labels]),
    ]
    grid = np.array([row for _name, row in sources], dtype=float)

    fig = plt.figure(figsize=(13.0, 6.0))
    ax = fig.add_axes([0.175, 0.395, 0.68, 0.285])

    ax.imshow(grid, cmap=ps.COVERAGE, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([lab.replace("_", " ") for lab in labels], rotation=42,
                       ha="right", fontsize=9.6)
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels([name for name, _ in sources], fontsize=10.6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sources), 1), minor=True)
    ax.grid(which="minor", color=ps.PAPER, linewidth=2.4)

    # A legend rather than a word in every cell: the colour already says it, and
    # rotated 7pt text in a 42-cell grid is not readable.
    for k, (colour, text) in enumerate(((ps.BLUE, "supervised, contributes to the loss"),
                                        ("#F1F5F9", "masked, excluded from the loss"))):
        fig.patches.append(plt.Rectangle((0.175 + k * 0.30, 0.715), 0.015, 0.024,
                                         transform=fig.transFigure, facecolor=colour,
                                         edgecolor=ps.HAIR, linewidth=0.8, zorder=5))
        fig.text(0.196 + k * 0.30, 0.727, text, fontsize=9.6, color=ps.MUTED,
                 va="center")

    counts = [int(sum(row)) for _n, row in sources]
    for i, count in enumerate(counts):
        ax.text(len(labels) - 0.30, i, f"{count} of {len(labels)}", va="center",
                ha="left", fontsize=10.0, color=ps.INK, fontweight="600")

    ps.title_block(
        fig, "Three datasets, one label space, and most of it unlabelled",
        "The model predicts the 14 NIH findings. Each source supervises only the "
        "findings it actually annotates, and\neverything else is excluded from the "
        "loss rather than counted as a confident negative.", y=0.955, size=20)
    ps.footnote(fig, [
        "Treating an unannotated finding as absent is the easy mistake here. It "
        "would tell the model that every CheXpert image is free of the seven "
        "findings CheXpert never assessed.",
        "CheXpert also marks individual findings uncertain. Those entries are "
        "masked per image, so the coverage above is an upper bound rather than a "
        "count of usable labels.",
        "Source: SELECTED_LABELS and CHEXPERT_TO_NIH in train_phase4_optimized.py."],
        y=0.175)
    ps.save(fig, OUT, "01_label_coverage")
    return counts


def fig_pipeline(script):
    """What the training script does, in the order it does it."""
    fig = plt.figure(figsize=(13.0, 6.3))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 100)
    ax.set_ylim(11, 100)
    ax.axis("off")

    def box(x, y, w, h, title, lines, edge, fill):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                                   linewidth=1.4, zorder=3,
                                   joinstyle="round"))
        ax.text(x + 1.6, y + h - 3.2, title, fontsize=10.6, color=ps.INK,
                fontweight="600", va="top", zorder=4)
        for i, line in enumerate(lines):
            ax.text(x + 1.6, y + h - 7.4 - i * 3.4, line, fontsize=8.8,
                    color=ps.MUTED, va="top", zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                    arrowprops={"arrowstyle": "-|>", "color": ps.FAINT, "lw": 1.4})

    box(3, 62, 27, 22, "Three sources", [
        "CheXpert, patient split",
        "NIH ChestX-ray14, patient split",
        "Kermany pneumonia, patient grouped",
        "no images are stored here",
    ], ps.SLATE, ps.PAPER)
    box(36.5, 62, 27, 22, "Labels and masks", [
        "map each source to 14 findings",
        "mask uncertain CheXpert entries",
        "mask findings a source omits",
        "masked entries carry label 0",
    ], ps.BLUE, ps.BLUE_BG)
    box(70, 62, 27, 22, "Splits", [
        "grouped by patient, never by image",
        "verified: no patient in two splits",
        "NIH redrawn until every finding",
        "has enough positives to score",
    ], ps.GREEN, ps.GREEN_BG)

    arrow(30, 73, 36.2, 73)
    arrow(63.7, 73, 69.8, 73)
    arrow(83, 61.6, 83, 52)

    box(70, 30, 27, 21, "Training", [
        "DenseNet-121, grayscale input",
        f"{script.IMG_SIZE} x {script.IMG_SIZE}, batch {script.BATCH_SIZE}",
        f"accumulation {script.ACCUMULATION_STEPS}",
        "resumes from a Phase 3 checkpoint",
    ], ps.AMBER, ps.AMBER_BG)
    box(36.5, 30, 27, 21, "Masked focal loss", [
        f"alpha {script.FOCAL_ALPHA}, gamma {script.FOCAL_GAMMA}",
        f"label smoothing {script.FOCAL_SMOOTHING}",
        "masked entries contribute nothing",
        "class weights on positives only",
    ], ps.AMBER, ps.AMBER_BG)
    box(3, 30, 27, 21, "Validation", [
        "per finding area under the curve",
        "computed on unmasked entries only",
        "checkpoint on the weakest finding",
        "not a clinical evaluation",
    ], ps.SLATE, ps.PAPER)

    arrow(69.8, 40, 63.7, 40)
    arrow(36.2, 40, 30.2, 40)

    ax.text(3, 22.5, "Not included in this repository", fontsize=10.6, color=ps.INK,
            fontweight="600")
    for i, line in enumerate([
        "The images, the CSV metadata, the Phase 3 checkpoint it resumes from, and any "
        "trained weights. None are redistributable,",
        "and none are committed. Without them the script validates its paths and stops, "
        "which is why this repository publishes no metrics.",
    ]):
        ax.text(3, 18.0 - i * 3.6, line, fontsize=9.0, color=ps.MUTED)

    ps.title_block(
        fig, "What the training script actually does",
        "Drawn from the script's own constants. Boxes describe implemented "
        "behaviour, not intended behaviour.", y=0.965, size=20)
    ps.save(fig, OUT, "02_pipeline")


def main() -> int:
    ps.apply()
    script = load_script()
    print(f"\n  {len(script.SELECTED_LABELS)} findings, "
          f"{len(script.CHEXPERT_TO_NIH)} mapped from CheXpert\n")
    counts = fig_label_coverage(script)
    fig_pipeline(script)
    print(f"\n  supervised findings per source: {counts}")
    print(f"  figures written to {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
