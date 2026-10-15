# Figures

Two rendered reports, kept as the worked example of what the demo produces.

| File | Source radiograph |
| --- | --- |
| `report_multi_finding.png` | NIH ChestX-ray14, image `00001989_001` |
| `report_no_finding.png` | NIH ChestX-ray14, image `00013144_004` |

## Image attribution

Both radiographs come from the **NIH ChestX-ray14** dataset, released by the
**NIH Clinical Center**:

> Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. *ChestX-ray8: Hospital-scale Chest X-ray
> Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax
> Diseases.* IEEE CVPR, 2017. [arXiv:1705.02315](https://arxiv.org/abs/1705.02315)

Dataset: [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) ·
[Kaggle mirror](https://www.kaggle.com/datasets/nih-chest-xrays/data)

The overlays, panels and report layout are this repository's work; the underlying radiographs
are not. Anyone reusing these figures should carry the citation above and check the NIH terms
for their own use.

**No CheXpert image appears in this repository or its history.** CheXpert is read from a local
path by `../../train_phase4_optimized.py` and nothing from it is committed. The same holds for
the Kermany pneumonia set.

## Regenerating them

Obtain NIH ChestX-ray14 from the links above and accept its terms. Then, from the repository
root:

```bash
# a one-page report for every finding the model detects
python ai-radiologist/visualize.py path/to/00001989_001.png --all --output ai-radiologist/figures

# scores only, no rendering
python ai-radiologist/predict.py path/to/00001989_001.png
```

`visualize.py` writes `<image>_disease_report.png` when a finding clears its threshold and
`<image>_healthy_report.png` when none does. The two files here are those outputs, renamed.

## What each one shows

`report_multi_finding.png` — the source radiograph, then one panel per detected finding
carrying the Grad-CAM++ overlay, the model's confidence, that finding's calibrated threshold,
the anatomical region and the reference text from `disease_info.py`.

`report_no_finding.png` — the radiograph plus all five scores, each below its threshold.

Read [`../README.md`](../README.md) before reading a report: the overlay is Grad-CAM
constrained by an anatomical template rather than raw attribution, and the displayed
percentage is a score, not a calibrated probability.

The measured performance behind those thresholds is in [`../eval/metrics.json`](../eval/metrics.json):
AUROC, AUPRC and Youden's J over 4,128 NIH images, with `Data_Entry_2017.csv` as ground truth.
