# Medico: chest X-ray multi-label training

A research training script that fine-tunes a DenseNet-121 across the 14 NIH chest
X-ray findings, learning from three datasets that label different subsets of them.

## Scope, stated first

This is experimental research code. It is not a clinical product, it has not been
clinically validated, and it must not be used for diagnosis, triage, treatment
decisions or any other clinical purpose.

No trained weights, patient data or held-out metrics are included here, so this
repository publishes no performance numbers. There are no result figures below
for the same reason: a score this repository cannot produce is not worth drawing.

## The problem it explores

Three public chest X-ray datasets annotate overlapping but different findings.
NIH ChestX-ray14 labels all 14. CheXpert labels 7 of them, and marks some of
those uncertain. The Kermany pneumonia set labels 1.

The naive way to combine them is to treat an unlabelled finding as absent. That
tells the model every CheXpert image is free of the seven findings CheXpert never
assessed, which is false for a large share of them, and the model learns it.

![Label coverage across the three sources](docs/figures/01_label_coverage.png)

The approach here is to carry a mask alongside every label. A finding a source
does not annotate, and a finding CheXpert marks uncertain, contributes nothing to
the loss and nothing to the metrics for that image.

## What the script does

![Training pipeline](docs/figures/02_pipeline.png)

Fine-tunes DenseNet-121 with a grayscale input stem, resuming from a Phase 3
checkpoint. The loss is a masked focal loss with label smoothing, class weights
applied to positives only, and masked entries excluded from both the loss and its
normalisation.

## Patient splits

Several images of one person are normal in a chest X-ray dataset. Splitting over
images rather than people puts the same patient on both sides, and the resulting
score measures memorisation as much as generalisation.

All three splits are grouped by patient. CheXpert and NIH publish an identifier.
The Kermany pneumonia set does not, but its filenames carry one, and that is
recovered rather than ignored.

**This was not previously true.** The pneumonia split stratified by class alone
and had no notion of a patient, so images of one person could land in both
training and evaluation. `docs/splits-and-leakage.md` records what was found, the
fix, and what the fix does not cover. The training script now verifies all three
splits and stops rather than training on a leaking one.

## Data and permissions

Download the datasets from their owners and comply with their licences, access
terms, privacy requirements and any institutional approvals. Do not commit images,
CSV metadata, checkpoints or logs to this repository.

Expected local layout:

```text
data/
  chexpert/train.csv
  nih/Data_Entry_2017.csv
  nih/images/images/
  pneumonia/train/{NORMAL,PNEUMONIA}/
  pneumonia/test/{NORMAL,PNEUMONIA}/
checkpoints_phase3_fulldata/best_model_phase3_fulldata.pt
```

The Phase 3 checkpoint is required and is intentionally not tracked. Establish its
provenance and compatibility before a run.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Point the script at your data when it does not use the layout above:

```powershell
$env:MEDICO_DATA_DIR = "D:\medical-data"
$env:PHASE3_DIR = "D:\models\phase3"
python train_phase4_optimized.py
```

`CHEXPERT_ROOT`, `NIH_IMAGE_DIR`, `NIH_CSV_PATH`, `PNEUMONIA_ROOT` and
`CHECKPOINT_DIR` override individual locations. The script validates every path
and the checkpoint before it starts, and it is a long-running job that writes
split CSVs, logs and checkpoints into `CHECKPOINT_DIR`.

## Tests

The tests need no data and no checkpoint. They run on generated noise and on
synthetic metadata shaped like the real files.

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

They cover the parts that were wrong or that would be silent if they broke:
patient grouping and the absence of leakage, the label mask for each source, the
model's output shape, that the masked loss ignores masked entries while still
reacting to unmasked ones, that one rule decides whether a finding is present
rather than four copies of it, and that an available Arc GPU is actually used.

Figures regenerate from the script's own constants:

```bash
python scripts/figures/generate_figures.py
```

## Hardware

The script prefers an Intel XPU, then DirectML, then CUDA, then CPU. The XPU check asks
torch directly: native XPU support has been in torch since 2.5, and requiring Intel
Extension for PyTorch to reach the GPU meant a machine with a working Arc and no IPEX
trained on the CPU instead.

Measured by [`scripts/benchmark_device.py`](scripts/benchmark_device.py) on an Arc Pro
140T, DenseNet-121 with the grayscale stem at 224x224, median of 15 optimiser steps on
generated noise. Throughput only; nothing here says anything about accuracy.

| Configuration | Device | Images/s | Peak device memory |
| --- | --- | --- | --- |
| Script settings: batch 4, checkpointing on | CPU | 4.7 | |
| Script settings: batch 4, checkpointing on | Arc Pro 140T | **19.3** | 0.37 GB |
| Batch 4, checkpointing off | Arc Pro 140T | 22.6 | 0.58 GB |
| Batch 32, checkpointing off | Arc Pro 140T | 35.3 | 4.00 GB |
| Batch 32, bfloat16 autocast | Arc Pro 140T | 62.3 | 2.09 GB |

The hyperparameters are tuned for a 2 GB card, which is what they say: batch 4, gradient
checkpointing whenever the device is XPU, and 32 accumulation steps to hold the effective
batch at 128. On a 48 GB card that configuration uses 0.37 GB and gives up throughput for
memory it has. Turning checkpointing off is worth 17% on its own, and batch 32 with
bfloat16 autocast, which the script never enables because AMP is gated on CUDA, is 3.2
times the current rate while still using 2 GB of 48.

None of that is changed here. Batch size alters BatchNorm statistics, so a faster
configuration is not automatically the right one, and this repository cannot run training
to completion to check. The numbers are recorded so the choice is an informed one.

## Limitations

No metrics are published here and none should be inferred. The script has not been
run to completion in this repository, and its output depends on dataset versions,
the Phase 3 checkpoint, hardware and library versions.

Combining three sources with different labelling protocols means a finding learned
mostly from one source carries that source's definition and its biases. The masking
handles missing labels, not disagreement between annotators.

Chest X-ray datasets of this kind carry known label noise, since most labels were
extracted from radiology reports by text mining rather than read from the image.

Nothing here has been evaluated for clinical use, on any population, at any site.

## Provenance

The training script, the split module, the tests, the benchmark and the figures
are the author's own work. `docs/splits-and-leakage.md` records the split audit and
the correction.

The datasets belong to their publishers: CheXpert to Stanford ML Group, ChestX-ray14
to the NIH Clinical Center, and the pneumonia set to Kermany and colleagues. None
is redistributed here. DenseNet-121 and its ImageNet weights come from torchvision.

## Licence

MIT for the code, see [LICENSE](LICENSE). Dataset and checkpoint licences are
separate and are not granted by this repository.
