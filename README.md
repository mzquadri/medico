# Medico: Chest X-Ray Multi-Label Fine-Tuning

This repository contains a research training script for Phase 4 fine-tuning of a DenseNet-121 chest X-ray classifier across 14 findings. It combines CheXpert, NIH ChestX-ray14, and a binary pneumonia dataset, with masked focal loss for uncertain or unavailable labels.

## Two components, two different models

Read this before comparing any numbers between them.

| | `train_phase4_optimized.py` | [`ai-radiologist/`](ai-radiologist/) |
| --- | --- | --- |
| What it is | The fine-tuning job | A screening and explainability demo |
| Model | DenseNet-121 fine-tuned here, from a Phase 3 checkpoint | DenseNet-121 **pre-trained by [TorchXRayVision](https://github.com/mlmed/torchxrayvision)**, used as shipped |
| Findings | 14 | 5 |
| Weights | Not tracked; produced by running the script | Not tracked; fetched by the loader |

They share an architecture and a problem, not a checkpoint. The demo was built to exercise
the inference, attribution and reporting path end to end, and it runs on public pre-trained
weights so it works without the training artefacts. Accuracy figures from one say nothing
about the other.

## Datasets

Obtain each from its owner and comply with its licence and access terms. None are
redistributed here.

| Dataset | Content | Source |
| --- | --- | --- |
| **NIH ChestX-ray14** | 112,120 frontal radiographs, 30,805 patients, 14 findings; labels in `Data_Entry_2017.csv` | [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) · [paper](https://arxiv.org/abs/1705.02315) · [Kaggle mirror](https://www.kaggle.com/datasets/nih-chest-xrays/data) |
| **CheXpert** | 224,316 radiographs, 65,240 patients, 14 observations with explicit uncertainty labels | [Stanford AIMI](https://stanfordmlgroup.github.io/competitions/chexpert/) · [paper](https://arxiv.org/abs/1901.07031) |
| **Chest X-Ray Pneumonia** | Binary pneumonia / normal paediatric set | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |

## Status and scope

This is experimental research code, not a clinical product. No trained weights, patient data, held-out metrics, or clinical validation are included in this repository. The implementation must not be used for diagnosis, triage, treatment decisions, or any other clinical purpose.

The script creates deterministic dataset splits with seed 42 where source metadata permits, but final results also depend on dataset versions, hardware, library versions, and the Phase 3 checkpoint. Reproduce and review all splits and metrics independently before making any research claim.

## Data and permissions

Download the source datasets directly from their owners and comply with their licenses, access terms, privacy requirements, and institutional approvals. Do not commit images, CSV metadata, checkpoints, logs, or derived records to this repository.

Expected local layout:

```text
data/
  chexpert/
    train.csv
  nih/
    Data_Entry_2017.csv
    images/images/
  pneumonia/
    train/{NORMAL,PNEUMONIA}/
    test/{NORMAL,PNEUMONIA}/
checkpoints_phase3_fulldata/
  best_model_phase3_fulldata.pt
```

The supplied Phase 3 checkpoint is required by default and is intentionally not tracked. Its provenance and compatibility must be established before a run.

## Setup

Use a dedicated Python environment, then install the runtime dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set locations with environment variables when your data does not use the layout above:

```powershell
$env:MEDICO_DATA_DIR = "D:\medical-data"
$env:PHASE3_DIR = "D:\models\phase3"
python train_phase4_optimized.py
```

More specific overrides are available: `CHEXPERT_ROOT`, `NIH_IMAGE_DIR`, `NIH_CSV_PATH`, `PNEUMONIA_ROOT`, and `CHECKPOINT_DIR`. The script validates data paths and the required checkpoint before training.

## Training behavior

- Fine-tunes DenseNet-121 for the 14 NIH findings.
- Uses masked labels so CheXpert uncertainty and unsupported labels are excluded from loss/metrics.
- Uses patient-level splits for CheXpert and NIH, and a stratified split for the pneumonia dataset.
- Prefers Intel XPU, then DirectML, CUDA, and CPU.

Review `train_phase4_optimized.py` before running: it is a long-running training job and will write split CSVs, logs, and checkpoints to `CHECKPOINT_DIR`.

## License

The source code is available under the MIT License. Dataset and checkpoint licenses are separate and are not granted by this repository.
