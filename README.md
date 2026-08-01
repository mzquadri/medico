# Medico: Chest X-Ray Multi-Label Fine-Tuning

This repository contains a research training script for Phase 4 fine-tuning of a DenseNet-121 chest X-ray classifier across 14 findings. It combines CheXpert, NIH ChestX-ray14, and a binary pneumonia dataset, with masked focal loss for uncertain or unavailable labels.

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
