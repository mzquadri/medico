"""End to end smoke test on synthetic images, no real patient data involved.

The training script needs three datasets and a checkpoint that cannot be
committed, so nothing here proves the model is any good. What it does prove is
that the pieces still fit together: the loaders produce the shapes the model
expects, the masking rules hold, and the loss ignores what it is told to ignore.

Every image is generated noise. No patient data is read, written or required.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL", reason="Pillow is required")
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def script():
    """Import the training script without running main()."""
    spec = importlib.util.spec_from_file_location("t4", ROOT / "train_phase4_optimized.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["t4"] = module
    spec.loader.exec_module(module)
    return module


def write_noise(path: Path, size=(64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(abs(hash(path.name)) % (2**31))
    Image.fromarray(rng.randint(0, 255, size, dtype=np.uint8), mode="L").save(path)


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    """A miniature stand-in shaped like the three real datasets."""
    root = tmp_path_factory.mktemp("fake")

    chex = root / "chexpert"
    rows = []
    for patient in range(1, 9):
        for study in range(1, 3):
            rel = f"CheXpert-v1.0/train/patient{patient:05d}/study{study}/view1_frontal.jpg"
            write_noise(chex / rel)
            rows.append({
                "Path": rel,
                "Atelectasis": [0.0, 1.0, -1.0, np.nan][patient % 4],
                "Cardiomegaly": [1.0, 0.0, np.nan, -1.0][patient % 4],
                "Consolidation": 0.0, "Edema": 1.0,
                "Pleural Effusion": [1.0, -1.0][patient % 2],
                "Pneumonia": 0.0, "Pneumothorax": 0.0,
            })
    pd.DataFrame(rows).to_csv(chex / "train.csv", index=False)

    nih_images = root / "nih" / "images" / "images"
    nih_rows = []
    findings = ["Atelectasis", "Cardiomegaly", "Effusion", "Pleural Thickening",
                "Mass", "Nodule", "No Finding"]
    for patient in range(1, 13):
        for k in range(2):
            name = f"{patient:08d}_{k:03d}.png"
            write_noise(nih_images / name)
            nih_rows.append({"Image Index": name, "Patient ID": patient,
                             "Finding Labels": findings[(patient + k) % len(findings)]})
    pd.DataFrame(nih_rows).to_csv(root / "nih" / "Data_Entry_2017.csv", index=False)

    pneu = root / "pneumonia"
    for split in ("train", "test"):
        for i in range(6):
            write_noise(pneu / split / "NORMAL" / f"IM-{i:04d}-0001.jpeg")
            write_noise(pneu / split / "PNEUMONIA" / f"person{i}_bacteria_{i}.jpeg")
    return root


def test_chexpert_masks_uncertain_and_unsupported_labels(script, fake_data):
    ds = script.CheXpertDataset(
        csv_path=str(fake_data / "chexpert" / "train.csv"),
        root_dir=str(fake_data / "chexpert"),
        transform=script.get_transforms(train=False, enable_clahe=False),
    )
    assert ds.masks.shape[1] == len(script.SELECTED_LABELS)

    # CheXpert carries no column for these, so they must never reach the loss.
    for disease in ("Emphysema", "Fibrosis", "Hernia", "Infiltration",
                    "Mass", "Nodule", "Pleural_Thickening"):
        column = ds.masks[:, script.LABEL_TO_IDX[disease]]
        assert (column == 0).all(), f"{disease} should be masked on CheXpert"

    # Uncertain entries are excluded, and anything masked carries label 0.
    assert (ds.masks[:, script.LABEL_TO_IDX["Cardiomegaly"]] == 0).any()
    assert (ds.labels[ds.masks == 0] == 0).all()
    assert set(np.unique(ds.masks)).issubset({0.0, 1.0})


def test_pneumonia_supervises_only_pneumonia(script, fake_data):
    ds = script.PneumoniaDataset(
        root_dir=str(fake_data / "pneumonia"), split="train",
        transform=script.get_transforms(train=False, enable_clahe=False),
    )
    _image, _labels, mask = ds[0]

    assert mask.sum().item() == 1.0
    assert mask[script.LABEL_TO_IDX["Pneumonia"]].item() == 1.0


def test_nih_supervises_every_finding(script, fake_data):
    ds = script.NIHDataset(
        csv_path=str(fake_data / "nih" / "Data_Entry_2017.csv"),
        image_dir=str(fake_data / "nih" / "images" / "images"),
        transform=script.get_transforms(train=False, enable_clahe=False),
    )
    _image, _labels, mask = ds[0]

    assert mask.sum().item() == len(script.SELECTED_LABELS)


def test_loaders_produce_the_shape_the_model_expects(script, fake_data):
    ds = script.PneumoniaDataset(
        root_dir=str(fake_data / "pneumonia"), split="train",
        transform=script.get_transforms(train=False, enable_clahe=False),
    )
    image, labels, mask = ds[0]

    assert image.shape == (1, script.IMG_SIZE, script.IMG_SIZE)
    assert labels.shape == (len(script.SELECTED_LABELS),)
    assert mask.shape == (len(script.SELECTED_LABELS),)


def test_model_accepts_a_batch_and_returns_one_logit_per_finding(script):
    model = script.DenseNet121(num_classes=len(script.SELECTED_LABELS), grayscale=True)
    model.eval()
    batch = torch.randn(2, 1, script.IMG_SIZE, script.IMG_SIZE)

    with torch.no_grad():
        out = model(batch)

    assert out.shape == (2, len(script.SELECTED_LABELS))
    assert torch.isfinite(out).all()


def test_masked_loss_ignores_masked_entries(script):
    """A masked column must not change the loss, whatever the target says."""
    loss_fn = script.MaskedSmoothedFocalLoss(alpha=0.75, gamma=2.0, smoothing=0.02)
    logits = torch.randn(4, len(script.SELECTED_LABELS))
    targets = torch.zeros_like(logits)
    masks = torch.ones_like(logits)
    masks[:, 3] = 0.0

    base = loss_fn(logits, targets, masks)
    targets_flipped = targets.clone()
    targets_flipped[:, 3] = 1.0
    flipped = loss_fn(logits, targets_flipped, masks)

    assert torch.isfinite(base)
    assert torch.allclose(base, flipped, atol=1e-6)


def test_masked_loss_reacts_to_unmasked_entries(script):
    """The mirror of the test above, otherwise it would pass on a broken loss."""
    loss_fn = script.MaskedSmoothedFocalLoss()
    logits = torch.randn(4, len(script.SELECTED_LABELS))
    targets = torch.zeros_like(logits)
    masks = torch.ones_like(logits)

    base = loss_fn(logits, targets, masks)
    targets_flipped = targets.clone()
    targets_flipped[:, 3] = 1.0

    assert not torch.allclose(base, loss_fn(logits, targets_flipped, masks), atol=1e-6)


def test_a_fully_masked_batch_does_not_produce_nan(script):
    loss_fn = script.MaskedSmoothedFocalLoss()
    logits = torch.randn(3, len(script.SELECTED_LABELS))
    targets = torch.zeros_like(logits)

    value = loss_fn(logits, targets, torch.zeros_like(logits))

    assert torch.isfinite(value), "an all-masked batch must not return NaN"
