"""Tests for the split logic, written around the leakage that was there.

The pneumonia split previously grouped only by class, so images of one person
could land in both training and test. These fix that property in place.
"""

import numpy as np
import pandas as pd
import pytest

from medico.splits import (
    SplitSizes,
    count_positive,
    grouped_stratified_split,
    nih_patient_split_with_min_positives,
    overlap,
    patient_split,
    pneumonia_group_key,
)

# --------------------------------------------------------------------------
# recovering the patient identity the filenames carry
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("person1_bacteria_1.jpeg", "person1"),
    ("person1_bacteria_2.jpeg", "person1"),
    ("person1_virus_6.jpeg", "person1"),
    ("PERSON23_VIRUS_54.JPEG", "person23"),
    ("person007_bacteria_31.jpeg", "person7"),
    ("IM-0115-0001.jpeg", "study115"),
    ("NORMAL2-IM-1440-0001.jpeg", "study1440"),
])
def test_group_key_recovers_the_identity_in_the_filename(name, expected):
    assert pneumonia_group_key(f"data/pneumonia/train/PNEUMONIA/{name}") == expected


def test_all_images_of_one_person_share_a_key():
    names = ["person12_bacteria_46.jpeg", "person12_bacteria_47.jpeg", "person12_virus_50.jpeg"]
    keys = {pneumonia_group_key(n) for n in names}

    assert len(keys) == 1


def test_unrecognised_filenames_become_their_own_group():
    """Never worse than the per-image split it replaces."""
    a = pneumonia_group_key("data/x/odd_name_a.png")
    b = pneumonia_group_key("data/x/odd_name_b.png")

    assert a != b


def test_windows_and_posix_paths_agree():
    assert (pneumonia_group_key(r"data\pneumonia\train\PNEUMONIA\person5_virus_1.jpeg")
            == pneumonia_group_key("data/pneumonia/train/PNEUMONIA/person5_virus_1.jpeg"))


# --------------------------------------------------------------------------
# the property that was broken
# --------------------------------------------------------------------------

def kermany_like(n_people=60, n_normal=90, images_per_person=4):
    """A dataset shaped like the real one: several images per positive patient."""
    records = []
    for p in range(1, n_people + 1):
        for k in range(images_per_person):
            records.append({
                "image_path": f"data/pneumonia/train/PNEUMONIA/person{p}_bacteria_{k}.jpeg",
                "label": 1.0,
            })
    for i in range(n_normal):
        records.append({
            "image_path": f"data/pneumonia/train/NORMAL/IM-{i:04d}-0001.jpeg",
            "label": 0.0,
        })
    return records


def test_no_person_appears_in_two_splits():
    train, val, test = grouped_stratified_split(kermany_like(), seed=42)

    clashes = overlap(train, val, test, key=lambda r: pneumonia_group_key(r["image_path"]))
    assert clashes == set()


def test_a_class_stratified_split_would_have_leaked():
    """Shows the failure the grouped split exists to prevent."""
    records = kermany_like()
    rng = np.random.RandomState(0)
    order = rng.permutation(len(records))
    shuffled = [records[i] for i in order]
    cut = int(0.7 * len(shuffled))
    naive_train, naive_test = shuffled[:cut], shuffled[cut:]

    leaked = overlap(naive_train, naive_test,
                     key=lambda r: pneumonia_group_key(r["image_path"]))
    assert leaked, "the naive split should leak, otherwise this test proves nothing"


def test_every_image_lands_in_exactly_one_split():
    records = kermany_like()
    train, val, test = grouped_stratified_split(records, seed=42)

    paths = [r["image_path"] for r in train + val + test]
    assert len(paths) == len(records)
    assert len(set(paths)) == len(records)


def test_split_sizes_are_close_to_the_request():
    records = kermany_like()
    train, val, test = grouped_stratified_split(records, SplitSizes(0.7, 0.15, 0.15), seed=42)
    total = len(records)

    assert abs(len(train) / total - 0.70) < 0.06
    assert abs(len(val) / total - 0.15) < 0.06
    assert abs(len(test) / total - 0.15) < 0.06


def test_both_classes_reach_every_split():
    train, val, test = grouped_stratified_split(kermany_like(), seed=42)

    for part in (train, val, test):
        labels = {float(r["label"]) for r in part}
        assert labels == {0.0, 1.0}


def test_the_split_is_deterministic():
    a = grouped_stratified_split(kermany_like(), seed=42)
    b = grouped_stratified_split(kermany_like(), seed=42)

    assert [r["image_path"] for r in a[0]] == [r["image_path"] for r in b[0]]


def test_a_patient_with_mixed_labels_still_lands_in_one_split():
    records = kermany_like(n_people=20, n_normal=40)
    records.append({"image_path": "data/pneumonia/train/NORMAL/person3_bacteria_9.jpeg",
                    "label": 0.0})
    train, val, test = grouped_stratified_split(records, seed=1)

    where = [name for name, part in (("train", train), ("val", val), ("test", test))
             if any(pneumonia_group_key(r["image_path"]) == "person3" for r in part)]
    assert len(where) == 1


# --------------------------------------------------------------------------
# the two datasets that publish a patient identifier
# --------------------------------------------------------------------------

def nih_like(n_patients=400, seed=0):
    rng = np.random.RandomState(seed)
    diseases = ["Atelectasis", "Cardiomegaly", "Effusion", "Pleural Thickening", "Hernia"]
    rows = []
    for pid in range(1, n_patients + 1):
        for _ in range(rng.randint(1, 5)):
            picked = [d for d in diseases if rng.rand() < 0.35]
            rows.append({"Patient ID": pid,
                         "Finding Labels": "|".join(picked) if picked else "No Finding"})
    return pd.DataFrame(rows)


def test_patient_split_keeps_patients_whole():
    frame = nih_like()
    train, val, test = patient_split(frame, "Patient ID", SplitSizes(0.8, 0.1, 0.1))

    assert overlap(train, val, test, key="Patient ID") == set()
    assert len(train) + len(val) + len(test) == len(frame)


def test_count_positive_maps_the_underscored_name():
    frame = pd.DataFrame({"Finding Labels": ["Pleural Thickening|Mass", "Mass", "No Finding"]})

    assert count_positive(frame, "Pleural_Thickening") == 1


def test_count_positive_does_not_match_a_substring():
    frame = pd.DataFrame({"Finding Labels": ["Pneumothorax", "Pneumonia"]})

    assert count_positive(frame, "Pneumonia") == 1


def test_nih_split_reports_whether_the_minimum_was_met():
    frame = nih_like(n_patients=500, seed=3)
    diseases = ["Atelectasis", "Cardiomegaly", "Effusion"]
    train, val, test, info = nih_patient_split_with_min_positives(
        frame, diseases, min_pos=5, max_tries=20)

    assert overlap(train, val, test, key="Patient ID") == set()
    assert isinstance(info["met_minimum"], bool)
    if info["met_minimum"]:
        for disease in diseases:
            assert count_positive(val, disease) >= 5
            assert count_positive(test, disease) >= 5


def test_nih_split_does_not_claim_success_when_it_fails():
    """An impossible minimum must be reported, not silently returned as a split."""
    frame = nih_like(n_patients=60, seed=5)
    _, _, _, info = nih_patient_split_with_min_positives(
        frame, ["Cardiomegaly"], min_pos=10_000, max_tries=3)

    assert info["met_minimum"] is False
    assert info["short"] == ["Cardiomegaly"]


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        SplitSizes(0.8, 0.3, 0.3).validate()
