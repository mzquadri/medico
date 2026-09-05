"""Dataset splitting, kept separate from the training script so it can be tested.

The one property that matters here is that no patient appears in more than one
split. A chest X-ray dataset almost always contains several images of the same
person, and a split drawn at random over images will put some of them on both
sides. The model then gets credit at evaluation time for having already seen that
patient, and the reported score is not a measure of generalisation.

CheXpert and NIH both publish a patient identifier, so grouping on it is direct.
The Kermany pneumonia dataset publishes none, but its filenames carry one: the
positive images are named `personN_bacteria_M.jpeg` and `personN_virus_M.jpeg`,
and one person contributes several images. That is recovered here rather than
ignored.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

#: NIH writes some findings with a space where the label list uses an underscore.
NIH_NAME_MAP = {"Pleural_Thickening": "Pleural Thickening"}

#: `personN` in the Kermany positive filenames, and the `IM-NNNN` study token the
#: negatives carry. Matching is case insensitive because both cases appear.
_PERSON = re.compile(r"person[\s_-]?(\d+)", re.IGNORECASE)
_STUDY = re.compile(r"IM[\s_-]?(\d+)", re.IGNORECASE)


def pneumonia_group_key(path: str) -> str:
    """A stable per-patient key for one Kermany pneumonia filename.

    Falls back to the filename itself when nothing is recognisable, which makes
    that image its own group. That is never worse than splitting per image.

    The person number is used without its source folder. Numbering restarts
    between the published train and test folders, so keying on the number alone
    can merge two different people into one group. Merging is safe: it only keeps
    images together. Separating them would risk the opposite, and that is the
    error worth avoiding.
    """
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    match = _PERSON.search(name)
    if match:
        return f"person{int(match.group(1))}"
    match = _STUDY.search(name)
    if match:
        return f"study{int(match.group(1))}"
    return f"file:{name}"


@dataclass(frozen=True)
class SplitSizes:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")


#: The default used when a caller does not pass one.
DEFAULT_SIZES = SplitSizes()


def grouped_stratified_split(records, sizes: SplitSizes | None = None, seed: int = 42):
    """Split records into train, val and test without splitting any group.

    `records` is a sequence of mappings with at least `image_path` and `label`.
    Groups are assigned whole. Within a class, groups are handed to whichever
    split is furthest below its target, which keeps the class balance close to
    the requested fractions without ever dividing a patient.
    """
    sizes = sizes or DEFAULT_SIZES
    sizes.validate()
    groups: dict[str, list] = defaultdict(list)
    for record in records:
        groups[pneumonia_group_key(record["image_path"])].append(record)

    # A group's class is its majority label, so a patient with mixed labels still
    # lands in exactly one split.
    summarised = []
    for key, items in groups.items():
        positives = sum(1 for r in items if float(r["label"]) == 1.0)
        summarised.append((key, items, 1 if positives * 2 >= len(items) else 0))

    rng = np.random.RandomState(seed)
    out: dict[str, list] = {"train": [], "val": [], "test": []}
    targets = {"train": sizes.train, "val": sizes.val, "test": sizes.test}

    for cls in (0, 1):
        of_class = [g for g in summarised if g[2] == cls]
        of_class.sort(key=lambda g: g[0])  # deterministic before shuffling
        order = rng.permutation(len(of_class))
        # Largest groups first, so one big patient cannot overshoot a small split.
        chosen = sorted((of_class[i] for i in order), key=lambda g: -len(g[1]))
        counts = {"train": 0, "val": 0, "test": 0}
        total = sum(len(g[1]) for g in chosen)
        for _key, items, _ in chosen:
            deficit = {
                name: targets[name] - (counts[name] / total if total else 0.0)
                for name in counts
            }
            pick = max(deficit, key=lambda name: deficit[name])
            out[pick].extend(items)
            counts[pick] += len(items)

    for name in out:
        rng.shuffle(out[name])
    return out["train"], out["val"], out["test"]


def patient_split(frame, patient_column: str, sizes: SplitSizes, seed: int = 42):
    """Split a dataframe by unique patient, 80/10/10 style fractions."""
    sizes.validate()
    rng = np.random.RandomState(seed)
    patients = frame[patient_column].dropna().unique()
    patients = np.array(sorted(patients))  # deterministic before shuffling
    rng.shuffle(patients)

    n = len(patients)
    train_end = int(sizes.train * n)
    val_end = int((sizes.train + sizes.val) * n)
    assign = {
        "train": set(patients[:train_end]),
        "val": set(patients[train_end:val_end]),
        "test": set(patients[val_end:]),
    }
    return tuple(
        frame[frame[patient_column].isin(assign[name])].copy()
        for name in ("train", "val", "test")
    )


def count_positive(frame, disease: str) -> int:
    """Positive rows for one finding in an NIH-style `Finding Labels` column."""
    csv_name = NIH_NAME_MAP.get(disease, disease)
    # Non-capturing groups: pandas warns that a pattern with match groups is
    # probably meant for str.extract, and the boundaries here are only anchors.
    pattern = rf"(?:^|\|){re.escape(csv_name)}(?:\||$)"
    return int(frame["Finding Labels"].str.contains(pattern, regex=True, na=False).sum())


def nih_patient_split_with_min_positives(nih_df, diseases, min_pos: int = 20,
                                         max_tries: int = 50, seed: int = 42,
                                         skip: tuple[str, ...] = ("Hernia",)):
    """Patient-level split that also keeps enough positives to score each finding.

    A validation split holding two positives for a finding produces an area under
    the curve that swings on a single image. Splits are redrawn until every
    finding except the ones named in `skip` clears `min_pos` in both validation
    and test.

    Returns the split and a record of how it was obtained, including whether the
    minimum was actually met, so a caller cannot mistake a fallback for success.
    """
    sizes = SplitSizes(0.8, 0.1, 0.1)
    for attempt in range(1, max_tries + 1):
        train, val, test = patient_split(nih_df, "Patient ID", sizes, seed=seed + attempt)
        shortfall = [
            disease for disease in diseases
            if disease not in skip
            and min(count_positive(val, disease), count_positive(test, disease)) < min_pos
        ]
        if not shortfall:
            return train, val, test, {"attempts": attempt, "met_minimum": True,
                                      "min_pos": min_pos, "short": []}
    return train, val, test, {"attempts": max_tries, "met_minimum": False,
                              "min_pos": min_pos, "short": shortfall}


def overlap(*frames_or_lists, key) -> set:
    """Any key present in more than one split. Empty means no leakage."""
    seen: list[set] = []
    for part in frames_or_lists:
        if hasattr(part, "columns"):
            seen.append(set(part[key].dropna().unique()))
        else:
            seen.append({key(r) if callable(key) else r[key] for r in part})
    clashes: set = set()
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            clashes |= seen[i] & seen[j]
    return clashes
