"""One rule decides whether a finding is present, and both users share it.

`Finding Labels` is a pipe-separated list, so a finding counts only as a whole
element: "Mass" is present in "Pleural Thickening|Mass" and absent from "Mass Effect".

Two places needed that rule. `medico.splits.count_positive` uses it to decide whether a
split contains enough positives to score a finding, and the NIH dataset class uses it to
build the label matrix the model trains on. They were separate implementations, and the
one in the training script was the older and weaker of the two: capturing groups, which
made pandas emit a warning per finding on every dataset load, no `re.escape`, and no
explicit handling of a missing cell.

Nothing was mislabelled by it. None of the fourteen NIH findings contains a regex
metacharacter, and 4,000 synthetic label rows produced identical matrices either way,
56,000 cells with no disagreement. The problem was that a split's positive counts and the
labels trained on were computed by two functions that nothing kept in step.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from medico.splits import NIH_NAME_MAP, count_positive, label_matches

COLUMN = "Finding Labels"


@pytest.fixture
def labels():
    return pd.Series(
        [
            "Atelectasis|Effusion",
            "Mass",
            "Mass Effect",
            "Pleural Thickening|Mass",
            "No Finding",
            "",
            None,
        ]
    )


class TestWholeElementMatching:
    def test_a_finding_alone_in_the_cell_matches(self, labels):
        assert bool(label_matches(labels, "Mass").iloc[1])

    def test_a_finding_at_the_end_of_a_list_matches(self, labels):
        assert bool(label_matches(labels, "Mass").iloc[3])

    def test_a_finding_at_the_start_of_a_list_matches(self, labels):
        assert bool(label_matches(labels, "Atelectasis").iloc[0])

    def test_a_longer_phrase_starting_with_the_finding_does_not_match(self, labels):
        """"Mass Effect" is not the finding "Mass"."""
        assert not bool(label_matches(labels, "Mass").iloc[2])

    def test_no_finding_matches_nothing(self, labels):
        for disease in ("Mass", "Effusion", "Pneumonia"):
            assert not bool(label_matches(labels, disease).iloc[4])

    def test_an_empty_cell_is_not_a_match(self, labels):
        assert not bool(label_matches(labels, "Mass").iloc[5])

    def test_a_missing_cell_is_false_rather_than_missing(self, labels):
        """na=False. Without it the mask is three-valued and surprises its callers."""
        result = label_matches(labels, "Mass")
        assert result.iloc[6] is False or not bool(result.iloc[6])
        assert result.dtype == bool


class TestRobustness:
    def test_the_nih_spelling_is_mapped(self, labels):
        """The CSV writes a space where the label list uses an underscore."""
        assert NIH_NAME_MAP["Pleural_Thickening"] == "Pleural Thickening"
        assert bool(label_matches(labels, "Pleural_Thickening").iloc[3])

    def test_a_name_containing_a_metacharacter_is_escaped(self):
        """No current finding needs this. An unescaped one would fail silently."""
        series = pd.Series(["Mass|Nodule (calcified)", "Mass|Nodule X"])
        result = label_matches(series, "Nodule (calcified)")
        assert list(result) == [True, False]

    def test_it_emits_no_pandas_warning(self, labels):
        """The capturing form warned once per finding, on every dataset load."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for disease in ("Mass", "Effusion", "Pleural_Thickening"):
                label_matches(labels, disease)
        assert [str(w.message) for w in caught] == []


class TestBothUsersShareIt:
    def test_count_positive_agrees_with_the_mask(self, labels):
        frame = pd.DataFrame({COLUMN: labels})
        for disease in ("Mass", "Effusion", "Pleural_Thickening", "Hernia"):
            assert count_positive(frame, disease) == int(label_matches(labels, disease).sum())

    def test_the_training_script_has_no_second_implementation(self):
        """The duplicate this file exists because of must not come back.

        Checked as text because the alternative is importing a 2,800-line script that
        builds a device and a model at import time.
        """
        from pathlib import Path

        source = Path(__file__).resolve().parents[1] / "train_phase4_optimized.py"
        text = source.read_text(encoding="utf-8")
        assert "label_matches" in text, "the script must use the shared matcher"
        assert "str.contains(rf'(^|" not in text, "the capturing-group pattern is back"
        assert text.count("NIH_NAME_MAP = {") == 0, "the name map is defined twice again"

    def test_the_script_uses_the_tested_split_rather_than_its_own(self):
        """The important one.

        The script carried its own make_nih_patient_split_with_min_positives and called
        it, while medico.splits held a tested implementation that nothing invoked. The
        copy returned three values and only printed a warning when it gave up, so a
        fallback split reached training indistinguishable from a successful one.
        """
        from pathlib import Path

        text = (Path(__file__).resolve().parents[1] / "train_phase4_optimized.py").read_text(
            encoding="utf-8"
        )
        assert "def make_nih_patient_split_with_min_positives" not in text, (
            "the untested duplicate split is back"
        )
        assert "nih_patient_split_with_min_positives(" in text, (
            "the script must call the tested split"
        )
        assert "split_info" in text, "the caller must read what the split reported"
