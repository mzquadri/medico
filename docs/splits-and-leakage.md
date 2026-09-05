# Splits and patient leakage

A chest X-ray dataset contains several images of the same person. If a split is
drawn over images rather than over people, some of that person's images land in
training and others in evaluation. The model is then scored partly on patients it
has already seen, and the number that comes out is not a measure of
generalisation. It is the most common way a medical imaging result is quietly
wrong, and it is not visible in the metric itself.

This document records what each split does and what was found when they were
audited.

## The three sources

| Source | Patient identifier | Split |
| --- | --- | --- |
| CheXpert | `patientNNNNN` inside the `Path` column | By patient, 80/10/10 |
| NIH ChestX-ray14 | `Patient ID` column | By patient, 80/10/10, redrawn for positives |
| Kermany pneumonia | none published | By patient recovered from the filename, 70/15/15 |

## What the audit found

**CheXpert and NIH were already grouped by patient.** Both extract a patient
identifier and assign whole patients to one split. Rows whose patient identifier
cannot be parsed are dropped rather than allowed into several splits.

**The pneumonia split was not.** It collected every image from the published
train and test folders, separated them into two lists by class, shuffled each
list and cut it 70/15/15. That is stratification by class, and it has no notion
of a patient at all. Nothing in the file linked an image to a person.

The Kermany dataset publishes no patient column, which is presumably why. It does
publish the identity in the filenames: the positive images are named
`personN_bacteria_M.jpeg` and `personN_virus_M.jpeg`, and one person contributes
several images. A class-stratified shuffle over those files splits people across
the boundary.

## The fix

`medico/splits.py` recovers the group from the filename and assigns whole groups.

- `personN` for the positive images.
- The `IM-NNNN` study token for the negatives.
- Anything unrecognised becomes its own group, which is never worse than the
  per-image split it replaces.

The person number is used without its source folder. Numbering restarts between
the published train and test folders, so keying on the number alone can merge two
different people into one group. That is the safe direction: merging only keeps
images together. Separating them would risk the opposite.

Within a class, groups are handed to whichever split is furthest below its
target, largest group first, so the class balance stays close to the requested
fractions without ever dividing a patient.

## Verification

The training script now checks all three splits and stops rather than training on
a leaking split:

```python
leaked = overlap(pneu_train_list, pneu_val_list, pneu_test_list,
                 key=lambda r: pneumonia_group_key(r['image_path']))
if leaked:
    raise RuntimeError(...)
```

The same check runs on the CheXpert and NIH splits, which were already correct.
A guard that only runs where a bug was found does not protect the other two.

`tests/test_splits.py` covers the property directly. One test asserts that no
person appears in two splits. Another builds the naive class-stratified split and
asserts that it *does* leak, so the first test cannot pass by accident on a
dataset where grouping makes no difference.

## What this does not fix

Grouping is only as good as the identity recovered from the filename. If two
different people share a person number across the published folders, they are
treated as one, which costs a little flexibility and no correctness. If one person
appears under two different numbers, the grouping will not catch it, and nothing
in the published data would reveal that.

The published train and test folders are pooled before splitting. That is a
deliberate choice for a multi-dataset fine-tune, but it means results here are not
comparable with papers that report on the dataset's official test split.

No result in this repository has been recomputed against the corrected split,
because no trained weights or metrics exist here to recompute. The correction
applies to any future run.
