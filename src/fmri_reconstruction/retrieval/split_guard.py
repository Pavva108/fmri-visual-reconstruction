from __future__ import annotations

from typing import Iterable


def assert_no_test_overlap(train_paths: Iterable[str], test_paths: Iterable[str]) -> None:
    train = {str(p) for p in train_paths}
    test = {str(p) for p in test_paths}
    overlap = train.intersection(test)
    if overlap:
        raise AssertionError(f"Retrieval database contains {len(overlap)} test images.")


def assert_subject_split(train_subjects, val_subjects, test_subjects):
    train = set(train_subjects)
    val = set(val_subjects)
    test = set(test_subjects)
    if train & val or train & test or val & test:
        raise AssertionError("Train/validation/test subject sets overlap.")
