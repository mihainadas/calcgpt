"""Lightweight dataset loading, augmentation, and leakage-safe splitting."""

from __future__ import annotations

import random
import re
from pathlib import Path

_EQUATION_RE = re.compile(r"^(\d+)([+-])(\d+)=(.+)$")


def load_dataset(dataset_path: Path) -> list[str]:
    """Load non-empty examples from a UTF-8 text dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    examples = [
        line.strip()
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not examples:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    return examples


def augment_data(examples: list[str]) -> list[str]:
    """Add missing commutative addition twins to a training partition."""
    augmented = list(examples)
    known = set(augmented)
    for example in examples:
        match = _EQUATION_RE.fullmatch(example)
        if match is None:
            continue
        left, operator, right, answer = match.groups()
        if operator != "+":
            continue
        twin = f"{right}+{left}={answer}"
        if twin not in known:
            known.add(twin)
            augmented.append(twin)
    return augmented


def canonical_group_key(example: str) -> str:
    """Return a semantic group key for exact and commutative twins."""
    match = _EQUATION_RE.fullmatch(example)
    if match is None:
        return example
    left, operator, right, _ = match.groups()
    if operator == "+":
        a, b = sorted((int(left), int(right)))
        return f"+:{a}:{b}"
    return f"-:{int(left)}:{int(right)}"


def split_examples_grouped(
    examples: list[str], validation_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split deterministically without placing semantic twins on both sides."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if validation_fraction == 0 or len(examples) < 2:
        return list(examples), []

    groups: dict[str, list[str]] = {}
    for example in examples:
        groups.setdefault(canonical_group_key(example), []).append(example)
    if len(groups) < 2:
        raise ValueError("at least two semantic groups are required for validation")

    group_items = sorted(groups.items())
    random.Random(seed).shuffle(group_items)
    target = max(1, round(len(examples) * validation_fraction))
    validation: list[str] = []
    training: list[str] = []
    for _, members in group_items:
        destination = validation if len(validation) < target else training
        destination.extend(members)

    if not training:
        _, moved = group_items[-1]
        for member in moved:
            validation.remove(member)
        training.extend(moved)
    return training, validation
