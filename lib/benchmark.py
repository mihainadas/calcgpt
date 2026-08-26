"""Deterministic task-space and held-out sampling helpers for CalcGPT."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

from .representation import RepresentationSpec, Task, task_roster_sha256, validate_task

BucketedTask = Tuple[int, int, int, str]
_EXAMPLE_PATTERN = re.compile(r"^(\d+)([+-])(\d+)=\d+$")
HOLDOUT_MANIFEST_SCHEMA = "calcgpt-semantic-holdout"
HOLDOUT_MANIFEST_VERSION = 1


def _require_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def task_space_size(operand_width: int) -> int:
    """Count all ordered additions and canonical nonnegative subtractions."""
    _require_positive(operand_width, "operand_width")
    operand_count = 10 ** operand_width
    return operand_count**2 + operand_count * (operand_count + 1) // 2


def decode_task_id(task_id: int, operand_width: int) -> Task:
    """Map a dense full-domain task ID to a normalized task."""
    capacity = task_space_size(operand_width)
    if not isinstance(task_id, int) or isinstance(task_id, bool) or not 0 <= task_id < capacity:
        raise ValueError("task_id is outside the task space")
    operand_count = 10**operand_width
    additions = operand_count**2
    if task_id < additions:
        return task_id // operand_count, "+", task_id % operand_count

    subtraction_id = task_id - additions
    left = (math.isqrt(8 * subtraction_id + 1) - 1) // 2
    row_start = left * (left + 1) // 2
    right = subtraction_id - row_start
    return left, "-", right


def sample_task_roster(count: int, operand_width: int, seed: int = 42) -> List[Task]:
    """Sample an ordered, normalized task roster uniformly without replacement."""
    _require_positive(count, "count")
    capacity = task_space_size(operand_width)
    if count > capacity:
        raise ValueError(
            f"requested {count:,} tasks, but width {operand_width} has only "
            f"{capacity:,} distinct tasks"
        )
    task_ids = random.Random(seed).sample(range(capacity), count)
    return [decode_task_id(task_id, operand_width) for task_id in task_ids]


def magnitude_bounds(digit_count: int) -> Tuple[int, int]:
    """Return the inclusive operand bounds used by a magnitude bucket."""
    _require_positive(digit_count, "digit_count")
    return (0 if digit_count == 1 else 10 ** (digit_count - 1), 10**digit_count - 1)


def magnitude_task_space_size(digit_count: int) -> int:
    """Count tasks where both operands belong to one magnitude bucket."""
    lo, hi = magnitude_bounds(digit_count)
    operand_count = hi - lo + 1
    return operand_count**2 + operand_count * (operand_count + 1) // 2


def magnitude_semantic_group_count(digit_count: int) -> int:
    """Count addition-equivalence groups plus canonical subtraction tasks."""
    lo, hi = magnitude_bounds(digit_count)
    operand_count = hi - lo + 1
    triangular = operand_count * (operand_count + 1) // 2
    return triangular * 2


def parse_example(example: str) -> Task:
    """Extract the task from a complete plain or reversed-answer equation."""
    match = _EXAMPLE_PATTERN.fullmatch(example.strip())
    if match is None:
        raise ValueError(f"invalid arithmetic example: {example!r}")
    a_text, op, b_text = match.groups()
    return int(a_text), op, int(b_text)


def load_examples(path: Union[str, Path]) -> List[str]:
    """Load nonempty examples from a dataset file."""
    with Path(path).open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def format_task(task: Task, operand_width: int) -> str:
    """Format a task in CalcGPT's fixed-width, reversed-answer representation."""
    return RepresentationSpec.from_name("padded-reversed", operand_width).format_task(task)


def semantic_task_key(task: Task) -> str:
    """Return the equivalence key used for leakage-safe arithmetic holdouts."""
    left, operator, right = task
    if operator == "+":
        low, high = sorted((left, right))
        return f"+:{low}:{high}"
    return f"-:{left}:{right}"


def _encode_semantic_bucket_group(task: Task, digit_count: int) -> Optional[int]:
    left, operator, right = task
    lo, hi = magnitude_bounds(digit_count)
    if not (lo <= left <= hi and lo <= right <= hi):
        return None
    operand_count = hi - lo + 1
    triangular = operand_count * (operand_count + 1) // 2
    if operator == "+":
        high, low = sorted((left - lo, right - lo), reverse=True)
        return high * (high + 1) // 2 + low
    if operator == "-" and left >= right:
        left_offset = left - lo
        return triangular + left_offset * (left_offset + 1) // 2 + (right - lo)
    return None


def _decode_semantic_bucket_group(
    group_id: int, digit_count: int, reverse_addition: bool
) -> Task:
    lo, hi = magnitude_bounds(digit_count)
    operand_count = hi - lo + 1
    triangular = operand_count * (operand_count + 1) // 2
    if not 0 <= group_id < triangular * 2:
        raise ValueError("group_id is outside the semantic magnitude bucket")
    operator = "+" if group_id < triangular else "-"
    triangular_id = group_id if operator == "+" else group_id - triangular
    high = (math.isqrt(8 * triangular_id + 1) - 1) // 2
    row_start = high * (high + 1) // 2
    low = triangular_id - row_start
    left, right = lo + high, lo + low
    if operator == "+" and reverse_addition and left != right:
        left, right = right, left
    return left, operator, right


def _allowed_id(rank: int, blocked_ids: Sequence[int], capacity: int) -> int:
    """Map a rank in the compressed allowed set back to its task ID."""
    low = rank
    high = rank + len(blocked_ids)
    while low < high:
        middle = (low + high) // 2
        allowed_through_middle = middle + 1 - bisect.bisect_right(blocked_ids, middle)
        if allowed_through_middle > rank:
            high = middle
        else:
            low = middle + 1
    if low >= capacity or low in blocked_ids:
        raise RuntimeError("failed to map held-out task rank")
    return low


def sample_heldout_tasks(
    count: int,
    digit_count: int,
    excluded_examples: Iterable[Union[str, Task]],
    seed: int = 42,
) -> List[Task]:
    """Sample unique tasks absent from every excluded semantic task group."""
    _require_positive(count, "count")
    capacity = magnitude_semantic_group_count(digit_count)
    blocked = set()
    for value in excluded_examples:
        task = parse_example(value) if isinstance(value, str) else value
        group_id = _encode_semantic_bucket_group(task, digit_count)
        if group_id is not None:
            blocked.add(group_id)

    available = capacity - len(blocked)
    if count > available:
        raise ValueError(
            f"requested {count:,} held-out tasks, but only {available:,} "
            f"remain in the {digit_count}-digit bucket"
        )

    blocked_ids = sorted(blocked)
    rng = random.Random(seed)
    allowed_ranks = rng.sample(range(available), count)
    group_ids = [_allowed_id(rank, blocked_ids, capacity) for rank in allowed_ranks]
    return [
        _decode_semantic_bucket_group(group_id, digit_count, bool(rng.getrandbits(1)))
        for group_id in group_ids
    ]


def sample_heldout_by_magnitude(
    per_bucket: int,
    max_digits: int,
    excluded_examples: Iterable[Union[str, Task]],
    seed: int = 42,
) -> List[BucketedTask]:
    """Sample a guaranteed-held-out benchmark for every digit-count bucket."""
    _require_positive(per_bucket, "per_bucket")
    _require_positive(max_digits, "max_digits")
    excluded = list(excluded_examples)
    seed_rng = random.Random(seed)
    output: List[BucketedTask] = []
    for digit_count in range(1, max_digits + 1):
        bucket_seed = seed_rng.randrange(2**63)
        for a, op, b in sample_heldout_tasks(
            per_bucket, digit_count, excluded, bucket_seed
        ):
            output.append((digit_count, a, b, op))
    return output


def arithmetic_features(task: Task, operand_width: int) -> dict[str, object]:
    """Describe carry/borrow structure and simple arithmetic edge cases."""
    validate_task(task, operand_width)
    left, operator, right = task
    left_digits = [int(char) for char in str(left).zfill(operand_width)][::-1]
    right_digits = [int(char) for char in str(right).zfill(operand_width)][::-1]

    incoming = 0
    event_count = 0
    current_chain = 0
    longest_chain = 0
    for left_digit, right_digit in zip(left_digits, right_digits):
        if operator == "+":
            event = left_digit + right_digit + incoming >= 10
        else:
            event = left_digit - incoming < right_digit
        incoming = 1 if event else 0
        if event:
            event_count += 1
            current_chain += 1
            longest_chain = max(longest_chain, current_chain)
        else:
            current_chain = 0

    result = left + right if operator == "+" else left - right
    return {
        "operation": "addition" if operator == "+" else "subtraction",
        "event_kind": "carry" if operator == "+" else "borrow",
        "event_count": event_count,
        "longest_chain": longest_chain,
        "overflow": result >= 10**operand_width,
        "has_zero_operand": left == 0 or right == 0,
        "equal_operands": left == right,
    }


def build_semantic_holdout_manifest(
    per_bucket: int,
    max_digits: int,
    excluded_examples: Iterable[Union[str, Task]],
    seed: int = 42,
) -> dict[str, Any]:
    """Build a deterministic, versioned manifest for one semantic holdout."""
    excluded_values = list(excluded_examples)
    excluded_tasks = [
        parse_example(value) if isinstance(value, str) else value
        for value in excluded_values
    ]
    sampled = sample_heldout_by_magnitude(
        per_bucket, max_digits, excluded_tasks, seed=seed
    )
    tasks = [(left, operator, right) for _, left, right, operator in sampled]
    records = []
    for (digit_count, left, right, operator), task in zip(sampled, tasks):
        records.append(
            {
                "digit_count": digit_count,
                "left": left,
                "operator": operator,
                "right": right,
                "semantic_key": semantic_task_key(task),
                "features": arithmetic_features(task, max_digits),
            }
        )
    return {
        "schema": HOLDOUT_MANIFEST_SCHEMA,
        "schema_version": HOLDOUT_MANIFEST_VERSION,
        "operand_width": max_digits,
        "seed": seed,
        "samples_per_digit_bucket": per_bucket,
        "exclusion_policy": "exact-tasks-and-commutative-addition-twins",
        "excluded_task_roster_sha256": task_roster_sha256(excluded_tasks),
        "task_roster_sha256": task_roster_sha256(tasks),
        "task_count": len(tasks),
        "tasks": records,
    }


def holdout_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """Serialize a holdout manifest canonically for storage or hashing."""
    if manifest.get("schema") != HOLDOUT_MANIFEST_SCHEMA:
        raise ValueError("unsupported holdout manifest schema")
    if manifest.get("schema_version") != HOLDOUT_MANIFEST_VERSION:
        raise ValueError("unsupported holdout manifest schema version")
    return (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def holdout_manifest_sha256(manifest: dict[str, Any]) -> str:
    """Hash the canonical bytes of a semantic holdout manifest."""
    return hashlib.sha256(holdout_manifest_bytes(manifest)).hexdigest()
