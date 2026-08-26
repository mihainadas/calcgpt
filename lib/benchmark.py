"""Deterministic task-space and held-out sampling helpers for CalcGPT."""

from __future__ import annotations

import bisect
import math
import random
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

Task = Tuple[int, str, int]
BucketedTask = Tuple[int, int, int, str]
_EXAMPLE_PATTERN = re.compile(r"^(\d+)([+-])(\d+)=\d+$")


def _require_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def task_space_size(operand_width: int) -> int:
    """Count all ordered additions and canonical nonnegative subtractions."""
    _require_positive(operand_width, "operand_width")
    operand_count = 10 ** operand_width
    return operand_count**2 + operand_count * (operand_count + 1) // 2


def magnitude_bounds(digit_count: int) -> Tuple[int, int]:
    """Return the inclusive operand bounds used by a magnitude bucket."""
    _require_positive(digit_count, "digit_count")
    return (0 if digit_count == 1 else 10 ** (digit_count - 1), 10**digit_count - 1)


def magnitude_task_space_size(digit_count: int) -> int:
    """Count tasks where both operands belong to one magnitude bucket."""
    lo, hi = magnitude_bounds(digit_count)
    operand_count = hi - lo + 1
    return operand_count**2 + operand_count * (operand_count + 1) // 2


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
    _require_positive(operand_width, "operand_width")
    a, op, b = task
    limit = 10**operand_width
    if op not in {"+", "-"}:
        raise ValueError("op must be '+' or '-'")
    if not (0 <= a < limit and 0 <= b < limit):
        raise ValueError("operands must be nonnegative and fit operand_width")
    if op == "-" and a < b:
        raise ValueError("subtraction tasks must have a >= b")
    result = a + b if op == "+" else a - b
    answer = str(result).zfill(operand_width + 1)[::-1]
    return f"{a:0{operand_width}d}{op}{b:0{operand_width}d}={answer}"


def _decode_bucket_task(task_id: int, digit_count: int) -> Task:
    lo, hi = magnitude_bounds(digit_count)
    operand_count = hi - lo + 1
    additions = operand_count**2
    if not 0 <= task_id < magnitude_task_space_size(digit_count):
        raise ValueError("task_id is outside the magnitude bucket")
    if task_id < additions:
        return lo + task_id // operand_count, "+", lo + task_id % operand_count

    subtraction_id = task_id - additions
    a_offset = (math.isqrt(8 * subtraction_id + 1) - 1) // 2
    row_start = a_offset * (a_offset + 1) // 2
    b_offset = subtraction_id - row_start
    return lo + a_offset, "-", lo + b_offset


def _encode_bucket_task(task: Task, digit_count: int) -> Optional[int]:
    a, op, b = task
    lo, hi = magnitude_bounds(digit_count)
    if not (lo <= a <= hi and lo <= b <= hi):
        return None
    operand_count = hi - lo + 1
    if op == "+":
        return (a - lo) * operand_count + (b - lo)
    if op == "-" and a >= b:
        a_offset = a - lo
        return operand_count**2 + a_offset * (a_offset + 1) // 2 + (b - lo)
    return None


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
    seed: int = 0,
) -> List[Task]:
    """Sample unique tasks guaranteed absent from ``excluded_examples``."""
    _require_positive(count, "count")
    capacity = magnitude_task_space_size(digit_count)
    blocked = set()
    for value in excluded_examples:
        task = parse_example(value) if isinstance(value, str) else value
        task_id = _encode_bucket_task(task, digit_count)
        if task_id is not None:
            blocked.add(task_id)

    available = capacity - len(blocked)
    if count > available:
        raise ValueError(
            f"requested {count:,} held-out tasks, but only {available:,} "
            f"remain in the {digit_count}-digit bucket"
        )

    blocked_ids = sorted(blocked)
    rng = random.Random(seed)
    allowed_ranks = rng.sample(range(available), count)
    return [
        _decode_bucket_task(_allowed_id(rank, blocked_ids, capacity), digit_count)
        for rank in allowed_ranks
    ]


def sample_heldout_by_magnitude(
    per_bucket: int,
    max_digits: int,
    excluded_examples: Iterable[Union[str, Task]],
    seed: int = 0,
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
