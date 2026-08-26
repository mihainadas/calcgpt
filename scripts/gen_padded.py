#!/usr/bin/env python3
"""
Generate an arithmetic dataset for within-width generalization via fixed-width
zero-padding and a reversed answer.

Why this format generalizes
---------------------------

Standard text "7+8=15" forces the model to discover where the units digit
lives — and that position depends on operand length.  With fixed-width
zero-padding:

    "0000007+0000008=51000000"

every digit lives at a fixed position from left to right:

   position 0   millions of operand A
   position 1   hundred-thousands of operand A
   ...
   position 6   units of operand A
   position 7   '+'
   position 8   millions of operand B
   ...

This representation makes a reusable digit-and-carry procedure easier to
learn and allows exact held-out evaluation across the configured range.

Reversing the answer lets the decoder emit units first (carries flow
naturally left to right in the output sequence).
"""

import argparse
import math
import random
from pathlib import Path


def task_space_size(operand_width: int) -> int:
    """Return the number of distinct addition and nonnegative-subtraction tasks."""
    if operand_width <= 0:
        raise ValueError("operand_width must be positive")
    operand_count = 10 ** operand_width
    additions = operand_count * operand_count
    subtractions = operand_count * (operand_count + 1) // 2
    return additions + subtractions


def _task_from_id(task_id: int, operand_width: int) -> tuple[int, int, str]:
    """Map a dense task ID to one task without materializing the task space."""
    operand_count = 10 ** operand_width
    additions = operand_count * operand_count
    if task_id < additions:
        return task_id // operand_count, task_id % operand_count, "+"

    # Subtraction rows have lengths 1, 2, ..., operand_count for a=0,1,...
    subtraction_id = task_id - additions
    a = (math.isqrt(8 * subtraction_id + 1) - 1) // 2
    row_start = a * (a + 1) // 2
    b = subtraction_id - row_start
    return a, b, "-"


def format_example(a: int, b: int, op: str, operand_width: int, answer_width: int) -> str:
    if operand_width <= 0 or answer_width <= 0:
        raise ValueError("operand_width and answer_width must be positive")
    if op not in {"+", "-"}:
        raise ValueError("op must be '+' or '-'")
    limit = 10 ** operand_width
    if not (0 <= a < limit and 0 <= b < limit):
        raise ValueError("operands must be nonnegative and fit operand_width")
    if op == "-" and a < b:
        raise ValueError("subtraction tasks must have a >= b")
    if op == "+":
        r = a + b
    else:
        r = a - b
    a_s = str(a).zfill(operand_width)
    b_s = str(b).zfill(operand_width)
    r_s = str(r).zfill(answer_width)
    return f"{a_s}{op}{b_s}={r_s[::-1]}"


def sample_examples(
    n: int, operand_width: int, seed: int
) -> list[str]:
    if n <= 0:
        raise ValueError("n must be positive")
    capacity = task_space_size(operand_width)
    if n > capacity:
        raise ValueError(
            f"requested {n:,} examples, but width {operand_width} has "
            f"only {capacity:,} distinct tasks"
        )

    rng = random.Random(seed)
    # The largest sum has at most one more digit than either operand.
    answer_width = operand_width + 1
    task_ids = rng.sample(range(capacity), n)
    return [
        format_example(a, b, op, operand_width, answer_width)
        for task_id in task_ids
        for a, b, op in [_task_from_id(task_id, operand_width)]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("-n", "--num-examples", type=int, default=40000)
    parser.add_argument(
        "-w",
        "--operand-width",
        type=int,
        default=7,
        help="Zero-pad operands to this many digits (default 7)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("datasets/ds-calcgpt-padded.txt"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        examples = sample_examples(args.num_examples, args.operand_width, args.seed)
    except ValueError as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as f:
        f.write("\n".join(examples) + "\n")

    print(f"wrote {len(examples):,} examples -> {args.output}")
    print(f"operand width: {args.operand_width} digits  (max value {10**args.operand_width - 1:,})")
    print(f"task space: {task_space_size(args.operand_width):,} distinct tasks")
    print(f"sequence length: {len(examples[0])} characters")
    print("first 5 samples:")
    for ex in examples[:5]:
        print(f"  {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
