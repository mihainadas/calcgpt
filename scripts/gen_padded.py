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

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.benchmark import sample_task_roster, task_space_size
from lib.representation import RepresentationSpec


def format_example(a: int, b: int, op: str, operand_width: int, answer_width: int) -> str:
    if answer_width != operand_width + 1:
        raise ValueError("answer_width must equal operand_width + 1")
    spec = RepresentationSpec.from_name("padded-reversed", operand_width)
    return spec.format_task((a, op, b))


def sample_examples(
    n: int, operand_width: int, seed: int
) -> list[str]:
    roster = sample_task_roster(n, operand_width, seed)
    spec = RepresentationSpec.from_name("padded-reversed", operand_width)
    return spec.render_dataset(roster)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("-n", "--num-examples", type=int, default=40000)
    parser.add_argument(
        "-w",
        "--operand-width",
        type=int,
        default=3,
        help="Zero-pad operands to this many digits (default 3)",
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

    dataset_bytes = ("\n".join(examples) + "\n").encode("utf-8")
    args.output.write_bytes(dataset_bytes)

    print(f"wrote {len(examples):,} examples -> {args.output}")
    print(
        f"operand width: {args.operand_width} digits  "
        f"(max value {10**args.operand_width - 1:,})"
    )
    print(f"task space: {task_space_size(args.operand_width):,} distinct tasks")
    print(f"sequence length: {len(examples[0])} characters")
    print("first 5 samples:")
    for ex in examples[:5]:
        print(f"  {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
