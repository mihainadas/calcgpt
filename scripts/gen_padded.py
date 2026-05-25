#!/usr/bin/env python3
"""
Generate an arithmetic dataset for length-generalization via fixed-width
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

The model learns ONE algorithm — add digit at position p_A to digit at
position p_B with carry — and applies it everywhere.  Because every
randomly sampled (a, b) pair in the training set drives the same circuit,
the model interpolates over the entire numerical range, not over a table
of seen pairs.

Reversing the answer lets the decoder emit units first (carries flow
naturally left to right in the output sequence).
"""

import argparse
import random
from pathlib import Path


def format_example(a: int, b: int, op: str, operand_width: int, answer_width: int) -> str:
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
    rng = random.Random(seed)
    lo = 0
    hi = 10 ** operand_width - 1
    answer_width = operand_width + 1  # max sum is 2*hi which has at most width+1 digits
    out: set[str] = set()
    while len(out) < n:
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        op = rng.choice(["+", "-"])
        if op == "-" and a < b:
            a, b = b, a
        out.add(format_example(a, b, op, operand_width, answer_width))
    examples = list(out)
    rng.shuffle(examples)
    return examples


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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    examples = sample_examples(args.num_examples, args.operand_width, args.seed)

    with args.output.open("w") as f:
        f.write("\n".join(examples) + "\n")

    print(f"wrote {len(examples):,} examples -> {args.output}")
    print(f"operand width: {args.operand_width} digits  (max value {10**args.operand_width - 1:,})")
    print(f"sequence length: {len(examples[0])} characters")
    print(f"first 5 samples:")
    for ex in examples[:5]:
        print(f"  {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
