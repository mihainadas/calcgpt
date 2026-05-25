#!/usr/bin/env python3
"""
Generate an arithmetic dataset designed for length generalization.

Two ideas, both well-known in the transformer-arithmetic literature:

  1. **Reverse the answer.**  Standard `12+34=46` forces the decoder to emit
     the most significant digit FIRST, before it can possibly know the
     carries.  Writing `12+34=64` (the result `46` reversed) lets the model
     decode digit by digit in the natural carry order — units, tens,
     hundreds — which is the algorithm humans use.

  2. **Train across many digit counts.**  Random pairs are drawn uniformly
     over digit lengths 1..max_digits so the model sees the full range of
     sequence lengths, not just one.

Together these enable a small transformer to learn an actual carry
algorithm instead of memorizing a lookup table.
"""

import argparse
import random
from pathlib import Path


def sample_examples(n: int, max_digits: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out: set[str] = set()
    digit_choices = list(range(1, max_digits + 1))
    while len(out) < n:
        da = rng.choice(digit_choices)
        db = rng.choice(digit_choices)
        lo_a = 0 if da == 1 else 10 ** (da - 1)
        lo_b = 0 if db == 1 else 10 ** (db - 1)
        a = rng.randint(lo_a, 10 ** da - 1)
        b = rng.randint(lo_b, 10 ** db - 1)
        op = rng.choice(["+", "-"])
        if op == "-" and a < b:
            a, b = b, a
        result = a + b if op == "+" else a - b
        out.add(f"{a}{op}{b}={str(result)[::-1]}")
    examples = list(out)
    rng.shuffle(examples)
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("-n", "--num-examples", type=int, default=60000)
    parser.add_argument("--max-digits", type=int, default=6)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("datasets/ds-calcgpt-v2.txt"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    examples = sample_examples(args.num_examples, args.max_digits, args.seed)

    with args.output.open("w") as f:
        f.write("\n".join(examples) + "\n")

    print(f"wrote {len(examples):,} examples -> {args.output}")
    print(f"longest sequence: {max(len(e) for e in examples)} characters")
    print(f"first 5 samples (note the reversed RHS):")
    for ex in examples[:5]:
        print(f"  {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
