# CalcGPT

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A small from-scratch GPT-2 that learned to add and subtract — and, with
the right data format, **actually generalizes** instead of memorizing
the answer table.

## The result

Two models, same architecture (≈ 534K parameters, 128-dim, 4 layers,
8 heads), same hardware (CPU):

| Model | Training data | Format | 0–100 | 0–999 |
|---|---|---|---|---|
| `calcgpt-demo` | all pairs in `[0, 100]` | `12+34=46` | 99.5 % | 27 % |
| `calcgpt-padded` | 40k random pairs in `[0, 999]` | `012+034=064` reversed | 100 % | **100 %** |

Each held-out test pair is almost certainly absent from the new
model's training set (40 k samples out of 10⁶ possibilities), so 100 %
accuracy means the network has learned a real algorithm. The technique
is documented in [`docs/generalization.md`](docs/generalization.md).

## Live demo

```bash
pip install -r requirements.txt

# 1. The "memorizer" — trains in ~4 minutes on CPU
python calcgpt_train.py \
    --epochs 25 --batch-size 64 \
    --embedding-dim 128 --num-layers 4 --num-heads 8 \
    -o models/calcgpt-demo

# 2. The "generalizer" — trains in ~16 minutes on CPU
python scripts/gen_padded.py -w 3
python calcgpt_train.py \
    -d datasets/ds-calcgpt-padded.txt \
    -o models/calcgpt-padded \
    --epochs 30 --batch-size 64 \
    --embedding-dim 128 --num-layers 4 --num-heads 8 --feedforward-dim 256 \
    --learning-rate 1e-3 --warmup-steps 100 \
    --n-positions 20 --save-steps 2000 --no-augmentation

python demo.py
```

`demo.py` shows the model's architecture, streams a few generations
token-by-token, evaluates accuracy on 100 unseen pairs per digit-count
bucket, runs a head-to-head against the memorizer, peeks at the top-k
probabilities for one decoding step, and drops you into an interactive
prompt.

## The trick, in one paragraph

Standard `7+8=15` makes the decoder emit the most significant digit
first, which requires knowing the whole sum before writing anything.
Zero-pad operands to a fixed width and reverse the answer
(`007+008=51000`) and the decoder emits units, then tens, then
hundreds — the natural carry direction. Padding also pins every digit
to a known absolute position, so GPT-2's learned positional embeddings
line up with place value. The model learns one small "add digit at
position p with carry" circuit and applies it across the whole input
space.

See [`docs/generalization.md`](docs/generalization.md) for the full
write-up, ablations, and references.

## Project layout

```
calcgpt/
├── demo.py                    # Live walkthrough (run this)
├── calcgpt.py                 # Inference CLI
├── calcgpt_train.py           # Training CLI
├── calcgpt_eval.py            # Evaluation CLI
├── calcgpt_dategen.py         # Dataset generation CLI (for the memorizer)
├── scripts/
│   └── gen_padded.py          # Fixed-width zero-padded dataset (for the generalizer)
├── lib/                       # Library: tokenizer, training, inference, evaluation
├── datasets/                  # Training data
├── docs/
│   └── generalization.md      # Why the new model generalizes
└── calcgpt.ipynb              # Tutorial notebook
```

## CLI reference

### `calcgpt.py` — inference

```bash
# Interactive
python calcgpt.py -i

# Batch
python calcgpt.py -b "1+1" "23+58" "99-50"

# From file, JSON output
python calcgpt.py -f problems.txt -o out.json --format json

# Pick a specific model
python calcgpt.py -m models/calcgpt-padded -b "100+200"
```

When the model uses zero-padded operands, run `demo.py` instead — it
handles the padding/un-reversing automatically.

### `calcgpt_train.py` — training

```bash
python calcgpt_train.py \
    -d datasets/ds-calcgpt.txt \
    -o models/my-model \
    --epochs 30 --batch-size 64 \
    --embedding-dim 128 --num-layers 4 --num-heads 8 \
    --learning-rate 1e-3 \
    --n-positions 20    # explicit context window (defaults to data maxlen + 10)
```

Pass `--help` to see every flag. Models are saved at the end of
training and at `--save-steps` intervals along the way.

### `calcgpt_eval.py` — evaluation

```bash
python calcgpt_eval.py --sample 200
```

Runs three test types — `first_operand`, `expression_complete`,
`answer_complete` — and reports format validity, arithmetic
correctness, and latency.

### `scripts/gen_padded.py` — fixed-width dataset

```bash
python scripts/gen_padded.py -n 40000 -w 3 -o datasets/ds-calcgpt-padded.txt
```

Generates `N` random `(a, op, b)` pairs with operands in `[0, 10^W − 1]`,
zero-padded to width `W` and the answer written in reverse.

### `calcgpt_dategen.py` — exhaustive dataset (for the memorizer)

```bash
python calcgpt_dategen.py -m 100
```

Generates every pair in `[0, M]` with both operations. Used for the
old `calcgpt-demo` model.

## Tokenizer

Character-level by default — vocab is just the digits, `+`, `-`, `=`,
plus `<pad>` and `<eos>` (15 tokens). The tokenizer also supports a
number-level mode (`0`–`99` as single tokens) via
`CalcGPTTokenizer.from_dataset(mode='number')`, but the demo and the
padded format only use the character tokenizer.

## Hardware and performance

Everything in this repo trains and runs on CPU. The numbers in the
table above were measured on a 4-core CPU container with no GPU; the
generalizer's full training run takes ≈ 16 minutes there, and the demo
itself runs at ≈ 7 ms per problem.

## License

MIT — see [`LICENSE`](LICENSE).

## References

The generalization technique combines ideas from:

- Lee et al., *Teaching Arithmetic to Small Transformers* (2023)
  — [arXiv:2307.03381](https://arxiv.org/abs/2307.03381)
- Nogueira et al., *Investigating the Limitations of Transformers with
  Simple Arithmetic Tasks* (2021)
  — [arXiv:2102.13019](https://arxiv.org/abs/2102.13019)
- McLeish et al., *Transformers Can Do Arithmetic with the Right
  Embeddings (Abacus)* (2024)
  — [arXiv:2405.17399](https://arxiv.org/abs/2405.17399)
