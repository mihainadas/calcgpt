# CalcGPT

[![CI](https://github.com/mihainadas/calcgpt/actions/workflows/ci.yml/badge.svg)](https://github.com/mihainadas/calcgpt/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

CalcGPT is a small GPT-2-style language model for controlled experiments in
arithmetic representation and within-width generalization. The core experiment
asks a practical question: can a compact causal transformer learn a reusable
digit-and-carry procedure when arithmetic is represented in a decoder-friendly
format?

The repository includes deterministic dataset generation, leakage-safe splits,
strict held-out sampling, training and evaluation CLIs, self-contained model
artifacts, tests, and reproducibility workflows.

## Representation

For a three-digit task, operands are zero-padded and the answer is reversed:

```text
007+008=5100
```

The model emits units before tens, following the natural carry direction. Fixed
positions also align each place value with the same learned positional embedding.
This setup tests combinatorial generalization inside the trained operand width; it
does **not** establish generalization to wider operands.

## Current evidence standard

The canonical width-3 task space contains 1,500,500 distinct tasks: 1,000,000
ordered additions and 500,500 nonnegative subtractions. The committed training
dataset contains 40,000 tasks generated without replacement.

`demo.py` constructs 300 unique benchmark tasks—100 per operand-magnitude
bucket—and verifies exact zero overlap with the training dataset before scoring.
The repository deliberately does not present a headline accuracy until a model
artifact, manifest, and benchmark report are published together. Earlier results
should be treated as observations pending that reproducibility bundle.

## Quick start

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train,demo]"
```

Generate the canonical dataset and train the model:

```bash
python scripts/gen_padded.py \
  --num-examples 40000 \
  --operand-width 3 \
  --seed 42 \
  --output datasets/ds-calcgpt-padded.txt

python calcgpt_train.py \
  --dataset datasets/ds-calcgpt-padded.txt \
  --output-dir models/calcgpt-padded \
  --task-format padded-reversed \
  --operand-width 3 \
  --epochs 30 \
  --batch-size 64 \
  --embedding-dim 128 \
  --num-layers 4 \
  --num-heads 8 \
  --feedforward-dim 256 \
  --learning-rate 1e-3 \
  --warmup-steps 100 \
  --n-positions 20 \
  --save-steps 2000 \
  --no-augmentation

python demo.py
```

Because the model bundle includes `task_spec.json`, the regular inference CLI
automatically pads inputs and reverses outputs for this model:

```bash
python calcgpt.py --model models/calcgpt-padded --batch "7+8" "123+456"
```

The reviewable experiment configuration is also recorded in
[`configs/padded-3digit.toml`](configs/padded-3digit.toml).

## Model artifacts

Training writes a self-contained directory containing:

- Hugging Face model configuration and Safetensors weights;
- `tokenizer.json` with the exact ordered vocabulary and special-token IDs;
- `task_spec.json` with operand width, answer direction, and supported operators;
- `training_manifest.json` with the dataset SHA-256, grouped split counts,
  configuration, seed, environment, metrics, and Git revision.

Inference refuses to guess a missing tokenizer. For a legacy model, provide the
exact training dataset explicitly:

```bash
python calcgpt.py \
  --model models/legacy-model \
  --legacy-dataset datasets/ds-calcgpt.txt \
  --batch "7+8"
```

See [`ARTIFACTS.md`](ARTIFACTS.md) and [`MODEL_CARD.md`](MODEL_CARD.md) for the
publication contract and limitations.

## Commands

```bash
python calcgpt_dategen.py --help   # exhaustive plain-format data
python scripts/gen_padded.py --help
python calcgpt_train.py --help
python calcgpt_eval.py --help
python calcgpt.py --help
```

`examples/complete_workflow.py` is a tiny end-to-end plumbing smoke test. It is
not intended as a benchmark.

## Verification

The core tests avoid heavyweight ML dependencies:

```bash
python -m compileall -q calcgpt.py calcgpt_train.py calcgpt_eval.py \
  calcgpt_dategen.py demo.py lib scripts examples
python -m unittest discover -s tests -p "test_*.py" -v
```

CI runs these checks on Python 3.11, 3.12, and 3.13. A separate scheduled/manual
workflow regenerates the canonical dataset and compares it byte-for-byte with the
tracked copy.

## Project layout

```text
configs/                 canonical experiment specifications
datasets/                tracked canonical datasets
docs/                    methodology notes
examples/                executable workflow example
lib/                     tokenizer, data, training, inference, evaluation
scripts/gen_padded.py    deterministic padded/reversed generator
tests/                   fast reproducibility and contract tests
demo.py                  strict held-out interactive walkthrough
```

## Research roadmap

The next scientific milestone is a controlled, multi-seed ablation comparing:

1. plain operands and normal answers;
2. reversed answers only;
3. padding only;
4. padding plus reversed answers.

Each run should use the same architecture, data budget, task space, grouped
splits, and benchmark manifest. Results should report exact task counts,
dataset/model hashes, multiple seeds, accuracy by carry/borrow structure, and
throughput separately from accuracy.

## Contributing and security

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow and
[`SECURITY.md`](SECURITY.md) for private vulnerability reporting. Model weights,
optimizer state, and large checkpoints do not belong in Git history.

## License

MIT. See [`LICENSE`](LICENSE).
