# CalcGPT model card

## Status

CalcGPT is an educational arithmetic-language-model experiment, not a production calculator. The repository currently provides training and evaluation code; canonical pretrained weights should be published separately with a release before any result is considered independently reproducible.

## Model family

CalcGPT uses a small GPT-2-style causal transformer trained from scratch on character sequences. The primary experiment uses fixed-width, zero-padded operands and reverses the answer so decoding follows the direction of arithmetic carries.

The canonical architecture and training settings are recorded in [`configs/padded-3digit.toml`](configs/padded-3digit.toml).

## Intended uses

- Teaching transformer training and decoding concepts.
- Studying interpolation and length/generalization behavior on synthetic arithmetic.
- Reproducing small CPU-friendly experiments.
- Comparing data representations and evaluation protocols.

## Out-of-scope uses

- General-purpose or high-assurance calculation.
- Financial, medical, legal, safety-critical, or authorization decisions.
- Claims about mathematical reasoning beyond the evaluated task and operand range.
- Processing untrusted model artifacts as though they were safe data.

## Training data

Training data is synthetic. See [DATA_CARD.md](DATA_CARD.md) for formats, generation parameters, checksums, and known limitations.

## Evaluation

Published artifacts must be evaluated on an explicitly disjoint set of operand/operator tuples. An evaluation report should include:

- Model and dataset SHA-256 checksums.
- Git revision and experiment configuration.
- Evaluation seed, sample size, and decoding parameters.
- Accuracy by operation and operand-width bucket.
- Format validity, exact match, and arithmetic correctness.
- Hardware and dependency versions.

Existing prose results in the repository should be treated as historical observations until accompanied by such an artifact and report.

## Limitations

- The learned behavior is tied to the trained token format and positional range.
- Subtraction data excludes negative answers.
- Wider operands, multiplication, division, decimals, and malformed input are not covered by the canonical experiment.
- A language model can produce plausible but incorrect arithmetic.
- Results may vary across dependency versions, devices, and nondeterministic kernels.

## Release artifact requirements

A released model must contain `model.safetensors`, model configuration,
`tokenizer.json`, `task_spec.json`, `training_manifest.json`, this model card or a
versioned derivative, and a checksum file. Optimizer and scheduler state are
development checkpoints and must not be part of the public release artifact.
