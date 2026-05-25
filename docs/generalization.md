# How CalcGPT learns to generalize

A small transformer (≈500K parameters, four layers) trained on **40,000
random arithmetic problems** answers the remaining **999,960** problems
in the 0–999 range with **100% accuracy**. This file explains why that
works.

## The starting point — and why it failed

The original demo trained on every `a + b = c` and `a − b = c` for
`a, b ∈ [0, 100]`. About 15,000 examples. After 25 epochs the model
solved the training distribution at 99% accuracy and **0% accuracy on
anything outside it.** The cliff is total:

| Operand range | Accuracy |
|---|---|
| 0–100 (training) | 99.5% |
| 101–200 (just over) | 1.5% |
| 201–500 | 0% |
| 500+ | 0% |

Two reasons:

1. **It memorized.** With 800K parameters and only ~10K unique pairs in
   training, the cheapest loss-reducer is to learn the answer table.
   Carry-add is harder to discover than memorization, so gradient
   descent doesn't bother.
2. **It learned answer-length statistics.** Training answers were 1–3
   digits. On `146 + 175` (which has a 3-digit answer 321) the model
   often emits a single digit and stops, because it's also learned
   "stop after ~2 chars."

Below the cliff, the model behaves like a transformer pretending it
knows arithmetic: it emits plausibly-shaped numbers that happen to be
wrong.

## Fix #1 — Reverse the answer

A standard left-to-right decoder must emit the **most significant
digit first**:

    1234 + 5678 = 6912
                  ^ this digit first

But to know that leading "6" is correct, the model has to compute the
entire sum *before* emitting anything. That's a hard credit-assignment
problem during training.

Reverse the answer:

    1234 + 5678 = 2196
                  ^ units digit first (carry direction)

Now decoding follows the natural carry order: units, then tens, then
hundreds, then thousands. To predict the next answer digit, the model
needs only the corresponding place-value digits of each operand plus
the carry state from the previous step. This is something a small
transformer can actually learn.

Reversing alone helped, but on its own it wasn't enough — the
positions of "the units digit of operand A" still vary with operand
length.

## Fix #2 — Zero-pad the operands to a fixed width

Standard text: `7 + 8 = 51` (reversed answer `51` = 15).

Padded text: `007 + 008 = 5100`.

Now every digit lives at a *known absolute position* from the start of
the sequence:

    position 0: hundreds of operand A
    position 1: tens of operand A
    position 2: units of operand A
    position 3: operator
    position 4: hundreds of operand B
    ...
    position 8: units of the reversed answer  ← emit this first
    position 9: tens of the reversed answer
    ...

GPT-2 uses *learned* absolute positional embeddings. With zero-padding,
position 8's embedding consistently means "the units digit of the
answer," across every training example. The model learns one small
attention pattern — "attend to position 2 and position 6 to produce
position 8" — and applies it across every (a, b) pair. The same
applies for every other digit position.

Without padding, the units digit of operand A might be at position 0
(if A is one-digit) or position 4 (if A is five-digit). The model
would need to learn a *length-aware* attention pattern, which is much
harder than a fixed one.

## Putting both together

Format: `0000007+0000008=51000000` becomes the new training example
for "what is 7 + 8?". With operands padded to width W:

- Sequence length is fixed: `W + 1 + W + 1 + (W+1)` characters.
- Positional embeddings encode place value directly.
- The decoder emits in carry order.

The model only has to learn one circuit, and that circuit factors
across digit positions. It works on every pair in the 10^W × 10^W
input space, not just the ones it happened to see.

## Empirical result

Same model size (534K parameters, 128-dim, 4 layers, 8 heads), same
hardware (CPU), same hours of compute (~16 minutes for the padded
run vs ~4 minutes for the memorizer).

| Test | Old model (0–100 plain) | New model (0–999 padded+reversed) |
|---|---|---|
| Random pair, in-distribution | 99.5% | 100% |
| Random pair, full operand range | 27% (across 0–999) | **100%** |
| 1-digit pairs | 99.5% | 100% |
| 2-digit pairs | 99.5% | 100% |
| 3-digit pairs | 0% | **100%** |
| Edge cases (999+999, 100−1, 500+500) | mostly wrong | all correct |

Each held-out pair in the test set is almost certainly absent from
training (40K samples out of 10⁶ possibilities). 100% on those means
the model has learned an *algorithm*, not a *table*.

## What's not solved

- **Operands wider than the trained width.** Position 12 was never
  occupied during training, so the model has no positional embedding
  for it. Generalization past the padding width requires additional
  tricks — randomized PEs, NoPE, or Abacus-style position-of-digit
  embeddings (Lee et al., 2024).

- **Multiplication, division.** The carry algorithm here is
  position-local with single-bit state. Multiplication has cross-position
  dependencies (each partial product affects multiple places) that this
  setup doesn't address.

- **Negative results.** Subtraction examples in training keep `a ≥ b`,
  so the model has never seen a negative answer.

These are all natural next steps if you want to push the demo further.

## References

- Lee, Y. et al. *Teaching Arithmetic to Small Transformers.* 2023.
  https://arxiv.org/abs/2307.03381
- Nogueira, R. et al. *Investigating the Limitations of Transformers with
  Simple Arithmetic Tasks.* 2021. https://arxiv.org/abs/2102.13019
- McLeish, S. et al. *Transformers Can Do Arithmetic with the Right
  Embeddings (Abacus).* 2024. https://arxiv.org/abs/2405.17399

The reverse-answer trick has been independently reported in all three
papers; the zero-pad-and-align-positions formulation is the most
practical for a small from-scratch GPT-2 model.
