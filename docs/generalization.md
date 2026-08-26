# Arithmetic representation and within-width generalization

CalcGPT studies how the textual representation of arithmetic changes what a
small causal transformer can learn. This note describes the hypothesis and the
evaluation protocol; it does not treat a single model run as proof that a neural
network has learned a symbolic algorithm.

## Hypothesis

A conventional decoder must emit the most-significant result digit first:

```text
123+456=579
        ^ first output token
```

That token can depend on carries originating several positions away. Reversing
the answer changes the generation order to units, tens, hundreds, and so on:

```text
123+456=9750
        ^ units digit first
```

Using fixed-width fields zero-pads both operands to `W` digits and the answer to
`W + 1` digits. This assigns each place value a stable absolute position:

```text
007+008=5100
```

For a width-3 addition, the two units digits always occupy positions 2 and 6,
and the first answer token always occupies position 8. This makes a reusable
digit-and-carry computation easier to represent with learned positional
embeddings.

The expected benefit is **within-width combinatorial generalization**: transfer
to unseen operand pairs inside the fixed width. Wider operands occupy positions
that were not trained for, so this setup is not a length-generalization method.

## Canonical task space

For operand width `W`, let `N = 10**W`.

- ordered additions: `N²`;
- nonnegative subtractions with `a >= b`: `N(N+1)/2`;
- total: `N² + N(N+1)/2`.

At width 3, this is 1,500,500 distinct tasks. The canonical training dataset
contains 40,000 tasks sampled uniformly without replacement from that space.
Generation is deterministic for a fixed seed and independent of Python hash
randomization.

## Split and benchmark requirements

A credible result must satisfy all of the following:

1. Generate train, validation, and benchmark membership deterministically.
2. Keep commutative addition twins such as `012+034` and `034+012` in the same
   split.
3. Sample benchmark tasks without replacement.
4. Verify zero semantic overlap with every task used for training or validation:
   addition twins share a sorted-operand group, while subtraction remains
   directional.
5. Save dataset/split hashes, configuration, seed, package versions, Git revision,
   and metrics with the model.
6. Decode outputs according to the model's task format before computing numerical
   accuracy.
7. Report exact sequence match, numerical accuracy, format validity, and EOS
   behavior separately.

`lib/benchmark.py` implements deterministic held-out sampling. `demo.py` uses
benchmark seed 42 to select 100 unique tasks from each of the one-, two-, and
three-digit operand buckets after excluding the committed training dataset. The
benchmark seed stays fixed while training seeds vary.

## What can be concluded

High accuracy on a strictly held-out benchmark is evidence consistent with a
reusable arithmetic procedure. It is not, by itself, proof of a symbolic
algorithm or correctness over the complete finite domain. A 300-task benchmark
estimates performance; exhaustive evaluation is required to make a full-domain
accuracy statement.

The strongest experimental comparison should be a controlled ablation using the
same architecture, data budget, task space, splits, and seeds:

| Condition | Fixed-width operands and answer | Reversed answer |
|---|---:|---:|
| Plain baseline | no | no |
| Reverse only | no | yes |
| Fixed-width only | yes | no |
| Combined | yes | yes |

Run each condition over multiple seeds and report the distribution, not only the
best run.

This matrix compares whole representations, not operand padding in isolation.
With answer-only loss, minimal-layout answers contain a variable number of target
digits, while fixed-width answers always contain `W + 1`; EOS is supervised in
both cases. Each run report must include the number of supervised answer tokens
and EOS targets in its training and validation splits. A causal claim about
fixed-width layout independent of target-token budget would require a separate
matched-budget design.

The experiment plan and result set should be plain versioned files. Declare all
conditions and seeds in advance, publish an immutable report for every run, and
retain failures and low-accuracy outcomes. A small model scoring poorly is still a
valid experimental result when its data, configuration, artifact, and evaluation
provenance are intact; it should not be silently rerun or omitted.

Every planned cell must produce a status record. Completed records bind the model,
training manifest, normalized train/validation rosters, and evaluation output to
the declared benchmark-manifest hash. Failed records retain the stage, error or
exit status, configuration, Git revision, and available logs or partial-artifact
hashes. Completed reports include exact match, numerical accuracy, strict format,
EOS behavior, and counts with denominators by operation, operand-width bucket,
carry/borrow count and chain length, overflow, zero operand, and equal operands.

Reinforcement learning is deferred until after the supervised four-way baseline
is complete. Any later RL study needs its own predeclared reward, optimization,
seed, stopping, and failure-reporting protocol while keeping the benchmark
manifest frozen and outside reward selection. No RL implementation or result is
claimed here.

## Known limitations

- Operands wider than the configured width are out of distribution.
- Subtraction is restricted to nonnegative results.
- Multiplication and division are not represented.
- Learned absolute positions may encourage shortcuts tied to the fixed layout.
- A model artifact and benchmark report must be published before results are
  independently reproducible.

## References

- Lee et al., *Teaching Arithmetic to Small Transformers* (2023),
  <https://arxiv.org/abs/2307.03381>
- Nogueira et al., *Investigating the Limitations of Transformers with Simple
  Arithmetic Tasks* (2021), <https://arxiv.org/abs/2102.13019>
- McLeish et al., *Transformers Can Do Arithmetic with the Right Embeddings*
  (2024), <https://arxiv.org/abs/2405.17399>
