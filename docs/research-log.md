# CalcGPT research log

This log captures decisions and findings while the experiments are still moving.
Entries distinguish implementation status from empirical evidence: a merged check
is not a model-quality result, and a planned comparison is not a completed study.

## 2026-08-27 — Exact overlap was too weak

**Status:** semantic exclusion is implemented and the integrated dependency-light
suite passes; canonical model evaluation remains pending.

The first held-out check compared exact normalized task tuples. Under canonical
benchmark seed 42, that still admitted 2 of 300 addition tasks whose commutative
twins were present in training. For example, training on `012+034` and testing on
`034+012` is not a string overlap, but it exposes the same arithmetic fact.

This matters because a benchmark can be exactly disjoint while still leaking
semantic equivalents. Addition tasks now belong to a commutative group keyed by
the sorted operand pair; excluding either ordering excludes both. Subtraction
remains directional. The episode is a useful reminder that split integrity must
follow the invariances of the task, not merely its serialized text.

## 2026-08-27 — Low accuracy is data, not an operational failure

**Status:** evaluator exit semantics are corrected, covered by a CLI regression,
and pass the integrated dependency-light suite; canonical model evaluation
remains pending.

The earlier evaluator treated arithmetic accuracy below 50% as a failing process
exit. That conflated two different outcomes. A command can successfully load a
model, run a declared benchmark, validate every record, and write a complete
report even when the model scores 0%. In an ablation, that low score may be the
most informative outcome.

Process failure should mean the experiment could not be executed or recorded:
invalid configuration, missing artifacts, malformed output, or an I/O/runtime
error. Model quality belongs in the report. Scheduled smoke tests therefore judge
the artifact round trip, not whether an intentionally tiny one-epoch model learns
arithmetic.

## 2026-08-27 — Four representations, one task roster

**Status:** representation and plan contracts are implemented and pass the
integrated dependency-light suite; training runs and result records remain pending.

The planned ablation compares four representations:

1. minimal fields with normal answers (`plain`);
2. minimal fields with reversed answers (`reversed`);
3. fixed-width operands and answers with normal answer order (`padded`);
4. fixed-width operands and answers with reversed answer order
   (`padded-reversed`).

The comparison will use predeclared training seeds 41, 42, and 43. Within each
seed, every representation must receive the identical roster of normalized
`(left, operator, right)` tasks and the same grouped split. Serialization happens
only after roster membership is fixed. The benchmark roster remains fixed at seed
42 across all training seeds, so evaluation noise is not mixed with training
variation.

The first factor is fixed-width layout, not operand padding alone. It pads both
operands and answers and therefore changes positions and the number of supervised
answer tokens. Under answer-only loss, minimal answers have variable length,
fixed-width answers always contain `operand_width + 1` digits, and EOS is a target
in both cases. Reports must include these target counts. The planned matrix is a
whole-representation comparison; a padding-only causal claim would need a later
token-budget-matched design.

Each planned cell remains visible as a `completed` or `failed` record, including
low scores. Completed reports bind normalized train/validation roster hashes and
the evaluation roster to one immutable benchmark-manifest hash, then report exact,
numerical, strict-format, EOS, throughput, and arithmetic-stratum metrics with
denominators. Failed records retain the stage, error or exit status, configuration,
Git revision, and available diagnostics. A compact file-based summary will
aggregate across all three seeds. No dashboard or experiment service is needed.

## 2026-08-27 — Identity is not metric completeness

**Status:** implemented and covered by regression tests; the integrated
dependency-light suite passes 75 of 75 tests.

Final adversarial review found that the aggregator could bind a report to the
canonical benchmark and roster hashes while still accepting only 100 evaluated
examples where the manifest declares 300. It could also accept a completed record
without EOS, arithmetic-stratum, throughput, or supervised-target evidence.

The hashes answer “which benchmark was intended?” They do not answer “was every
declared task evaluated, and were all required measurements recorded?” The
aggregator now requires the canonical 300-task denominator; versioned train and
validation target-token counts that match the training manifest; EOS evidence;
all seven declared arithmetic-stratum partitions, each reconciling to the primary
denominator and correct count; and finite, internally consistent throughput under
the declared successful-completions scope. Regression tests cover missing and
inconsistent evidence. These are reporting guarantees, not model-quality results.

## 2026-08-27 — Reinforcement learning stays after the baseline

**Status:** deferred protocol only; no RL implementation, run, or result.

Reinforcement learning can answer a different question after the supervised
four-way baseline is complete. It must not be introduced as an unrecorded repair
for weak baseline runs or mixed into their summary.

Any later RL experiment needs a separate, predeclared reward, optimization,
stopping, seed, and failure-reporting protocol. The supervised baseline artifacts
stay frozen, and the benchmark manifest remains evaluation-only rather than a
reward-selection signal.

The smallest useful follow-up is a three-arm continuation study, declared before
inspecting the ablation winner:

1. use the padded-reversed representation with seeds 41, 42, and 43 from its
   frozen supervised checkpoints;
2. compare the frozen checkpoint, an equal-compute supervised continuation, and
   a verifier-reward policy-optimization continuation;
3. train the continuation arms on a separate roster whose semantic groups are
   excluded from the canonical benchmark;
4. award 1 only for an exact canonical answer followed by EOS and 0 otherwise,
   with the KL coefficient and schedule fixed before any run;
5. keep the 300-task canonical benchmark untouched until final evaluation, using
   the existing exact, numerical, strict-format, EOS, throughput, and arithmetic-
   stratum report contract; and
6. retain failed runs and stop on predeclared verifier mismatch, reward exploit,
   or output-distribution collapse rather than repairing the protocol mid-run.

The equal-compute supervised arm is essential: without it, additional optimization
would be confounded with reinforcement learning. Padded-reversed is named before
the baseline result so the RL condition is not chosen post hoc. CalcGPT currently
emits only final answers, so an outcome verifier is the natural first test; process
reward would require a new intermediate-reasoning representation and a different
research question.

This direction is informed by [DeepSeekMath](https://arxiv.org/abs/2402.03300),
which introduced Group Relative Policy Optimization for mathematical reasoning,
and [DeepSeek-R1](https://arxiv.org/abs/2501.12948), which studied verifier-based
reinforcement learning on reasoning tasks. [Let's Verify Step by
Step](https://arxiv.org/abs/2305.20050) motivates treating process supervision as
distinct from final-answer outcome supervision. These papers motivate the design;
they do not imply that their large-model results transfer to CalcGPT.

Until the supervised matrix is complete and this protocol is made executable, RL
is a future study, not a current CalcGPT capability claim.

## 2026-08-27 — No model-quality result yet

**Status:** unchanged; canonical training and publication pending.

Repository tests establish deterministic generation, split and benchmark
contracts, packaging, and artifact plumbing. A tiny end-to-end run can establish
that the software path works. Neither is evidence that the canonical model has
learned the arithmetic task.

CalcGPT will not publish a headline accuracy until the full run artifacts,
versioned manifests, fixed benchmark report, and multi-seed context are available
together. Until then, the repository contains a method and an evidence standard,
not a model-quality claim.
