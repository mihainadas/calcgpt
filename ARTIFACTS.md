# Artifact and repository policy

## What belongs in Git

- Source code, tests, documentation, and experiment configuration.
- Small canonical synthetic datasets and test fixtures.
- Checksums, model/data cards, and lightweight evaluation summaries.

## What does not belong in Git

- Model weights and intermediate checkpoints.
- Optimizer, scheduler, trainer, or mixed-precision state.
- Generated datasets outside the explicitly reviewed canonical set.
- Logs, caches, notebook scratch output, and local evaluation results.
- Credentials, private data, or unreviewed third-party artifacts.

The ignore rules cover common artifact names, but contributors must still inspect staged files before committing.

## Public model releases

Publish final models through a GitHub Release or a model registry such as Hugging Face. Each release should include:

1. `model.safetensors` rather than pickle-based weights.
2. Model configuration and generation configuration.
3. The exact tokenizer vocabulary and tokenizer settings.
4. `task_spec.json` with representation, operand/answer widths, answer direction,
   supported operators, and negative-result policy.
5. `training_manifest.json` with Git revision, dataset checksums, seed, full training configuration, dependency versions, hardware, and metrics.
6. `SHA256SUMS.txt` and a version-specific model card.

Do not release optimizer or scheduler state unless resumable training is an explicit, separately documented use case.

## Multi-seed experiment reports

Controlled ablations should remain a collection of reviewable files. Declare the
complete representation and seed matrix before running the experiment. For each run,
publish a versioned report containing:

- representation name, training seed, benchmark seed, and `completed` or `failed`
  status;
- Git revision and whether the source tree was dirty;
- dataset, normalized train/validation roster, benchmark manifest,
  training-manifest, and model SHA-256 values;
- task specification, training configuration, decoding configuration, and
  dependency/hardware environment;
- supervised answer-token and EOS-target counts for the training and validation
  splits;
- for completed runs, exact-match, numerical, strict-format, EOS, throughput, and
  operation/operand-width/carry-or-borrow/chain-length/overflow/zero/equal-operand
  strata with numerators and denominators;
- for failed runs, the failed stage, error or exit status, and references or hashes
  for available logs and partial artifacts.

The benchmark-manifest hash is the evaluation identity. The manifest must bind
the semantic exclusion policy, excluded-training-roster hash, benchmark seed,
sampling parameters, and normalized evaluation-roster hash. A shared seed or
matching row count is not a substitute for this binding.

The four-way baseline compares minimal versus fixed-width fields and normal
versus reversed answers. Fixed-width layout pads both operands and answers; under
answer-only loss it therefore changes the number of supervised target tokens.
This must be reported as a whole-representation comparison, not as an isolated
operand-padding effect.

A compact summary may be committed under `reports/<experiment>/`, while weights
and detailed per-example output remain external release assets. The summary must
list every planned run—including failures and low-accuracy outcomes—and aggregate
across seeds rather than selecting the best run. Reports need a schema version and
must be validated in dependency-light CI before publication. No online dashboard
or experiment database is required.

Reinforcement learning is deferred and belongs in a separate post-baseline
protocol. Do not mix RL runs into the supervised baseline summary. Any later RL
record must declare its reward, optimization and stopping rules, seeds, and
failure policy while keeping the benchmark manifest outside reward selection.

## Historical repository cleanup

Old checkpoints remain in Git history even after deletion from the current branch. Rewriting history is a destructive repository-maintenance operation and must be coordinated separately from ordinary feature work.

Before a rewrite:

- Freeze pushes and merge all intended cleanup changes.
- Work from a fresh full mirror clone, never a shallow or partial checkout.
- Create and verify an offline bundle containing every branch and tag.
- Analyze history and remove only explicitly identified artifact paths.
- Compare the exported default-branch tree before and after filtering.
- Run repository integrity and size checks before any force-push.
- Obtain explicit maintainer approval, announce invalidated commit identifiers, and require collaborators to re-clone.

Keep the pre-rewrite bundle private because removed objects remain readable from it. Git LFS is not a substitute for removing disposable checkpoints; it is appropriate only when versioning large artifacts in Git is an intentional product requirement.
