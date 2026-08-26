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
