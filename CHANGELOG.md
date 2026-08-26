# Changelog

All notable changes to CalcGPT will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and future releases should follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-08-27

### Added

- Python project metadata and dependency groups in `pyproject.toml`.
- Lightweight Python 3.11-3.13 CI and a scheduled/manual reproducibility workflow.
- Contributor, security, conduct, model, data, artifact, and citation guidance.
- GitHub issue and pull-request templates plus dependency update configuration.
- A reviewable configuration for the canonical padded three-digit experiment.
- Self-contained tokenizer, task-format, and training-provenance artifacts.
- Strict held-out benchmark sampling and dependency-light contract tests.
- Complete source-distribution contents with extracted-archive verification.
- A scheduled/manual tiny real-ML artifact round-trip check.
- A completed/failed ablation-run reporting contract bound to one benchmark
  manifest, with EOS, supervised-target, and arithmetic-stratum requirements.

### Changed

- Replaced the transitive dependency snapshot with a project-managed convenience install.
- Made the `demo` extra include its complete ML runtime and terminal UI.
- Consolidated ignore rules while preserving explicitly tracked canonical datasets.
- Regenerated the canonical padded dataset deterministically and replaced the
  plain baseline's symlink/long-filename pair with one canonical regular file.
- Replaced the stale workflow example and library guide with APIs that match the
  current implementation.
- Masked padding labels, added attention masks, and moved augmentation after a
  deterministic commutative-group split.
- Fixed the canonical benchmark seed at 42 and documented a file-based multi-seed
  ablation/report contract without claiming model-quality results.
- Aligned the canonical training command with split seed 42 and answer-only loss.
- Described the ablation as a whole-representation comparison of minimal versus
  fixed-width fields, with reinforcement learning deferred to a separate
  post-baseline protocol.
- Made ablation summaries enforce metric completeness as well as provenance: the
  canonical 300-task denominator, versioned target-token counts, EOS, seven
  reconciling strata partitions, and finite scoped throughput are required.

### Removed

- Removed the outdated tutorial notebook; it targeted pre-3.0 APIs and contained
  stale execution output. Its history remains available in Git.

## Release history

A complete historical release record has not yet been reconstructed. Future releases should add dated entries here and link to immutable tags and artifacts.
