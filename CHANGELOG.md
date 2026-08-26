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

### Changed

- Replaced the transitive dependency snapshot with a project-managed convenience install.
- Consolidated ignore rules while preserving explicitly tracked canonical datasets.
- Regenerated the canonical padded dataset deterministically and replaced the
  plain baseline's symlink/long-filename pair with one canonical regular file.
- Replaced the stale workflow example and library guide with APIs that match the
  current implementation.
- Masked padding labels, added attention masks, and moved augmentation after a
  deterministic commutative-group split.

### Removed

- Removed the outdated tutorial notebook; it targeted pre-3.0 APIs and contained
  stale execution output. Its history remains available in Git.

## Release history

A complete historical release record has not yet been reconstructed. Future releases should add dated entries here and link to immutable tags and artifacts.
