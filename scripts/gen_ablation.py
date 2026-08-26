#!/usr/bin/env python3
"""Validate and expand the four-way ablation matrix without launching jobs."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.benchmark import (
    build_semantic_holdout_manifest,
    holdout_manifest_sha256,
)
from lib.representation import (
    REPRESENTATION_NAMES,
    RepresentationSpec,
    Task,
    task_roster_sha256,
)

PLAN_SCHEMA = "calcgpt-ablation-plan"
PLAN_SCHEMA_VERSION = 1
EXPANDED_PLAN_SCHEMA = "calcgpt-expanded-ablation-plan"
EXPANDED_PLAN_VERSION = 1
CANONICAL_MODEL_SEEDS = [41, 42, 43]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ablations-v1.toml"
CANONICAL_SOURCE_PATH = (PROJECT_ROOT / "datasets" / "ds-calcgpt-padded.txt").resolve()
CANONICAL_SOURCE_SHA256 = "de810f1c93fba69e2d0fc3985bcfb9fc62e88d9083df320ce069e7ffc5a9667c"

_TOP_LEVEL_KEYS = {
    "schema",
    "schema_version",
    "name",
    "data",
    "experiment",
    "benchmark",
    "model",
    "training",
}
_CANONICAL_MODEL = {
    "embedding_dim": 128,
    "num_layers": 4,
    "num_heads": 8,
    "feedforward_dim": 256,
    "n_positions": 20,
}
_CANONICAL_TRAINING = {
    "epochs": 30,
    "batch_size": 64,
    "learning_rate": 0.001,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "save_steps": 2000,
    "validation_fraction": 0.2,
    "augmentation": False,
    "loss_scope": "answer-only",
}
_CANONICAL_BENCHMARK = {
    "seed": 42,
    "samples_per_digit_bucket": 100,
    "semantic_addition_groups": True,
    "decoding": "greedy",
}


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _require_keys(mapping: dict[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(mapping))
    extra = sorted(set(mapping) - expected)
    if missing:
        raise ValueError(f"{name} is missing required field(s): {', '.join(missing)}")
    if extra:
        raise ValueError(f"{name} has unsupported field(s): {', '.join(extra)}")


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_path(value: object, name: str) -> str:
    path = _require_string(value, name)
    if "\x00" in path:
        raise ValueError(f"{name} must not contain a null byte")
    return path


def _require_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_float(value: object, name: str) -> float:
    if not isinstance(value, float):
        raise ValueError(f"{name} must be a float")
    return value


def _require_exact_controls(
    mapping: dict[str, Any], expected: dict[str, object], name: str
) -> None:
    _require_keys(mapping, set(expected), name)
    for field, expected_value in expected.items():
        value = mapping[field]
        field_name = f"{name}.{field}"
        if isinstance(expected_value, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{field_name} must be a boolean")
        elif isinstance(expected_value, int):
            _require_int(value, field_name)
        elif isinstance(expected_value, float):
            _require_float(value, field_name)
        elif isinstance(expected_value, str):
            _require_string(value, field_name)
        if value != expected_value:
            raise ValueError(f"{field_name} must be {expected_value!r}")


def _require_unique(values: Iterable[str], name: str) -> list[str]:
    items = list(values)
    if len(set(items)) != len(items):
        raise ValueError(f"{name} contains a collision: values must be unique")
    return items


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve_project_path(configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_ablation_config(config_path: Path) -> dict[str, Any]:
    """Load one TOML plan and reject incomplete or broadened matrices."""
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    _require_keys(config, _TOP_LEVEL_KEYS, "plan")
    if config.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported ablation plan schema")
    _require_int(config.get("schema_version"), "schema_version")
    if config["schema_version"] != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported ablation plan schema version")
    _require_string(config.get("name"), "name")

    data = _require_mapping(config.get("data"), "data")
    experiment = _require_mapping(config.get("experiment"), "experiment")
    benchmark = _require_mapping(config.get("benchmark"), "benchmark")
    model = _require_mapping(config.get("model"), "model")
    training = _require_mapping(config.get("training"), "training")

    _require_keys(
        data,
        {
            "source",
            "source_representation",
            "source_sha256",
            "operand_width",
            "task_count",
            "seed",
        },
        "data",
    )
    _require_path(data["source"], "data.source")
    _require_string(data["source_representation"], "data.source_representation")
    source_sha256 = _require_string(data["source_sha256"], "data.source_sha256")
    if len(source_sha256) != 64 or any(char not in "0123456789abcdef" for char in source_sha256):
        raise ValueError("data.source_sha256 must be a lowercase SHA-256 digest")
    if _resolve_project_path(data["source"]) != CANONICAL_SOURCE_PATH:
        raise ValueError(f"data.source must resolve to {CANONICAL_SOURCE_PATH}")
    if source_sha256 != CANONICAL_SOURCE_SHA256:
        raise ValueError(
            f"data.source_sha256 must be the canonical digest {CANONICAL_SOURCE_SHA256}"
        )
    for field in ("operand_width", "task_count", "seed"):
        _require_int(data[field], f"data.{field}")

    _require_keys(
        experiment,
        {"representations", "model_seeds", "split_seed", "output_root"},
        "experiment",
    )
    _require_path(experiment["output_root"], "experiment.output_root")
    _require_int(experiment["split_seed"], "experiment.split_seed")

    _require_exact_controls(model, _CANONICAL_MODEL, "model")
    _require_exact_controls(training, _CANONICAL_TRAINING, "training")
    _require_exact_controls(benchmark, _CANONICAL_BENCHMARK, "benchmark")

    representations = experiment.get("representations")
    if not isinstance(representations, list) or not all(
        isinstance(value, str) for value in representations
    ):
        raise ValueError("experiment.representations must be a string array")
    representations = _require_unique(representations, "experiment.representations")
    if representations != list(REPRESENTATION_NAMES):
        raise ValueError(
            "experiment.representations must be exactly plain, reversed, padded, "
            "padded-reversed in that order"
        )

    model_seeds = experiment.get("model_seeds")
    if not isinstance(model_seeds, list) or any(
        not isinstance(seed, int) or isinstance(seed, bool) for seed in model_seeds
    ):
        raise ValueError("experiment.model_seeds must be an integer array")
    if model_seeds != CANONICAL_MODEL_SEEDS:
        raise ValueError("experiment.model_seeds must be exactly [41, 42, 43]")
    for field, expected in (("seed", 42), ("operand_width", 3), ("task_count", 40_000)):
        if data.get(field) != expected:
            raise ValueError(f"data.{field} must be {expected}")
    if experiment.get("split_seed") != 42:
        raise ValueError("experiment.split_seed must be 42")
    if data.get("source_representation") != "padded-reversed":
        raise ValueError("data.source_representation must be 'padded-reversed'")
    return config


def expand_ablation_plan(config_path: Path) -> dict[str, Any]:
    """Expand exactly four representations by three model seeds."""
    config_path = Path(config_path).resolve()
    config = load_ablation_config(config_path)
    data = config["data"]
    experiment = config["experiment"]
    width = data["operand_width"]

    source_path = _resolve_project_path(data["source"])
    source_bytes = source_path.read_bytes()
    source_hash = _sha256_bytes(source_bytes)
    if source_hash != data["source_sha256"]:
        raise ValueError(
            f"source dataset checksum mismatch: expected {data['source_sha256']}, "
            f"got {source_hash}"
        )
    source_examples = [line for line in source_bytes.decode("utf-8").splitlines() if line]
    source_spec = RepresentationSpec.from_name(data["source_representation"], width)
    tasks = source_spec.validate_dataset(source_examples)
    if len(tasks) != data["task_count"]:
        raise ValueError(
            f"source dataset has {len(tasks):,} tasks, expected {data['task_count']:,}"
        )

    output_root = _resolve_project_path(experiment["output_root"])
    benchmark_manifest = build_semantic_holdout_manifest(
        config["benchmark"]["samples_per_digit_bucket"],
        width,
        tasks,
        seed=config["benchmark"]["seed"],
    )
    benchmark_manifest_hash = holdout_manifest_sha256(benchmark_manifest)
    benchmark_manifest_path = output_root / "benchmarks" / "semantic-holdout-manifest.json"
    benchmark_tasks: list[Task] = [
        (record["left"], record["operator"], record["right"])
        for record in benchmark_manifest["tasks"]
    ]
    evaluation_roster_hash = task_roster_sha256(benchmark_tasks)
    if evaluation_roster_hash != benchmark_manifest["task_roster_sha256"]:
        raise RuntimeError("semantic holdout manifest task roster is internally inconsistent")

    datasets = []
    benchmark_renderings = []
    runs = []
    for representation in experiment["representations"]:
        spec = RepresentationSpec.from_name(representation, width)
        dataset_path = output_root / "datasets" / f"{representation}.txt"
        rendered_bytes = spec.render_dataset_bytes(tasks)
        benchmark_path = output_root / "benchmarks" / f"{representation}.txt"
        benchmark_bytes = spec.render_dataset_bytes(benchmark_tasks)
        decoded_benchmark_tasks = spec.validate_dataset(
            benchmark_bytes.decode("utf-8").splitlines()
        )
        rendering_roster_hash = task_roster_sha256(decoded_benchmark_tasks)
        if rendering_roster_hash != evaluation_roster_hash:
            raise RuntimeError("benchmark rendering changed normalized task membership")
        benchmark_rendering = {
            "representation": representation,
            "path": str(benchmark_path),
            "sha256": _sha256_bytes(benchmark_bytes),
            "task_roster_sha256": rendering_roster_hash,
            "task_count": len(decoded_benchmark_tasks),
        }
        benchmark_renderings.append(benchmark_rendering)
        datasets.append(
            {
                "representation": representation,
                "representation_spec": spec.to_dict(),
                "path": str(dataset_path),
                "sha256": _sha256_bytes(rendered_bytes),
                "task_roster_sha256": task_roster_sha256(tasks),
                "task_count": len(tasks),
                "benchmark_path": str(benchmark_path),
                "benchmark_sha256": benchmark_rendering["sha256"],
                "benchmark_manifest_path": str(benchmark_manifest_path),
                "benchmark_manifest_sha256": benchmark_manifest_hash,
                "evaluation_task_roster_sha256": evaluation_roster_hash,
            }
        )
        for model_seed in experiment["model_seeds"]:
            run_id = f"{representation}-seed-{model_seed}"
            runs.append(
                {
                    "run_id": run_id,
                    "representation": representation,
                    "model_seed": model_seed,
                    "data_seed": data["seed"],
                    "split_seed": experiment["split_seed"],
                    "benchmark_seed": config["benchmark"]["seed"],
                    "dataset_path": str(dataset_path),
                    "benchmark_path": str(benchmark_path),
                    "benchmark_sha256": benchmark_rendering["sha256"],
                    "benchmark_manifest_path": str(benchmark_manifest_path),
                    "benchmark_manifest_sha256": benchmark_manifest_hash,
                    "evaluation_task_roster_sha256": evaluation_roster_hash,
                    "model_dir": str(output_root / "models" / run_id),
                    "result_path": str(output_root / "results" / f"{run_id}.json"),
                }
            )

    if len(datasets) != 4 or len(runs) != 12:
        raise RuntimeError("ablation expansion must produce four datasets and twelve runs")
    _require_unique((item["path"] for item in datasets), "dataset paths")
    _require_unique((item["path"] for item in benchmark_renderings), "benchmark paths")
    _require_unique((item["run_id"] for item in runs), "run IDs")
    _require_unique((item["model_dir"] for item in runs), "model directories")
    _require_unique((item["result_path"] for item in runs), "result paths")

    return {
        "schema": EXPANDED_PLAN_SCHEMA,
        "schema_version": EXPANDED_PLAN_VERSION,
        "name": config["name"],
        "config_path": str(config_path),
        "source_dataset": {
            "path": str(source_path),
            "sha256": source_hash,
            "task_roster_sha256": task_roster_sha256(tasks),
            "task_count": len(tasks),
        },
        "benchmark": {
            **config["benchmark"],
            "manifest": benchmark_manifest,
            "manifest_path": str(benchmark_manifest_path),
            "manifest_sha256": benchmark_manifest_hash,
            "task_roster_sha256": evaluation_roster_hash,
            "task_count": len(benchmark_tasks),
            "renderings": benchmark_renderings,
        },
        "model": config["model"],
        "training": config["training"],
        "datasets": datasets,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Ablation TOML to validate and expand (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()
    try:
        plan = expand_ablation_plan(args.config)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
