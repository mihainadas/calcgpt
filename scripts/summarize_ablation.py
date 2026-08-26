#!/usr/bin/env python3
"""Summarize one complete CalcGPT representation ablation descriptively."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gen_ablation import DEFAULT_CONFIG, expand_ablation_plan

SUMMARY_SCHEMA = "calcgpt-ablation-summary"
SUMMARY_SCHEMA_VERSION = 1
FAILED_RUN_SCHEMA = "calcgpt-ablation-failed-run"
FAILED_RUN_SCHEMA_VERSION = 1
METRICS = {
    "numerical_accuracy": "correct_arithmetic",
    "strict_format": "valid_format",
    "exact_match": "exact_matches",
    "eos_termination": "eos_terminated",
}
STRATA_DIMENSIONS = {
    "digit_bucket",
    "operation",
    "event_count",
    "longest_chain",
    "overflow",
    "zero_operand",
    "equal_operands",
}
TARGET_TOKEN_SCHEMA = "calcgpt-target-token-counts"
TARGET_TOKEN_SCHEMA_VERSION = 1
REPRESENTATIONS = ("plain", "reversed", "padded", "padded-reversed")
CONTRASTS = {
    "reversal_minimal": ("reversed", "plain"),
    "reversal_fixed": ("padded-reversed", "padded"),
    "layout_normal": ("padded", "plain"),
    "layout_reversed": ("padded-reversed", "reversed"),
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} object is required")
    return value


def _exact_equal(left: object, right: object) -> bool:
    """Compare JSON provenance without Python's bool/integer equivalence."""
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _load_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: report must be a JSON object")
    status = payload.get("status")
    if status == "completed":
        if (
            payload.get("schema") != "calcgpt-evaluation-report"
            or payload.get("schema_version") != 1
            or payload.get("report_type") != "calcgpt-evaluation"
        ):
            raise ValueError(f"{path}: unsupported completed evaluation report schema")
    elif status == "failed":
        if (
            payload.get("schema") != FAILED_RUN_SCHEMA
            or payload.get("schema_version") != FAILED_RUN_SCHEMA_VERSION
        ):
            raise ValueError(f"{path}: unsupported failed-run record schema")
    else:
        raise ValueError(f"{path}: status must be completed or failed")
    return payload


def _metric_record(metrics: dict[str, Any], key: str, denominator: int) -> dict[str, Any]:
    numerator = metrics.get(key)
    if (
        not isinstance(numerator, int)
        or isinstance(numerator, bool)
        or not 0 <= numerator <= denominator
    ):
        raise ValueError(f"invalid {key} numerator {numerator!r}")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "percent": numerator / denominator * 100,
    }


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _validate_strata(
    path: Path, metrics: dict[str, Any], denominator: int
) -> dict[str, Any]:
    strata = _mapping(metrics.get("answer_complete_strata"), "answer_complete_strata")
    if set(strata) != STRATA_DIMENSIONS:
        raise ValueError(f"{path}: answer_complete_strata must contain exactly all seven dimensions")
    expected_correct = metrics.get("correct_arithmetic")
    validated: dict[str, Any] = {}
    for dimension in sorted(STRATA_DIMENSIONS):
        categories = _mapping(strata.get(dimension), f"stratum {dimension}")
        dimension_n = 0
        dimension_correct = 0
        validated_categories = {}
        for category, raw_record in categories.items():
            record = _mapping(raw_record, f"stratum {dimension}/{category}")
            if set(record) != {"n", "correct"}:
                raise ValueError(
                    f"{path}: stratum {dimension}/{category} must contain n and correct"
                )
            count = _nonnegative_int(record.get("n"), f"{dimension}/{category}.n")
            correct = _nonnegative_int(
                record.get("correct"), f"{dimension}/{category}.correct"
            )
            if correct > count:
                raise ValueError(f"{path}: stratum {dimension}/{category} correct exceeds n")
            dimension_n += count
            dimension_correct += correct
            validated_categories[category] = {"n": count, "correct": correct}
        if dimension_n != denominator:
            raise ValueError(
                f"{path}: stratum {dimension} n sums to {dimension_n}, expected {denominator}"
            )
        if dimension_correct != expected_correct:
            raise ValueError(
                f"{path}: stratum {dimension} correct sums to {dimension_correct}, "
                f"expected {expected_correct}"
            )
        validated[dimension] = validated_categories
    return validated


def _validate_timing(
    path: Path, metrics: dict[str, Any], denominator: int
) -> dict[str, Any]:
    timing = _mapping(metrics.get("timing"), "timing")
    if timing.get("scope") != "successful_completions_all_prompts":
        raise ValueError(f"{path}: unsupported timing scope")
    count = _nonnegative_int(timing.get("count"), f"{path}: timing.count")
    if count > denominator * 3:
        raise ValueError(f"{path}: timing.count exceeds the three-prompt evaluation scope")
    total_seconds = _finite_nonnegative(
        timing.get("total_seconds"), f"{path}: timing.total_seconds"
    )
    tests_per_second = _finite_nonnegative(
        timing.get("tests_per_second"), f"{path}: timing.tests_per_second"
    )
    expected_rate = count / total_seconds if total_seconds > 0 else 0.0
    if not math.isclose(tests_per_second, expected_rate, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{path}: timing throughput is inconsistent with count and total_seconds")
    return {
        "scope": timing["scope"],
        "count": count,
        "total_seconds": total_seconds,
        "tests_per_second": tests_per_second,
    }


def _validate_target_tokens(
    path: Path,
    report: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    training = _mapping(report.get("training"), "training")
    target_tokens = _mapping(training.get("target_tokens"), "target_tokens")
    if target_tokens.get("schema") != TARGET_TOKEN_SCHEMA:
        raise ValueError(f"{path}: unsupported target-token schema")
    if target_tokens.get("schema_version") != TARGET_TOKEN_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported target-token schema version")
    evaluation = _mapping(report.get("evaluation"), "evaluation")
    training_controls = _mapping(evaluation.get("training_controls"), "training_controls")
    loss_scope = target_tokens.get("loss_scope")
    if loss_scope not in {"full-sequence", "answer-only"}:
        raise ValueError(f"{path}: invalid target-token loss_scope")
    if loss_scope != training_controls.get("loss_scope"):
        raise ValueError(f"{path}: target-token loss_scope does not match training controls")

    manifest = artifacts["training_manifest"]["content"]
    if manifest.get("target_tokens") != target_tokens:
        raise ValueError(f"{path}: target-token counts differ from the training manifest")
    manifest_splits = manifest.get("splits")
    if not isinstance(manifest_splits, dict):
        manifest_splits = {}

    validated: dict[str, Any] = {
        "schema": TARGET_TOKEN_SCHEMA,
        "schema_version": TARGET_TOKEN_SCHEMA_VERSION,
        "loss_scope": loss_scope,
    }
    for split, examples_field in (
        ("train", "train_examples"),
        ("validation", "validation_examples"),
    ):
        record = _mapping(target_tokens.get(split), f"target_tokens.{split}")
        if set(record) != {
            "active_target_tokens",
            "active_answer_tokens",
            "active_eos_tokens",
        }:
            raise ValueError(f"{path}: target_tokens.{split} has unsupported fields")
        total = _nonnegative_int(
            record.get("active_target_tokens"), f"target_tokens.{split}.active_target_tokens"
        )
        answer = _nonnegative_int(
            record.get("active_answer_tokens"), f"target_tokens.{split}.active_answer_tokens"
        )
        eos = _nonnegative_int(
            record.get("active_eos_tokens"), f"target_tokens.{split}.active_eos_tokens"
        )
        if answer > total or eos > total or answer + eos > total:
            raise ValueError(f"{path}: target_tokens.{split} component count exceeds total")
        if loss_scope == "answer-only" and answer + eos != total:
            raise ValueError(f"{path}: answer-only target count must equal answer plus EOS")
        expected_examples = manifest_splits.get(examples_field)
        if expected_examples is not None:
            expected_examples = _nonnegative_int(
                expected_examples, f"training manifest {examples_field}"
            )
            if eos != expected_examples:
                raise ValueError(
                    f"{path}: target_tokens.{split} EOS count does not match split size"
                )
        validated[split] = {
            "active_target_tokens": total,
            "active_answer_tokens": answer,
            "active_eos_tokens": eos,
        }
    return validated


def _descriptive(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "stdev": None,
            "min": None,
            "max": None,
            "range": None,
        }
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def _planned_dataset(plan: dict[str, Any], representation: str) -> dict[str, Any]:
    return next(
        item for item in plan["datasets"] if item["representation"] == representation
    )


def _validate_common_provenance(
    path: Path,
    provenance: dict[str, Any],
    plan: dict[str, Any],
    plan_sha256: str,
    planned_run: dict[str, Any],
    dataset: dict[str, Any],
) -> tuple[str, str]:
    expected = {
        "plan_sha256": plan_sha256,
        "split_seed": planned_run["split_seed"],
        "model_controls": plan["model"],
        "training_controls": plan["training"],
        "training_dataset_sha256": dataset["sha256"],
        "full_task_roster_sha256": dataset["task_roster_sha256"],
        "benchmark_dataset_sha256": planned_run["benchmark_sha256"],
        "benchmark_manifest_sha256": planned_run["benchmark_manifest_sha256"],
        "evaluation_task_roster_sha256": planned_run[
            "evaluation_task_roster_sha256"
        ],
    }
    for field, expected_value in expected.items():
        if not _exact_equal(provenance.get(field), expected_value):
            raise ValueError(f"{path}: {field} does not match the ablation plan")
    train_roster = _require_sha256(
        provenance.get("train_task_roster_sha256"),
        f"{path}: train task roster",
    )
    validation_roster = _require_sha256(
        provenance.get("validation_task_roster_sha256"),
        f"{path}: validation task roster",
    )
    return train_roster, validation_roster


def _completed_provenance(report: dict[str, Any]) -> dict[str, Any]:
    evaluation = _mapping(report.get("evaluation"), "evaluation")
    training = _mapping(report.get("training"), "training")
    dataset = _mapping(report.get("dataset"), "dataset")
    benchmark = _mapping(report.get("benchmark"), "benchmark")
    return {
        "plan_sha256": evaluation.get("plan_sha256"),
        "split_seed": evaluation.get("split_seed"),
        "model_controls": evaluation.get("model_controls"),
        "training_controls": evaluation.get("training_controls"),
        "training_dataset_sha256": training.get("dataset_sha256"),
        "full_task_roster_sha256": training.get("full_task_roster_sha256"),
        "train_task_roster_sha256": training.get("train_task_roster_sha256"),
        "validation_task_roster_sha256": training.get(
            "validation_task_roster_sha256"
        ),
        "benchmark_dataset_sha256": dataset.get("sha256"),
        "benchmark_manifest_sha256": benchmark.get("manifest_sha256"),
        "evaluation_task_roster_sha256": dataset.get(
            "evaluation_task_roster_sha256"
        ),
    }


def _validate_completed_artifacts(
    path: Path,
    report: dict[str, Any],
    planned_dataset: dict[str, Any],
) -> dict[str, Any]:
    model = _mapping(report.get("model"), "model")
    model_hash = _require_sha256(model.get("sha256"), f"{path}: model hash")
    artifacts = _mapping(report.get("artifacts"), "artifacts")
    artifact_hashes: dict[str, str] = {}
    for name in ("model_config", "tokenizer", "task_spec", "training_manifest"):
        record = _mapping(artifacts.get(name), f"artifact {name}")
        artifact_hashes[name] = _require_sha256(
            record.get("sha256"), f"{path}: {name} hash"
        )
        if not isinstance(record.get("content"), dict):
            raise ValueError(f"{path}: {name} content must be exposed in the report")

    evaluation = _mapping(report.get("evaluation"), "evaluation")
    spec = evaluation.get("representation_spec")
    if not isinstance(spec, dict) or any(
        spec.get(field) != value
        for field, value in planned_dataset["representation_spec"].items()
    ):
        raise ValueError(f"{path}: task specification does not match the plan")
    if spec.get("task_roster_sha256") != planned_dataset["task_roster_sha256"]:
        raise ValueError(f"{path}: task specification roster does not match the plan")
    if artifacts["task_spec"]["content"] != spec:
        raise ValueError(f"{path}: task specification content is inconsistent")
    target_tokens = _validate_target_tokens(path, report, artifacts)
    return {
        "model_sha256": model_hash,
        "artifact_sha256": artifact_hashes,
        "target_tokens": target_tokens,
    }


def summarize_reports(
    report_paths: Iterable[Path], plan_path: Path = DEFAULT_CONFIG
) -> dict[str, Any]:
    """Validate all planned cells and return deterministic descriptive statistics."""
    plan_path = Path(plan_path).resolve()
    plan_sha256 = _sha256_file(plan_path)
    plan = expand_ablation_plan(plan_path)
    planned = {(run["representation"], run["model_seed"]): run for run in plan["runs"]}
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    split_rosters: tuple[str, str] | None = None

    for raw_path in report_paths:
        path = Path(raw_path).resolve()
        report = _load_record(path)
        if report["status"] == "completed":
            evaluation = _mapping(report.get("evaluation"), "evaluation")
            representation = evaluation.get("representation")
            model_seed = evaluation.get("model_seed")
            provenance = _completed_provenance(report)
        else:
            run = _mapping(report.get("run"), "run")
            representation = run.get("representation")
            model_seed = run.get("model_seed")
            provenance = _mapping(report.get("provenance"), "provenance")
        if not isinstance(representation, str) or not isinstance(model_seed, int):
            raise ValueError(f"{path}: representation and model_seed are required")
        cell = (representation, model_seed)
        if cell not in planned:
            raise ValueError(f"{path}: unplanned ablation cell {cell!r}")
        if cell in observed:
            raise ValueError(f"duplicate ablation cell {cell!r}")

        planned_run = planned[cell]
        dataset = _planned_dataset(plan, representation)
        current_rosters = _validate_common_provenance(
            path, provenance, plan, plan_sha256, planned_run, dataset
        )
        if split_rosters is None:
            split_rosters = current_rosters
        elif current_rosters != split_rosters:
            raise ValueError(f"{path}: normalized train/validation rosters differ across cells")

        if report["status"] == "failed":
            failure = _mapping(report.get("failure"), "failure")
            stage = failure.get("stage")
            reason = failure.get("reason")
            if not isinstance(stage, str) or not stage.strip():
                raise ValueError(f"{path}: failed run requires failure.stage")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"{path}: failed run requires failure.reason")
            observed[cell] = {
                "report_path": str(path),
                "representation": representation,
                "model_seed": model_seed,
                "status": "failed",
                "failure": {"stage": stage, "reason": reason},
                "provenance": provenance,
            }
            continue

        artifacts = _validate_completed_artifacts(path, report, dataset)
        metrics = _mapping(report.get("metrics"), "metrics")
        if metrics.get("primary_task_type") != "answer_complete":
            raise ValueError(f"{path}: primary task must be answer_complete")
        denominator = metrics.get("total_tests")
        if (
            not isinstance(denominator, int)
            or isinstance(denominator, bool)
            or denominator != plan["benchmark"]["task_count"]
        ):
            raise ValueError(
                f"{path}: primary denominator must equal canonical benchmark task_count "
                f"{plan['benchmark']['task_count']}"
            )
        metric_records = {
            name: _metric_record(metrics, source, denominator)
            for name, source in METRICS.items()
        }
        strata = _validate_strata(path, metrics, denominator)
        throughput = _validate_timing(path, metrics, denominator)
        observed[cell] = {
            "report_path": str(path),
            "representation": representation,
            "model_seed": model_seed,
            "status": "completed",
            "denominator": denominator,
            "metrics": metric_records,
            "answer_complete_strata": strata,
            "throughput": throughput,
            **artifacts,
        }

    missing = sorted(set(planned) - set(observed))
    if missing:
        formatted = ", ".join(f"{name}/seed-{seed}" for name, seed in missing)
        raise ValueError(f"incomplete ablation: missing {formatted}")

    representations = []
    for representation in REPRESENTATIONS:
        runs = [observed[(representation, seed)] for seed in (41, 42, 43)]
        aggregates: dict[str, Any] = {}
        for metric_name in METRICS:
            completed = [run for run in runs if run["status"] == "completed"]
            values = [run["metrics"][metric_name]["percent"] for run in completed]
            aggregate = _descriptive(values)
            aggregate.update(
                {
                    "total_numerator": sum(
                        run["metrics"][metric_name]["numerator"] for run in completed
                    ),
                    "total_denominator": sum(run["denominator"] for run in completed),
                    "denominators_by_seed": {
                        str(run["model_seed"]): run["denominator"] for run in completed
                    },
                }
            )
            aggregates[metric_name] = aggregate
        completed = [run for run in runs if run["status"] == "completed"]
        throughput_values = [
            run["throughput"]["tests_per_second"] for run in completed
        ]
        throughput_aggregate = _descriptive(throughput_values)
        throughput_count = sum(run["throughput"]["count"] for run in completed)
        throughput_seconds = sum(
            run["throughput"]["total_seconds"] for run in completed
        )
        throughput_aggregate.update(
            {
                "scope": "successful_completions_all_prompts",
                "total_count": throughput_count,
                "total_seconds": throughput_seconds,
                "pooled_tests_per_second": (
                    throughput_count / throughput_seconds
                    if throughput_seconds > 0
                    else 0.0
                ),
            }
        )
        representations.append(
            {
                "representation": representation,
                "runs": runs,
                "aggregate": aggregates,
                "throughput_aggregate": throughput_aggregate,
            }
        )

    contrasts_by_seed = []
    contrast_values = {name: [] for name in (*CONTRASTS, "interaction")}
    for seed in (41, 42, 43):
        seed_runs = {name: observed[(name, seed)] for name in REPRESENTATIONS}
        failed = sorted(name for name, run in seed_runs.items() if run["status"] == "failed")
        if failed:
            contrasts_by_seed.append(
                {"model_seed": seed, "status": "unavailable", "failed_cells": failed}
            )
            continue
        accuracy = {
            name: run["metrics"]["numerical_accuracy"]["percent"]
            for name, run in seed_runs.items()
        }
        values = {
            name: accuracy[positive] - accuracy[negative]
            for name, (positive, negative) in CONTRASTS.items()
        }
        values["interaction"] = values["reversal_fixed"] - values["reversal_minimal"]
        for name, value in values.items():
            contrast_values[name].append(value)
        contrasts_by_seed.append(
            {"model_seed": seed, "status": "completed", "percentage_points": values}
        )

    return {
        "schema": SUMMARY_SCHEMA,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "summary_kind": "descriptive",
        "inference_claims": False,
        "plan": {
            "path": str(plan_path),
            "sha256": plan_sha256,
            "name": plan["name"],
            "planned_cells": len(planned),
            "model_controls": plan["model"],
            "training_controls": plan["training"],
            "benchmark_manifest_sha256": plan["benchmark"]["manifest_sha256"],
            "evaluation_task_roster_sha256": plan["benchmark"][
                "task_roster_sha256"
            ],
        },
        "normalized_split_rosters": {
            "train_task_roster_sha256": split_rosters[0] if split_rosters else None,
            "validation_task_roster_sha256": split_rosters[1]
            if split_rosters
            else None,
        },
        "representations": representations,
        "paired_contrasts": {
            "unit": "percentage_points",
            "metric": "numerical_accuracy",
            "by_seed": contrasts_by_seed,
            "aggregate": {
                name: _descriptive(values) for name, values in contrast_values.items()
            },
        },
    }


def _fmt(value: object) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render every seed, failures, aggregates, and paired descriptive contrasts."""
    lines = [
        "# CalcGPT representation ablation",
        "",
        f"Summary schema: `{summary['schema']}` v{summary['schema_version']}",
        "",
        "Descriptive aggregation only; no statistical-significance claims are made.",
        "",
        "## Runs",
        "",
        "| Representation | Seed | Status | N | Numerical % | Strict format % | Exact match % | EOS % | Tests/s | Failure |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for group in summary["representations"]:
        for run in group["runs"]:
            if run["status"] == "completed":
                metrics = run["metrics"]
                lines.append(
                    f"| {group['representation']} | {run['model_seed']} | completed "
                    f"| {run['denominator']} | {metrics['numerical_accuracy']['percent']:.3f} "
                    f"| {metrics['strict_format']['percent']:.3f} "
                    f"| {metrics['exact_match']['percent']:.3f} "
                    f"| {metrics['eos_termination']['percent']:.3f} "
                    f"| {run['throughput']['tests_per_second']:.3f} | |"
                )
            else:
                reason = run["failure"]["reason"].replace("|", "\\|")
                lines.append(
                    f"| {group['representation']} | {run['model_seed']} | failed "
                    f"| — | — | — | — | — | — | {run['failure']['stage']}: {reason} |"
                )
    lines.extend(
        [
            "",
            "## Descriptive summary across completed seeds",
            "",
            "| Representation | Metric | Seeds | Mean % | Stdev % | Min % | Max % | Range % | Total N |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in summary["representations"]:
        for metric_name, aggregate in group["aggregate"].items():
            lines.append(
                f"| {group['representation']} | {metric_name} | {aggregate['n']} "
                f"| {_fmt(aggregate['mean'])} | {_fmt(aggregate['stdev'])} "
                f"| {_fmt(aggregate['min'])} | {_fmt(aggregate['max'])} "
                f"| {_fmt(aggregate['range'])} | {aggregate['total_denominator']} |"
            )
    lines.extend(
        [
            "",
            "## Throughput across completed seeds",
            "",
            "| Representation | Seeds | Mean tests/s | Stdev | Min | Max | Range | Pooled tests/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in summary["representations"]:
        aggregate = group["throughput_aggregate"]
        lines.append(
            f"| {group['representation']} | {aggregate['n']} "
            f"| {_fmt(aggregate['mean'])} | {_fmt(aggregate['stdev'])} "
            f"| {_fmt(aggregate['min'])} | {_fmt(aggregate['max'])} "
            f"| {_fmt(aggregate['range'])} "
            f"| {_fmt(aggregate['pooled_tests_per_second'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired 2×2 contrasts",
            "",
            "Positive values favor reversal for reversal contrasts and fixed-width for layout contrasts.",
            "",
            "| Seed | Status | Reversal minimal | Reversal fixed | Layout normal | Layout reversed | Interaction |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["paired_contrasts"]["by_seed"]:
        if row["status"] == "completed":
            values = row["percentage_points"]
            lines.append(
                f"| {row['model_seed']} | completed | {values['reversal_minimal']:.3f} "
                f"| {values['reversal_fixed']:.3f} | {values['layout_normal']:.3f} "
                f"| {values['layout_reversed']:.3f} | {values['interaction']:.3f} |"
            )
        else:
            lines.append(
                f"| {row['model_seed']} | unavailable ({', '.join(row['failed_cells'])}) "
                "| — | — | — | — | — |"
            )
    lines.extend(
        [
            "",
            "| Contrast | Seeds | Mean pp | Stdev pp | Min pp | Max pp | Range pp |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, aggregate in summary["paired_contrasts"]["aggregate"].items():
        lines.append(
            f"| {name} | {aggregate['n']} | {_fmt(aggregate['mean'])} "
            f"| {_fmt(aggregate['stdev'])} | {_fmt(aggregate['min'])} "
            f"| {_fmt(aggregate['max'])} | {_fmt(aggregate['range'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--plan", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = summarize_reports(args.reports, args.plan)
        args.json_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        args.markdown_output.write_text(render_markdown(summary), encoding="utf-8")
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
