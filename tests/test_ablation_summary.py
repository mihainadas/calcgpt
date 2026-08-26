"""Dependency-light completeness and aggregation tests for ablation reports."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.gen_ablation import expand_ablation_plan
from scripts.summarize_ablation import (
    FAILED_RUN_SCHEMA,
    render_markdown,
    summarize_reports,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "ablations-v1.toml"
REPRESENTATIONS = ("plain", "reversed", "padded", "padded-reversed")
SEEDS = (41, 42, 43)
BASE_CORRECT = {"plain": 30, "reversed": 60, "padded": 90, "padded-reversed": 150}
TRAIN_ROSTER = "c" * 64
VALIDATION_ROSTER = "d" * 64


class AblationSummaryTests(unittest.TestCase):
    def _reports(self, directory: Path) -> tuple[list[Path], dict]:
        plan = expand_ablation_plan(PLAN)
        plan_hash = hashlib.sha256(PLAN.read_bytes()).hexdigest()
        datasets = {item["representation"]: item for item in plan["datasets"]}
        paths = []
        for run in plan["runs"]:
            representation = run["representation"]
            seed = run["model_seed"]
            dataset = datasets[representation]
            task_spec = {
                **dataset["representation_spec"],
                "task_roster_sha256": dataset["task_roster_sha256"],
            }
            target_tokens = {
                "schema": "calcgpt-target-token-counts",
                "schema_version": 1,
                "loss_scope": "answer-only",
                "train": {
                    "active_target_tokens": 82_000,
                    "active_answer_tokens": 50_000,
                    "active_eos_tokens": 32_000,
                },
                "validation": {
                    "active_target_tokens": 20_000,
                    "active_answer_tokens": 12_000,
                    "active_eos_tokens": 8_000,
                },
            }
            training_manifest = {
                "splits": {
                    "train_examples": 32_000,
                    "validation_examples": 8_000,
                },
                "target_tokens": target_tokens,
            }
            path = directory / f"{representation}-{seed}.json"
            numerator = BASE_CORRECT[representation] + (seed - 41) * 3
            strata = {
                dimension: {"all": {"n": 300, "correct": numerator}}
                for dimension in (
                    "digit_bucket",
                    "operation",
                    "event_count",
                    "longest_chain",
                    "overflow",
                    "zero_operand",
                    "equal_operands",
                )
            }
            path.write_text(
                json.dumps(
                    {
                        "schema": "calcgpt-evaluation-report",
                        "schema_version": 1,
                        "report_type": "calcgpt-evaluation",
                        "status": "completed",
                        "dataset": {
                            "sha256": run["benchmark_sha256"],
                            "evaluation_task_roster_sha256": run[
                                "evaluation_task_roster_sha256"
                            ],
                        },
                        "training": {
                            "dataset_sha256": dataset["sha256"],
                            "full_task_roster_sha256": dataset[
                                "task_roster_sha256"
                            ],
                            "train_task_roster_sha256": TRAIN_ROSTER,
                            "validation_task_roster_sha256": VALIDATION_ROSTER,
                            "target_tokens": target_tokens,
                        },
                        "benchmark": {
                            "manifest_sha256": run[
                                "benchmark_manifest_sha256"
                            ],
                        },
                        "model": {"sha256": "e" * 64},
                        "artifacts": {
                            name: {
                                "sha256": "f" * 64,
                                "content": (
                                    task_spec
                                    if name == "task_spec"
                                    else training_manifest
                                    if name == "training_manifest"
                                    else {}
                                ),
                            }
                            for name in (
                                "model_config",
                                "tokenizer",
                                "task_spec",
                                "training_manifest",
                            )
                        },
                        "evaluation": {
                            "representation": representation,
                            "representation_spec": task_spec,
                            "model_seed": seed,
                            "split_seed": run["split_seed"],
                            "plan_sha256": plan_hash,
                            "model_controls": plan["model"],
                            "training_controls": plan["training"],
                        },
                        "metrics": {
                            "primary_task_type": "answer_complete",
                            "total_tests": 300,
                            "correct_arithmetic": numerator,
                            "valid_format": 240,
                            "exact_matches": numerator,
                            "eos_terminated": 270,
                            "answer_complete_strata": strata,
                            "timing": {
                                "scope": "successful_completions_all_prompts",
                                "count": 900,
                                "total_seconds": 9.0,
                                "tests_per_second": 100.0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            paths.append(path)
        return paths, plan

    def _make_failed(self, path: Path, plan: dict) -> None:
        completed = json.loads(path.read_text(encoding="utf-8"))
        evaluation = completed["evaluation"]
        training = completed["training"]
        dataset = completed["dataset"]
        benchmark = completed["benchmark"]
        path.write_text(
            json.dumps(
                {
                    "schema": FAILED_RUN_SCHEMA,
                    "schema_version": 1,
                    "status": "failed",
                    "run": {
                        "representation": evaluation["representation"],
                        "model_seed": evaluation["model_seed"],
                    },
                    "failure": {"stage": "training", "reason": "out of memory"},
                    "provenance": {
                        "plan_sha256": evaluation["plan_sha256"],
                        "split_seed": evaluation["split_seed"],
                        "model_controls": plan["model"],
                        "training_controls": plan["training"],
                        "training_dataset_sha256": training["dataset_sha256"],
                        "full_task_roster_sha256": training[
                            "full_task_roster_sha256"
                        ],
                        "train_task_roster_sha256": TRAIN_ROSTER,
                        "validation_task_roster_sha256": VALIDATION_ROSTER,
                        "benchmark_dataset_sha256": dataset["sha256"],
                        "benchmark_manifest_sha256": benchmark[
                            "manifest_sha256"
                        ],
                        "evaluation_task_roster_sha256": dataset[
                            "evaluation_task_roster_sha256"
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_complete_matrix_emits_artifacts_and_paired_contrasts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reports, _ = self._reports(Path(directory))
            summary = summarize_reports(reports, PLAN)
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["plan"]["planned_cells"], 12)
        plain = summary["representations"][0]
        self.assertEqual([run["model_seed"] for run in plain["runs"]], [41, 42, 43])
        aggregate = plain["aggregate"]["numerical_accuracy"]
        self.assertEqual(aggregate["mean"], 11.0)
        self.assertEqual(aggregate["stdev"], 1.0)
        self.assertEqual(aggregate["range"], 2.0)
        self.assertEqual(aggregate["total_denominator"], 900)
        contrasts = summary["paired_contrasts"]
        self.assertEqual(
            contrasts["by_seed"][0]["percentage_points"]["interaction"], 10.0
        )
        self.assertEqual(contrasts["aggregate"]["layout_reversed"]["mean"], 30.0)
        markdown = render_markdown(summary)
        self.assertIn("no statistical-significance claims", markdown)
        self.assertIn("| plain | 41 | completed | 300 |", markdown)
        self.assertIn("| plain | 3 | 100.000 |", markdown)

    def test_failed_cell_stays_visible_and_is_excluded_from_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reports, plan = self._reports(Path(directory))
            self._make_failed(reports[0], plan)
            summary = summarize_reports(reports, PLAN)
        failed = summary["representations"][0]["runs"][0]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(
            summary["representations"][0]["aggregate"]["numerical_accuracy"]["n"],
            2,
        )
        self.assertEqual(summary["paired_contrasts"]["by_seed"][0]["status"], "unavailable")
        self.assertEqual(summary["paired_contrasts"]["aggregate"]["interaction"]["n"], 2)

    def test_incomplete_duplicate_and_plan_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reports, _ = self._reports(Path(directory))
            with self.assertRaisesRegex(ValueError, "incomplete ablation"):
                summarize_reports(reports[:-1], PLAN)
            with self.assertRaisesRegex(ValueError, "duplicate ablation cell"):
                summarize_reports([*reports, reports[0]], PLAN)

            changed = json.loads(reports[-1].read_text(encoding="utf-8"))
            changed["evaluation"]["model_controls"]["embedding_dim"] = 999
            reports[-1].write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model_controls does not match"):
                summarize_reports(reports, PLAN)

    def test_missing_artifact_hash_and_split_roster_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reports, _ = self._reports(Path(directory))
            changed = json.loads(reports[-1].read_text(encoding="utf-8"))
            changed["artifacts"]["tokenizer"]["sha256"] = None
            reports[-1].write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tokenizer hash"):
                summarize_reports(reports, PLAN)

            reports, _ = self._reports(Path(directory))
            changed = json.loads(reports[-1].read_text(encoding="utf-8"))
            changed["training"]["train_task_roster_sha256"] = "a" * 64
            reports[-1].write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rosters differ"):
                summarize_reports(reports, PLAN)

    def test_denominator_must_equal_the_canonical_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for denominator in (299, 301):
                with self.subTest(denominator=denominator):
                    reports, _ = self._reports(root)
                    changed = json.loads(reports[0].read_text(encoding="utf-8"))
                    changed["metrics"]["total_tests"] = denominator
                    reports[0].write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "canonical benchmark task_count"):
                        summarize_reports(reports, PLAN)

    def test_missing_required_scientific_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mutations = {
                "EOS": lambda report: report["metrics"].pop("eos_terminated"),
                "strata": lambda report: report["metrics"].pop(
                    "answer_complete_strata"
                ),
                "timing": lambda report: report["metrics"].pop("timing"),
                "target_tokens": lambda report: report["training"].pop(
                    "target_tokens"
                ),
            }
            for label, mutate in mutations.items():
                with self.subTest(field=label):
                    reports, _ = self._reports(root)
                    changed = json.loads(reports[0].read_text(encoding="utf-8"))
                    mutate(changed)
                    reports[0].write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        summarize_reports(reports, PLAN)

    def test_bad_strata_timing_and_target_counts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            reports, _ = self._reports(root)
            changed = json.loads(reports[0].read_text(encoding="utf-8"))
            changed["metrics"]["answer_complete_strata"]["operation"]["all"][
                "n"
            ] = 299
            reports[0].write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "n sums to 299"):
                summarize_reports(reports, PLAN)

            reports, _ = self._reports(root)
            changed = json.loads(reports[0].read_text(encoding="utf-8"))
            changed["metrics"]["timing"]["tests_per_second"] = float("inf")
            reports[0].write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite nonnegative"):
                summarize_reports(reports, PLAN)

            reports, _ = self._reports(root)
            changed = json.loads(reports[0].read_text(encoding="utf-8"))
            for target in (
                changed["training"]["target_tokens"],
                changed["artifacts"]["training_manifest"]["content"][
                    "target_tokens"
                ],
            ):
                target["train"]["active_eos_tokens"] = 82_001
            reports[0].write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "component count exceeds total"):
                summarize_reports(reports, PLAN)


if __name__ == "__main__":
    unittest.main()
