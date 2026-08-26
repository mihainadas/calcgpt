"""Dependency-light CLI contract smoke tests."""

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def run_cli(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / script), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class CliTests(unittest.TestCase):
    def test_help_does_not_require_ml_dependencies(self) -> None:
        for script in (
            "calcgpt.py",
            "calcgpt_train.py",
            "calcgpt_eval.py",
            "calcgpt_dategen.py",
            "scripts/gen_padded.py",
            "scripts/summarize_ablation.py",
        ):
            with self.subTest(script=script):
                result = run_cli(script, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_versions_share_one_release_number(self) -> None:
        for script in (
            "calcgpt.py",
            "calcgpt_train.py",
            "calcgpt_eval.py",
            "calcgpt_dategen.py",
        ):
            with self.subTest(script=script):
                result = run_cli(script, "--version")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("3.0.0", result.stdout)

    def test_batch_requires_at_least_one_problem(self) -> None:
        result = run_cli("calcgpt.py", "--batch")
        self.assertEqual(result.returncode, 2)
        self.assertIn("expected at least one argument", result.stderr)

    def test_generator_fails_fast_when_request_exceeds_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "should-not-exist.txt"
            result = run_cli(
                "scripts/gen_padded.py",
                "--num-examples",
                "156",
                "--operand-width",
                "1",
                "--output",
                str(output),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("only 155 distinct tasks", result.stderr)
            self.assertFalse(output.exists())

    def test_fail_under_range_is_validated_before_loading_ml_stack(self) -> None:
        result = run_cli("calcgpt_eval.py", "--fail-under", "101")
        self.assertEqual(result.returncode, 2)
        self.assertIn("fail-under must be between 0 and 100", result.stderr)

    def test_missing_ml_dependencies_have_actionable_errors(self) -> None:
        code = r"""
import builtins
import runpy
import sys

script, *arguments = sys.argv[1:]
real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'transformers'}:
        raise ModuleNotFoundError(f"No module named '{name}'")
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
sys.argv = [script, *arguments]
runpy.run_path(script, run_name='__main__')
"""
        invocations = (
            ("calcgpt.py", "--batch", "1+1"),
            ("calcgpt_train.py",),
            ("calcgpt_eval.py",),
        )
        for invocation in invocations:
            with self.subTest(script=invocation[0]):
                result = subprocess.run(
                    [sys.executable, "-c", code, str(ROOT / invocation[0]), *invocation[1:]],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn("Install them with", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_batch_json_is_emitted_before_operational_error_status(self) -> None:
        import calcgpt

        class FakeCalcGPT:
            def solve_batch(self, problems):
                return [
                    {
                        "problem": problems[0],
                        "answer": "3",
                        "is_correct": False,
                        "inference_time": 0.001,
                    },
                    {
                        "problem": problems[1],
                        "error": "unsupported operator",
                        "inference_time": 0.001,
                    },
                ]

            def get_model_info(self):
                return {"model_path": "fake"}

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            errors = calcgpt.batch_mode(
                FakeCalcGPT(), ["1+1", "1*1"], "json", None, quiet=True
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(errors, 1)
        self.assertEqual(payload["metadata"]["errors"], 1)
        self.assertEqual(len(payload["results"]), 2)

    def test_eval_json_is_clean_and_low_accuracy_is_scientific_result(self) -> None:
        import calcgpt_eval

        @dataclass
        class FakeEvaluationConfig:
            max_tokens: int = 15
            device: str = "auto"
            sample_size: int | None = None
            sample_seed: int = 42
            verbose: bool = False

        metrics = {
            "primary_task_type": "answer_complete",
            "total_tests": 1,
            "successful_completions": 1,
            "successful_completions_pct": 100.0,
            "valid_format": 1,
            "valid_format_pct": 100.0,
            "correct_arithmetic": 0,
            "correct_arithmetic_pct": 0.0,
            "complete_expressions": 1,
            "complete_expressions_pct": 100.0,
            "exact_matches": 0,
            "exact_matches_pct": 0.0,
            "contains_input": 1,
            "contains_input_pct": 100.0,
            "diagnostic_all_prompts": {"total_tests": 1},
            "by_type": {"answer_complete": {"total": 1, "correct": 0, "valid_format": 1}},
        }
        results = [
            {
                "test_case": {
                    "input": "1+1=",
                    "expected": "1+1=2",
                    "type": "answer_complete",
                },
                "completion_result": {
                    "success": True,
                    "completion": "1+1=3",
                    "inference_time": 0.001,
                },
                "validation": {"correct_arithmetic": False},
            }
        ]

        class FakeEvaluator:
            def __init__(self, model_path, config, **kwargs):
                self.model_path = Path(model_path)
                self.loaded_model_path = self.model_path
                self.device = "cpu"
                self.evaluation_task_roster_sha256 = "b" * 64

            def evaluate_dataset(self, dataset_path):
                return results, metrics

        evaluation_module = types.ModuleType("lib.evaluation")
        evaluation_module.EvaluationConfig = FakeEvaluationConfig
        evaluation_module.CalcGPTEvaluator = FakeEvaluator
        inference_module = types.ModuleType("lib.inference")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.txt"
            dataset.write_text("1+1=2\n", encoding="utf-8")
            model = root / "model"
            model.mkdir()
            model_config = {
                "n_embd": 128,
                "n_layer": 4,
                "n_head": 8,
                "n_inner": 256,
                "n_positions": 20,
            }
            task_spec = {
                "schema_version": 1,
                "name": "plain",
                "format": "plain",
                "layout": "minimal",
                "answer_order": "normal",
                "operand_width": 3,
                "task_roster_sha256": "a" * 64,
            }
            training_manifest = {
                "dataset": {
                    "path": "training.txt",
                    "sha256": "c" * 64,
                    "task_roster_sha256": "a" * 64,
                },
                "splits": {
                    "train_task_roster_sha256": "d" * 64,
                    "validation_task_roster_sha256": "e" * 64,
                },
                "target_tokens": {"loss_scope": "answer-only"},
                "training_config": {
                    "seed": 41,
                    "split_seed": 42,
                    "epochs": 30,
                    "batch_size": 64,
                    "learning_rate": 0.001,
                    "warmup_steps": 100,
                    "weight_decay": 0.01,
                    "save_steps": 2000,
                    "test_split": 0.2,
                    "no_augmentation": True,
                    "loss_scope": "answer-only",
                },
            }
            artifact_payloads = {
                "config.json": model_config,
                "tokenizer.json": {"schema_version": 1},
                "task_spec.json": task_spec,
                "training_manifest.json": training_manifest,
            }
            for filename, payload in artifact_payloads.items():
                (model / filename).write_text(
                    json.dumps(payload) + "\n", encoding="utf-8"
                )
            (model / "model.safetensors").write_bytes(b"fake weights")
            benchmark_manifest = root / "benchmark-manifest.json"
            benchmark_manifest.write_text(
                json.dumps(
                    {
                        "schema": "calcgpt-semantic-holdout",
                        "schema_version": 1,
                        "task_roster_sha256": "b" * 64,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            ablation_plan = root / "ablation.toml"
            ablation_plan.write_text('schema = "test"\n', encoding="utf-8")
            inference_module.get_model_path = lambda requested: str(model)

            def run(*extra_args):
                stdout = io.StringIO()
                stderr = io.StringIO()
                argv = [
                    "calcgpt_eval.py",
                    "--json",
                    "--dataset",
                    str(dataset),
                    *extra_args,
                ]
                with (
                    mock.patch.dict(
                        sys.modules,
                        {
                            "lib.evaluation": evaluation_module,
                            "lib.inference": inference_module,
                        },
                    ),
                    mock.patch.object(sys, "argv", argv),
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    status = calcgpt_eval.main()
                return status, stdout.getvalue(), stderr.getvalue()

            status, stdout, stderr = run(
                "--benchmark-manifest",
                str(benchmark_manifest),
                "--ablation-plan",
                str(ablation_plan),
            )
            self.assertEqual(status, 0, stderr)
            report = json.loads(stdout)
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["metrics"]["correct_arithmetic_pct"], 0.0)
            self.assertTrue(report["created_at"].endswith("+00:00"))
            self.assertIsNotNone(report["dataset"]["sha256"])
            self.assertIsNotNone(report["model"]["sha256"])
            self.assertTrue(
                all(
                    artifact["sha256"]
                    for artifact in report["artifacts"].values()
                )
            )
            self.assertEqual(report["evaluation"]["representation"], "plain")
            self.assertEqual(report["evaluation"]["model_controls"]["num_layers"], 4)
            self.assertEqual(report["training"]["train_task_roster_sha256"], "d" * 64)
            self.assertEqual(report["benchmark"]["task_roster_sha256"], "b" * 64)
            self.assertEqual(
                report["evaluation"]["plan_sha256"],
                hashlib.sha256(ablation_plan.read_bytes()).hexdigest(),
            )
            self.assertTrue(
                all(isinstance(artifact["content"], dict) for artifact in report["artifacts"].values())
            )
            self.assertEqual(stderr, "")

            status, stdout, stderr = run("--fail-under", "1")
            self.assertEqual(status, 1, stderr)
            self.assertEqual(json.loads(stdout)["schema_version"], 1)

    def test_evaluation_report_write_error_is_not_swallowed(self) -> None:
        import calcgpt_eval

        with tempfile.TemporaryDirectory() as directory:
            missing_parent = Path(directory) / "missing" / "report.json"
            with self.assertRaises(OSError):
                calcgpt_eval.save_results({"schema_version": 1}, str(missing_parent))


if __name__ == "__main__":
    unittest.main()
