"""Research-contract tests for semantic holdouts, strata, and ablation plans."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lib.benchmark import (
    arithmetic_features,
    build_semantic_holdout_manifest,
    holdout_manifest_sha256,
    load_examples,
    parse_example,
    sample_heldout_tasks,
    semantic_task_key,
)
from scripts.gen_ablation import expand_ablation_plan, load_ablation_config

ROOT = Path(__file__).resolve().parents[1]


class ResearchBenchmarkTests(unittest.TestCase):
    def _write_plan(self, directory: Path, content: str) -> Path:
        path = directory / "plan.toml"
        path.write_text(content, encoding="utf-8", newline="\n")
        return path

    def test_semantic_exclusion_blocks_both_addition_orientations(self) -> None:
        tasks = sample_heldout_tasks(109, 1, ["3+9=12"], seed=42)
        self.assertEqual(len(tasks), 109)
        self.assertNotIn((3, "+", 9), tasks)
        self.assertNotIn((9, "+", 3), tasks)

    def test_canonical_holdout_has_no_semantic_training_overlap(self) -> None:
        examples = load_examples(ROOT / "datasets" / "ds-calcgpt-padded.txt")
        training_keys = {semantic_task_key(parse_example(example)) for example in examples}
        manifest = build_semantic_holdout_manifest(100, 3, examples, seed=42)
        benchmark_keys = {record["semantic_key"] for record in manifest["tasks"]}
        self.assertEqual(manifest["task_count"], 300)
        self.assertEqual(len(benchmark_keys), 300)
        self.assertTrue(training_keys.isdisjoint(benchmark_keys))

    def test_arithmetic_features_count_chains_and_edges(self) -> None:
        self.assertEqual(
            arithmetic_features((999, "+", 1), 3),
            {
                "operation": "addition",
                "event_kind": "carry",
                "event_count": 3,
                "longest_chain": 3,
                "overflow": True,
                "has_zero_operand": False,
                "equal_operands": False,
            },
        )
        separated = arithmetic_features((505, "+", 505), 3)
        self.assertEqual(separated["event_count"], 2)
        self.assertEqual(separated["longest_chain"], 1)
        self.assertTrue(separated["equal_operands"])
        borrow = arithmetic_features((100, "-", 1), 3)
        self.assertEqual(borrow["event_count"], 2)
        self.assertEqual(borrow["longest_chain"], 2)
        self.assertFalse(borrow["overflow"])
        self.assertTrue(arithmetic_features((7, "+", 0), 3)["has_zero_operand"])

    def test_holdout_manifest_hash_is_deterministic_and_seed_sensitive(self) -> None:
        excluded = ["003+009=2100", "009-003=6000"]
        first = build_semantic_holdout_manifest(10, 2, excluded, seed=42)
        second = build_semantic_holdout_manifest(10, 2, excluded, seed=42)
        other = build_semantic_holdout_manifest(10, 2, excluded, seed=43)
        self.assertEqual(holdout_manifest_sha256(first), holdout_manifest_sha256(second))
        self.assertNotEqual(holdout_manifest_sha256(first), holdout_manifest_sha256(other))

    def test_ablation_plan_expands_four_by_three_without_collisions(self) -> None:
        plan = expand_ablation_plan(ROOT / "configs" / "ablations-v1.toml")
        self.assertEqual(len(plan["datasets"]), 4)
        self.assertEqual(len(plan["runs"]), 12)
        self.assertEqual(
            [dataset["representation"] for dataset in plan["datasets"]],
            ["plain", "reversed", "padded", "padded-reversed"],
        )
        self.assertEqual({run["model_seed"] for run in plan["runs"]}, {41, 42, 43})
        self.assertEqual(len({run["model_dir"] for run in plan["runs"]}), 12)
        self.assertEqual(len({run["result_path"] for run in plan["runs"]}), 12)
        self.assertEqual(
            len({dataset["task_roster_sha256"] for dataset in plan["datasets"]}),
            1,
        )
        self.assertEqual(len({dataset["sha256"] for dataset in plan["datasets"]}), 4)
        manifest = build_semantic_holdout_manifest(
            100,
            3,
            load_examples(ROOT / "datasets" / "ds-calcgpt-padded.txt"),
            seed=42,
        )
        manifest_hash = holdout_manifest_sha256(manifest)
        evaluation_roster_hash = manifest["task_roster_sha256"]
        self.assertEqual(plan["benchmark"]["manifest"], manifest)
        self.assertEqual(plan["benchmark"]["manifest_sha256"], manifest_hash)
        self.assertEqual(
            plan["benchmark"]["task_roster_sha256"], evaluation_roster_hash
        )
        self.assertEqual(len(plan["benchmark"]["renderings"]), 4)
        self.assertEqual(
            {
                rendering["task_roster_sha256"]
                for rendering in plan["benchmark"]["renderings"]
            },
            {evaluation_roster_hash},
        )
        self.assertEqual(
            len(
                {
                    rendering["sha256"]
                    for rendering in plan["benchmark"]["renderings"]
                }
            ),
            4,
        )
        renderings = {
            rendering["representation"]: rendering
            for rendering in plan["benchmark"]["renderings"]
        }
        for dataset in plan["datasets"]:
            self.assertTrue(Path(dataset["path"]).is_absolute())
            self.assertTrue(Path(dataset["benchmark_path"]).is_absolute())
            self.assertEqual(dataset["benchmark_manifest_sha256"], manifest_hash)
            self.assertEqual(
                dataset["evaluation_task_roster_sha256"], evaluation_roster_hash
            )
            rendering = renderings[dataset["representation"]]
            self.assertEqual(dataset["benchmark_path"], rendering["path"])
            self.assertEqual(dataset["benchmark_sha256"], rendering["sha256"])
        for run in plan["runs"]:
            for field in (
                "dataset_path",
                "benchmark_path",
                "benchmark_manifest_path",
                "model_dir",
                "result_path",
            ):
                self.assertTrue(Path(run[field]).is_absolute())
            self.assertEqual(run["benchmark_manifest_sha256"], manifest_hash)
            self.assertEqual(
                run["evaluation_task_roster_sha256"], evaluation_roster_hash
            )
            rendering = renderings[run["representation"]]
            self.assertEqual(run["benchmark_path"], rendering["path"])
            self.assertEqual(run["benchmark_sha256"], rendering["sha256"])

    def test_ablation_config_rejects_duplicate_representation_collision(self) -> None:
        canonical = (ROOT / "configs" / "ablations-v1.toml").read_text(encoding="utf-8")
        invalid = canonical.replace(
            '["plain", "reversed", "padded", "padded-reversed"]',
            '["plain", "plain", "padded", "padded-reversed"]',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.toml"
            path.write_text(invalid, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "collision"):
                load_ablation_config(path)

    def test_ablation_config_rejects_mutated_or_malformed_controls(self) -> None:
        canonical = (ROOT / "configs" / "ablations-v1.toml").read_text(encoding="utf-8")
        mutations = {
            "missing name": canonical.replace(
                'name = "representations-width3-v1"\n', ""
            ),
            "wrong model": canonical.replace("num_layers = 4", "num_layers = 5"),
            "wrong model type": canonical.replace(
                "embedding_dim = 128", 'embedding_dim = "128"'
            ),
            "missing model field": canonical.replace("n_positions = 20\n", ""),
            "extra model field": canonical.replace(
                "n_positions = 20", "n_positions = 20\ndropout = 0.1"
            ),
            "wrong training": canonical.replace("epochs = 30", "epochs = 31"),
            "wrong float type": canonical.replace(
                "learning_rate = 0.001", "learning_rate = 1"
            ),
            "wrong benchmark": canonical.replace(
                "samples_per_digit_bucket = 100", "samples_per_digit_bucket = 99"
            ),
            "wrong data type": canonical.replace("task_count = 40000", 'task_count = "40000"'),
            "wrong source": canonical.replace(
                'source = "datasets/ds-calcgpt-padded.txt"',
                'source = "datasets/ds-calcgpt.txt"',
            ),
            "wrong source checksum": canonical.replace(
                "de810f1c93fba69e2d0fc3985bcfb9fc62e88d9083df320ce069e7ffc5a9667c",
                "a" * 64,
            ),
            "wrong output path type": canonical.replace(
                'output_root = "outputs/ablations-v1"', "output_root = 42"
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in mutations.items():
                with self.subTest(name=name):
                    path = self._write_plan(root, content)
                    with self.assertRaises(ValueError):
                        load_ablation_config(path)

    def test_cli_rejects_malformed_plan_without_traceback(self) -> None:
        canonical = (ROOT / "configs" / "ablations-v1.toml").read_text(encoding="utf-8")
        invalid_plans = {
            "malformed TOML": 'schema = "calcgpt-ablation-plan"\nname = [\n',
            "missing name": canonical.replace(
                'name = "representations-width3-v1"\n', ""
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in invalid_plans.items():
                with self.subTest(name=name):
                    path = self._write_plan(root, content)
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "scripts" / "gen_ablation.py"),
                            "--config",
                            str(path),
                        ],
                        cwd=root,
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertIn("error:", result.stderr)

    def test_plan_paths_resolve_from_project_root_outside_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "gen_ablation.py")],
                cwd=directory,
                capture_output=True,
                text=True,
                check=True,
            )
        plan = json.loads(result.stdout)
        expected_output = (ROOT / "outputs" / "ablations-v1").resolve()
        self.assertEqual(
            Path(plan["source_dataset"]["path"]),
            (ROOT / "datasets" / "ds-calcgpt-padded.txt").resolve(),
        )
        output_paths = [
            *(Path(dataset["path"]) for dataset in plan["datasets"]),
            *(Path(dataset["benchmark_path"]) for dataset in plan["datasets"]),
            *(Path(run["model_dir"]) for run in plan["runs"]),
            *(Path(run["result_path"]) for run in plan["runs"]),
        ]
        self.assertTrue(
            all(path.is_relative_to(expected_output) for path in output_paths)
        )


if __name__ == "__main__":
    unittest.main()
