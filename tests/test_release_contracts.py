"""Dependency-light checks for packaging and canonical experiment metadata."""

import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseContractTests(unittest.TestCase):
    def test_source_distribution_contains_the_demo(self) -> None:
        manifest_lines = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        self.assertIn("include demo.py", manifest_lines)
        self.assertTrue((ROOT / "demo.py").is_file())

    def test_canonical_experiment_has_versioned_fixed_benchmark(self) -> None:
        with (ROOT / "configs" / "padded-3digit.toml").open("rb") as handle:
            config = tomllib.load(handle)

        self.assertEqual(config["schema_version"], 1)
        self.assertEqual(config["data"]["seed"], 42)
        self.assertEqual(config["training"]["seed"], 42)
        self.assertEqual(config["evaluation"]["seed"], 42)
        self.assertEqual(config["evaluation"]["samples_per_digit_bucket"], 100)
        self.assertTrue(config["evaluation"]["require_disjoint_training_pairs"])
        self.assertEqual(config["training"]["split_seed"], 42)
        self.assertEqual(config["training"]["loss_scope"], "answer-only")

    def test_ablation_locks_split_and_answer_only_loss(self) -> None:
        with (ROOT / "configs" / "ablations-v1.toml").open("rb") as handle:
            config = tomllib.load(handle)
        self.assertEqual(config["experiment"]["split_seed"], 42)
        self.assertEqual(config["training"]["loss_scope"], "answer-only")

    def test_demo_extra_includes_ml_runtime_and_terminal_ui(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]

        demo_dependencies = set(project["optional-dependencies"]["demo"])
        self.assertIn("calc-gpt[train]", demo_dependencies)
        self.assertTrue(any(item.startswith("rich") for item in demo_dependencies))


if __name__ == "__main__":
    unittest.main()
