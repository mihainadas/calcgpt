"""Dependency-light CLI contract smoke tests."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
