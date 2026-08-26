import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.gen_padded import sample_examples, task_space_size

ROOT = Path(__file__).resolve().parents[1]


class GeneratorTests(unittest.TestCase):
    def test_task_space_size(self):
        self.assertEqual(task_space_size(1), 155)
        self.assertEqual(task_space_size(2), 15_050)

    def test_sample_is_unique_and_valid(self):
        examples = sample_examples(155, 1, seed=42)
        self.assertEqual(len(examples), 155)
        self.assertEqual(len(set(examples)), 155)
        self.assertTrue(all(len(example) == 6 for example in examples))

    def test_rejects_invalid_size_or_width(self):
        for count, width in [(0, 1), (-1, 1), (1, 0), (156, 1)]:
            with self.subTest(count=count, width=width):
                with self.assertRaises(ValueError):
                    sample_examples(count, width, seed=0)

    def test_same_seed_is_byte_identical_across_hash_seeds(self):
        script = ROOT / "scripts" / "gen_padded.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = [Path(temp_dir) / "first.txt", Path(temp_dir) / "second.txt"]
            for hash_seed, path in zip(("1", "999"), paths):
                env = os.environ.copy()
                env["PYTHONHASHSEED"] = hash_seed
                subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "-n",
                        "1000",
                        "-w",
                        "3",
                        "--seed",
                        "42",
                        "-o",
                        str(path),
                    ],
                    cwd=ROOT,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            digests = [hashlib.sha256(path.read_bytes()).digest() for path in paths]
            self.assertEqual(digests[0], digests[1])

    def test_committed_dataset_matches_canonical_generation(self):
        expected = "\n".join(sample_examples(40_000, 3, seed=42)) + "\n"
        actual = (ROOT / "datasets" / "ds-calcgpt-padded.txt").read_text(
            encoding="utf-8"
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
