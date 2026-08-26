import unittest
from pathlib import Path

from lib.benchmark import (
    format_task,
    load_examples,
    magnitude_task_space_size,
    parse_example,
    sample_heldout_by_magnitude,
    sample_heldout_tasks,
    task_space_size,
)

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTests(unittest.TestCase):
    def test_task_space_accounting(self):
        self.assertEqual(task_space_size(1), 155)
        self.assertEqual(task_space_size(3), 1_500_500)
        self.assertEqual(magnitude_task_space_size(1), 155)
        self.assertEqual(magnitude_task_space_size(2), 12_195)

    def test_parse_accepts_reversed_answer_format(self):
        self.assertEqual(parse_example("012+034=6400"), (12, "+", 34))
        self.assertEqual(parse_example("034-012=2200"), (34, "-", 12))
        with self.assertRaises(ValueError):
            parse_example("12 + 34")

    def test_heldout_tasks_are_unique_and_excluded(self):
        excluded = ["002+004=6000", "008-001=7000", "044-039=5000"]
        tasks = sample_heldout_by_magnitude(100, 3, excluded, seed=7)
        self.assertEqual(len(tasks), 300)
        self.assertEqual(len(set(tasks)), 300)
        excluded_tasks = {parse_example(example) for example in excluded}
        self.assertTrue(all((a, op, b) not in excluded_tasks for _, a, b, op in tasks))
        for digits, a, b, op in tasks:
            lo = 0 if digits == 1 else 10 ** (digits - 1)
            hi = 10**digits - 1
            self.assertTrue(lo <= a <= hi and lo <= b <= hi)
            if op == "-":
                self.assertGreaterEqual(a, b)

    def test_heldout_sampling_is_repeatable(self):
        first = sample_heldout_tasks(50, 2, ["010+010=0200"], seed=11)
        second = sample_heldout_tasks(50, 2, ["010+010=0200"], seed=11)
        self.assertEqual(first, second)

    def test_canonical_demo_benchmark_is_absent_from_training_dataset(self):
        excluded = load_examples(ROOT / "datasets" / "ds-calcgpt-padded.txt")
        tasks = sample_heldout_by_magnitude(100, 3, excluded, seed=0)
        excluded_tasks = {parse_example(example) for example in excluded}
        self.assertEqual(len(tasks), 300)
        self.assertEqual(len(set(tasks)), 300)
        self.assertTrue(
            all((a, op, b) not in excluded_tasks for _, a, b, op in tasks)
        )

    def test_rejects_when_bucket_is_exhausted(self):
        all_tasks = sample_heldout_tasks(155, 1, [], seed=0)
        excluded = [format_task(task, 1) for task in all_tasks]
        with self.assertRaises(ValueError):
            sample_heldout_tasks(1, 1, excluded, seed=0)


if __name__ == "__main__":
    unittest.main()
