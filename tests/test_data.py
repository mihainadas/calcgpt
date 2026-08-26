"""Tests for deterministic, leakage-safe dataset preparation."""

import tempfile
import unittest
from pathlib import Path

from lib.data import augment_data, canonical_group_key, load_dataset, split_examples_grouped


class DataTests(unittest.TestCase):
    def test_load_dataset_rejects_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.txt"
            path.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                load_dataset(path)

    def test_augmentation_adds_only_missing_addition_twin(self) -> None:
        examples = ["01+02=30", "03-01=20"]
        self.assertEqual(
            augment_data(examples),
            ["01+02=30", "03-01=20", "02+01=30"],
        )

    def test_grouped_split_has_no_commutative_leakage(self) -> None:
        examples = [
            "01+02=30",
            "02+01=30",
            "01+03=40",
            "03+01=40",
            "03-01=20",
            "02-01=10",
        ]
        training, validation = split_examples_grouped(examples, 0.5, seed=42)
        train_groups = {canonical_group_key(example) for example in training}
        validation_groups = {canonical_group_key(example) for example in validation}
        self.assertTrue(training)
        self.assertTrue(validation)
        self.assertTrue(train_groups.isdisjoint(validation_groups))

    def test_grouped_split_is_deterministic(self) -> None:
        examples = [f"{value}+0={value}" for value in range(20)]
        first = split_examples_grouped(examples, 0.2, seed=7)
        second = split_examples_grouped(examples, 0.2, seed=7)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
