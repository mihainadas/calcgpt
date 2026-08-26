"""Tests for the four orthogonal arithmetic representations."""

import unittest

from lib.representation import (
    REPRESENTATION_NAMES,
    RepresentationSpec,
    task_roster_sha256,
)


class RepresentationTests(unittest.TestCase):
    def test_all_four_formats_are_exact_and_round_trip(self) -> None:
        task = (7, "+", 8)
        expected = {
            "plain": "7+8=15",
            "reversed": "7+8=51",
            "padded": "007+008=0015",
            "padded-reversed": "007+008=5100",
        }
        for name in REPRESENTATION_NAMES:
            with self.subTest(name=name):
                spec = RepresentationSpec.from_name(name, operand_width=3)
                encoded = spec.format_task(task)
                self.assertEqual(encoded, expected[name])
                self.assertEqual(spec.parse_example(encoded), task)

    def test_schema_round_trip_and_cross_field_validation(self) -> None:
        spec = RepresentationSpec.from_name("padded-reversed", 3)
        self.assertEqual(RepresentationSpec.from_dict(spec.to_dict()), spec)

        invalid = spec.to_dict()
        invalid["answer_order"] = "normal"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            RepresentationSpec.from_dict(invalid)

    def test_parser_rejects_wrong_layout_order_or_answer(self) -> None:
        spec = RepresentationSpec.from_name("padded-reversed", 3)
        for invalid in ("7+8=51", "007+008=0015", "007+008=6100"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    spec.parse_example(invalid)

    def test_structural_decoder_does_not_require_correct_arithmetic(self) -> None:
        spec = RepresentationSpec.from_name("padded-reversed", 3)
        task, decoded_answer = spec.decode_example("007+008=6100")
        self.assertEqual(task, (7, "+", 8))
        self.assertEqual(decoded_answer, 16)
        with self.assertRaisesRegex(ValueError, "wrong width"):
            spec.decode_example("7+8=61")

    def test_dataset_validation_preserves_one_normalized_roster(self) -> None:
        roster = [(7, "+", 8), (12, "-", 3), (0, "+", 0)]
        roster_hash = task_roster_sha256(roster)
        for name in REPRESENTATION_NAMES:
            spec = RepresentationSpec.from_name(name, 3)
            rendered = spec.render_dataset(roster)
            parsed = spec.validate_dataset(rendered)
            self.assertEqual(parsed, roster)
            self.assertEqual(task_roster_sha256(parsed), roster_hash)

    def test_dataset_validation_rejects_duplicate_tasks(self) -> None:
        spec = RepresentationSpec.from_name("plain", 3)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            spec.validate_dataset(["7+8=15", "7+8=15"])

    def test_task_domain_rejects_negative_subtraction_and_width_overflow(self) -> None:
        spec = RepresentationSpec.from_name("plain", 3)
        for task in ((1, "-", 2), (1000, "+", 0)):
            with self.subTest(task=task):
                with self.assertRaises(ValueError):
                    spec.format_task(task)


if __name__ == "__main__":
    unittest.main()
