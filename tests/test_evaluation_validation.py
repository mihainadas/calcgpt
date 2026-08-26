"""Pure evaluation-contract tests that run without loading a model."""

# ruff: noqa: E402

import sys
import types
import unittest

# lib.evaluation imports optional ML modules. Minimal stubs keep these pure tests
# independent from the heavyweight training extra.
torch_stub = types.ModuleType("torch")
torch_stub.device = object
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
torch_stub.backends = types.SimpleNamespace()
sys.modules.setdefault("torch", torch_stub)
transformers_stub = types.ModuleType("transformers")
transformers_stub.GPT2LMHeadModel = object
sys.modules.setdefault("transformers", transformers_stub)

from lib.evaluation import calculate_metrics, validate_completion
from lib.representation import RepresentationSpec


class EvaluationValidationTests(unittest.TestCase):
    def test_wrong_task_does_not_receive_arithmetic_credit(self) -> None:
        case = {"input": "12+34=", "expected": "12+34=46", "type": "answer_complete"}
        result = validate_completion(case, "12+99=111")
        self.assertFalse(result["correct_arithmetic"])
        self.assertFalse(result["details"]["same_task"])

    def test_reversed_padded_answer_is_decoded(self) -> None:
        case = {
            "input": "012+034=",
            "expected": "012+034=6400",
            "type": "answer_complete",
        }
        result = validate_completion(case, "012+034=6400")
        self.assertTrue(result["correct_arithmetic"])
        self.assertEqual(result["details"]["answer_order"], "reversed")
        self.assertEqual(result["details"]["actual_result"], 46)

    def test_plain_answer_still_works(self) -> None:
        case = {"input": "12-5=", "expected": "12-5=7", "type": "answer_complete"}
        self.assertTrue(validate_completion(case, "12-5=7")["correct_arithmetic"])

    def test_fixed_width_validity_is_separate_from_numerical_correctness(self) -> None:
        spec = RepresentationSpec.from_name("padded-reversed", 3)
        case = {
            "input": "012+034=",
            "expected": "012+034=6400",
            "type": "answer_complete",
        }
        truncated = validate_completion(case, "12+34=64", spec)
        self.assertTrue(truncated["correct_arithmetic"])
        self.assertFalse(truncated["valid_format"])
        self.assertFalse(truncated["exact_match"])

        wrong_but_well_formed = validate_completion(case, "012+034=7400", spec)
        self.assertFalse(wrong_but_well_formed["correct_arithmetic"])
        self.assertTrue(wrong_but_well_formed["valid_format"])

    def test_primary_metrics_exclude_underspecified_first_operand_prompt(self) -> None:
        def result(prompt_type: str, correct: bool):
            details = {}
            if prompt_type == "answer_complete":
                details = validate_completion(
                    {
                        "input": "1+1=",
                        "expected": "1+1=2",
                        "type": "answer_complete",
                    },
                    "1+1=2" if correct else "1+1=3",
                )["details"]
            return {
                "test_case": {"type": prompt_type},
                "completion_result": {
                    "success": True,
                    "inference_time": 0.001,
                    "terminated_by_eos": prompt_type == "answer_complete",
                },
                "validation": {
                    "valid_format": True,
                    "correct_arithmetic": correct,
                    "complete_expression": True,
                    "exact_match": correct,
                    "contains_input": True,
                    "details": details,
                },
            }

        metrics = calculate_metrics(
            [
                result("first_operand", False),
                result("expression_complete", False),
                result("answer_complete", True),
            ]
        )
        self.assertEqual(metrics["primary_task_type"], "answer_complete")
        self.assertEqual(metrics["total_tests"], 1)
        self.assertEqual(metrics["correct_arithmetic_pct"], 100.0)
        self.assertEqual(metrics["diagnostic_all_prompts"]["total_tests"], 3)
        self.assertAlmostEqual(
            metrics["diagnostic_all_prompts"]["correct_arithmetic_pct"], 100 / 3
        )
        self.assertEqual(metrics["eos_terminated"], 1)
        self.assertEqual(
            metrics["timing"]["scope"], "successful_completions_all_prompts"
        )
        self.assertEqual(metrics["timing"]["count"], 3)
        self.assertAlmostEqual(metrics["timing"]["total_seconds"], 0.003)
        self.assertAlmostEqual(metrics["timing"]["tests_per_second"], 1000.0)
        self.assertEqual(
            metrics["answer_complete_strata"]["operation"]["addition"],
            {"n": 1, "correct": 1},
        )

    def test_answer_complete_strata_count_correctness_by_feature(self) -> None:
        cases = [
            ({"input": "999+1=", "expected": "999+1=1000", "type": "answer_complete"}, "999+1=1000", True),
            ({"input": "0+0=", "expected": "0+0=0", "type": "answer_complete"}, "0+0=1", False),
        ]
        results = []
        for case, completion, eos in cases:
            validation = validate_completion(case, completion)
            results.append(
                {
                    "test_case": case,
                    "completion_result": {
                        "success": True,
                        "inference_time": 0.001,
                        "terminated_by_eos": eos,
                    },
                    "validation": validation,
                }
            )
        metrics = calculate_metrics(results)
        self.assertEqual(metrics["eos_terminated"], 1)
        self.assertEqual(metrics["eos_terminated_pct"], 50.0)
        self.assertEqual(
            metrics["answer_complete_strata"]["overflow"]["true"],
            {"n": 1, "correct": 1},
        )
        self.assertEqual(
            metrics["answer_complete_strata"]["zero_operand"]["true"],
            {"n": 1, "correct": 0},
        )

    def test_metrics_require_a_determined_answer_case(self) -> None:
        result = {
            "test_case": {"type": "first_operand"},
            "completion_result": {"success": True, "inference_time": 0.001},
            "validation": {
                "valid_format": True,
                "correct_arithmetic": True,
                "complete_expression": True,
                "exact_match": True,
                "contains_input": True,
            },
        }
        with self.assertRaisesRegex(ValueError, "no determined answer_complete"):
            calculate_metrics([result])

    def test_zero_success_throughput_is_finite_and_scoped(self) -> None:
        case = {"input": "1+1=", "expected": "1+1=2", "type": "answer_complete"}
        metrics = calculate_metrics(
            [
                {
                    "test_case": case,
                    "completion_result": {
                        "success": False,
                        "inference_time": 0.001,
                        "terminated_by_eos": False,
                    },
                    "validation": validate_completion(case, ""),
                }
            ]
        )
        self.assertEqual(metrics["timing"]["count"], 0)
        self.assertEqual(metrics["timing"]["total_seconds"], 0)
        self.assertEqual(metrics["timing"]["tests_per_second"], 0.0)


if __name__ == "__main__":
    unittest.main()
