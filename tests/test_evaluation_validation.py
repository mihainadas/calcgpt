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

from lib.evaluation import validate_completion


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


if __name__ == "__main__":
    unittest.main()
