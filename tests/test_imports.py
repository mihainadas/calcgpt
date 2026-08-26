"""Verify that data-only CalcGPT imports do not load optional ML packages."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LazyImportTests(unittest.TestCase):
    def test_package_and_data_exports_do_not_import_ml_stack(self) -> None:
        code = """
import sys
import lib
from lib import CalcGPTTokenizer, DatagenConfig, DatasetGenerator
assert 'torch' not in sys.modules, 'torch was imported eagerly'
assert 'transformers' not in sys.modules, 'transformers was imported eagerly'
assert CalcGPTTokenizer.__module__ == 'lib.tokenizer'
assert DatagenConfig.__module__ == 'lib.dategen'
assert DatasetGenerator.__module__ == 'lib.dategen'
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_package_attribute_has_normal_error(self) -> None:
        code = """
import lib
try:
    lib.not_a_real_export
except AttributeError as exc:
    assert "not_a_real_export" in str(exc)
else:
    raise AssertionError('missing AttributeError')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_compatibility_validation_rejects_mismatched_model_metadata(self) -> None:
        code = """
import sys
import types

torch = types.ModuleType('torch')
transformers = types.ModuleType('transformers')
transformers.GPT2LMHeadModel = object
sys.modules['torch'] = torch
sys.modules['transformers'] = transformers

from lib.inference import validate_tokenizer_compatibility
from lib.tokenizer import CalcGPTTokenizer

tokenizer = CalcGPTTokenizer(['1+1=2'])
good_config = types.SimpleNamespace(
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    n_positions=tokenizer.max_length + 2,
)
validate_tokenizer_compatibility(tokenizer, types.SimpleNamespace(config=good_config))

bad_config = types.SimpleNamespace(
    vocab_size=tokenizer.vocab_size + 1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    n_positions=tokenizer.max_length + 2,
)
try:
    validate_tokenizer_compatibility(tokenizer, types.SimpleNamespace(config=bad_config))
except ValueError as exc:
    assert 'vocabulary mismatch' in str(exc)
else:
    raise AssertionError('mismatched model metadata was accepted')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_task_spec_round_trip_and_validation(self) -> None:
        code = """
import json
import sys
import tempfile
import types
from pathlib import Path

torch = types.ModuleType('torch')
transformers = types.ModuleType('transformers')
transformers.GPT2LMHeadModel = object
sys.modules['torch'] = torch
sys.modules['transformers'] = transformers

from lib.inference import _load_task_spec

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)
    payload = {
        'schema_version': 1,
        'format': 'padded-reversed',
        'operand_width': 3,
        'answer_width': 4,
        'answer_order': 'reversed',
        'operators': ['+', '-'],
        'negative_results': False,
    }
    (path / 'task_spec.json').write_text(json.dumps(payload), encoding='utf-8')
    assert _load_task_spec(path, path) == payload
    payload['answer_width'] = 5
    (path / 'task_spec.json').write_text(json.dumps(payload), encoding='utf-8')
    try:
        _load_task_spec(path, path)
    except ValueError as exc:
        assert 'answer_width' in str(exc)
    else:
        raise AssertionError('invalid task spec was accepted')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
