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
from lib.representation import REPRESENTATION_NAMES, RepresentationSpec

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

    for name in REPRESENTATION_NAMES:
        modern = RepresentationSpec.from_name(name, 3).to_dict()
        modern['task_roster_sha256'] = 'a' * 64
        (path / 'task_spec.json').write_text(json.dumps(modern), encoding='utf-8')
        assert _load_task_spec(path, path) == modern

    modern['layout'] = 'minimal'
    (path / 'task_spec.json').write_text(json.dumps(modern), encoding='utf-8')
    try:
        _load_task_spec(path, path)
    except ValueError as exc:
        assert 'conflicts' in str(exc)
    else:
        raise AssertionError('cross-field conflict was accepted')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_inference_policy_applies_before_representation_encoding(self) -> None:
        code = """
import sys
import types

torch = types.ModuleType('torch')
transformers = types.ModuleType('transformers')
transformers.GPT2LMHeadModel = object
sys.modules['torch'] = torch
sys.modules['transformers'] = transformers

from lib.inference import CalcGPT, InferenceConfig

engine = CalcGPT.__new__(CalcGPT)
engine.config = InferenceConfig()
engine.task_spec = {
    'format': 'plain',
    'operators': ['+'],
    'negative_results': False,
}

unsupported = engine.solve('2-1')
assert 'not supported' in unsupported['error']

engine.task_spec['operators'] = ['+', '-']
negative = engine.solve('1-2')
assert 'negative subtraction' in negative['error']
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_inference_formats_and_decodes_all_four_representations(self) -> None:
        code = """
import sys
import types

class NoGrad:
    def __enter__(self): return None
    def __exit__(self, *args): return False

torch = types.ModuleType('torch')
torch.long = object()
torch.tensor = lambda values, **kwargs: values
torch.no_grad = NoGrad
transformers = types.ModuleType('transformers')
transformers.GPT2LMHeadModel = object
sys.modules['torch'] = torch
sys.modules['transformers'] = transformers

from lib.inference import CalcGPT, InferenceConfig
from lib.representation import REPRESENTATION_NAMES, RepresentationSpec
from lib.tokenizer import CalcGPTTokenizer

class Output:
    def __init__(self, values): self.values = values
    def tolist(self): return self.values

class Model:
    def __init__(self, output):
        self.output = output
        self.config = types.SimpleNamespace(n_positions=100)
        self.kwargs = None
    def generate(self, input_ids, **kwargs):
        self.kwargs = kwargs
        return [Output(self.output)]

for name in REPRESENTATION_NAMES:
    spec = RepresentationSpec.from_name(name, 3)
    equation = spec.format_task((7, '+', 8))
    tokenizer = CalcGPTTokenizer([equation])
    engine = CalcGPT.__new__(CalcGPT)
    engine.config = InferenceConfig()
    engine.task_spec = spec.to_dict()
    engine.representation_spec = spec
    engine.tokenizer = tokenizer
    engine.device = 'cpu'
    engine.loaded_model_path = 'fake'
    engine.model = Model(tokenizer.encode(equation, add_eos=False))
    result = engine.solve('7+8')
    assert result['model_prompt'] == equation.split('=', 1)[0] + '='
    assert result['answer'] == '15'
    assert result['is_correct'] is True
    assert result['task_format'] == name
    assert engine.model.kwargs['do_sample'] is False

spec = RepresentationSpec.from_name('plain', 3)
equation = spec.format_task((7, '+', 8))
prompt = equation.split('=', 1)[0] + '='
tokenizer = CalcGPTTokenizer([equation])
engine = CalcGPT.__new__(CalcGPT)
engine.config = InferenceConfig()
engine.task_spec = spec.to_dict()
engine.representation_spec = spec
engine.tokenizer = tokenizer
engine.device = 'cpu'
engine.loaded_model_path = 'fake'
engine.model = Model(
    tokenizer.encode(prompt, add_eos=False) + [tokenizer.vocab['=']]
)
malformed = engine.solve('7+8')
assert 'non-numeric answer' in malformed['error']
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_training_rejects_context_shorter_than_encoded_data(self) -> None:
        code = """
import json
import sys
import tempfile
import types
from pathlib import Path

torch = types.ModuleType('torch')
torch.Tensor = object
torch.cuda = types.SimpleNamespace(is_available=lambda: False)
torch.backends = types.SimpleNamespace()
torch_utils = types.ModuleType('torch.utils')
torch_data = types.ModuleType('torch.utils.data')
torch_data.Dataset = object
transformers = types.ModuleType('transformers')
transformers.GPT2Config = object
transformers.GPT2LMHeadModel = object
transformers.Trainer = object
transformers.TrainingArguments = object
transformers.set_seed = lambda seed: None
sys.modules['torch'] = torch
sys.modules['torch.utils'] = torch_utils
sys.modules['torch.utils.data'] = torch_data
sys.modules['transformers'] = transformers

from lib.tokenizer import CalcGPTTokenizer
from lib.train import CalcGPTTrainer, OptimizedDataset, TrainingConfig

assert TrainingConfig(seed=17).split_seed == 17
assert TrainingConfig(seed=17, split_seed=42).split_seed == 42

tokenizer = CalcGPTTokenizer(['12+3=15'])
answer_only = OptimizedDataset(
    ['12+3=15'], tokenizer.max_length, tokenizer, 'answer-only'
)
prompt_length = len(tokenizer.encode('12+3=', add_eos=False))
labels = answer_only.data[0]['labels']
assert labels[:prompt_length] == [-100] * prompt_length
assert labels[prompt_length] != -100
assert labels[len(tokenizer.encode('12+3=15')) - 1] == tokenizer.eos_token_id

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    dataset = root / 'dataset.txt'
    dataset.write_text('7+8=15\\n', encoding='utf-8')
    trainer = CalcGPTTrainer(
        TrainingConfig(n_positions=1), dataset, root / 'model', verbose=False
    )
    try:
        trainer.load_and_prepare_data()
    except ValueError as exc:
        assert 'smaller than the longest encoded training sequence' in str(exc)
    else:
        raise AssertionError('undersized context was accepted')

    dataset.write_text('7+8=15\\n', encoding='utf-8')
    wrong_representation = CalcGPTTrainer(
        TrainingConfig(task_format='reversed'),
        dataset,
        root / 'wrong-model',
        verbose=False,
    )
    try:
        wrong_representation.load_and_prepare_data()
    except ValueError:
        pass
    else:
        raise AssertionError('training accepted a row from the wrong representation')

    dataset.write_text('7+8=51\\n', encoding='utf-8')
    artifact_dir = root / 'artifact'
    artifact_dir.mkdir()
    artifact_trainer = CalcGPTTrainer(
        TrainingConfig(
            seed=17,
            split_seed=42,
            task_format='reversed',
            loss_scope='answer-only',
        ),
        dataset,
        artifact_dir,
        verbose=False,
    )
    artifact_trainer.load_and_prepare_data()
    artifact_trainer.train_examples = list(artifact_trainer.examples)
    artifact_trainer.validation_examples = []
    artifact_trainer._write_training_manifest(1.0, 0.5, None)
    task_spec = json.loads((artifact_dir / 'task_spec.json').read_text())
    manifest = json.loads((artifact_dir / 'training_manifest.json').read_text())
    assert task_spec['name'] == 'reversed'
    assert task_spec['layout'] == 'minimal'
    assert len(task_spec['task_roster_sha256']) == 64
    assert manifest['training_config']['split_seed'] == 42
    assert manifest['training_config']['loss_scope'] == 'answer-only'
    assert manifest['dataset']['task_roster_sha256'] == task_spec['task_roster_sha256']
    assert len(manifest['splits']['train_task_roster_sha256']) == 64
    assert len(manifest['splits']['validation_task_roster_sha256']) == 64
    assert manifest['target_tokens']['loss_scope'] == 'answer-only'
    assert manifest['target_tokens']['schema'] == 'calcgpt-target-token-counts'
    assert manifest['target_tokens']['schema_version'] == 1
    assert manifest['target_tokens']['train'] == {
        'active_target_tokens': 3,
        'active_answer_tokens': 2,
        'active_eos_tokens': 1,
    }
    assert manifest['target_tokens']['validation'] == {
        'active_target_tokens': 0,
        'active_answer_tokens': 0,
        'active_eos_tokens': 0,
    }
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
