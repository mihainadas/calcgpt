"""
CalcGPT Evaluation Library

Core evaluation functionality for CalcGPT models with proper separation of concerns.
"""

import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import GPT2LMHeadModel

from .benchmark import arithmetic_features
from .inference import (
    _load_artifact_tokenizer,
    _load_task_spec,
    _representation_from_task_spec,
    _resolve_model_artifact,
    get_device,
    validate_tokenizer_compatibility,
)
from .representation import RepresentationSpec, task_roster_sha256


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters"""
    max_tokens: int = 15
    device: str = 'auto'
    sample_size: Optional[int] = None
    sample_seed: int = 42
    verbose: bool = False

    def validate(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.sample_size is not None and self.sample_size < 1:
            raise ValueError("sample_size must be positive")


def load_evaluation_dataset(dataset_path: str) -> List[str]:
    """Load the evaluation dataset"""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    with path.open('r', encoding='utf-8') as dataset_file:
        equations = [line.strip() for line in dataset_file if line.strip()]
    if not equations:
        raise ValueError(f"Evaluation dataset is empty: {path}")
    return equations


def create_test_cases(equations: List[str]) -> List[Dict[str, str]]:
    """Create test cases from full equations"""
    test_cases = []
    
    for equation in equations:
        if '=' in equation:
            parts = equation.split('=')
            if len(parts) == 2:
                left_side = parts[0].strip()
                full_equation = equation
                
                # Create different types of test cases
                test_cases.extend([
                    # Complete from just the first operand
                    {'input': left_side.split('+')[0] if '+' in left_side else left_side.split('-')[0] if '-' in left_side else left_side,
                     'expected': full_equation,
                     'type': 'first_operand'},
                    
                    # Complete from the operation without equals
                    {'input': left_side,
                     'expected': full_equation,
                     'type': 'expression_complete'},
                    
                    # Complete from partial equation
                    {'input': left_side + '=',
                     'expected': full_equation,
                     'type': 'answer_complete'}
                ])
    
    return test_cases


def _infer_representation(expected: str) -> Optional[RepresentationSpec]:
    """Infer a contract only for the legacy public validation helper."""
    match = re.fullmatch(r'(\d+)([+-])(\d+)=(\d+)', expected.strip())
    if match is None:
        return None
    left_text, operator, right_text, _ = match.groups()
    width = max(len(left_text), len(right_text))
    task = (int(left_text), operator, int(right_text))
    for name in ('plain', 'reversed', 'padded', 'padded-reversed'):
        spec = RepresentationSpec.from_name(name, width)
        try:
            if spec.format_task(task) == expected.strip():
                return spec
        except ValueError:
            pass
    return None


def validate_completion(
    test_case: Dict[str, str],
    completion: str,
    representation_spec: Optional[RepresentationSpec] = None,
) -> Dict[str, Any]:
    """Validate a completion against the requested task and answer encoding."""
    input_text = test_case['input']
    expected = test_case['expected']
    
    result = {
        'valid_format': False,
        'correct_arithmetic': False,
        'complete_expression': False,
        'exact_match': False,
        'contains_input': False,
        'details': {}
    }
    
    # Check if completion contains the input
    result['contains_input'] = input_text in completion
    
    expected_match = re.fullmatch(r'(\d+)([+-])(\d+)=(\d+)', expected.strip())
    completion_match = re.fullmatch(r'(\d+)([+-])(\d+)=(\d+)', completion.strip())
    result['complete_expression'] = completion_match is not None
    spec = representation_spec or _infer_representation(expected)

    if expected_match:
        expected_left, expected_op, expected_right, _ = expected_match.groups()
        expected_task = (int(expected_left), expected_op, int(expected_right))
        feature_width = spec.operand_width if spec is not None else max(
            len(expected_left), len(expected_right)
        )
        try:
            result['details'].update(
                {
                    'expected_task': {
                        'left': expected_task[0],
                        'operator': expected_task[1],
                        'right': expected_task[2],
                    },
                    'digit_bucket': max(
                        len(str(expected_task[0])), len(str(expected_task[2]))
                    ),
                    'arithmetic_features': arithmetic_features(
                        expected_task, feature_width
                    ),
                }
            )
        except ValueError as exc:
            result['details']['expected_task_error'] = str(exc)

    if completion_match:
        if spec is None:
            result['valid_format'] = True
        else:
            try:
                spec.decode_example(completion)
                result['valid_format'] = True
            except ValueError as exc:
                result['details']['format_error'] = str(exc)

    if expected_match and completion_match:
        expected_left, expected_op, expected_right, _ = expected_match.groups()
        actual_left, actual_op, actual_right, actual_answer_text = (
            completion_match.groups()
        )
        expected_value = (
            int(expected_left) + int(expected_right)
            if expected_op == '+'
            else int(expected_left) - int(expected_right)
        )
        answer_order = spec.answer_order if spec is not None else 'normal'
        try:
            actual_value = int(
                actual_answer_text[::-1]
                if answer_order == 'reversed'
                else actual_answer_text
            )
        except ValueError:
            actual_value = None

        same_task = (
            int(actual_left) == int(expected_left)
            and actual_op == expected_op
            and int(actual_right) == int(expected_right)
        )
        result['correct_arithmetic'] = same_task and actual_value == expected_value
        result['details'].update(
            {
                'answer_order': answer_order,
                'expected_result': expected_value,
                'actual_result': actual_value,
                'same_task': same_task,
            }
        )
    
    # Check exact match
    result['exact_match'] = completion.strip() == expected.strip()
    
    return result


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate primary task-answer metrics and separate prompt diagnostics."""
    if not results:
        return {}

    def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(items)
        summary = {
            'total_tests': total,
            'successful_completions': sum(
                1 for result in items if result['completion_result']['success']
            ),
            'valid_format': sum(1 for result in items if result['validation']['valid_format']),
            'correct_arithmetic': sum(
                1 for result in items if result['validation']['correct_arithmetic']
            ),
            'complete_expressions': sum(
                1 for result in items if result['validation']['complete_expression']
            ),
            'exact_matches': sum(1 for result in items if result['validation']['exact_match']),
            'contains_input': sum(1 for result in items if result['validation']['contains_input']),
            'eos_terminated': sum(
                1
                for result in items
                if result['completion_result'].get('terminated_by_eos', False)
            ),
        }
        for key in (
            'successful_completions',
            'valid_format',
            'correct_arithmetic',
            'complete_expressions',
            'exact_matches',
            'contains_input',
            'eos_terminated',
        ):
            summary[f'{key}_pct'] = (summary[key] / total) * 100 if total else 0.0
        return summary

    primary_results = [
        result for result in results if result['test_case']['type'] == 'answer_complete'
    ]
    if not primary_results:
        raise ValueError("evaluation results contain no determined answer_complete cases")

    # Top-level fields intentionally describe the determined arithmetic task for
    # compatibility with existing report consumers. Other prompts are diagnostics.
    metrics = summarize(primary_results)
    metrics['primary_task_type'] = 'answer_complete'
    metrics['diagnostic_all_prompts'] = summarize(results)

    strata = {
        'digit_bucket': defaultdict(lambda: {'n': 0, 'correct': 0}),
        'operation': defaultdict(lambda: {'n': 0, 'correct': 0}),
        'event_count': defaultdict(lambda: {'n': 0, 'correct': 0}),
        'longest_chain': defaultdict(lambda: {'n': 0, 'correct': 0}),
        'overflow': defaultdict(lambda: {'n': 0, 'correct': 0}),
        'zero_operand': defaultdict(lambda: {'n': 0, 'correct': 0}),
        'equal_operands': defaultdict(lambda: {'n': 0, 'correct': 0}),
    }
    for result in primary_results:
        details = result['validation'].get('details', {})
        features = details.get('arithmetic_features', {})
        values = {
            'digit_bucket': details.get('digit_bucket'),
            'operation': features.get('operation'),
            'event_count': features.get('event_count'),
            'longest_chain': features.get('longest_chain'),
            'overflow': features.get('overflow'),
            'zero_operand': features.get('has_zero_operand'),
            'equal_operands': features.get('equal_operands'),
        }
        if any(value is None for value in values.values()):
            raise ValueError("answer_complete result lacks arithmetic strata metadata")
        for name, value in values.items():
            key = str(value).lower() if isinstance(value, bool) else str(value)
            strata[name][key]['n'] += 1
            if result['validation']['correct_arithmetic']:
                strata[name][key]['correct'] += 1
    metrics['answer_complete_strata'] = {
        name: {key: dict(counts) for key, counts in values.items()}
        for name, values in strata.items()
    }
    
    # Calculate by test type
    by_type = defaultdict(
        lambda: {'total': 0, 'correct': 0, 'valid_format': 0}
    )
    for result in results:
        test_type = result['test_case']['type']
        by_type[test_type]['total'] += 1
        if result['validation']['correct_arithmetic']:
            by_type[test_type]['correct'] += 1
        if result['validation']['valid_format']:
            by_type[test_type]['valid_format'] += 1
    
    metrics['by_type'] = {name: dict(values) for name, values in by_type.items()}
    
    # Throughput is scoped to successful completions across all prompt types.
    # Failed generations remain visible in diagnostic_all_prompts but have no
    # meaningful completion latency to include here.
    import statistics

    times = [
        result['completion_result']['inference_time']
        for result in results
        if result['completion_result']['success']
    ]
    total_seconds = sum(times)
    metrics['timing'] = {
        'scope': 'successful_completions_all_prompts',
        'count': len(times),
        'total_seconds': total_seconds,
        'tests_per_second': len(times) / total_seconds if total_seconds > 0 else 0.0,
    }
    if times:
        metrics['timing'].update(
            {
                'mean_ms': statistics.mean(times) * 1000,
                'median_ms': statistics.median(times) * 1000,
                'min_ms': min(times) * 1000,
                'max_ms': max(times) * 1000,
            }
        )
        if len(times) > 1:
            metrics['timing']['std_ms'] = statistics.stdev(times) * 1000
    
    return metrics


class CalcGPTEvaluator:
    """CalcGPT model evaluator"""
    
    def __init__(
        self,
        model_path: str,
        config: Optional[EvaluationConfig] = None,
        verbose: bool = False,
        *,
        tokenizer_path: Optional[Union[str, Path]] = None,
        legacy_dataset_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize CalcGPT evaluator
        
        Args:
            model_path: Path to trained model
            config: Evaluation configuration
            verbose: Enable verbose output
        """
        self.model_path = Path(model_path)
        self.config = config or EvaluationConfig()
        self.config.validate()
        self.verbose = verbose
        self.tokenizer_path = tokenizer_path
        self.legacy_dataset_path = legacy_dataset_path
        
        # Initialize device
        self.device = get_device(self.config.device)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.task_spec: Dict[str, Any] = {}
        self.representation_spec: Optional[RepresentationSpec] = None
        self.evaluation_task_roster_sha256: Optional[str] = None
        self.loaded_model_path = self.model_path
        
        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()
        self.task_spec = _load_task_spec(self.model_path, self.loaded_model_path)
        self.representation_spec = _representation_from_task_spec(self.task_spec)
        
    def log(self, message: str):
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def _load_model(self):
        """Load the trained model"""
        try:
            self.loaded_model_path = _resolve_model_artifact(self.model_path)
            self.log(f"Loading model from: {self.loaded_model_path}")
            self.model = GPT2LMHeadModel.from_pretrained(str(self.loaded_model_path))
            self.model.to(self.device)
            self.model.eval()
            
            if self.verbose:
                total_params = sum(p.numel() for p in self.model.parameters())
                self.log("✅ Model loaded successfully!")
                self.log(f"   Parameters: {total_params:,}")
                self.log(f"   Device: {self.device}")
                
        except Exception as exc:
            raise RuntimeError(f"Error loading model: {exc}") from exc
    
    def _load_tokenizer(self):
        """Load the tokenizer saved with the selected model artifact."""
        try:
            self.tokenizer = _load_artifact_tokenizer(
                self.model_path,
                self.loaded_model_path,
                self.tokenizer_path,
                self.legacy_dataset_path,
            )
            validate_tokenizer_compatibility(self.tokenizer, self.model)
            
            if self.verbose:
                self.log("✅ Tokenizer loaded:")
                self.log(f"   Vocab size: {self.tokenizer.vocab_size}")
                self.log(f"   Max length: {self.tokenizer.max_length}")
                
        except Exception as exc:
            raise RuntimeError(f"Error loading tokenizer: {exc}") from exc
    
    def complete_expression(self, partial_expr: str) -> Dict[str, Any]:
        """Complete a partial arithmetic expression
        
        Args:
            partial_expr: Partial expression to complete
            
        Returns:
            Dictionary with completion results
        """
        start_time = time.time()
        
        # Clean input
        partial_expr = partial_expr.strip()
        
        try:
            # Encode input (remove EOS for generation)
            input_tokens = self.tokenizer.encode(partial_expr, add_eos=False)
            if not input_tokens:
                raise ValueError("Expression produced no input tokens")
            remaining_training_length = self.tokenizer.max_length - len(input_tokens)
            if remaining_training_length < 1:
                raise ValueError("Expression leaves no room for a generated completion")
            max_new_tokens = min(
                self.config.max_tokens, remaining_training_length
            )
            context_length = getattr(self.model.config, 'n_positions', None)
            if context_length is None:
                context_length = getattr(self.model.config, 'max_position_embeddings', None)
            requested_length = len(input_tokens) + max_new_tokens
            if context_length is not None and requested_length > context_length:
                raise ValueError(
                    f"Input plus max_tokens requires {requested_length} positions, "
                    f"but model context length is {context_length}"
                )
            input_ids = torch.tensor([input_tokens], dtype=torch.long).to(self.device)
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy for consistent evaluation
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bad_words_ids=[[self.tokenizer.pad_token_id]],  # Prevent padding
                    num_return_sequences=1
                )
            
            result_tokens = generated[0].tolist()
            new_tokens = result_tokens[len(input_tokens):]
            completion = self.tokenizer.decode(result_tokens)
            
            # Calculate timing
            inference_time = time.time() - start_time
            
            return {
                'input': partial_expr,
                'completion': completion,
                'inference_time': inference_time,
                'input_tokens': input_tokens,
                'output_tokens': result_tokens,
                'terminated_by_eos': self.tokenizer.eos_token_id in new_tokens,
                'success': True
            }
            
        except Exception as e:
            return {
                'input': partial_expr,
                'completion': '',
                'inference_time': time.time() - start_time,
                'error': str(e),
                'terminated_by_eos': False,
                'success': False
            }
    
    def evaluate_test_cases(self, test_cases: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Evaluate multiple test cases
        
        Args:
            test_cases: List of test case dictionaries
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for test_case in test_cases:
            # Get model completion
            completion_result = self.complete_expression(test_case['input'])
            
            # Validate the completion
            validation = validate_completion(
                test_case,
                completion_result['completion'],
                self.representation_spec,
            )
            
            result = {
                'test_case': test_case,
                'completion_result': completion_result,
                'validation': validation
            }
            
            results.append(result)
        
        return results
    
    def evaluate_dataset(self, dataset_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Evaluate model on a complete dataset
        
        Args:
            dataset_path: Path to evaluation dataset
            
        Returns:
            Tuple of (results, metrics)
        """
        # Load equations
        equations = load_evaluation_dataset(dataset_path)

        validation_spec = self.representation_spec
        if validation_spec is not None:
            validation_spec.validate_dataset(equations)
        else:
            # Legacy plain artifacts had no operand-width field. Infer only the
            # validation bound while retaining their explicit compatibility path.
            matches = [
                re.fullmatch(r'(\d+)([+-])(\d+)=(\d+)', equation)
                for equation in equations
            ]
            if any(match is None for match in matches):
                raise ValueError("legacy plain evaluation dataset has an invalid row")
            width = max(
                max(len(match.group(1)), len(match.group(3)))
                for match in matches
                if match is not None
            )
            validation_spec = RepresentationSpec.from_name('plain', width)
            validation_spec.validate_dataset(equations)

        # Sample source tasks first so every prompt mode remains balanced.
        if self.config.sample_size and self.config.sample_size < len(equations):
            equations = random.Random(self.config.sample_seed).sample(
                equations, self.config.sample_size
            )

        evaluated_tasks = validation_spec.validate_dataset(equations)
        self.evaluation_task_roster_sha256 = task_roster_sha256(evaluated_tasks)

        test_cases = create_test_cases(equations)
        if not test_cases:
            raise ValueError(f"Evaluation dataset contains no valid equations: {dataset_path}")
        
        # Run evaluation
        results = self.evaluate_test_cases(test_cases)
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        
        return results, metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_path': str(self.loaded_model_path),
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vocab_size': self.tokenizer.vocab_size if self.tokenizer else 0,
            'max_length': self.tokenizer.max_length if self.tokenizer else 0,
            'config': {
                'max_tokens': self.config.max_tokens,
                'device': self.config.device,
                'sample_size': self.config.sample_size,
                'sample_seed': self.config.sample_seed,
            }
        }
