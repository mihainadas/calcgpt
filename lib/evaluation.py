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

from .inference import (
    _load_artifact_tokenizer,
    _resolve_model_artifact,
    get_device,
    validate_tokenizer_compatibility,
)


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


def validate_completion(test_case: Dict[str, str], completion: str) -> Dict[str, Any]:
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
    
    # Check if it's a valid arithmetic expression format
    arithmetic_pattern = r'^\d+[\+\-]\d+=\d+$'
    result['valid_format'] = bool(re.match(arithmetic_pattern, completion))
    
    expected_match = re.fullmatch(r'(\d+)([+-])(\d+)=(\d+)', expected.strip())
    completion_match = re.fullmatch(r'(\d+)([+-])(\d+)=(\d+)', completion.strip())
    result['complete_expression'] = completion_match is not None

    if expected_match and completion_match:
        expected_left, expected_op, expected_right, expected_answer_text = (
            expected_match.groups()
        )
        actual_left, actual_op, actual_right, actual_answer_text = (
            completion_match.groups()
        )
        expected_value = (
            int(expected_left) + int(expected_right)
            if expected_op == '+'
            else int(expected_left) - int(expected_right)
        )
        normal_answer = str(expected_value)
        reversed_answer = normal_answer.zfill(len(expected_answer_text))[::-1]
        answer_order = (
            'reversed'
            if expected_answer_text == reversed_answer
            and expected_answer_text != normal_answer
            else 'normal'
        )
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
    """Calculate comprehensive evaluation metrics"""
    total = len(results)
    if total == 0:
        return {}
    
    metrics = {
        'total_tests': total,
        'successful_completions': sum(1 for r in results if r['completion_result']['success']),
        'valid_format': sum(1 for r in results if r['validation']['valid_format']),
        'correct_arithmetic': sum(1 for r in results if r['validation']['correct_arithmetic']),
        'complete_expressions': sum(1 for r in results if r['validation']['complete_expression']),
        'exact_matches': sum(1 for r in results if r['validation']['exact_match']),
        'contains_input': sum(1 for r in results if r['validation']['contains_input']),
    }
    
    # Calculate percentages
    for key in ['successful_completions', 'valid_format', 'correct_arithmetic', 
                'complete_expressions', 'exact_matches', 'contains_input']:
        metrics[f'{key}_pct'] = (metrics[key] / total) * 100
    
    # Calculate by test type
    by_type = defaultdict(lambda: defaultdict(int))
    for result in results:
        test_type = result['test_case']['type']
        by_type[test_type]['total'] += 1
        if result['validation']['correct_arithmetic']:
            by_type[test_type]['correct'] += 1
        if result['validation']['valid_format']:
            by_type[test_type]['valid_format'] += 1
    
    metrics['by_type'] = dict(by_type)
    
    # Calculate timing statistics
    times = [r['completion_result']['inference_time'] for r in results if r['completion_result']['success']]
    if times:
        try:
            import statistics
            metrics['timing'] = {
                'mean_ms': statistics.mean(times) * 1000,
                'median_ms': statistics.median(times) * 1000,
                'min_ms': min(times) * 1000,
                'max_ms': max(times) * 1000,
            }
            if len(times) > 1:
                metrics['timing']['std_ms'] = statistics.stdev(times) * 1000
        except ImportError:
            metrics['timing'] = {
                'mean_ms': sum(times) / len(times) * 1000,
                'min_ms': min(times) * 1000,
                'max_ms': max(times) * 1000
            }
    
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
        self.loaded_model_path = self.model_path
        
        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()
        
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
            context_length = getattr(self.model.config, 'n_positions', None)
            if context_length is None:
                context_length = getattr(self.model.config, 'max_position_embeddings', None)
            requested_length = len(input_tokens) + self.config.max_tokens
            if context_length is not None and requested_length > context_length:
                raise ValueError(
                    f"Input plus max_tokens requires {requested_length} positions, "
                    f"but model context length is {context_length}"
                )
            input_ids = torch.tensor([input_tokens], dtype=torch.long).to(self.device)
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=False,  # Greedy for consistent evaluation
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bad_words_ids=[[self.tokenizer.pad_token_id]],  # Prevent padding
                    num_return_sequences=1
                )
            
            result_tokens = generated[0].tolist()
            completion = self.tokenizer.decode(result_tokens)
            
            # Calculate timing
            inference_time = time.time() - start_time
            
            return {
                'input': partial_expr,
                'completion': completion,
                'inference_time': inference_time,
                'input_tokens': input_tokens,
                'output_tokens': result_tokens,
                'success': True
            }
            
        except Exception as e:
            return {
                'input': partial_expr,
                'completion': '',
                'inference_time': time.time() - start_time,
                'error': str(e),
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
            validation = validate_completion(test_case, completion_result['completion'])
            
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

        # Sample source tasks first so every prompt mode remains balanced.
        if self.config.sample_size and self.config.sample_size < len(equations):
            equations = random.Random(self.config.sample_seed).sample(
                equations, self.config.sample_size
            )

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
