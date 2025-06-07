"""
CalcGPT Evaluation Library

Core evaluation functionality for CalcGPT models with proper separation of concerns.
"""

import time
import re
import torch
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
from transformers import GPT2LMHeadModel

from .inference import load_vocabulary_from_dataset, get_device


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters"""
    max_tokens: int = 15
    device: str = 'auto'
    sample_size: Optional[int] = None
    verbose: bool = False


def load_evaluation_dataset(dataset_path: str) -> List[str]:
    """Load the evaluation dataset"""
    try:
        with open(dataset_path, 'r') as f:
            equations = [line.strip() for line in f if line.strip()]
        return equations
    except Exception as e:
        raise FileNotFoundError(f"Error loading dataset: {e}")


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
    """Validate if a completion is correct"""
    input_text = test_case['input']
    expected = test_case['expected']
    test_type = test_case['type']
    
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
    
    # Check if it's a complete expression (has = and result)
    result['complete_expression'] = '=' in completion and len(completion.split('=')) == 2
    
    if result['complete_expression']:
        parts = completion.split('=')
        if len(parts) == 2:
            try:
                left_side = parts[0].strip()
                right_side = parts[1].strip()
                
                # Evaluate the left side
                if '+' in left_side:
                    operands = left_side.split('+')
                    if len(operands) == 2:
                        expected_result = int(operands[0]) + int(operands[1])
                        actual_result = int(right_side)
                        result['correct_arithmetic'] = expected_result == actual_result
                        result['details']['expected_result'] = expected_result
                        result['details']['actual_result'] = actual_result
                        
                elif '-' in left_side:
                    operands = left_side.split('-')
                    if len(operands) == 2:
                        expected_result = int(operands[0]) - int(operands[1])
                        actual_result = int(right_side)
                        result['correct_arithmetic'] = expected_result == actual_result
                        result['details']['expected_result'] = expected_result
                        result['details']['actual_result'] = actual_result
                        
            except (ValueError, IndexError):
                result['correct_arithmetic'] = False
    
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
    
    def __init__(self, model_path: str, config: EvaluationConfig = None, verbose: bool = False):
        """Initialize CalcGPT evaluator
        
        Args:
            model_path: Path to trained model
            config: Evaluation configuration
            verbose: Enable verbose output
        """
        self.model_path = Path(model_path)
        self.config = config or EvaluationConfig()
        self.verbose = verbose
        
        # Initialize device
        self.device = get_device(self.config.device)
        
        # Initialize model and vocabulary
        self.model = None
        self.vocab = None
        self.id2char = None
        self.maxlen = None
        
        # Load model and vocabulary
        self._load_model()
        self._load_vocabulary()
        
    def log(self, message: str):
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def _load_model(self):
        """Load the trained model"""
        self.log(f"Loading model from: {self.model_path}")
        
        try:
            if self.model_path.exists():
                # Check for model files
                model_files = list(self.model_path.glob("*.bin")) + list(self.model_path.glob("*.safetensors"))
                
                if model_files:
                    self.model = GPT2LMHeadModel.from_pretrained(str(self.model_path))
                else:
                    # Try to find checkpoint directories
                    checkpoints = [d for d in self.model_path.iterdir() if d.is_dir() and d.name.startswith('checkpoint')]
                    if checkpoints:
                        # Use the latest checkpoint
                        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
                        self.log(f"Using checkpoint: {latest_checkpoint}")
                        self.model = GPT2LMHeadModel.from_pretrained(str(latest_checkpoint))
                    else:
                        raise FileNotFoundError("No model files found in directory")
            else:
                raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
                
            self.model.to(self.device)
            self.model.eval()
            
            if self.verbose:
                total_params = sum(p.numel() for p in self.model.parameters())
                self.log(f"✅ Model loaded successfully!")
                self.log(f"   Parameters: {total_params:,}")
                self.log(f"   Device: {self.device}")
                
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    def _load_vocabulary(self):
        """Load vocabulary from dataset"""
        try:
            self.vocab, self.id2char, self.maxlen = load_vocabulary_from_dataset()
            
            if self.verbose:
                self.log(f"✅ Vocabulary loaded:")
                self.log(f"   Vocab size: {len(self.vocab)}")
                self.log(f"   Max length: {self.maxlen}")
                
        except Exception as e:
            raise RuntimeError(f"Error loading vocabulary: {e}")
    
    def encode(self, s: str) -> List[int]:
        """Encode string to token IDs"""
        return [self.vocab[c] for c in s if c in self.vocab] + [self.vocab['<eos>']]
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to string"""
        return ''.join([self.id2char[i] for i in ids if i != self.vocab['<pad>'] and i != self.vocab['<eos>']])
    
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
        
        # Encode input (remove EOS for generation)
        input_tokens = self.encode(partial_expr)[:-1]
        input_ids = torch.tensor([input_tokens], dtype=torch.long).to(self.device)
        
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_length=len(input_tokens) + self.config.max_tokens,
                    do_sample=False,  # Greedy for consistent evaluation
                    pad_token_id=self.vocab['<pad>'],
                    eos_token_id=self.vocab['<eos>'],
                    bad_words_ids=[[self.vocab['<pad>']]],  # Prevent padding
                    num_return_sequences=1
                )
            
            result_tokens = generated[0].tolist()
            completion = self.decode(result_tokens)
            
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
        
        # Create test cases
        test_cases = create_test_cases(equations)
        
        # Sample if requested
        if self.config.sample_size and self.config.sample_size < len(test_cases):
            import random
            test_cases = random.sample(test_cases, self.config.sample_size)
        
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
            'model_path': str(self.model_path),
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vocab_size': len(self.vocab) if self.vocab else 0,
            'max_length': self.maxlen,
            'config': {
                'max_tokens': self.config.max_tokens,
                'device': self.config.device,
                'sample_size': self.config.sample_size
            }
        } 