#!/usr/bin/env python3
"""
CalcGPT Evaluation Tool - Comprehensive model assessment CLI

A powerful command-line interface for evaluating CalcGPT models on arithmetic tasks.
Provides detailed accuracy metrics, completion analysis, and performance benchmarks.

Author: Mihai NADAS
Version: 1.0.0
"""

import argparse
import os
import sys
import json
import time
import torch
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from transformers import GPT2LMHeadModel, GPT2Config
from datetime import datetime
from collections import defaultdict
import math

# ANSI color codes for beautiful output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class CalcGPTEvaluator:
    """CalcGPT model evaluator"""
    
    def __init__(self, model_path: str, device: str = 'auto', verbose: bool = False):
        self.model_path = Path(model_path)
        self.device = self._get_device(device)
        self.verbose = verbose
        self.model = None
        self.vocab = None
        self.id2char = None
        self.maxlen = None
        
        # Load model and vocabulary
        self._load_model()
        self._load_vocabulary()
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for inference"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def _load_model(self):
        """Load the trained model"""
        if self.verbose:
            print(f"{Colors.CYAN}Loading model from: {self.model_path}{Colors.ENDC}")
        
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
                        if self.verbose:
                            print(f"{Colors.WARNING}Using checkpoint: {latest_checkpoint}{Colors.ENDC}")
                        self.model = GPT2LMHeadModel.from_pretrained(str(latest_checkpoint))
                    else:
                        raise FileNotFoundError("No model files found in directory")
            else:
                raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
                
            self.model.to(self.device)
            self.model.eval()
            
            if self.verbose:
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"{Colors.GREEN}✅ Model loaded successfully!")
                print(f"   Parameters: {total_params:,}")
                print(f"   Device: {self.device}{Colors.ENDC}")
                
        except Exception as e:
            print(f"{Colors.FAIL}❌ Error loading model: {e}{Colors.ENDC}")
            sys.exit(1)
    
    def _load_vocabulary(self):
        """Load vocabulary from dataset"""
        dataset_path = Path('datasets/ds-calcgpt.txt')
        
        if not dataset_path.exists():
            print(f"{Colors.FAIL}❌ Dataset file not found: {dataset_path}")
            print(f"   Please ensure the dataset file exists for vocabulary loading.{Colors.ENDC}")
            sys.exit(1)
            
        try:
            with open(dataset_path) as f:
                examples = [line.strip() for line in f if line.strip()]
            
            # Recreate the same vocabulary used during training
            special_tokens = ['<pad>', '<eos>']
            chars = sorted(set(''.join(examples)))
            self.vocab = {c: i for i, c in enumerate(special_tokens + chars)}
            self.id2char = {i: c for c, i in self.vocab.items()}
            self.maxlen = max(len(self.encode(x)) for x in examples)
            
            if self.verbose:
                print(f"{Colors.GREEN}✅ Vocabulary loaded:")
                print(f"   Vocab size: {len(self.vocab)}")
                print(f"   Max length: {self.maxlen}")
                print(f"   Vocabulary: {self.vocab}{Colors.ENDC}")
                
        except Exception as e:
            print(f"{Colors.FAIL}❌ Error loading vocabulary: {e}{Colors.ENDC}")
            sys.exit(1)
    
    def encode(self, s: str) -> List[int]:
        """Encode string to token IDs"""
        return [self.vocab[c] for c in s if c in self.vocab] + [self.vocab['<eos>']]
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to string"""
        return ''.join([self.id2char[i] for i in ids if i != self.vocab['<pad>'] and i != self.vocab['<eos>']])
    
    def complete_expression(self, partial_expr: str, max_tokens: int = 15) -> Dict[str, Any]:
        """Complete a partial arithmetic expression"""
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
                    max_length=len(input_tokens) + max_tokens,
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

def load_evaluation_dataset(dataset_path: str) -> List[str]:
    """Load the evaluation dataset"""
    try:
        with open(dataset_path, 'r') as f:
            equations = [line.strip() for line in f if line.strip()]
        return equations
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error loading dataset: {e}{Colors.ENDC}")
        sys.exit(1)

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

def print_banner():
    """Print the evaluation tool banner"""
    banner = f"""
{Colors.BLUE}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════╗
║                        CalcGPT Eval                          ║
║                   Model Evaluation Tool                      ║
║                         v1.0.0                               ║
╚═══════════════════════════════════════════════════════════════╝
{Colors.ENDC}"""
    print(banner)

def run_evaluation(evaluator: CalcGPTEvaluator, test_cases: List[Dict[str, str]], 
                  max_tokens: int, verbose: bool) -> List[Dict[str, Any]]:
    """Run the evaluation on all test cases"""
    results = []
    total = len(test_cases)
    
    print(f"\n{Colors.GREEN}🧪 Running evaluation on {total} test cases{Colors.ENDC}")
    
    for i, test_case in enumerate(test_cases, 1):
        if not verbose:
            print(f"\r{Colors.CYAN}Progress: {i}/{total} ({i/total*100:.1f}%){Colors.ENDC}", end='', flush=True)
        
        # Get model completion
        completion_result = evaluator.complete_expression(test_case['input'], max_tokens)
        
        # Validate the completion
        validation = validate_completion(test_case, completion_result['completion'])
        
        result = {
            'test_case': test_case,
            'completion_result': completion_result,
            'validation': validation
        }
        
        results.append(result)
        
        if verbose:
            status = "✅" if validation['correct_arithmetic'] else "❌"
            print(f"{status} '{test_case['input']}' → '{completion_result['completion']}' [{test_case['type']}]")
    
    if not verbose:
        print()  # New line after progress
    
    return results

def print_results(metrics: Dict[str, Any], verbose: bool):
    """Print evaluation results in a beautiful format"""
    print(f"\n{Colors.BOLD}📊 EVALUATION RESULTS{Colors.ENDC}")
    print("=" * 60)
    
    # Overall metrics
    print(f"\n{Colors.CYAN}Overall Performance:{Colors.ENDC}")
    print(f"  Total test cases: {metrics['total_tests']}")
    print(f"  Successful completions: {metrics['successful_completions']} ({metrics['successful_completions_pct']:.1f}%)")
    print(f"  Valid format: {metrics['valid_format']} ({metrics['valid_format_pct']:.1f}%)")
    print(f"  Correct arithmetic: {Colors.GREEN}{metrics['correct_arithmetic']}{Colors.ENDC} ({Colors.GREEN}{metrics['correct_arithmetic_pct']:.1f}%{Colors.ENDC})")
    print(f"  Complete expressions: {metrics['complete_expressions']} ({metrics['complete_expressions_pct']:.1f}%)")
    print(f"  Exact matches: {metrics['exact_matches']} ({metrics['exact_matches_pct']:.1f}%)")
    
    # Performance by test type
    if 'by_type' in metrics:
        print(f"\n{Colors.CYAN}Performance by Test Type:{Colors.ENDC}")
        for test_type, stats in metrics['by_type'].items():
            accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            format_acc = (stats['valid_format'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"  {test_type.replace('_', ' ').title()}:")
            print(f"    Arithmetic accuracy: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
            print(f"    Format accuracy: {stats['valid_format']}/{stats['total']} ({format_acc:.1f}%)")
    
    # Timing statistics
    if 'timing' in metrics:
        timing = metrics['timing']
        print(f"\n{Colors.CYAN}Performance Timing:{Colors.ENDC}")
        print(f"  Mean: {timing['mean_ms']:.1f}ms")
        if 'median_ms' in timing:
            print(f"  Median: {timing['median_ms']:.1f}ms")
        print(f"  Range: {timing['min_ms']:.1f}ms - {timing['max_ms']:.1f}ms")
        if 'std_ms' in timing:
            print(f"  Std Dev: {timing['std_ms']:.1f}ms")

def save_results(results: List[Dict[str, Any]], metrics: Dict[str, Any], 
                output_file: str, model_path: str):
    """Save detailed results to JSON file"""
    output_data = {
        'metadata': {
            'model_path': model_path,
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_test_cases': len(results),
            'evaluator_version': '1.0.0'
        },
        'metrics': metrics,
        'detailed_results': results
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n{Colors.GREEN}📄 Detailed results saved to: {output_file}{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error saving results: {e}{Colors.ENDC}")

def find_latest_model(models_dir: str = "models") -> Optional[str]:
    """Find the latest trained model in the models directory"""
    models_path = Path(models_dir)
    
    if not models_path.exists():
        return None
    
    # Find all calcgpt model directories
    model_dirs = [d for d in models_path.iterdir() 
                  if d.is_dir() and d.name.startswith('calcgpt')]
    
    if not model_dirs:
        return None
    
    # Sort by modification time (most recent first)
    model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Return the most recently modified model
    return str(model_dirs[0])

def get_model_path(specified_path: Optional[str]) -> str:
    """Get the model path, using auto-detection if not specified"""
    if specified_path and specified_path != 'auto':
        return specified_path
    
    # Try to find latest model in models directory
    latest_model = find_latest_model()
    if latest_model:
        return latest_model
    
    # Fallback to legacy out directory
    if Path('./out').exists():
        return './out'
    
    # If nothing found, show helpful error
    raise FileNotFoundError(
        "No trained models found. Please:\n"
        "  1. Train a model using: python calcgpt_train.py\n"
        "  2. Or specify a model path with: -m /path/to/model"
    )

def main():
    parser = argparse.ArgumentParser(
        description="CalcGPT Evaluation Tool - Comprehensive model assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Evaluate default model on default dataset
  %(prog)s -m ./custom_model                  # Evaluate custom model
  %(prog)s -d datasets/custom.txt             # Use custom dataset
  %(prog)s --max-tokens 20 --verbose          # Detailed output with longer generations
  %(prog)s -o evaluation_results.json         # Save detailed results to JSON
  %(prog)s --sample 100                       # Evaluate on random sample of 100 cases
        """
    )
    
    # Model options
    parser.add_argument('-m', '--model', default='auto',
                       help='Path to model directory (default: auto-detect latest model)')
    parser.add_argument('--device', default='auto', 
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use for inference (default: auto)')
    
    # Dataset options
    parser.add_argument('-d', '--dataset', default='datasets/ds-calcgpt.txt',
                       help='Path to evaluation dataset (default: datasets/ds-calcgpt.txt)')
    parser.add_argument('--sample', type=int,
                       help='Evaluate on random sample of N test cases')
    
    # Generation parameters
    parser.add_argument('--max-tokens', type=int, default=15,
                       help='Maximum tokens to generate (default: 15)')
    
    # Output options
    parser.add_argument('-o', '--output', type=str,
                       help='Save detailed results to JSON file')
    parser.add_argument('--summary-only', action='store_true',
                       help='Show only summary metrics (no detailed output)')
    
    # Utility options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output with individual test results')
    parser.add_argument('--version', action='version', version='CalcGPT Eval 1.0.0')
    parser.add_argument('--no-banner', action='store_true',
                       help='Suppress banner output')
    
    args = parser.parse_args()
    
    # Print banner unless suppressed
    if not args.no_banner:
        print_banner()
    
    # Get model path
    try:
        model_path = get_model_path(args.model)
        if args.verbose or args.model == 'auto':
            if args.model == 'auto':
                print(f"{Colors.GREEN}🎯 Auto-detected model: {Colors.CYAN}{Path(model_path).name}{Colors.ENDC}")
            else:
                print(f"{Colors.CYAN}📁 Using model: {model_path}{Colors.ENDC}")
    except FileNotFoundError as e:
        print(f"{Colors.FAIL}❌ {e}{Colors.ENDC}")
        sys.exit(1)
    
    # Initialize evaluator
    if args.verbose:
        print(f"{Colors.CYAN}Initializing CalcGPT evaluator...{Colors.ENDC}")
    
    evaluator = CalcGPTEvaluator(model_path, args.device, args.verbose)
    
    # Load evaluation dataset
    if args.verbose:
        print(f"{Colors.CYAN}Loading evaluation dataset: {args.dataset}{Colors.ENDC}")
    
    equations = load_evaluation_dataset(args.dataset)
    print(f"{Colors.GREEN}✅ Loaded {len(equations)} equations from dataset{Colors.ENDC}")
    
    # Create test cases
    test_cases = create_test_cases(equations)
    print(f"{Colors.GREEN}✅ Generated {len(test_cases)} test cases{Colors.ENDC}")
    
    # Sample if requested
    if args.sample and args.sample < len(test_cases):
        import random
        test_cases = random.sample(test_cases, args.sample)
        print(f"{Colors.WARNING}📝 Using random sample of {len(test_cases)} test cases{Colors.ENDC}")
    
    # Run evaluation
    results = run_evaluation(evaluator, test_cases, args.max_tokens, args.verbose)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print results
    print_results(metrics, args.verbose and not args.summary_only)
    
    # Save detailed results if requested
    if args.output:
        save_results(results, metrics, args.output, str(evaluator.model_path))
    
    # Exit with appropriate code
    if metrics.get('correct_arithmetic_pct', 0) < 50:
        print(f"\n{Colors.WARNING}⚠️ Low accuracy detected - consider additional training{Colors.ENDC}")
        sys.exit(1)
    else:
        print(f"\n{Colors.GREEN}🎉 Evaluation completed successfully{Colors.ENDC}")
        sys.exit(0)

if __name__ == "__main__":
    main() 