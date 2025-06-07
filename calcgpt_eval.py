#!/usr/bin/env python3
"""
CalcGPT Evaluation Tool - CLI Interface

A command-line interface for evaluating CalcGPT models on arithmetic tasks.
Provides detailed accuracy metrics, completion analysis, and performance benchmarks.

Author: Mihai NADAS
Version: 2.0.0
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from lib.evaluation import CalcGPTEvaluator, EvaluationConfig
from lib.inference import get_model_path

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

def print_banner():
    """Print the evaluation tool banner"""
    banner = f"""
{Colors.BLUE}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════╗
║                        CalcGPT Eval                          ║
║                   Model Evaluation Tool                      ║
║                         v2.0.0                               ║
╚═══════════════════════════════════════════════════════════════╝
{Colors.ENDC}"""
    print(banner)

def create_config_from_args(args) -> EvaluationConfig:
    """Create EvaluationConfig from command line arguments"""
    return EvaluationConfig(
        max_tokens=args.max_tokens,
        device=args.device,
        sample_size=args.sample,
        verbose=args.verbose
    )

def run_evaluation_with_progress(evaluator: CalcGPTEvaluator, test_cases: List[Dict[str, str]], 
                                verbose: bool) -> List[Dict[str, Any]]:
    """Run the evaluation with progress display"""
    total = len(test_cases)
    print(f"\n{Colors.GREEN}🧪 Running evaluation on {total} test cases{Colors.ENDC}")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        if not verbose:
            print(f"\r{Colors.CYAN}Progress: {i}/{total} ({i/total*100:.1f}%){Colors.ENDC}", end='', flush=True)
        
        # Get model completion
        completion_result = evaluator.complete_expression(test_case['input'])
        
        # Validate the completion
        from lib.evaluation import validate_completion
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
            'evaluator_version': '2.0.0'
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
    parser.add_argument('--version', action='version', version='CalcGPT Eval 2.0.0')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress banner and verbose output')
    
    args = parser.parse_args()
    
    # Print banner unless suppressed
    if not args.quiet:
        print_banner()
    
    # Get model path
    try:
        model_path = get_model_path(args.model)
        if not args.quiet:
            if args.model == 'auto':
                print(f"{Colors.GREEN}🎯 Auto-detected model: {Colors.CYAN}{Path(model_path).name}{Colors.ENDC}")
            else:
                print(f"{Colors.CYAN}📁 Using model: {model_path}{Colors.ENDC}")
    except FileNotFoundError as e:
        print(f"{Colors.FAIL}❌ {e}{Colors.ENDC}")
        return 1
    
    # Create configuration
    config = create_config_from_args(args)
    config.verbose = args.verbose and not args.quiet
    
    # Initialize evaluator
    if not args.quiet:
        print(f"{Colors.CYAN}Initializing CalcGPT evaluator...{Colors.ENDC}")
    
    try:
        evaluator = CalcGPTEvaluator(model_path, config, verbose=not args.quiet)
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error initializing evaluator: {e}{Colors.ENDC}")
        return 1
    
    # Load evaluation dataset and run evaluation
    if not args.quiet:
        print(f"{Colors.CYAN}Loading evaluation dataset: {args.dataset}{Colors.ENDC}")
    
    try:
        # Use the evaluator's built-in dataset evaluation method
        results, metrics = evaluator.evaluate_dataset(args.dataset)
        
        if not args.quiet:
            equations_count = len(set(r['test_case']['expected'] for r in results))
            print(f"{Colors.GREEN}✅ Loaded {equations_count} equations from dataset{Colors.ENDC}")
            print(f"{Colors.GREEN}✅ Generated {metrics['total_tests']} test cases{Colors.ENDC}")
            
            if config.sample_size:
                print(f"{Colors.WARNING}📝 Using random sample of {metrics['total_tests']} test cases{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error during evaluation: {e}{Colors.ENDC}")
        return 1
    
    # Print results
    print_results(metrics, args.verbose and not args.summary_only and not args.quiet)
    
    # Save detailed results if requested
    if args.output:
        save_results(results, metrics, args.output, str(evaluator.model_path))
    
    # Exit with appropriate code
    if metrics.get('correct_arithmetic_pct', 0) < 50:
        if not args.quiet:
            print(f"\n{Colors.WARNING}⚠️ Low accuracy detected - consider additional training{Colors.ENDC}")
        return 1
    else:
        if not args.quiet:
            print(f"\n{Colors.GREEN}🎉 Evaluation completed successfully{Colors.ENDC}")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 