#!/usr/bin/env python3
"""
CalcGPT - CLI Interface

A command-line interface for running inference with CalcGPT models.
Supports interactive mode, batch processing, and various output formats.

Author: Mihai NADAS
Version: 2.0.0
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import readline  # For better input experience
from datetime import datetime

from lib.inference import CalcGPT, InferenceConfig, get_model_path

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
    """Print the CalcGPT banner"""
    banner = f"""
{Colors.BLUE}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════╗
║                            CalcGPT                            ║
║                   Arithmetic Language Model                   ║
║                         CLI Tool v2.0.0                       ║
╚═══════════════════════════════════════════════════════════════╝
{Colors.ENDC}"""
    print(banner)

def create_config_from_args(args) -> InferenceConfig:
    """Create InferenceConfig from command line arguments"""
    return InferenceConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        device=args.device,
        show_tokens=args.show_tokens
    )

def interactive_mode(calcgpt: CalcGPT):
    """Run CalcGPT in interactive mode"""
    print(f"\n{Colors.GREEN}🚀 Interactive Mode Started{Colors.ENDC}")
    print(f"{Colors.CYAN}Enter arithmetic problems (e.g., '1+1', '15+27', '100+200'){Colors.ENDC}")
    print(f"{Colors.WARNING}Type 'quit', 'exit', or Ctrl+C to exit{Colors.ENDC}\n")
    
    try:
        while True:
            try:
                # Get user input with colored prompt
                problem = input(f"{Colors.BOLD}CalcGPT> {Colors.ENDC}").strip()
                
                if problem.lower() in ['quit', 'exit', 'q']:
                    print(f"\n{Colors.GREEN}👋 Goodbye!{Colors.ENDC}")
                    break
                
                if not problem:
                    continue
                
                # Solve the problem
                result = calcgpt.solve(problem)
                
                if 'error' in result:
                    print(f"{Colors.FAIL}❌ Error: {result['error']}{Colors.ENDC}")
                else:
                    # Pretty print result
                    status_icon = "✅" if result.get('is_correct') else "💡"
                    print(f"{Colors.GREEN}{status_icon} {result['problem']} {Colors.BOLD}{result['answer']}{Colors.ENDC}")
                    
                    if calcgpt.config.show_tokens:
                        print(f"{Colors.CYAN}   Input tokens: {result['input_tokens']}")
                        print(f"   Output tokens: {result['output_tokens']}{Colors.ENDC}")
                    
                    print(f"{Colors.CYAN}   ⏱️ {result['inference_time']*1000:.1f}ms{Colors.ENDC}")
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print(f"\n\n{Colors.GREEN}👋 Goodbye!{Colors.ENDC}")
                break
            except EOFError:
                print(f"\n\n{Colors.GREEN}👋 Goodbye!{Colors.ENDC}")
                break
                
    except Exception as e:
        print(f"{Colors.FAIL}❌ Unexpected error: {e}{Colors.ENDC}")

def batch_mode(calcgpt: CalcGPT, problems: List[str], output_format: str, output_file: Optional[str]):
    """Run CalcGPT in batch mode"""
    print(f"\n{Colors.GREEN}📊 Batch Mode - Processing {len(problems)} problems{Colors.ENDC}")
    
    results = calcgpt.solve_batch(problems)
    
    # Calculate statistics
    correct = sum(1 for r in results if r.get('is_correct', False))
    total = len(results)
    errors = sum(1 for r in results if 'error' in r)
    
    print(f"\r{Colors.CYAN}Completed: {total}/{total}{Colors.ENDC}")
    
    # Output results
    if output_format == 'json':
        output_data = {
            'metadata': {
                'total_problems': total,
                'correct_answers': correct,
                'errors': errors,
                'accuracy': correct / total if total > 0 else 0,
                'model_info': calcgpt.get_model_info(),
                'timestamp': datetime.now().isoformat()
            },
            'results': results
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"{Colors.GREEN}📄 Results saved to: {output_file}{Colors.ENDC}")
        else:
            print(json.dumps(output_data, indent=2))
    
    else:  # table format
        print(f"\n{Colors.BOLD}Results Summary:{Colors.ENDC}")
        print(f"{'Problem':<15} {'Answer':<10} {'Time (ms)':<10} {'Status':<10}")
        print("-" * 50)
        
        for result in results:
            if 'error' in result:
                status = "❌ ERROR"
                answer = "N/A"
            elif result.get('is_correct'):
                status = "✅ CORRECT"
                answer = result['answer']
            else:
                status = "❓ UNKNOWN"
                answer = result['answer']
            
            time_ms = result['inference_time'] * 1000
            print(f"{result['problem']:<15} {answer:<10} {time_ms:<10.1f} {status:<10}")
        
        print(f"\n{Colors.GREEN}📊 Statistics:{Colors.ENDC}")
        print(f"  Correct: {correct}/{total} ({correct/total*100:.1f}%)")
        print(f"  Errors: {errors}/{total} ({errors/total*100:.1f}%)")
        
        if output_file:
            # Save table format to file
            with open(output_file, 'w') as f:
                f.write("CalcGPT Batch Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"{'Problem':<15} {'Answer':<10} {'Time (ms)':<10} {'Status':<10}\n")
                f.write("-" * 50 + "\n")
                
                for result in results:
                    if 'error' in result:
                        status = "ERROR"
                        answer = "N/A"
                    elif result.get('is_correct'):
                        status = "CORRECT"
                        answer = result['answer']
                    else:
                        status = "UNKNOWN"
                        answer = result['answer']
                    
                    time_ms = result['inference_time'] * 1000
                    f.write(f"{result['problem']:<15} {answer:<10} {time_ms:<10.1f} {status:<10}\n")
                
                f.write(f"\nStatistics:\n")
                f.write(f"  Correct: {correct}/{total} ({correct/total*100:.1f}%)\n")
                f.write(f"  Errors: {errors}/{total} ({errors/total*100:.1f}%)\n")
            
            print(f"{Colors.GREEN}📄 Results saved to: {output_file}{Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(
        description="CalcGPT - Arithmetic language model CLI tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode with default model
  %(prog)s -m ./custom_model                  # Use custom model path
  %(prog)s -b "1+1" "2+3" "10+5"             # Batch mode with problems
  %(prog)s -f problems.txt                    # Batch mode from file
  %(prog)s -i --temperature 0.2 --verbose    # Interactive with custom settings
  %(prog)s -b "1+1" "2+3" -o results.json    # Save results to JSON
        """
    )
    
    # Model options
    parser.add_argument('-m', '--model', default='auto',
                       help='Path to model directory (default: auto-detect latest model)')
    parser.add_argument('-d', '--device', default='auto', 
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use for inference (default: auto)')
    
    # Mode options
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Run in interactive mode (default)')
    parser.add_argument('-b', '--batch', nargs='*',
                       help='Run in batch mode with specified problems')
    parser.add_argument('-f', '--file', type=str,
                       help='Read problems from file (one per line)')
    
    # Generation parameters
    parser.add_argument('-t', '--temperature', type=float, default=0.1,
                       help='Sampling temperature (default: 0.1, use 0 for greedy)')
    parser.add_argument('--max-tokens', type=int, default=10,
                       help='Maximum tokens to generate (default: 10)')
    
    # Output options
    parser.add_argument('-o', '--output', type=str,
                       help='Output file for batch results')
    parser.add_argument('--format', choices=['table', 'json'], default='table',
                       help='Output format for batch mode (default: table)')
    parser.add_argument('--show-tokens', action='store_true',
                       help='Show token details in output')
    
    # Utility options
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    parser.add_argument('--version', action='version', version='CalcGPT 2.0.0')
    
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
    
    # Initialize CalcGPT
    if not args.quiet:
        print(f"{Colors.CYAN}Initializing CalcGPT...{Colors.ENDC}")
    
    try:
        calcgpt = CalcGPT(model_path, config, verbose=not args.quiet)
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error initializing CalcGPT: {e}{Colors.ENDC}")
        return 1
    
    # Determine mode and execute
    try:
        if args.batch is not None:
            # Batch mode with command line problems
            batch_mode(calcgpt, args.batch, args.format, args.output)
        elif args.file:
            # Batch mode with file input
            try:
                with open(args.file, 'r') as f:
                    problems = [line.strip() for line in f if line.strip()]
                batch_mode(calcgpt, problems, args.format, args.output)
            except Exception as e:
                print(f"{Colors.FAIL}❌ Error reading file: {e}{Colors.ENDC}")
                return 1
        else:
            # Interactive mode (default)
            interactive_mode(calcgpt)
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠️ Interrupted by user{Colors.ENDC}")
        return 1
    except Exception as e:
        print(f"{Colors.FAIL}❌ Unexpected error: {e}{Colors.ENDC}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 