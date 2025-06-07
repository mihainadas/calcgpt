#!/usr/bin/env python3
"""
CalcGPT - Arithmetic language model CLI tool

A powerful command-line interface for running inference with CalcGPT models.
Supports interactive mode, batch processing, and various output formats.

Author: Mihai NADAS
Version: 1.0.0
"""

import argparse
import os
import sys
import json
import time
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from transformers import GPT2LMHeadModel, GPT2Config
import readline  # For better input experience
from datetime import datetime

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

class CalcGPT:
    """CalcGPT inference engine"""
    
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
            # Try loading from the specified path
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
        # This needs to match the training vocabulary
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
    
    def solve(self, problem: str, temperature: float = 0.1, max_tokens: int = 10, 
              show_tokens: bool = False) -> Dict[str, any]:
        """Solve an arithmetic problem"""
        start_time = time.time()
        
        # Clean and validate input
        problem = problem.strip()
        if not problem.endswith('='):
            problem += '='
        
        # Encode input (remove EOS for generation)
        input_tokens = self.encode(problem)[:-1]
        input_ids = torch.tensor([input_tokens], dtype=torch.long).to(self.device)
        
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_length=len(input_tokens) + max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 1.0,
                    pad_token_id=self.vocab['<pad>'],
                    eos_token_id=self.vocab['<eos>'],
                    bad_words_ids=[[self.vocab['<pad>']]],  # Prevent padding
                    num_return_sequences=1
                )
            
            result_tokens = generated[0].tolist()
            new_tokens = result_tokens[len(input_tokens):]
            
            # Decode results
            full_result = self.decode(result_tokens)
            answer_part = self.decode(new_tokens) if new_tokens else ""
            
            # Calculate timing
            inference_time = time.time() - start_time
            
            # Extract just the numerical answer
            if '=' in full_result:
                parts = full_result.split('=')
                if len(parts) >= 2:
                    numerical_answer = parts[1].strip()
                else:
                    numerical_answer = answer_part
            else:
                numerical_answer = answer_part
            
            return {
                'problem': problem,
                'full_result': full_result,
                'answer': numerical_answer,
                'input_tokens': input_tokens if show_tokens else None,
                'output_tokens': new_tokens if show_tokens else None,
                'inference_time': inference_time,
                'model_path': str(self.model_path),
                'device': str(self.device)
            }
            
        except Exception as e:
            return {
                'problem': problem,
                'error': str(e),
                'inference_time': time.time() - start_time
            }

def print_banner():
    """Print the CalcGPT banner"""
    banner = f"""
{Colors.BLUE}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════╗
║                            CalcGPT                            ║
║                   Arithmetic Language Model                   ║
║                         CLI Tool v1.0.0                       ║
╚═══════════════════════════════════════════════════════════════╝
{Colors.ENDC}"""
    print(banner)

def interactive_mode(calcgpt: CalcGPT, temperature: float, max_tokens: int, show_tokens: bool):
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
                result = calcgpt.solve(problem, temperature, max_tokens, show_tokens)
                
                if 'error' in result:
                    print(f"{Colors.FAIL}❌ Error: {result['error']}{Colors.ENDC}")
                else:
                    # Pretty print result
                    print(f"{Colors.GREEN}💡 {result['problem']} {Colors.BOLD}{result['answer']}{Colors.ENDC}")
                    
                    if show_tokens:
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

def batch_mode(calcgpt: CalcGPT, problems: List[str], temperature: float, 
               max_tokens: int, show_tokens: bool, output_format: str, output_file: Optional[str]):
    """Run CalcGPT in batch mode"""
    print(f"\n{Colors.GREEN}📊 Batch Mode - Processing {len(problems)} problems{Colors.ENDC}")
    
    results = []
    correct = 0
    total = len(problems)
    
    for i, problem in enumerate(problems, 1):
        print(f"\r{Colors.CYAN}Processing: {i}/{total}{Colors.ENDC}", end='', flush=True)
        
        result = calcgpt.solve(problem, temperature, max_tokens, show_tokens)
        results.append(result)
        
        # Simple correctness check for basic problems
        if not result.get('error') and validate_simple_arithmetic(problem, result['answer']):
            correct += 1
    
    print()  # New line after progress
    
    # Output results
    if output_format == 'json':
        output_data = {
            'metadata': {
                'total_problems': total,
                'correct_answers': correct,
                'accuracy': correct / total if total > 0 else 0,
                'model_path': str(calcgpt.model_path),
                'device': str(calcgpt.device),
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
            status = "✅" if not result.get('error') else "❌"
            time_ms = result['inference_time'] * 1000
            print(f"{result['problem']:<15} {result.get('answer', 'ERROR'):<10} {time_ms:<10.1f} {status:<10}")
        
        print(f"\n{Colors.GREEN}📊 Accuracy: {correct}/{total} ({correct/total*100:.1f}%){Colors.ENDC}")
        
        if output_file:
            # Save table format to file
            with open(output_file, 'w') as f:
                f.write("CalcGPT Batch Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"{'Problem':<15} {'Answer':<10} {'Time (ms)':<10} {'Status':<10}\n")
                f.write("-" * 50 + "\n")
                
                for result in results:
                    status = "OK" if not result.get('error') else "ERROR"
                    time_ms = result['inference_time'] * 1000
                    f.write(f"{result['problem']:<15} {result.get('answer', 'ERROR'):<10} {time_ms:<10.1f} {status:<10}\n")
                
                f.write(f"\nAccuracy: {correct}/{total} ({correct/total*100:.1f}%)\n")
            
            print(f"{Colors.GREEN}📄 Results saved to: {output_file}{Colors.ENDC}")

def validate_simple_arithmetic(problem: str, answer: str) -> bool:
    """Simple validation for basic arithmetic problems"""
    try:
        # Remove = and spaces
        clean_problem = problem.replace('=', '').strip()
        
        # Only handle simple + operations for now
        if '+' in clean_problem:
            parts = clean_problem.split('+')
            if len(parts) == 2:
                expected = int(parts[0].strip()) + int(parts[1].strip())
                return str(expected) == answer.strip()
    except:
        pass
    return False

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
        description="CalcGPT - State-of-the-art arithmetic language model CLI tool",
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
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--version', action='version', version='CalcGPT 1.0.0')
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
    
    # Initialize CalcGPT
    if args.verbose:
        print(f"{Colors.CYAN}Initializing CalcGPT...{Colors.ENDC}")
    
    calcgpt = CalcGPT(model_path, args.device, args.verbose)
    
    # Determine mode and execute
    if args.batch is not None:
        # Batch mode with command line problems
        batch_mode(calcgpt, args.batch, args.temperature, args.max_tokens, 
                  args.show_tokens, args.format, args.output)
    elif args.file:
        # Batch mode with file input
        try:
            with open(args.file, 'r') as f:
                problems = [line.strip() for line in f if line.strip()]
            batch_mode(calcgpt, problems, args.temperature, args.max_tokens, 
                      args.show_tokens, args.format, args.output)
        except Exception as e:
            print(f"{Colors.FAIL}❌ Error reading file: {e}{Colors.ENDC}")
            sys.exit(1)
    else:
        # Interactive mode (default)
        interactive_mode(calcgpt, args.temperature, args.max_tokens, args.show_tokens)

if __name__ == "__main__":
    main() 