"""
CalcGPT Inference Library

Core inference functionality for CalcGPT models with proper separation of concerns.
"""

import time
import torch
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from transformers import GPT2LMHeadModel, GPT2Config

from .tokenizer import CalcGPTTokenizer


@dataclass
class InferenceConfig:
    """Configuration for inference parameters"""
    temperature: float = 0.1
    max_tokens: int = 10
    device: str = 'auto'
    show_tokens: bool = False


def get_device(device_spec: str = 'auto') -> torch.device:
    """Get the best available device"""
    if device_spec == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_spec)


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
    
    # If nothing found, raise error
    raise FileNotFoundError(
        "No trained models found. Please:\n"
        "  1. Train a model using: python calcgpt_train.py\n"
        "  2. Or specify a model path with: -m /path/to/model"
    )





def validate_simple_arithmetic(problem: str, answer: str) -> bool:
    """Simple validation for basic arithmetic problems"""
    try:
        # Remove = and spaces
        clean_problem = problem.replace('=', '').strip()
        
        # Handle simple + operations
        if '+' in clean_problem:
            parts = clean_problem.split('+')
            if len(parts) == 2:
                expected = int(parts[0].strip()) + int(parts[1].strip())
                return str(expected) == answer.strip()
        
        # Handle simple - operations
        elif '-' in clean_problem:
            parts = clean_problem.split('-')
            if len(parts) == 2:
                expected = int(parts[0].strip()) - int(parts[1].strip())
                return str(expected) == answer.strip()
    except:
        pass
    return False


class CalcGPT:
    """CalcGPT inference engine"""
    
    def __init__(self, model_path: str, config: InferenceConfig = None, verbose: bool = False):
        """Initialize CalcGPT inference engine
        
        Args:
            model_path: Path to trained model
            config: Inference configuration
            verbose: Enable verbose output
        """
        self.model_path = Path(model_path)
        self.config = config or InferenceConfig()
        self.verbose = verbose
        
        # Initialize device
        self.device = get_device(self.config.device)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        
        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()
        
    def log(self, message: str):
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(message)
        
    def _load_model(self):
        """Load the trained model"""
        self.log(f"Loading model from: {self.model_path}")
        
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
    
    def _load_tokenizer(self):
        """Load tokenizer from dataset"""
        try:
            self.tokenizer = CalcGPTTokenizer.from_dataset()
            
            if self.verbose:
                self.log(f"✅ Tokenizer loaded:")
                self.log(f"   Vocab size: {self.tokenizer.vocab_size}")
                self.log(f"   Max length: {self.tokenizer.max_length}")
                
        except Exception as e:
            raise RuntimeError(f"Error loading tokenizer: {e}")
    
    def solve(self, problem: str) -> Dict[str, Any]:
        """Solve an arithmetic problem
        
        Args:
            problem: Arithmetic problem string (e.g., "1+1" or "1+1=")
            
        Returns:
            Dictionary with solution results
        """
        start_time = time.time()
        
        # Clean and validate input
        problem = problem.strip()
        if not problem.endswith('='):
            problem += '='
        
        # Encode input (remove EOS for generation)
        input_tokens = self.tokenizer.encode(problem, add_eos=False)
        input_ids = torch.tensor([input_tokens], dtype=torch.long).to(self.device)
        
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_length=len(input_tokens) + self.config.max_tokens,
                    do_sample=self.config.temperature > 0,
                    temperature=self.config.temperature if self.config.temperature > 0 else 1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bad_words_ids=[[self.tokenizer.pad_token_id]],  # Prevent padding
                    num_return_sequences=1
                )
            
            result_tokens = generated[0].tolist()
            new_tokens = result_tokens[len(input_tokens):]
            
            # Decode results
            full_result = self.tokenizer.decode(result_tokens)
            answer_part = self.tokenizer.decode(new_tokens) if new_tokens else ""
            
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
            
            # Validate result
            is_correct = validate_simple_arithmetic(problem, numerical_answer)
            
            return {
                'problem': problem,
                'full_result': full_result,
                'answer': numerical_answer,
                'input_tokens': input_tokens if self.config.show_tokens else None,
                'output_tokens': new_tokens if self.config.show_tokens else None,
                'inference_time': inference_time,
                'is_correct': is_correct,
                'model_path': str(self.model_path),
                'device': str(self.device)
            }
            
        except Exception as e:
            return {
                'problem': problem,
                'error': str(e),
                'inference_time': time.time() - start_time
            }
    
    def solve_batch(self, problems: List[str]) -> List[Dict[str, Any]]:
        """Solve multiple arithmetic problems"""
        return [self.solve(problem) for problem in problems]
    
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
            'vocab_size': self.tokenizer.vocab_size if self.tokenizer else 0,
            'max_length': self.tokenizer.max_length if self.tokenizer else 0,
            'config': {
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens,
                'show_tokens': self.config.show_tokens
            }
        } 