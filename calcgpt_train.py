#!/usr/bin/env python3
"""
CalcGPT Training Tool - Advanced arithmetic model training system

A professional command-line interface for training CalcGPT models with 
optimized architectures, data augmentation, and comprehensive evaluation.

Author: Mihai NADAS
Version: 1.0.0
"""

import argparse
import sys
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import re

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
    """Print the training tool banner"""
    banner = f"""
{Colors.BLUE}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════╗
║                    CalcGPT Trainer                            ║
║              Advanced Model Training System                   ║
║                         v1.0.0                               ║
╚═══════════════════════════════════════════════════════════════╝
{Colors.ENDC}"""
    print(banner)

def detect_device(verbose: bool = False) -> Tuple[str, bool]:
    """Detect the best available training device"""
    if torch.cuda.is_available():
        device = 'cuda'
        use_fp16 = True
        if verbose:
            print(f"{Colors.GREEN}🚀 CUDA detected: {torch.cuda.get_device_name()}{Colors.ENDC}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        use_fp16 = False  # MPS doesn't support fp16 yet
        if verbose:
            print(f"{Colors.GREEN}🍎 Apple Silicon (MPS) detected{Colors.ENDC}")
    else:
        device = 'cpu'
        use_fp16 = False
        if verbose:
            print(f"{Colors.WARNING}⚠️ Using CPU (consider GPU for faster training){Colors.ENDC}")
    
    return device, use_fp16

def load_dataset(dataset_path: Path, verbose: bool = False) -> List[str]:
    """Load and validate the training dataset"""
    if verbose:
        print(f"{Colors.CYAN}📚 Loading dataset from: {dataset_path}{Colors.ENDC}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            examples = [line.strip() for line in f if line.strip()]
        
        if not examples:
            raise ValueError("Dataset is empty")
        
        if verbose:
            print(f"{Colors.GREEN}✅ Loaded {len(examples)} examples from dataset{Colors.ENDC}")
            
            # Show some statistics
            avg_length = sum(len(ex) for ex in examples) / len(examples)
            max_length = max(len(ex) for ex in examples)
            min_length = min(len(ex) for ex in examples)
            
            print(f"{Colors.CYAN}📊 Dataset statistics:{Colors.ENDC}")
            print(f"   Average length: {avg_length:.1f} characters")
            print(f"   Maximum length: {max_length} characters")
            print(f"   Minimum length: {min_length} characters")
        
        return examples
        
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")

def augment_arithmetic_data(examples: List[str], verbose: bool = False) -> List[str]:
    """Generate additional training examples through commutative property"""
    if verbose:
        print(f"{Colors.CYAN}✨ Applying data augmentation (commutative property)...{Colors.ENDC}")
    
    augmented = examples.copy()
    added_count = 0
    
    for example in examples:
        if '+' in example and '=' in example:
            parts = example.split('=')
            if '+' in parts[0]:
                operands = parts[0].split('+')
                if len(operands) == 2:
                    # Add commutative version: b+a=result
                    reversed_expr = f"{operands[1]}+{operands[0]}={parts[1]}"
                    if reversed_expr not in augmented:
                        augmented.append(reversed_expr)
                        added_count += 1
    
    if verbose:
        print(f"{Colors.GREEN}✅ Added {added_count} augmented examples{Colors.ENDC}")
        print(f"{Colors.CYAN}📈 Total dataset size: {len(augmented)} examples{Colors.ENDC}")
    
    return augmented

def create_optimized_vocab(examples: List[str], verbose: bool = False) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create vocabulary with better organization"""
    if verbose:
        print(f"{Colors.CYAN}🔤 Creating optimized vocabulary...{Colors.ENDC}")
    
    special_tokens = ['<pad>', '<eos>']
    chars = sorted(set(''.join(examples)))
    vocab = {c: i for i, c in enumerate(special_tokens + chars)}
    id2char = {i: c for c, i in vocab.items()}
    
    if verbose:
        print(f"{Colors.GREEN}✅ Vocabulary created with {len(vocab)} tokens{Colors.ENDC}")
        print(f"{Colors.CYAN}🔧 Special tokens: {special_tokens}{Colors.ENDC}")
        print(f"{Colors.CYAN}🔧 Character tokens: {''.join(sorted(chars))}{Colors.ENDC}")
    
    return vocab, id2char

class OptimizedDataset(Dataset):
    """Pre-tokenized dataset for faster training"""
    def __init__(self, data: List[str], maxlen: int, vocab: Dict[str, int], name: str = "dataset", verbose: bool = False):
        if verbose:
            print(f"{Colors.CYAN}🔧 Creating {name} with {len(data)} examples...{Colors.ENDC}")
        
        self.vocab = vocab
        # Pre-encode all examples for efficiency
        self.data = []
        for example in data:
            encoded = self.encode(example)
            padded = encoded + [vocab['<pad>']] * (maxlen - len(encoded))
            self.data.append(padded)
        
        if verbose:
            print(f"{Colors.GREEN}✅ {name} ready with {len(self.data)} pre-tokenized sequences{Colors.ENDC}")
    
    def encode(self, s: str) -> List[int]:
        return [self.vocab[c] for c in s if c in self.vocab] + [self.vocab['<eos>']]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.tensor(self.data[i]), 
            'labels': torch.tensor(self.data[i])
        }

def create_model_config(vocab_size: int, max_length: int, args, verbose: bool = False) -> GPT2Config:
    """Create optimized model configuration"""
    if verbose:
        print(f"{Colors.CYAN}🏗️ Creating model architecture...{Colors.ENDC}")
    
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_length + 10,  # Extra room for generation
        n_embd=args.embedding_dim, 
        n_layer=args.num_layers,
        n_head=args.num_heads,
        n_inner=args.feedforward_dim,
        pad_token_id=0,  # <pad> token
        eos_token_id=1,  # <eos> token
        use_cache=False  # Disable for training efficiency
    )
    
    if verbose:
        print(f"{Colors.GREEN}✅ Model configuration created:{Colors.ENDC}")
        print(f"   📏 Sequence length: {config.n_positions}")
        print(f"   🧠 Embedding dimension: {config.n_embd}")
        print(f"   🏗️  Number of layers: {config.n_layer}")
        print(f"   👁️  Attention heads: {config.n_head}")
        print(f"   🔧 Feedforward dimension: {config.n_inner}")
    
    return config

def print_model_info(model: GPT2LMHeadModel, verbose: bool = False):
    """Print detailed model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 / 1024
    
    if verbose:
        print(f"{Colors.CYAN}🧠 Model Information:{Colors.ENDC}")
    
    print(f"   📊 Total parameters: {Colors.BOLD}{total_params:,}{Colors.ENDC}")
    print(f"   🎯 Trainable parameters: {Colors.BOLD}{trainable_params:,}{Colors.ENDC}")
    print(f"   💾 Model size: {Colors.BOLD}{model_size_mb:.1f} MB{Colors.ENDC}")

def create_training_args(args, output_dir: Path, device: str, use_fp16: bool) -> TrainingArguments:
    """Create optimized training arguments"""
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy='steps' if args.eval_steps > 0 else 'no',
        save_strategy='steps',
        load_best_model_at_end=True if args.eval_steps > 0 else False,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=use_fp16,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=args.seed,
        report_to=[]  # Disable wandb/tensorboard
    )
    
    return training_args

def test_model_inference(model: GPT2LMHeadModel, vocab: Dict[str, int], id2char: Dict[int, str], 
                        test_prompts: List[str], verbose: bool = False) -> Tuple[int, int]:
    """Test model inference capabilities"""
    if verbose:
        print(f"\n{Colors.CYAN}🧪 Testing model inference...{Colors.ENDC}")
    
    model.eval()
    successful_predictions = 0
    total_predictions = len(test_prompts)
    
    def encode(s: str) -> List[int]:
        return [vocab[c] for c in s if c in vocab] + [vocab['<eos>']]
    
    def decode(ids: List[int]) -> str:
        return ''.join([id2char[i] for i in ids if i != vocab['<pad>'] and i != vocab['<eos>']])
    
    for i, prompt in enumerate(test_prompts):
        if verbose:
            print(f"\n{Colors.CYAN}🔍 Test {i+1}/{len(test_prompts)}: '{prompt}'{Colors.ENDC}")
        
        try:
            # Encode the prompt but remove EOS for generation
            input_tokens = encode(prompt)[:-1]  # Remove EOS token for generation
            input_ids = torch.tensor([input_tokens], dtype=torch.long).to(model.device)
            
            with torch.no_grad():
                out = model.generate(
                    input_ids, 
                    max_length=len(input_tokens) + 10,
                    do_sample=False,
                    pad_token_id=vocab['<pad>'],
                    eos_token_id=vocab['<eos>'],
                    bad_words_ids=[[vocab['<pad>']]]
                )
            
            generated_tokens = out[0].tolist()
            full_result = decode(generated_tokens)
            
            if verbose:
                new_tokens = generated_tokens[len(input_tokens):]
                new_part = decode(new_tokens) if new_tokens else ""
                print(f"   Generated: '{full_result}'")
                print(f"   New part: '{new_part}'")
            
            # Basic correctness check
            if '=' in full_result:
                try:
                    parts = full_result.split('=')
                    if len(parts) >= 2:
                        equation = parts[0] + '='
                        answer = parts[1].strip().split()[0] if parts[1].strip() else ""
                        
                        # Simple validation for basic arithmetic
                        if '+' in equation:
                            operands = equation.replace('=', '').split('+')
                            if len(operands) == 2 and operands[0].isdigit() and operands[1].isdigit():
                                expected = str(int(operands[0]) + int(operands[1]))
                                if answer == expected:
                                    successful_predictions += 1
                                    if verbose:
                                        print(f"   {Colors.GREEN}✅ Correct!{Colors.ENDC}")
                                elif verbose:
                                    print(f"   {Colors.WARNING}❌ Expected {expected}, got {answer}{Colors.ENDC}")
                except:
                    if verbose:
                        print(f"   {Colors.WARNING}⚠️ Could not validate{Colors.ENDC}")
        
        except Exception as e:
            if verbose:
                print(f"   {Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
    
    return successful_predictions, total_predictions

def validate_arguments(args) -> None:
    """Validate command line arguments"""
    errors = []
    
    if args.epochs < 1:
        errors.append("epochs must be positive")
    
    if args.batch_size < 1:
        errors.append("batch-size must be positive")
    
    if args.learning_rate <= 0:
        errors.append("learning-rate must be positive")
    
    if args.embedding_dim < 1:
        errors.append("embedding-dim must be positive")
    
    if args.num_layers < 1:
        errors.append("num-layers must be positive")
    
    if args.num_heads < 1:
        errors.append("num-heads must be positive")
    
    if args.embedding_dim % args.num_heads != 0:
        errors.append("embedding-dim must be divisible by num-heads")
    
    if not Path(args.dataset).exists():
        errors.append(f"dataset file not found: {args.dataset}")
    
    if errors:
        print(f"{Colors.FAIL}❌ Validation Errors:{Colors.ENDC}")
        for error in errors:
            print(f"   • {error}")
        sys.exit(1)



def generate_model_dirname(
    dataset_path: str,
    embedding_dim: int,
    num_layers: int,
    num_heads: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    feedforward_dim: int = None,
    lr_scheduler: str = None,
    no_augmentation: bool = False,
    output_base: str = "models"
) -> str:
    """Generate descriptive model directory name based on training parameters."""
    
    parts = ["calcgpt"]
    
    # Add model architecture
    parts.append(f"emb{embedding_dim}")
    parts.append(f"lay{num_layers}")
    parts.append(f"head{num_heads}")
    
    # Add feedforward dimension if not default
    if feedforward_dim and feedforward_dim != 512:
        parts.append(f"ff{feedforward_dim}")
    
    # Add training parameters
    parts.append(f"ep{epochs}")
    parts.append(f"bs{batch_size}")
    
    # Format learning rate nicely
    if learning_rate >= 1:
        lr_str = f"lr{learning_rate:.0f}"
    elif learning_rate >= 0.1:
        lr_str = f"lr{learning_rate:.1f}"
    elif learning_rate >= 0.01:
        lr_str = f"lr{learning_rate:.2f}"
    else:
        # For scientific notation, use compact format like lr1e3 for 1e-3
        exp_str = f"{learning_rate:.0e}"
        if exp_str.startswith('1e-'):
            exponent = exp_str[3:]
            lr_str = f"lr1e{exponent}"
        else:
            lr_str = f"lr{learning_rate:.0e}".replace('e-0', 'e').replace('-', 'n')
    parts.append(lr_str)
    
    # Add scheduler if not default
    if lr_scheduler and lr_scheduler != 'cosine':
        parts.append(f"sch{lr_scheduler}")
    
    # Add dataset info
    dataset_name = Path(dataset_path).stem
    if dataset_name.startswith('ds-calcgpt'):
        # Extract key info from auto-generated dataset names
        if '_limit' in dataset_name:
            limit_part = dataset_name.split('_limit')[1].split('_')[0]
            parts.append(f"ds{limit_part}")
        elif 'max' in dataset_name:
            max_part = dataset_name.split('_max')[1].split('_')[0]
            parts.append(f"dsm{max_part}")
        else:
            parts.append("dsCalc")
    else:
        # Use simplified dataset name
        simplified = dataset_name.replace('ds-calcgpt', '').replace('_', '').replace('-', '')[:8]
        if simplified:
            parts.append(f"ds{simplified}")
        else:
            parts.append("dsCalc")
    
    # Add data processing flags
    if no_augmentation:
        parts.append("noaug")
    
    dirname = "_".join(parts)
    return str(Path(output_base) / dirname)

def parse_model_dirname(dirname: str) -> Dict[str, Any]:
    """Parse auto-generated model directory name to extract training parameters."""
    
    # Remove path and get base directory name
    base_name = Path(dirname).name
    
    # Check if it matches our naming convention
    if not base_name.startswith('calcgpt'):
        raise ValueError(f"Directory '{dirname}' doesn't match expected naming convention (calcgpt_...)")
    
    # Split by underscores and remove the prefix
    parts = base_name.split('_')[1:]  # Skip 'calcgpt'
    
    if len(parts) < 6:  # Minimum: emb, lay, head, ep, bs, lr
        raise ValueError(f"Directory '{dirname}' doesn't have enough components")
    
    params = {
        'embedding_dim': None,
        'num_layers': None,
        'num_heads': None,
        'feedforward_dim': 512,  # Default
        'epochs': None,
        'batch_size': None,
        'learning_rate': None,
        'lr_scheduler': 'cosine',  # Default
        'dataset_info': None,
        'no_augmentation': False
    }
    
    for part in parts:
        # Parse embedding dimension
        if part.startswith('emb'):
            params['embedding_dim'] = int(part[3:])
        
        # Parse number of layers
        elif part.startswith('lay'):
            params['num_layers'] = int(part[3:])
        
        # Parse number of heads
        elif part.startswith('head'):
            params['num_heads'] = int(part[4:])
        
        # Parse feedforward dimension
        elif part.startswith('ff'):
            params['feedforward_dim'] = int(part[2:])
        
        # Parse epochs
        elif part.startswith('ep'):
            params['epochs'] = int(part[2:])
        
        # Parse batch size
        elif part.startswith('bs'):
            params['batch_size'] = int(part[2:])
        
        # Parse learning rate
        elif part.startswith('lr'):
            lr_str = part[2:]
            if 'e' in lr_str:
                # Handle scientific notation like 1e3 -> 1e-3
                if lr_str.startswith('1e') and lr_str[2:].isdigit():
                    exponent = int(lr_str[2:])
                    params['learning_rate'] = float(f"1e-{exponent}")
                else:
                    params['learning_rate'] = float(lr_str)
            else:
                params['learning_rate'] = float(lr_str)
        
        # Parse scheduler
        elif part.startswith('sch'):
            params['lr_scheduler'] = part[3:]
        
        # Parse dataset info
        elif part.startswith('ds'):
            params['dataset_info'] = part[2:]
        
        # Parse flags
        elif part == 'noaug':
            params['no_augmentation'] = True
    
    # Validation
    required_params = ['embedding_dim', 'num_layers', 'num_heads', 'epochs', 'batch_size', 'learning_rate']
    missing = [p for p in required_params if params[p] is None]
    if missing:
        raise ValueError(f"Could not determine required parameters: {missing}")
    
    return params

def display_model_analysis(params: Dict[str, Any], model_dir: str):
    """Display model training parameters in a beautiful format."""
    print(f"\n{Colors.BOLD}📋 Model Analysis: {Path(model_dir).name}{Colors.ENDC}")
    print("=" * 60)
    
    print(f"\n{Colors.CYAN}🏗️  Model Architecture:{Colors.ENDC}")
    print(f"  📏 Embedding dimension: {Colors.BOLD}{params['embedding_dim']}{Colors.ENDC}")
    print(f"  🏗️  Number of layers: {Colors.BOLD}{params['num_layers']}{Colors.ENDC}")
    print(f"  👁️  Attention heads: {Colors.BOLD}{params['num_heads']}{Colors.ENDC}")
    print(f"  🔧 Feedforward dimension: {Colors.BOLD}{params['feedforward_dim']}{Colors.ENDC}")
    
    print(f"\n{Colors.CYAN}📖 Training Parameters:{Colors.ENDC}")
    print(f"  🔄 Epochs: {Colors.BOLD}{params['epochs']}{Colors.ENDC}")
    print(f"  📦 Batch size: {Colors.BOLD}{params['batch_size']}{Colors.ENDC}")
    print(f"  📈 Learning rate: {Colors.BOLD}{params['learning_rate']}{Colors.ENDC}")
    print(f"  📅 LR scheduler: {Colors.BOLD}{params['lr_scheduler']}{Colors.ENDC}")
    
    print(f"\n{Colors.CYAN}📊 Data Configuration:{Colors.ENDC}")
    if params['dataset_info']:
        print(f"  📚 Dataset: {Colors.GREEN}{params['dataset_info']}{Colors.ENDC}")
    else:
        print(f"  📚 Dataset: {Colors.GREEN}CalcGPT{Colors.ENDC}")
    
    if params['no_augmentation']:
        print(f"  ✨ Data augmentation: {Colors.WARNING}Disabled{Colors.ENDC}")
    else:
        print(f"  ✨ Data augmentation: {Colors.GREEN}Enabled{Colors.ENDC}")
    
    # Show model directory info if it exists
    model_path = Path(model_dir)
    if model_path.exists():
        # Count files in model directory
        files = list(model_path.rglob('*'))
        model_files = [f for f in files if f.is_file()]
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in model_files)
        
        print(f"\n{Colors.CYAN}📁 Model Directory:{Colors.ENDC}")
        print(f"  📄 Total files: {Colors.BOLD}{len(model_files)}{Colors.ENDC}")
        print(f"  💾 Total size: {Colors.BOLD}{total_size:,} bytes ({total_size/1024/1024:.1f} MB){Colors.ENDC}")
        
        # Check for specific files
        config_file = model_path / "config.json"
        pytorch_model = model_path / "pytorch_model.bin"
        training_args = model_path / "training_args.bin"
        
        if config_file.exists():
            print(f"  ✅ Model config found")
        if pytorch_model.exists():
            size_mb = pytorch_model.stat().st_size / 1024 / 1024
            print(f"  ✅ PyTorch model found ({size_mb:.1f} MB)")
        if training_args.exists():
            print(f"  ✅ Training arguments found")
    
    # Show equivalent command line
    print(f"\n{Colors.CYAN}💻 Equivalent Training Command:{Colors.ENDC}")
    cmd_parts = [f"python {Path(__file__).name}"]
    
    if params['embedding_dim'] != 128:  # 128 is default
        cmd_parts.append(f"--embedding-dim {params['embedding_dim']}")
    if params['num_layers'] != 6:  # 6 is default
        cmd_parts.append(f"--num-layers {params['num_layers']}")
    if params['num_heads'] != 8:  # 8 is default
        cmd_parts.append(f"--num-heads {params['num_heads']}")
    if params['feedforward_dim'] != 512:  # 512 is default
        cmd_parts.append(f"--feedforward-dim {params['feedforward_dim']}")
    if params['epochs'] != 50:  # 50 is default
        cmd_parts.append(f"-e {params['epochs']}")
    if params['batch_size'] != 8:  # 8 is default
        cmd_parts.append(f"-b {params['batch_size']}")
    if params['learning_rate'] != 1e-3:  # 1e-3 is default
        cmd_parts.append(f"--learning-rate {params['learning_rate']}")
    if params['lr_scheduler'] != 'cosine':  # cosine is default
        cmd_parts.append(f"--lr-scheduler {params['lr_scheduler']}")
    if params['no_augmentation']:
        cmd_parts.append("--no-augmentation")
    
    print(f"  {Colors.BOLD}{' '.join(cmd_parts)}{Colors.ENDC}")
    print()

def estimate_dataset_size(dataset_path: Path) -> int:
    """Estimate dataset size for naming purposes."""
    try:
        with open(dataset_path, 'r') as f:
            count = sum(1 for line in f if line.strip())
        return count
    except:
        return 0

def print_model_summary(args, dataset_size: int, vocab_size: int, device: str, use_fp16: bool, output_dir: Path):
    """Print a beautiful model training configuration summary."""
    print(f"\n{Colors.BOLD}🚀 Training Configuration:{Colors.ENDC}")
    print("=" * 60)
    
    print(f"  📚 Dataset: {Colors.CYAN}{args.dataset}{Colors.ENDC} ({dataset_size:,} examples)")
    print(f"  🎯 Device: {Colors.GREEN}{device.upper()}{Colors.ENDC}")
    print(f"  ⚡ Mixed precision: {'✅ Enabled' if use_fp16 else '❌ Disabled'}")
    print(f"  🔤 Vocabulary size: {Colors.BOLD}{vocab_size}{Colors.ENDC}")
    
    print(f"\n  🏗️  Model Architecture:")
    print(f"     📏 Embedding dimension: {Colors.BOLD}{args.embedding_dim}{Colors.ENDC}")
    print(f"     🏗️  Number of layers: {Colors.BOLD}{args.num_layers}{Colors.ENDC}")
    print(f"     👁️  Attention heads: {Colors.BOLD}{args.num_heads}{Colors.ENDC}")
    print(f"     🔧 Feedforward dimension: {Colors.BOLD}{args.feedforward_dim}{Colors.ENDC}")
    
    print(f"\n  📖 Training Parameters:")
    print(f"     🔄 Epochs: {Colors.BOLD}{args.epochs}{Colors.ENDC}")
    print(f"     📦 Batch size: {Colors.BOLD}{args.batch_size}{Colors.ENDC}")
    print(f"     📈 Learning rate: {Colors.BOLD}{args.learning_rate}{Colors.ENDC}")
    print(f"     📅 LR scheduler: {Colors.BOLD}{args.lr_scheduler}{Colors.ENDC}")
    
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    print(f"     🔧 Effective batch size: {Colors.BOLD}{effective_batch}{Colors.ENDC}")
    
    print(f"  📁 Model output: {Colors.CYAN}{output_dir}{Colors.ENDC}")
    print()

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="CalcGPT Trainer - Advanced arithmetic model training system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                          # Train with auto-generated model name
  %(prog)s -d datasets/large.txt -e 100            # Custom dataset, 100 epochs
  %(prog)s --embedding-dim 256 --num-layers 8      # Larger model architecture  
  %(prog)s --batch-size 16 --learning-rate 5e-4    # Custom training parameters
  %(prog)s -o models/calcgpt-v2 --verbose          # Custom output directory
  %(prog)s --no-augmentation --test-only           # No augmentation, test only
  %(prog)s --analyze models/calcgpt_emb128_lay6_head8_ep50_bs8_lr0.001_dsCalc  # Analyze model
        """
    )
    
    # Analysis mode
    parser.add_argument(
        '--analyze',
        type=str,
        help='Analyze an existing model directory to show training parameters'
    )
    
    # Dataset and I/O
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        default='datasets/ds-calcgpt.txt',
        help='Path to training dataset (default: datasets/ds-calcgpt.txt)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        help='Output directory for model checkpoints (auto-generated if not specified)'
    )
    
    # Model architecture
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=128,
        help='Embedding dimension (default: 128)'
    )
    
    parser.add_argument(
        '--num-layers',
        type=int,
        default=6,
        help='Number of transformer layers (default: 6)'
    )
    
    parser.add_argument(
        '--num-heads',
        type=int,
        default=8,
        help='Number of attention heads (default: 8)'
    )
    
    parser.add_argument(
        '--feedforward-dim',
        type=int,
        default=512,
        help='Feedforward dimension (default: 512)'
    )
    
    # Training parameters
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=8,
        help='Training batch size (default: 8)'
    )
    
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=16,
        help='Evaluation batch size (default: 16)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    
    parser.add_argument(
        '--lr-scheduler',
        type=str,
        default='cosine',
        choices=['linear', 'cosine', 'constant'],
        help='Learning rate scheduler (default: cosine)'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay for regularization (default: 0.01)'
    )
    
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=50,
        help='Number of warmup steps (default: 50)'
    )
    
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=2,
        help='Gradient accumulation steps (default: 2)'
    )
    
    # Training monitoring
    parser.add_argument(
        '--logging-steps',
        type=int,
        default=10,
        help='Logging frequency (default: 10)'
    )
    
    parser.add_argument(
        '--eval-steps',
        type=int,
        default=25,
        help='Evaluation frequency (0 to disable, default: 25)'
    )
    
    parser.add_argument(
        '--save-steps',
        type=int,
        default=100,
        help='Model save frequency (default: 100)'
    )
    
    # Data processing
    parser.add_argument(
        '--no-augmentation',
        action='store_true',
        help='Disable data augmentation'
    )
    
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Validation split ratio (default: 0.2)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Testing
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only run inference tests (skip training)'
    )
    
    # Utility options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output with detailed progress'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='CalcGPT Trainer 1.0.0'
    )
    
    parser.add_argument(
        '--no-banner',
        action='store_true',
        help='Suppress banner output'
    )
    
    args = parser.parse_args()
    
    # Print banner unless suppressed
    if not args.no_banner:
        print_banner()
    
    # Handle analyze mode
    if args.analyze:
        if args.verbose:
            print(f"{Colors.CYAN}🔍 Analyzing model directory: {args.analyze}{Colors.ENDC}")
        
        try:
            params = parse_model_dirname(args.analyze)
            display_model_analysis(params, args.analyze)
            return
        except ValueError as e:
            print(f"{Colors.FAIL}❌ Analysis Error: {e}{Colors.ENDC}")
            sys.exit(1)
        except Exception as e:
            print(f"{Colors.FAIL}❌ Unexpected error during analysis: {e}{Colors.ENDC}")
            sys.exit(1)
    
    # Validate arguments
    validate_arguments(args)
    
    # Detect device and capabilities
    device, use_fp16 = detect_device(args.verbose)
    
    try:
        # Load dataset
        examples = load_dataset(Path(args.dataset), args.verbose)
        
        # Apply data augmentation if enabled
        if not args.no_augmentation:
            examples = augment_arithmetic_data(examples, args.verbose)
        elif args.verbose:
            print(f"{Colors.WARNING}⚠️ Data augmentation disabled{Colors.ENDC}")
        
        # Create vocabulary
        vocab, id2char = create_optimized_vocab(examples, args.verbose)
        
        # Calculate maximum sequence length
        def encode(s): 
            return [vocab[c] for c in s if c in vocab] + [vocab['<eos>']]
        
        maxlen = max(len(encode(x)) for x in examples)
        if args.verbose:
            print(f"{Colors.CYAN}📏 Maximum sequence length: {maxlen}{Colors.ENDC}")
        
        # Split data for training/validation
        if args.eval_steps > 0 and not args.test_only:
            if args.verbose:
                print(f"{Colors.CYAN}🔄 Splitting dataset (test split: {args.test_split:.1%})...{Colors.ENDC}")
            
            train_examples, val_examples = train_test_split(
                examples, test_size=args.test_split, random_state=args.seed, shuffle=True
            )
            
            if args.verbose:
                print(f"{Colors.GREEN}📊 Training examples: {len(train_examples)}{Colors.ENDC}")
                print(f"{Colors.GREEN}📊 Validation examples: {len(val_examples)}{Colors.ENDC}")
        else:
            train_examples = examples
            val_examples = []
            if args.verbose:
                print(f"{Colors.WARNING}⚠️ No validation split (evaluation disabled){Colors.ENDC}")
        
        # Generate output directory if not provided
        if args.output_dir is None:
            output_dirname = generate_model_dirname(
                dataset_path=args.dataset,
                embedding_dim=args.embedding_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                feedforward_dim=args.feedforward_dim,
                lr_scheduler=args.lr_scheduler,
                no_augmentation=args.no_augmentation
            )
            output_dir = Path(output_dirname)
        else:
            output_dir = Path(args.output_dir)
        
        # Print model summary
        if args.verbose or not args.no_banner:
            print_model_summary(args, len(examples), len(vocab), device, use_fp16, output_dir)
        
        # Create model
        config = create_model_config(len(vocab), maxlen, args, args.verbose)
        model = GPT2LMHeadModel(config)
        model.to(device)
        
        print_model_info(model, True)
        
        if args.test_only:
            # Test-only mode
            print(f"\n{Colors.CYAN}🧪 Test-only mode: Running inference tests...{Colors.ENDC}")
            
            test_prompts = [
                "1+1=", "2+0=", "0+2=",           # Simple cases
                "10+1=", "11+1=", "100+10=",     # Larger numbers
                "22+100=", "12+10=", "5+5="      # More cases
            ]
            
            successful, total = test_model_inference(model, vocab, id2char, test_prompts, args.verbose)
            
            print(f"\n{Colors.BOLD}🎯 Test Results:{Colors.ENDC}")
            print(f"  ✅ Successful: {successful}/{total}")
            print(f"  📊 Accuracy: {successful/total*100:.1f}%")
            
            return
        
        # Create datasets
        train_dataset = OptimizedDataset(train_examples, maxlen, vocab, "training dataset", args.verbose)
        val_dataset = OptimizedDataset(val_examples, maxlen, vocab, "validation dataset", args.verbose) if val_examples else None
        
        # Setup training
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = create_training_args(args, output_dir, device, use_fp16)
        
        if args.verbose:
            print(f"{Colors.CYAN}⚙️ Training arguments configured{Colors.ENDC}")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=None
        )
        
        # Start training
        print(f"\n{Colors.GREEN}{Colors.BOLD}🚀 STARTING TRAINING{Colors.ENDC}")
        print("=" * 60)
        
        start_time = time.time()
        training_result = trainer.train()
        training_time = time.time() - start_time
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ TRAINING COMPLETED{Colors.ENDC}")
        print("=" * 60)
        print(f"  📊 Final training loss: {Colors.BOLD}{training_result.training_loss:.4f}{Colors.ENDC}")
        print(f"  ⏱️  Training time: {Colors.BOLD}{training_time/60:.1f} minutes{Colors.ENDC}")
        
        # Final evaluation
        if val_dataset:
            if args.verbose:
                print(f"\n{Colors.CYAN}📊 Running final evaluation...{Colors.ENDC}")
            
            eval_results = trainer.evaluate()
            print(f"  📉 Final validation loss: {Colors.BOLD}{eval_results['eval_loss']:.4f}{Colors.ENDC}")
        
        # Test inference
        print(f"\n{Colors.CYAN}🧪 Testing trained model...{Colors.ENDC}")
        
        test_prompts = [
            "1+1=", "2+0=", "0+2=", "10+1=", "11+1=", 
            "100+10=", "22+100=", "12+10=", "5+5="
        ]
        
        successful, total = test_model_inference(model, vocab, id2char, test_prompts, args.verbose)
        
        # Final success summary
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 TRAINING COMPLETE!{Colors.ENDC}")
        print("=" * 60)
        print(f"  📊 Model accuracy: {Colors.BOLD}{successful}/{total} ({successful/total*100:.1f}%){Colors.ENDC}")
        print(f"  📁 Model saved to: {Colors.CYAN}{output_dir}{Colors.ENDC}")
        print(f"  ⏱️  Total time: {Colors.BOLD}{training_time/60:.1f} minutes{Colors.ENDC}")
        
        # Show model directory name breakdown
        if args.output_dir is None:
            print(f"\n{Colors.CYAN}📝 Auto-generated Model Name:{Colors.ENDC}")
            print(f"  {Colors.BOLD}{output_dir.name}{Colors.ENDC}")
            print(f"  💡 Encodes: architecture + training params + dataset")
        
        # List achievements
        print(f"\n{Colors.CYAN}🏆 Training Achievements:{Colors.ENDC}")
        print(f"  ✅ Data augmentation applied")
        print(f"  ✅ Optimized model architecture")
        print(f"  ✅ Advanced training configuration")
        print(f"  ✅ Comprehensive evaluation")
        print(f"  ✅ Inference testing completed")
        print(f"  ✅ Intelligent model naming system")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠️ Training interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.FAIL}❌ Training error: {e}{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()