"""
CalcGPT Training Library

Core training functionality for CalcGPT models with proper separation of concerns.
"""

import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split


@dataclass
class TrainingConfig:
    """Configuration for CalcGPT training"""
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-3
    embedding_dim: int = 128
    num_layers: int = 6
    num_heads: int = 8
    feedforward_dim: int = 512
    warmup_steps: int = 50
    weight_decay: float = 0.01
    save_steps: int = 1000
    test_split: float = 0.2
    seed: int = 42
    no_augmentation: bool = False


def detect_device() -> Tuple[str, bool]:
    """Detect the best available training device
    
    Returns:
        Tuple of (device_name, use_fp16)
    """
    if torch.cuda.is_available():
        return 'cuda', True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps', False
    else:
        return 'cpu', False


def load_dataset(dataset_path: Path) -> List[str]:
    """Load training dataset from file
    
    Args:
        dataset_path: Path to dataset file
        
    Returns:
        List of training examples
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset is empty
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        examples = [line.strip() for line in f if line.strip()]
    
    if not examples:
        raise ValueError("Dataset is empty")
    
    return examples


def augment_data(examples: List[str]) -> List[str]:
    """Generate additional training examples through commutative property
    
    Args:
        examples: Original training examples
        
    Returns:
        Augmented dataset with commutative examples added
    """
    augmented = examples.copy()
    added_count = 0
    
    for example in examples:
        if '+' in example and '=' in example:
            parts = example.split('=')
            if '+' in parts[0]:
                operands = parts[0].split('+')
                if len(operands) == 2:
                    reversed_expr = f"{operands[1]}+{operands[0]}={parts[1]}"
                    if reversed_expr not in augmented:
                        augmented.append(reversed_expr)
                        added_count += 1
    
    return augmented


def create_vocab(examples: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create vocabulary from examples
    
    Args:
        examples: Training examples
        
    Returns:
        Tuple of (vocab_dict, id_to_char_dict)
    """
    special_tokens = ['<pad>', '<eos>']
    chars = sorted(set(''.join(examples)))
    vocab = {c: i for i, c in enumerate(special_tokens + chars)}
    id2char = {i: c for c, i in vocab.items()}
    
    return vocab, id2char


def encode(s: str, vocab: Dict[str, int]) -> List[int]:
    """Encode string to token IDs
    
    Args:
        s: String to encode
        vocab: Vocabulary dictionary
        
    Returns:
        List of token IDs
    """
    return [vocab[c] for c in s if c in vocab] + [vocab['<eos>']]


class OptimizedDataset(Dataset):
    """Pre-tokenized dataset for faster training"""
    
    def __init__(self, data: List[str], maxlen: int, vocab: Dict[str, int]):
        """Initialize dataset with pre-tokenized sequences
        
        Args:
            data: List of training examples
            maxlen: Maximum sequence length for padding
            vocab: Vocabulary dictionary
        """
        self.vocab = vocab
        self.data = []
        
        for example in data:
            encoded = encode(example, vocab)
            padded = encoded + [vocab['<pad>']] * (maxlen - len(encoded))
            self.data.append(padded)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.tensor(self.data[i]), 
            'labels': torch.tensor(self.data[i])
        }


def create_model_config(vocab_size: int, max_length: int, config: TrainingConfig) -> GPT2Config:
    """Create model configuration
    
    Args:
        vocab_size: Size of vocabulary
        max_length: Maximum sequence length
        config: Training configuration
        
    Returns:
        GPT2Config object
    """
    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_length + 10,
        n_embd=config.embedding_dim, 
        n_layer=config.num_layers,
        n_head=config.num_heads,
        n_inner=config.feedforward_dim,
        pad_token_id=0,
        eos_token_id=1,
        use_cache=False
    )


def print_model_info(model: GPT2LMHeadModel) -> Dict[str, int]:
    """Print and return model information
    
    Args:
        model: The GPT2 model
        
    Returns:
        Dictionary with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 / 1024
    
    stats = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': model_size_mb
    }
    
    return stats


class CalcGPTTrainer:
    """Main training class for CalcGPT models"""
    
    def __init__(self, config: TrainingConfig, dataset_path: Path, output_dir: Path, verbose: bool = True):
        """Initialize trainer
        
        Args:
            config: Training configuration
            dataset_path: Path to training dataset
            output_dir: Output directory for model
            verbose: Whether to print progress messages
        """
        self.config = config
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Initialize training components
        self.device, self.use_fp16 = detect_device()
        self.examples = None
        self.vocab = None
        self.id2char = None
        self.maxlen = None
        self.model = None
        self.trainer = None
        
    def log(self, message: str):
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def load_and_prepare_data(self):
        """Load and prepare training data"""
        self.log(f"Loading dataset from: {self.dataset_path}")
        self.examples = load_dataset(self.dataset_path)
        self.log(f"Loaded {len(self.examples)} examples")
        
        # Apply data augmentation if enabled
        if not self.config.no_augmentation:
            original_count = len(self.examples)
            self.examples = augment_data(self.examples)
            added = len(self.examples) - original_count
            self.log(f"Added {added} augmented examples (total: {len(self.examples)})")
        else:
            self.log("Data augmentation disabled")
        
        # Create vocabulary
        self.vocab, self.id2char = create_vocab(self.examples)
        self.log(f"Vocabulary created with {len(self.vocab)} tokens")
        
        # Calculate max sequence length
        self.maxlen = max(len(encode(x, self.vocab)) for x in self.examples)
        self.log(f"Maximum sequence length: {self.maxlen}")
    
    def create_datasets(self) -> Tuple[OptimizedDataset, Optional[OptimizedDataset]]:
        """Create training and validation datasets
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if self.config.test_split > 0:
            train_examples, val_examples = train_test_split(
                self.examples, 
                test_size=self.config.test_split, 
                random_state=self.config.seed, 
                shuffle=True
            )
            self.log(f"Training examples: {len(train_examples)}")
            self.log(f"Validation examples: {len(val_examples)}")
            
            train_dataset = OptimizedDataset(train_examples, self.maxlen, self.vocab)
            val_dataset = OptimizedDataset(val_examples, self.maxlen, self.vocab)
            return train_dataset, val_dataset
        else:
            self.log("No validation split")
            train_dataset = OptimizedDataset(self.examples, self.maxlen, self.vocab)
            return train_dataset, None
    
    def create_model(self) -> GPT2LMHeadModel:
        """Create and configure the model
        
        Returns:
            Initialized GPT2LMHeadModel
        """
        self.log("Creating model...")
        model_config = create_model_config(len(self.vocab), self.maxlen, self.config)
        model = GPT2LMHeadModel(model_config)
        model.to(self.device)
        
        # Print model information
        stats = print_model_info(model)
        self.log(f"Model: {stats['total_params']:,} parameters ({stats['size_mb']:.1f} MB)")
        
        return model
    
    def setup_trainer(self, train_dataset: OptimizedDataset, val_dataset: Optional[OptimizedDataset]) -> Trainer:
        """Setup the Hugging Face trainer
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            
        Returns:
            Configured Trainer object
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=50,
            save_steps=self.config.save_steps,
            eval_strategy='steps' if val_dataset else 'no',
            eval_steps=self.config.save_steps if val_dataset else None,
            fp16=self.use_fp16,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            seed=self.config.seed,
            report_to=[]
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=None
        )
        
        return trainer
    
    def run_quick_test(self) -> Dict[str, str]:
        """Run a quick inference test
        
        Returns:
            Dictionary mapping test prompts to results
        """
        self.log("\n=== QUICK TEST ===")
        self.model.eval()
        test_prompts = ["1+1=", "2+3=", "5+0="]
        results = {}
        
        with torch.no_grad():
            for prompt in test_prompts:
                try:
                    input_tokens = encode(prompt, self.vocab)[:-1]  # Remove EOS
                    input_ids = torch.tensor([input_tokens]).to(self.device)
                    
                    output = self.model.generate(
                        input_ids,
                        max_length=len(input_tokens) + 5,
                        do_sample=False,
                        pad_token_id=self.vocab['<pad>'],
                        eos_token_id=self.vocab['<eos>']
                    )
                    
                    result_tokens = output[0].tolist()
                    result = ''.join([self.id2char[i] for i in result_tokens 
                                    if i not in [self.vocab['<pad>'], self.vocab['<eos>']]])
                    
                    results[prompt] = result
                    self.log(f"  {prompt} -> {result}")
                    
                except Exception as e:
                    results[prompt] = f"Error: {e}"
                    self.log(f"  {prompt} -> Error: {e}")
        
        return results
    
    def train(self) -> Dict[str, any]:
        """Run the complete training process
        
        Returns:
            Dictionary with training results and statistics
        """
        start_time = time.time()
        
        self.log(f"=== CalcGPT Training ===")
        self.log(f"Device: {self.device} (fp16: {self.use_fp16})")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Create datasets
        train_dataset, val_dataset = self.create_datasets()
        
        # Create model
        self.model = self.create_model()
        
        # Setup trainer
        self.trainer = self.setup_trainer(train_dataset, val_dataset)
        
        # Start training
        self.log("\n=== STARTING TRAINING ===")
        training_result = self.trainer.train()
        
        training_time = time.time() - start_time
        
        self.log("\n=== TRAINING COMPLETED ===")
        self.log(f"Final training loss: {training_result.training_loss:.4f}")
        self.log(f"Training time: {training_time/60:.1f} minutes")
        
        # Final evaluation
        eval_loss = None
        if val_dataset:
            eval_results = self.trainer.evaluate()
            eval_loss = eval_results['eval_loss']
            self.log(f"Final validation loss: {eval_loss:.4f}")
        
        # Quick test
        test_results = self.run_quick_test()
        
        self.log(f"\nModel saved to: {self.output_dir}")
        
        # Return training statistics
        return {
            'training_loss': training_result.training_loss,
            'eval_loss': eval_loss,
            'training_time': training_time,
            'vocab_size': len(self.vocab),
            'dataset_size': len(self.examples),
            'test_results': test_results,
            'model_params': sum(p.numel() for p in self.model.parameters())
        } 