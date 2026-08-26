"""
CalcGPT Training Library

Core training functionality for CalcGPT models with proper separation of concerns.
"""

import hashlib
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .data import augment_data, load_dataset, split_examples_grouped
from .tokenizer import CalcGPTTokenizer
from .version import __version__


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
    n_positions: Optional[int] = None  # explicit context window; defaults to maxlen+10
    task_format: str = "plain"
    operand_width: Optional[int] = None


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch before model and dataset construction."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

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


class OptimizedDataset(Dataset):
    """Pre-tokenized dataset for faster training"""
    
    def __init__(self, data: List[str], maxlen: int, tokenizer: CalcGPTTokenizer):
        self.tokenizer = tokenizer
        self.data: List[Dict[str, List[int]]] = []
        
        for example in data:
            encoded = tokenizer.encode(example, add_eos=True)
            if len(encoded) > maxlen:
                raise ValueError(f"encoded example exceeds maxlen={maxlen}: {example!r}")
            padded = encoded + [tokenizer.pad_token_id] * (maxlen - len(encoded))
            attention_mask = [1] * len(encoded) + [0] * (maxlen - len(encoded))
            labels = encoded + [-100] * (maxlen - len(encoded))
            self.data.append(
                {
                    "input_ids": padded,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in self.data[i].items()
        }


def create_model_config(vocab_size: int, max_length: int, config: TrainingConfig) -> GPT2Config:
    """Create GPT2 model configuration"""
    n_positions = config.n_positions if config.n_positions else max_length + 10
    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=config.embedding_dim,
        n_layer=config.num_layers,
        n_head=config.num_heads,
        n_inner=config.feedforward_dim,
        pad_token_id=0,
        eos_token_id=1,
        use_cache=False
    )


def print_model_info(model: GPT2LMHeadModel) -> Dict[str, int]:
    """Return model statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 / 1024
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': model_size_mb
    }


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
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.train_examples: List[str] = []
        self.validation_examples: List[str] = []
        
    def log(self, message: str):
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def load_and_prepare_data(self):
        """Load and prepare training data"""
        self.log(f"Loading dataset from: {self.dataset_path}")
        self.examples = load_dataset(self.dataset_path)
        self.log(f"Loaded {len(self.examples)} examples")
        
        # Create tokenizer
        self.tokenizer = CalcGPTTokenizer(self.examples)
        self.log(f"Vocabulary created with {self.tokenizer.vocab_size} tokens")
        
        # Store max length
        self.maxlen = self.tokenizer.max_length
        self.log(f"Maximum sequence length: {self.maxlen}")
    
    def create_datasets(self) -> Tuple[OptimizedDataset, Optional[OptimizedDataset]]:
        """Create training and validation datasets
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        train_examples, val_examples = split_examples_grouped(
            self.examples, self.config.test_split, self.config.seed
        )
        if not self.config.no_augmentation:
            original_count = len(train_examples)
            train_examples = augment_data(train_examples)
            added = len(train_examples) - original_count
            self.log(f"Added {added} training-only augmented examples")
        else:
            self.log("Data augmentation disabled")

        self.train_examples = train_examples
        self.validation_examples = val_examples
        self.log(f"Training examples: {len(train_examples)}")
        self.log(f"Validation examples: {len(val_examples)}")
        train_dataset = OptimizedDataset(train_examples, self.maxlen, self.tokenizer)
        val_dataset = (
            OptimizedDataset(val_examples, self.maxlen, self.tokenizer)
            if val_examples
            else None
        )
        return train_dataset, val_dataset
    
    def create_model(self) -> GPT2LMHeadModel:
        """Create and configure the model
        
        Returns:
            Initialized GPT2LMHeadModel
        """
        self.log("Creating model...")
        model_config = create_model_config(self.tokenizer.vocab_size, self.maxlen, self.config)
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
            report_to=[],
            save_total_limit=2,
            save_safetensors=True,
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
        if self.config.task_format == "padded-reversed":
            width = self.config.operand_width or 1
            test_prompts = [
                f"{left:0{width}d}+{right:0{width}d}="
                for left, right in ((1, 1), (2, 3), (5, 0))
            ]
        else:
            test_prompts = ["1+1=", "2+3=", "5+0="]
        quick_test_tokens = (
            (self.config.operand_width or 1) + 1
            if self.config.task_format == "padded-reversed"
            else 5
        )
        results = {}
        
        with torch.no_grad():
            for prompt in test_prompts:
                try:
                    input_tokens = self.tokenizer.encode(prompt, add_eos=False)
                    input_ids = torch.tensor([input_tokens]).to(self.device)
                    
                    output = self.model.generate(
                        input_ids,
                        max_new_tokens=quick_test_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    result_tokens = output[0].tolist()
                    result = self.tokenizer.decode(result_tokens)
                    
                    results[prompt] = result
                    self.log(f"  {prompt} -> {result}")
                    
                except Exception as e:
                    results[prompt] = f"Error: {e}"
                    self.log(f"  {prompt} -> Error: {e}")
        
        return results
    
    def _write_training_manifest(
        self, training_time: float, training_loss: float, eval_loss: Optional[float]
    ) -> None:
        """Write provenance required to reproduce and audit a model artifact."""
        dataset_bytes = self.dataset_path.read_bytes()
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = None

        def package_version(name: str) -> Optional[str]:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                return None

        def examples_hash(examples: List[str]) -> str:
            payload = ("\n".join(examples) + "\n").encode("utf-8")
            return hashlib.sha256(payload).hexdigest()

        task_spec = {
            "schema_version": 1,
            "format": self.config.task_format,
            "operand_width": self.config.operand_width,
            "answer_order": (
                "reversed"
                if self.config.task_format == "padded-reversed"
                else "normal"
            ),
            "answer_width": (
                self.config.operand_width + 1
                if self.config.operand_width is not None
                else None
            ),
            "operators": ["+", "-"],
            "negative_results": False,
        }
        (self.output_dir / "task_spec.json").write_text(
            json.dumps(task_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        manifest = {
            "schema_version": 1,
            "calcgpt_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "dataset": {
                "path": str(self.dataset_path),
                "sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                "examples": len(self.examples),
            },
            "splits": {
                "train_examples": len(self.train_examples),
                "validation_examples": len(self.validation_examples),
                "train_sha256": examples_hash(self.train_examples),
                "validation_sha256": examples_hash(self.validation_examples),
                "grouped_by_commutative_equivalence": True,
            },
            "task_spec": task_spec,
            "training_config": asdict(self.config),
            "results": {
                "training_loss": training_loss,
                "validation_loss": eval_loss,
                "training_seconds": training_time,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "platform": platform.platform(),
                "device": self.device,
            },
        }
        manifest_path = self.output_dir / "training_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def train(self) -> Dict[str, Any]:
        """Run the complete training process
        
        Returns:
            Dictionary with training results and statistics
        """
        start_time = time.time()
        
        self.log("=== CalcGPT Training ===")
        self.log(f"Device: {self.device} (fp16: {self.use_fp16})")
        seed_everything(self.config.seed)
        
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
        
        # Persist the final model so it is always loadable, regardless of save_steps
        self.trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(self.output_dir)
        self._write_training_manifest(
            training_time, training_result.training_loss, eval_loss
        )

        # Quick test
        test_results = self.run_quick_test()

        self.log(f"\nModel saved to: {self.output_dir}")
        
        # Return training statistics
        return {
            'training_loss': training_result.training_loss,
            'eval_loss': eval_loss,
            'training_time': training_time,
            'vocab_size': self.tokenizer.vocab_size,
            'dataset_size': len(self.examples),
            'training_examples': len(self.train_examples),
            'validation_examples': len(self.validation_examples),
            'test_results': test_results,
            'model_params': sum(p.numel() for p in self.model.parameters())
        }
