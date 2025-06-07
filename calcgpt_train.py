#!/usr/bin/env python3
"""
CalcGPT Training Tool - CLI Interface

A command-line interface for training CalcGPT models with 
optimized architectures, data augmentation, and comprehensive evaluation.

Author: Mihai NADAS
Version: 2.0.0
"""

import argparse
import sys
from pathlib import Path

from lib.train import CalcGPTTrainer, TrainingConfig


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
    
    if args.test_split < 0 or args.test_split >= 1:
        errors.append("test-split must be between 0 and 1")
    
    if errors:
        print("❌ Validation Errors:")
        for error in errors:
            print(f"   • {error}")
        sys.exit(1)


def create_config_from_args(args) -> TrainingConfig:
    """Create TrainingConfig from command line arguments"""
    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feedforward_dim=args.feedforward_dim,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        test_split=args.test_split,
        seed=args.seed,
        no_augmentation=args.no_augmentation
    )


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="CalcGPT Trainer - Advanced arithmetic model training system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                          # Train with default settings
  %(prog)s -d datasets/large.txt -e 100            # Custom dataset, 100 epochs
  %(prog)s --embedding-dim 256 --num-layers 8      # Larger model architecture  
  %(prog)s --batch-size 16 --learning-rate 5e-4    # Custom training parameters
  %(prog)s -o models/calcgpt-v2 --verbose          # Custom output directory
  %(prog)s --no-augmentation                       # Disable data augmentation
        """
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
        default='models/calcgpt',
        help='Output directory for model checkpoints (default: models/calcgpt)'
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
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
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
        '--save-steps',
        type=int,
        default=5000,
        help='Model save frequency (default: 1000)'
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
    
    # Utility options
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='CalcGPT Trainer 2.0.0'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args)
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Setup paths
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    
    # Print configuration summary
    if not args.quiet:
        print("🚀 CalcGPT Training Configuration:")
        print(f"  📚 Dataset: {dataset_path}")
        print(f"  📁 Output: {output_dir}")
        print(f"  🏗️  Architecture: {config.embedding_dim}d, {config.num_layers}L, {config.num_heads}H")
        print(f"  📖 Training: {config.epochs} epochs, batch {config.batch_size}, lr {config.learning_rate}")
        print(f"  🔧 Data augmentation: {'Disabled' if config.no_augmentation else 'Enabled'}")
        print(f"  📊 Test split: {config.test_split:.1%}")
        print()
    
    try:
        # Initialize trainer
        trainer = CalcGPTTrainer(
            config=config,
            dataset_path=dataset_path,
            output_dir=output_dir,
            verbose=not args.quiet
        )
        
        # Run training
        results = trainer.train()
        
        # Print final summary
        if not args.quiet:
            print("\n🎉 Training Summary:")
            print(f"  📊 Final loss: {results['training_loss']:.4f}")
            if results['eval_loss']:
                print(f"  📉 Validation loss: {results['eval_loss']:.4f}")
            print(f"  ⏱️  Training time: {results['training_time']/60:.1f} minutes")
            print(f"  🧠 Model parameters: {results['model_params']:,}")
            print(f"  📚 Dataset size: {results['dataset_size']:,} examples")
            print(f"  🔤 Vocabulary size: {results['vocab_size']} tokens")
            
            print("\n🧪 Quick Test Results:")
            for prompt, result in results['test_results'].items():
                print(f"  {prompt} -> {result}")
            
            print(f"\n✅ Model saved to: {output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Training error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())