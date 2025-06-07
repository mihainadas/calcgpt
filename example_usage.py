#!/usr/bin/env python3
"""
Example: Using CalcGPT Training Library Directly

This demonstrates how to use the lib.train module programmatically
without going through the command-line interface.
"""

from pathlib import Path
from lib.train import CalcGPTTrainer, TrainingConfig

def main():
    # Create custom training configuration
    config = TrainingConfig(
        epochs=10,                # Quick training for demo
        batch_size=4,
        learning_rate=1e-3,
        embedding_dim=64,         # Smaller model
        num_layers=3,
        num_heads=4,
        feedforward_dim=256,
        test_split=0.15,
        no_augmentation=False     # Enable data augmentation
    )
    
    # Define paths
    dataset_path = Path("datasets/ds-calcgpt.txt")
    output_dir = Path("models/example_model")
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please ensure you have a dataset file.")
        return
    
    print("🚀 Starting programmatic CalcGPT training...")
    print(f"📚 Using dataset: {dataset_path}")
    print(f"🏗️  Model: {config.embedding_dim}d, {config.num_layers}L, {config.num_heads}H")
    print(f"📖 Training: {config.epochs} epochs")
    
    try:
        # Initialize trainer
        trainer = CalcGPTTrainer(
            config=config,
            dataset_path=dataset_path,
            output_dir=output_dir,
            verbose=True
        )
        
        # Run training
        results = trainer.train()
        
        # Process results
        print("\n✅ Training completed successfully!")
        print("\n📊 Final Results:")
        print(f"  Training loss: {results['training_loss']:.4f}")
        if results['eval_loss']:
            print(f"  Validation loss: {results['eval_loss']:.4f}")
        print(f"  Training time: {results['training_time']:.1f} seconds")
        print(f"  Model parameters: {results['model_params']:,}")
        print(f"  Dataset size: {results['dataset_size']} examples")
        
        # Show test results
        print("\n🧪 Model Test Results:")
        for prompt, result in results['test_results'].items():
            status = "✅" if not result.startswith("Error") else "❌"
            print(f"  {status} {prompt} -> {result}")
        
        # Calculate accuracy for basic tests
        correct = sum(1 for r in results['test_results'].values() 
                     if not r.startswith("Error") and '=' in r)
        total = len(results['test_results'])
        accuracy = correct / total * 100
        print(f"\n🎯 Basic test accuracy: {accuracy:.1f}% ({correct}/{total})")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

def quick_demo():
    """Quick demo with minimal configuration"""
    print("🎬 Quick Demo - Minimal Training")
    
    config = TrainingConfig(
        epochs=3,
        batch_size=2,
        embedding_dim=32,
        num_layers=2,
        num_heads=2,
        feedforward_dim=128,
        test_split=0.0,  # No validation for quick demo
        save_steps=100
    )
    
    trainer = CalcGPTTrainer(
        config=config,
        dataset_path=Path("datasets/ds-calcgpt.txt"),
        output_dir=Path("models/quick_demo"),
        verbose=True
    )
    
    return trainer.train()

if __name__ == "__main__":
    # You can run either the full example or quick demo
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_demo()
    else:
        main() 