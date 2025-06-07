#!/usr/bin/env python3
"""
Complete CalcGPT Workflow Example

This example demonstrates the entire CalcGPT pipeline:
1. Dataset generation
2. Model training  
3. Model evaluation
4. Interactive inference

Run from the main project directory:
    python examples/complete_workflow.py

Author: CalcGPT Team
Version: 1.0.0
"""

import sys
from pathlib import Path
import time

# Add the parent directory to Python path to import the lib module
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import CalcGPT library components
from lib import (
    DatasetGenerator, DatagenConfig,
    CalcGPTTrainer, TrainingConfig,
    CalcGPT, InferenceConfig,
    CalcGPTEvaluator, EvaluationConfig
)


def main():
    """
    Demonstrate the complete CalcGPT workflow.
    """
    print("🎬 Complete CalcGPT Workflow Example")
    print("=" * 50)
    
    # Step 1: Generate a dataset
    print("\n1️⃣ Generating Dataset")
    print("-" * 30)
    
    dataset_config = DatagenConfig(
        max_value=15,                          # Small range for quick demo
        operations=['addition', 'subtraction'], # Both operations
        max_expressions=100,                   # Reasonable size
        verbose=False                          # Quiet for cleaner output
    )
    
    generator = DatasetGenerator(dataset_config)
    dataset_path = generator.generate()
    
    # Analyze the dataset
    dataset = generator.load_dataset(dataset_path)
    analysis = generator.analyze_dataset(dataset)
    
    print(f"✅ Dataset generated: {len(dataset)} examples")
    print(f"   File: {Path(dataset_path).name}")
    print(f"   Vocabulary: {analysis['vocabulary']}")
    print(f"   Operations: {list(analysis['operations'].keys())}")
    
    # Step 2: Train a model
    print("\n2️⃣ Training Model")
    print("-" * 30)
    
    train_config = TrainingConfig(
        epochs=5,               # Quick training
        batch_size=4,           # Small batches
        embedding_dim=64,       # Modest size
        num_layers=2,           # Two layers
        num_heads=4,            # Four attention heads
        test_split=0.2,         # 20% validation
        verbose=False           # Quiet training
    )
    
    model_dir = PROJECT_ROOT / "models" / "example_workflow"
    trainer = CalcGPTTrainer(
        config=train_config,
        dataset_path=dataset_path,
        output_dir=model_dir,
        verbose=False
    )
    
    print("⏳ Training in progress...")
    start_time = time.time()
    results = trainer.train()
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.1f} seconds")
    print(f"   Model parameters: {results['model_params']:,}")
    print(f"   Final training loss: {results['training_loss']:.4f}")
    if results['eval_loss']:
        print(f"   Final validation loss: {results['eval_loss']:.4f}")
    
    # Step 3: Evaluate the model  
    print("\n3️⃣ Evaluating Model")
    print("-" * 30)
    
    eval_config = EvaluationConfig(
        sample_size=30,         # Test on 30 cases
        max_tokens=10,          # Allow up to 10 tokens
        verbose=False           # Quiet evaluation
    )
    
    evaluator = CalcGPTEvaluator(
        config=eval_config,
        model_path=str(model_dir),
        dataset_path=dataset_path,
        verbose=False
    )
    
    print("⏳ Evaluation in progress...")
    eval_results = evaluator.evaluate()
    
    print(f"✅ Evaluation completed")
    print(f"   Overall accuracy: {eval_results['accuracy_stats']['overall']:.1%}")
    print(f"   Arithmetic correctness: {eval_results['accuracy_stats']['arithmetic']:.1%}")
    print(f"   Format validity: {eval_results['accuracy_stats']['format']:.1%}")
    
    # Step 4: Interactive testing
    print("\n4️⃣ Interactive Testing")
    print("-" * 30)
    
    inference_config = InferenceConfig(
        temperature=0.0,        # Deterministic
        max_tokens=10,
        verbose=False
    )
    
    model = CalcGPT(
        config=inference_config,
        model_path=str(model_dir),
        verbose=False
    )
    
    # Test some problems
    test_problems = [
        "5+3=", "10-4=", "7+8=", "12-5=", 
        "6+6=", "15-3=", "4+9=", "11-2="
    ]
    
    print("Testing model on arithmetic problems:")
    print("\nProblem   → Prediction  Status")
    print("-" * 35)
    
    correct_count = 0
    for problem in test_problems:
        try:
            result = model.generate(problem)
            prediction = result['completion'].strip()
            
            # Calculate expected answer
            expr = problem.replace('=', '')
            if '+' in expr:
                operands = expr.split('+')
                expected = int(operands[0]) + int(operands[1])
            elif '-' in expr:
                operands = expr.split('-')
                expected = int(operands[0]) - int(operands[1])
            else:
                expected = None
            
            # Check correctness
            is_correct = str(prediction) == str(expected) if expected is not None else False
            status = "✅" if is_correct else "❌"
            if is_correct:
                correct_count += 1
            
            print(f"{problem:<9} → {prediction:<10} {status}")
            
        except Exception as e:
            print(f"{problem:<9} → ERROR      ❌")
    
    accuracy = (correct_count / len(test_problems)) * 100
    print(f"\nTest accuracy: {correct_count}/{len(test_problems)} ({accuracy:.1f}%)")
    
    # Summary
    print(f"\n🎉 Workflow Summary")
    print("=" * 50)
    print(f"✅ Dataset: {len(dataset)} examples generated")
    print(f"✅ Model: {results['model_params']:,} parameters trained")
    print(f"✅ Evaluation: {eval_results['accuracy_stats']['overall']:.1%} accuracy")
    print(f"✅ Testing: {accuracy:.1f}% on sample problems")
    print(f"\n📁 Files created:")
    print(f"   Dataset: {dataset_path}")
    print(f"   Model: {model_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Workflow interrupted by user")
    except Exception as e:
        print(f"❌ Error during workflow: {e}")
        import traceback
        traceback.print_exc() 