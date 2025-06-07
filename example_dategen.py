#!/usr/bin/env python3
"""
Example: Using CalcGPT Dataset Generation Library Directly

This demonstrates how to use the lib.dategen module programmatically
without going through the command-line interface.
"""

from pathlib import Path
from lib.dategen import DatasetGenerator, DatagenConfig, parse_digit_set

def main():
    """Main example demonstrating programmatic dataset generation."""
    print("🚀 CalcGPT Dataset Generation Examples")
    print("=" * 50)
    
    # Example 1: Simple dataset
    print("\n📊 Example 1: Simple Dataset (0-10, all operations)")
    
    config1 = DatagenConfig(
        max_value=10,
        min_value=0,
        allowed_digits=None,  # All digits allowed
        include_addition=True,
        include_subtraction=True,
        max_expressions=None,  # No limit
        output_dir="examples"
    )
    
    generator1 = DatasetGenerator(config1, verbose=True)
    results1 = generator1.generate_dataset(Path("examples/simple_dataset.txt"))
    
    print(f"✅ Generated {results1['expressions_generated']:,} expressions")
    print(f"📁 Saved to: {results1['output_path']}")
    print(f"⏱️  Time: {results1['generation_time']:.2f} seconds")
    
    # Example 2: Constrained digits dataset
    print("\n📊 Example 2: Constrained Digits (0-20, digits 1-3 only)")
    
    config2 = DatagenConfig(
        max_value=20,
        min_value=1,
        allowed_digits=parse_digit_set("1,2,3"),  # Only digits 1, 2, 3
        include_addition=True,
        include_subtraction=True,
        max_expressions=100,  # Limit to 100 expressions
        output_dir="examples"
    )
    
    generator2 = DatasetGenerator(config2, verbose=True)
    results2 = generator2.generate_dataset(Path("examples/constrained_dataset.txt"))
    
    print(f"✅ Generated {results2['expressions_generated']:,} expressions")
    print(f"📁 Saved to: {results2['output_path']}")
    print(f"⏱️  Time: {results2['generation_time']:.2f} seconds")
    
    # Example 3: Addition only with auto-generated filename
    print("\n📊 Example 3: Addition Only (auto-generated filename)")
    
    config3 = DatagenConfig(
        max_value=5,
        min_value=0,
        allowed_digits=None,
        include_addition=True,
        include_subtraction=False,  # Addition only
        max_expressions=50,
        output_dir="examples"
    )
    
    generator3 = DatasetGenerator(config3, verbose=False)  # Silent generation
    results3 = generator3.generate_dataset()  # Auto-generate filename
    
    print(f"✅ Generated {results3['expressions_generated']:,} expressions")
    print(f"📁 Auto-generated filename: {results3['output_path'].name}")
    print(f"⏱️  Time: {results3['generation_time']:.2f} seconds")
    
    # Example 4: Performance estimation
    print("\n📊 Example 4: Performance Estimation")
    
    config4 = DatagenConfig(
        max_value=100,
        min_value=0,
        allowed_digits=None,
        include_addition=True,
        include_subtraction=True,
        max_expressions=None,
        output_dir="examples"
    )
    
    generator4 = DatasetGenerator(config4, verbose=False)
    estimated = generator4.estimate_expressions()
    
    print(f"📈 Estimated expressions for 0-100 range: ~{estimated:,}")
    print(f"💾 Estimated file size: ~{estimated * 10} bytes (~{estimated * 10 / 1024:.1f} KB)")
    
    # Don't actually generate this large dataset in the example
    print("   (Skipping generation due to large size)")
    
    # Summary
    print(f"\n🎉 Summary:")
    print(f"  📊 Total examples generated: {results1['expressions_generated'] + results2['expressions_generated'] + results3['expressions_generated']:,}")
    print(f"  📁 Files created: 3")
    print(f"  📂 Output directory: examples/")
    
    # Show some sample expressions from the first dataset
    if results1['output_path'].exists():
        print(f"\n📝 Sample expressions from {results1['output_path'].name}:")
        with open(results1['output_path'], 'r') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Show first 5 expressions
                    break
                print(f"   {line.strip()}")
            if results1['expressions_generated'] > 5:
                print(f"   ... and {results1['expressions_generated'] - 5} more")

def quick_demo():
    """Quick demo with minimal dataset."""
    print("🎬 Quick Demo - Tiny Dataset")
    
    config = DatagenConfig(
        max_value=3,
        min_value=0,
        allowed_digits=None,
        include_addition=True,
        include_subtraction=True,
        max_expressions=None,
        output_dir="examples"
    )
    
    generator = DatasetGenerator(config, verbose=True)
    results = generator.generate_dataset(Path("examples/tiny_demo.txt"))
    
    print(f"\n📋 Complete dataset contents:")
    with open(results['output_path'], 'r') as f:
        for line_num, line in enumerate(f, 1):
            print(f"   {line_num:2d}. {line.strip()}")
    
    return results

def batch_generation_example():
    """Example showing how to generate multiple datasets programmatically."""
    print("\n🔄 Batch Generation Example")
    print("=" * 30)
    
    # Define multiple configurations
    configs = [
        ("tiny", DatagenConfig(max_value=2, include_subtraction=False)),
        ("small", DatagenConfig(max_value=5, allowed_digits=parse_digit_set("1,2"))),
        ("medium", DatagenConfig(max_value=10, max_expressions=50)),
    ]
    
    results = []
    for name, config in configs:
        print(f"\n📊 Generating {name} dataset...")
        generator = DatasetGenerator(config, verbose=False)
        result = generator.generate_dataset(Path(f"examples/batch_{name}.txt"))
        results.append((name, result))
        print(f"   ✅ {result['expressions_generated']:,} expressions in {result['generation_time']:.2f}s")
    
    print(f"\n📈 Batch Results:")
    for name, result in results:
        size_kb = result['file_stats']['file_size_kb'] if result['file_stats'] else 0
        print(f"   {name:6s}: {result['expressions_generated']:4,} expressions, {size_kb:5.1f} KB")

if __name__ == "__main__":
    import sys
    
    # Create examples directory
    Path("examples").mkdir(exist_ok=True)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_demo()
        elif sys.argv[1] == "batch":
            batch_generation_example()
        else:
            print("Options: quick, batch")
    else:
        main() 