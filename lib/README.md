# CalcGPT Library

This library provides the complete functionality for CalcGPT models including training, dataset generation, inference, and evaluation with proper separation of concerns between CLI interfaces and core logic.

## Architecture

```
lib/
├── __init__.py          # Package initialization with exports
├── train.py             # Core training functionality
├── dategen.py           # Core dataset generation functionality
├── inference.py         # Core inference functionality
├── evaluation.py        # Core evaluation functionality
└── README.md            # This documentation
```

## Core Components

### Training Module (`train.py`)

#### `TrainingConfig`
Configuration dataclass that holds all training parameters:

```python
from lib.train import TrainingConfig

config = TrainingConfig(
    epochs=50,
    batch_size=8,
    learning_rate=1e-3,
    embedding_dim=128,
    num_layers=6,
    num_heads=8,
    feedforward_dim=512,
    test_split=0.2,
    no_augmentation=False
)
```

#### `CalcGPTTrainer`
Main training class that orchestrates the entire training process:

```python
from lib.train import CalcGPTTrainer
from pathlib import Path

trainer = CalcGPTTrainer(
    config=config,
    dataset_path=Path("datasets/ds-calcgpt.txt"),
    output_dir=Path("models/my_model"),
    verbose=True
)

results = trainer.train()
```

### Dataset Generation Module (`dategen.py`)

#### `DatagenConfig`
Configuration for dataset generation parameters:

```python
from lib.dategen import DatagenConfig

config = DatagenConfig(
    max_value=100,
    min_value=0,
    allowed_digits={'0', '1', '2', '3', '4', '5'},
    include_addition=True,
    include_subtraction=True,
    max_expressions=1000,
    output_dir="datasets"
)
```

#### `DatasetGenerator`
Core dataset generation class:

```python
from lib.dategen import DatasetGenerator

generator = DatasetGenerator(config, verbose=True)
dataset_path = generator.generate_dataset("custom_dataset.txt")
```

### Inference Module (`inference.py`)

#### `InferenceConfig`
Configuration for inference parameters:

```python
from lib.inference import InferenceConfig

config = InferenceConfig(
    temperature=0.1,     # Sampling temperature (0 = greedy)
    max_tokens=10,       # Maximum tokens to generate
    device='auto',       # Device selection ('auto', 'cuda', 'mps', 'cpu')
    show_tokens=False    # Show token details in results
)
```

#### `CalcGPT`
Main inference engine for solving arithmetic problems:

```python
from lib.inference import CalcGPT, get_model_path

# Auto-detect latest model or specify path
model_path = get_model_path('auto')  # or specify: "/path/to/model"

# Initialize inference engine
calcgpt = CalcGPT(model_path, config, verbose=True)

# Solve single problem
result = calcgpt.solve("1+1")
print(f"{result['problem']} {result['answer']}")

# Batch processing
problems = ["1+1", "2+3", "5-2"]
results = calcgpt.solve_batch(problems)

# Get model information
info = calcgpt.get_model_info()
```

### Evaluation Module (`evaluation.py`)

#### `EvaluationConfig`
Configuration for evaluation parameters:

```python
from lib.evaluation import EvaluationConfig

config = EvaluationConfig(
    max_tokens=15,              # Maximum tokens to generate
    device='auto',              # Device selection
    sample_size=None,           # Sample size (None = use all)
    verbose=False              # Enable verbose output
)
```

#### `CalcGPTEvaluator`
Comprehensive model evaluation engine:

```python
from lib.evaluation import CalcGPTEvaluator

# Initialize evaluator
evaluator = CalcGPTEvaluator(model_path, config, verbose=True)

# Evaluate complete dataset
results, metrics = evaluator.evaluate_dataset("datasets/test.txt")

# Evaluate custom test cases
test_cases = [
    {'input': '1+1=', 'expected': '1+1=2', 'type': 'answer_complete'},
    {'input': '5+3', 'expected': '5+3=8', 'type': 'expression_complete'}
]
results = evaluator.evaluate_test_cases(test_cases)

# Get detailed metrics
from lib.evaluation import calculate_metrics
metrics = calculate_metrics(results)
print(f"Accuracy: {metrics['correct_arithmetic_pct']:.1f}%")
```

## Utility Functions

### Training Utilities
```python
from lib.train import detect_device, load_dataset, create_vocab

device, has_cuda = detect_device()
train_data, test_data = load_dataset("dataset.txt", test_split=0.2)
vocab, id2char = create_vocab(train_data)
```

### Dataset Generation Utilities
```python
from lib.dategen import generate_expressions, parse_digit_set

expressions = generate_expressions(max_val=50, operations=['+', '-'])
allowed_digits = parse_digit_set("0,1,2,3")
```

### Inference Utilities
```python
from lib.inference import find_latest_model, validate_simple_arithmetic

# Find most recent model
latest_model = find_latest_model("models/")

# Validate arithmetic
is_correct = validate_simple_arithmetic("1+1=", "2")
```

### Evaluation Utilities
```python
from lib.evaluation import create_test_cases, validate_completion

# Create test cases from equations
equations = ["1+1=2", "5+3=8", "10-4=6"]
test_cases = create_test_cases(equations)

# Validate a completion
test_case = {'input': '1+1=', 'expected': '1+1=2', 'type': 'answer_complete'}
validation = validate_completion(test_case, "1+1=2")
```

## Example Usage

### Complete Training Pipeline
```python
from lib.train import CalcGPTTrainer, TrainingConfig
from lib.dategen import DatasetGenerator, DatagenConfig
from pathlib import Path

# 1. Generate dataset
datagen_config = DatagenConfig(max_value=100, max_expressions=10000)
generator = DatasetGenerator(datagen_config)
dataset_path = generator.generate_dataset("training_data.txt")

# 2. Train model
train_config = TrainingConfig(epochs=50, batch_size=8)
trainer = CalcGPTTrainer(
    config=train_config,
    dataset_path=dataset_path,
    output_dir=Path("models/my_calcgpt")
)
results = trainer.train()

# 3. Use trained model
from lib.inference import CalcGPT
calcgpt = CalcGPT("models/my_calcgpt")
result = calcgpt.solve("25+17")
print(f"Answer: {result['answer']}")
```

### Complete Evaluation Pipeline
```python
from lib.evaluation import CalcGPTEvaluator, EvaluationConfig
from lib.inference import get_model_path

# 1. Configure evaluation
config = EvaluationConfig(max_tokens=20, sample_size=1000)

# 2. Initialize evaluator
model_path = get_model_path('auto')
evaluator = CalcGPTEvaluator(model_path, config)

# 3. Run comprehensive evaluation
results, metrics = evaluator.evaluate_dataset("datasets/test.txt")

# 4. Analyze results
print(f"Overall Accuracy: {metrics['correct_arithmetic_pct']:.1f}%")
print(f"Valid Format: {metrics['valid_format_pct']:.1f}%")
print(f"Mean Inference Time: {metrics['timing']['mean_ms']:.1f}ms")

# Performance by test type
for test_type, stats in metrics['by_type'].items():
    accuracy = (stats['correct'] / stats['total']) * 100
    print(f"{test_type.title()}: {accuracy:.1f}%")
```

### End-to-End Workflow
```python
# Complete workflow: Generate → Train → Evaluate
from lib import *

# 1. Generate training data
generator = DatasetGenerator(DatagenConfig(max_value=50, max_expressions=5000))
train_dataset = generator.generate_dataset("train.txt")

# 2. Train model
trainer = CalcGPTTrainer(
    TrainingConfig(epochs=20, batch_size=8),
    train_dataset,
    Path("models/workflow_model")
)
train_results = trainer.train()

# 3. Generate test data
test_generator = DatasetGenerator(DatagenConfig(max_value=50, max_expressions=1000))
test_dataset = test_generator.generate_dataset("test.txt")

# 4. Evaluate model
evaluator = CalcGPTEvaluator("models/workflow_model", EvaluationConfig(max_tokens=15))
eval_results, metrics = evaluator.evaluate_dataset(test_dataset)

# 5. Summary
print(f"Training Loss: {train_results['training_loss']:.4f}")
print(f"Evaluation Accuracy: {metrics['correct_arithmetic_pct']:.1f}%")
print(f"Model Parameters: {train_results['model_params']:,}")
```

### Inference-Only Usage
```python
from lib.inference import CalcGPT, InferenceConfig, get_model_path

# Configure inference
config = InferenceConfig(temperature=0.0, max_tokens=15)

# Load model (auto-detect latest)
model_path = get_model_path('auto')
calcgpt = CalcGPT(model_path, config)

# Interactive solving
while True:
    problem = input("Problem: ")
    if problem == 'quit':
        break
    
    result = calcgpt.solve(problem)
    if 'error' not in result:
        status = "✅" if result['is_correct'] else "❓"
        print(f"{status} {result['answer']} ({result['inference_time']*1000:.1f}ms)")
    else:
        print(f"Error: {result['error']}")
```

## Error Handling

All library functions use proper exception handling:

```python
try:
    calcgpt = CalcGPT("invalid/path")
except FileNotFoundError as e:
    print(f"Model not found: {e}")
except RuntimeError as e:
    print(f"Model loading error: {e}")

try:
    evaluator = CalcGPTEvaluator(model_path, config)
    results, metrics = evaluator.evaluate_dataset("test.txt")
except Exception as e:
    print(f"Evaluation error: {e}")
```

## CLI Integration

The library is designed to work seamlessly with CLI tools:

```bash
# Training
python calcgpt_train.py -d datasets/my_data.txt -o models/my_model

# Dataset generation  
python calcgpt_dategen.py -m 100 --max-expressions 5000

# Inference
python calcgpt.py -m models/my_model -b "1+1" "2+3" "5-2"

# Evaluation
python calcgpt_eval.py -m models/my_model -d datasets/test.txt -o results.json
```

## Configuration Files

You can save and load configurations:

```python
import json
from lib.train import TrainingConfig
from lib.evaluation import EvaluationConfig

# Save configurations
train_config = TrainingConfig(epochs=100, batch_size=16)
eval_config = EvaluationConfig(max_tokens=20, sample_size=500)

with open("configs.json", "w") as f:
    json.dump({
        'training': train_config.__dict__,
        'evaluation': eval_config.__dict__
    }, f, indent=2)

# Load configurations
with open("configs.json", "r") as f:
    configs = json.load(f)
    train_config = TrainingConfig(**configs['training'])
    eval_config = EvaluationConfig(**configs['evaluation'])
```

## Development

### Adding New Features
1. Add core functionality to appropriate module (`train.py`, `dategen.py`, `inference.py`, `evaluation.py`)
2. Export new functions in `__init__.py`
3. Update CLI tools to use new functionality
4. Add examples and documentation

### Testing
```python
# Test imports
from lib import *

# Test core functionality
print("Library imported successfully!")

# Test each module
evaluator = CalcGPTEvaluator("models/test")
generator = DatasetGenerator(DatagenConfig())
trainer = CalcGPTTrainer(TrainingConfig(), "data.txt", "models/test")
calcgpt = CalcGPT("models/test")
```

## Performance Metrics

The evaluation module provides comprehensive metrics:

- **Accuracy Metrics**: Correct arithmetic, valid format, complete expressions
- **Timing Metrics**: Mean, median, min/max inference times with standard deviation
- **Test Type Analysis**: Performance breakdown by completion type
- **Error Analysis**: Detailed validation results for debugging

## Advanced Usage

### Custom Validation Functions
```python
from lib.evaluation import CalcGPTEvaluator

class CustomEvaluator(CalcGPTEvaluator):
    def custom_validate(self, result, expected):
        # Add custom validation logic
        return custom_validation_result

evaluator = CustomEvaluator(model_path, config)
```

### Batch Evaluation with Progress Tracking
```python
from lib.evaluation import EvaluationConfig

# Large-scale evaluation with sampling
config = EvaluationConfig(sample_size=10000, max_tokens=25)
evaluator = CalcGPTEvaluator(model_path, config, verbose=True)

results, metrics = evaluator.evaluate_dataset("large_dataset.txt")
print(f"Evaluated {metrics['total_tests']} test cases")
``` 