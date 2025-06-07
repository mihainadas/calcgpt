# CalcGPT Library

This library provides the core functionality for CalcGPT models including training and dataset generation with proper separation of concerns between CLI interfaces and core logic.

## Architecture

```
lib/
├── __init__.py          # Package initialization with exports
├── train.py             # Core training functionality
├── dategen.py           # Core dataset generation functionality
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
    # ... more parameters
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
Configuration dataclass for dataset generation parameters:

```python
from lib.dategen import DatagenConfig, parse_digit_set

config = DatagenConfig(
    max_value=100,
    min_value=0,
    allowed_digits=parse_digit_set("1-5"),  # Only digits 1-5
    include_addition=True,
    include_subtraction=True,
    max_expressions=1000,
    output_dir="datasets"
)
```

#### `DatasetGenerator`
Main dataset generation class:

```python
from lib.datagen import DatasetGenerator
from pathlib import Path

generator = DatasetGenerator(config, verbose=True)
results = generator.generate_dataset(Path("datasets/my_dataset.txt"))
```

### Utility Functions

#### Training Utilities
- `detect_device()` - Auto-detect best available device (CUDA/MPS/CPU)
- `load_dataset()` - Load and validate training dataset
- `augment_data()` - Apply commutative property data augmentation
- `create_vocab()` - Create optimized character-level vocabulary
- `OptimizedDataset` - Pre-tokenized dataset class for efficient training

#### Dataset Generation Utilities
- `parse_digit_set()` - Parse digit specifications like "1,2,3" or "1-5"
- `generate_expressions()` - Core expression generation algorithm
- `generate_output_filename()` - Auto-generate descriptive filenames
- `parse_filename_parameters()` - Extract parameters from auto-generated filenames
- `get_file_stats()` - Get file statistics (size, line count, etc.)

## Usage Patterns

### 1. Training Models (Programmatic)

```python
from lib.train import CalcGPTTrainer, TrainingConfig
from pathlib import Path

# Configure training
config = TrainingConfig(
    epochs=10,
    batch_size=4,
    embedding_dim=64,
    num_layers=3,
    num_heads=4
)

# Initialize trainer
trainer = CalcGPTTrainer(
    config=config,
    dataset_path=Path("datasets/ds-calcgpt.txt"),
    output_dir=Path("models/my_model"),
    verbose=True
)

# Train model
results = trainer.train()

# Access results
print(f"Final loss: {results['training_loss']:.4f}")
print(f"Model parameters: {results['model_params']:,}")
```

### 2. Generating Datasets (Programmatic)

```python
from lib.dategen import DatasetGenerator, DatagenConfig, parse_digit_set
from pathlib import Path

# Configure generation
config = DatagenConfig(
    max_value=20,
    min_value=1,
    allowed_digits=parse_digit_set("1,2,3"),
    include_addition=True,
    include_subtraction=False,
    max_expressions=100
)

# Generate dataset
generator = DatasetGenerator(config, verbose=True)
results = generator.generate_dataset(Path("datasets/custom.txt"))

# Access results
print(f"Generated: {results['expressions_generated']:,} expressions")
print(f"Time: {results['generation_time']:.2f} seconds")
```

### 3. CLI Usage

```bash
# Training
python calcgpt_train.py -e 50 -b 8 --embedding-dim 128

# Dataset generation  
python calcgpt_dategen.py -m 10 -d "1,2,3" --max-expressions 100
```

### 4. Jupyter Notebook Usage

```python
# Perfect for experimentation in notebooks
from lib.train import CalcGPTTrainer, TrainingConfig
from lib.dategen import DatasetGenerator, DatagenConfig

# Generate custom dataset
dategen_config = DatagenConfig(max_value=5, max_expressions=50)
generator = DatasetGenerator(dategen_config, verbose=False)
dataset_results = generator.generate_dataset()

# Train model on generated dataset
train_config = TrainingConfig(epochs=5, batch_size=2)
trainer = CalcGPTTrainer(train_config, dataset_results['output_path'], 
                        Path("models/notebook_model"), verbose=False)
train_results = trainer.train()

# Analyze results
print(f"Dataset: {dataset_results['expressions_generated']} expressions")
print(f"Training: {train_results['training_loss']:.4f} final loss")
```

## Design Benefits

### 🔀 **Separation of Concerns**
- **CLI Scripts**: Argument parsing, user interface, error handling
- **Library Modules**: Core algorithms, reusable components, business logic

### 📦 **Modularity**
- Use individual functions for specific tasks
- Compose components for custom workflows
- Easy to test and maintain each module independently

### 🔄 **Reusability**
- Import and use in other projects
- Embed in larger ML workflows
- Easy integration with notebooks and scripts
- Share common utilities between training and generation

### 🛠️ **Extensibility**
- Subclass main classes for custom behavior
- Override specific methods as needed
- Add new utility functions easily
- Extend configurations with new parameters

## Integration Examples

### End-to-End Pipeline
```python
from lib.dategen import DatasetGenerator, DatagenConfig
from lib.train import CalcGPTTrainer, TrainingConfig
from pathlib import Path

# 1. Generate dataset
dategen_config = DatagenConfig(
    max_value=50,
    allowed_digits=parse_digit_set("1-5"),
    max_expressions=1000
)
generator = DatasetGenerator(dategen_config)
dataset_results = generator.generate_dataset()

# 2. Train model
train_config = TrainingConfig(epochs=20, batch_size=8)
trainer = CalcGPTTrainer(
    train_config, 
    dataset_results['output_path'], 
    Path("models/pipeline_model")
)
train_results = trainer.train()

# 3. Results
print(f"Pipeline complete!")
print(f"Dataset: {dataset_results['expressions_generated']:,} expressions")
print(f"Model: {train_results['model_params']:,} parameters")
print(f"Training loss: {train_results['training_loss']:.4f}")
```

### Hyperparameter Search with Custom Datasets
```python
from lib.dategen import DatasetGenerator, DatagenConfig
from lib.train import CalcGPTTrainer, TrainingConfig

best_loss = float('inf')
best_config = None

# Test different model sizes on the same dataset
dataset = DatasetGenerator(DatagenConfig(max_value=10)).generate_dataset()

for emb_dim in [32, 64, 128]:
    for layers in [2, 4, 6]:
        config = TrainingConfig(
            embedding_dim=emb_dim, 
            num_layers=layers, 
            epochs=10
        )
        trainer = CalcGPTTrainer(config, dataset['output_path'], 
                               f"models/search_{emb_dim}_{layers}", verbose=False)
        results = trainer.train()
        
        if results['training_loss'] < best_loss:
            best_loss = results['training_loss']
            best_config = config

print(f"Best config: {best_config.embedding_dim}d, {best_config.num_layers}L")
print(f"Best loss: {best_loss:.4f}")
```

### Multi-Dataset Training Comparison
```python
from lib.dategen import DatasetGenerator, DatagenConfig, parse_digit_set
from lib.train import CalcGPTTrainer, TrainingConfig

# Generate different types of datasets
datasets = [
    ("simple", DatagenConfig(max_value=5)),
    ("constrained", DatagenConfig(max_value=10, allowed_digits=parse_digit_set("1,2"))),
    ("addition_only", DatagenConfig(max_value=20, include_subtraction=False)),
]

# Train models on each dataset
train_config = TrainingConfig(epochs=15, embedding_dim=64, num_layers=3)
results = []

for name, dategen_config in datasets:
    # Generate dataset
    generator = DatasetGenerator(dategen_config, verbose=False)
    dataset = generator.generate_dataset()
    
    # Train model
    trainer = CalcGPTTrainer(train_config, dataset['output_path'], 
                           f"models/comparison_{name}", verbose=False)
    result = trainer.train()
    
    results.append((name, dataset['expressions_generated'], result['training_loss']))

# Compare results
print("Dataset Comparison:")
for name, dataset_size, loss in results:
    print(f"  {name:12s}: {dataset_size:4,} examples → {loss:.4f} loss")
```

## API Reference

See docstrings in `train.py` and `dategen.py` for detailed API documentation of all classes and functions. 