# CalcGPT Training Library

This library provides the core training functionality for CalcGPT models with proper separation of concerns between CLI interface and training logic.

## Architecture

```
lib/
├── __init__.py          # Package initialization with exports
└── train.py             # Core training functionality
```

## Core Components

### `TrainingConfig`
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

### `CalcGPTTrainer`
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

### Utility Functions

- `detect_device()` - Auto-detect best available device (CUDA/MPS/CPU)
- `load_dataset()` - Load and validate training dataset
- `augment_data()` - Apply commutative property data augmentation
- `create_vocab()` - Create optimized character-level vocabulary
- `OptimizedDataset` - Pre-tokenized dataset class for efficient training

## Usage Patterns

### 1. Programmatic Usage (Recommended)

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

### 2. CLI Usage

```bash
# Use the CLI interface (calcgpt_train.py)
python calcgpt_train.py -e 50 -b 8 --embedding-dim 128
```

### 3. Jupyter Notebook Usage

```python
# Perfect for experimentation in notebooks
from lib.train import CalcGPTTrainer, TrainingConfig

config = TrainingConfig(epochs=5, batch_size=2)  # Quick training
trainer = CalcGPTTrainer(config, dataset_path, output_dir, verbose=False)
results = trainer.train()

# Analyze results
import matplotlib.pyplot as plt
# ... visualization code
```

## Design Benefits

### 🔀 **Separation of Concerns**
- **CLI** (`calcgpt_train.py`): Argument parsing, user interface
- **Library** (`lib/train.py`): Core training logic, reusable components

### 📦 **Modularity**
- Use individual functions for specific tasks
- Compose components for custom training pipelines
- Easy to test and maintain

### 🔄 **Reusability**
- Import and use in other projects
- Embed in larger ML workflows
- Easy integration with notebooks and scripts

### 🛠️ **Extensibility**
- Subclass `CalcGPTTrainer` for custom behavior
- Override specific methods as needed
- Add new utility functions easily

## Integration Examples

### Custom Training Loop
```python
from lib.train import load_dataset, create_vocab, OptimizedDataset, create_model_config

# Use individual components
examples = load_dataset(dataset_path)
vocab, id2char = create_vocab(examples)
dataset = OptimizedDataset(examples, maxlen, vocab)
config = create_model_config(len(vocab), maxlen, training_config)

# Custom training logic here...
```

### Hyperparameter Search
```python
from lib.train import CalcGPTTrainer, TrainingConfig

best_config = None
best_loss = float('inf')

for lr in [1e-4, 5e-4, 1e-3]:
    for layers in [3, 6, 9]:
        config = TrainingConfig(learning_rate=lr, num_layers=layers, epochs=10)
        trainer = CalcGPTTrainer(config, dataset_path, output_dir, verbose=False)
        results = trainer.train()
        
        if results['training_loss'] < best_loss:
            best_loss = results['training_loss']
            best_config = config
```

### Model Evaluation Pipeline
```python
from lib.train import CalcGPTTrainer, TrainingConfig

# Train multiple models with different configurations
configs = [
    TrainingConfig(embedding_dim=64, num_layers=3),
    TrainingConfig(embedding_dim=128, num_layers=6),
    TrainingConfig(embedding_dim=256, num_layers=9),
]

results = []
for i, config in enumerate(configs):
    trainer = CalcGPTTrainer(config, dataset_path, f"models/variant_{i}")
    result = trainer.train()
    results.append((config, result))

# Compare results
for config, result in results:
    print(f"Model {config.embedding_dim}d/{config.num_layers}L: "
          f"Loss {result['training_loss']:.4f}, "
          f"Params {result['model_params']:,}")
```

## API Reference

See docstrings in `train.py` for detailed API documentation of all classes and functions. 