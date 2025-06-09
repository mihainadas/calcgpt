# CalcGPT: Transformer-Based Arithmetic Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**CalcGPT** is a comprehensive framework for building, training, and deploying transformer-based language models specialized in arithmetic operations. It demonstrates how to create domain-specific language models from scratch using modern deep learning techniques.

## 🌟 Features

### 🛠️ **Dual Interface Design**
- **📚 Python Library** (`lib/`): Professional programmatic API for integration
- **🖥️ CLI Tools**: User-friendly command-line interfaces for interactive usage

### 🧮 **Complete ML Pipeline**
- **Dataset Generation**: Intelligent arithmetic dataset creation with parameter encoding
- **Dual Tokenization**: Character-level and number-level (0-99) tokenization modes
- **Model Training**: Advanced transformer training with automatic naming conventions
- **Model Evaluation**: Comprehensive assessment across multiple test types
- **Production Inference**: High-performance model serving and batch processing
- **Comprehensive Logging**: High-traceability logging system for debugging and monitoring

### 🏗️ **Professional Architecture**
- **Modular Design**: Clean separation of concerns with reusable components
- **Configuration Management**: Type-safe dataclass configurations
- **Error Handling**: Robust error handling and validation throughout
- **Documentation**: Comprehensive inline documentation and examples

### 📊 **Advanced Features**
- **Dual Tokenization**: Character-level and number-level (0-99) tokenization modes
- **High-Traceability Logging**: Component-specific logs with timestamps, thread IDs, and performance monitoring
- **Data Augmentation**: Automatic commutative property expansion
- **Intelligent Naming**: Models auto-named with architecture and training parameters
- **Multi-format Output**: Support for JSON, plain text, and structured outputs
- **Device Optimization**: Automatic GPU/MPS/CPU detection and optimization

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/calcgpt.git
cd calcgpt

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 30-Second Demo

```bash
# 1. Generate a dataset
python calcgpt_dategen.py -m 10 --max-expressions 100

# 2. Train a model
python calcgpt_train.py --epochs 5 --verbose

# 3. Test the model
python calcgpt.py -i

# 4. Check logs for detailed traceability
ls logs/  # calcgpt.log, train.log, etc.
```

## 📖 Usage Guide

### 🖥️ CLI Tools

#### Dataset Generation
```bash
# Basic dataset (0-10, addition/subtraction)
python calcgpt_dategen.py -m 10

# Large dataset (0-100, all operations)
python calcgpt_dategen.py -m 100 --verbose

# Custom dataset (0-50, addition only, limited)
python calcgpt_dategen.py -m 50 --no-subtraction --max-expressions 1000
```

#### Model Training
```bash
# Quick training with defaults
python calcgpt_train.py --epochs 10

# Production training with custom architecture
python calcgpt_train.py \
    --embedding-dim 256 \
    --num-layers 8 \
    --num-heads 16 \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 1e-4

# Training with validation and checkpoints
python calcgpt_train.py \
    --epochs 100 \
    --test-split 0.2 \
    --save-steps 500 \
    --verbose
```

#### Model Evaluation
```bash
# Quick evaluation
python calcgpt_eval.py --sample 100

# Comprehensive evaluation
python calcgpt_eval.py \
    --sample 1000 \
    --max-tokens 20 \
    --verbose

# Evaluate specific model
python calcgpt_eval.py \
    -m models/calcgpt_emb128_lay6_head8_ep50_bs16_lr1e4_ds15k \
    --dataset datasets/test_set.txt
```

#### Interactive Inference
```bash
# Interactive mode
python calcgpt.py -i

# Batch processing
python calcgpt.py -b "25+25" "100-33" "67+12"

# File processing with JSON output
python calcgpt.py -f problems.txt -o results.json --format json

# Custom model and parameters
python calcgpt.py \
    -m models/my_model \
    --temperature 0.0 \
    --max-tokens 15 \
    -b "99+1" "50-25"

# Note: Tokenization mode is determined by the trained model
# Use character mode for learning, number mode for production
```

### 📚 Python Library

#### Dataset Generation
```python
from lib import DatasetGenerator, DatagenConfig

# Create configuration
config = DatagenConfig(
    max_value=100,
    operations=['addition', 'subtraction'],
    max_expressions=10000,
    verbose=True
)

# Generate dataset
generator = DatasetGenerator(config)
dataset_path = generator.generate()

# Analyze dataset
dataset = generator.load_dataset(dataset_path)
analysis = generator.analyze_dataset(dataset)
print(f"Generated {len(dataset)} examples")
print(f"Vocabulary: {analysis['vocabulary']}")
```

#### Tokenization Modes
```python
from lib import CalcGPTTokenizer

# Character-level tokenization (default)
examples = ['1+1=2', '12+34=46', '99-50=49']
char_tokenizer = CalcGPTTokenizer(examples, mode='char')
print(f"Character mode - Vocab size: {char_tokenizer.vocab_size}")

# Number-level tokenization (0-99 as single tokens)
num_tokenizer = CalcGPTTokenizer(examples, mode='number')
print(f"Number mode - Vocab size: {num_tokenizer.vocab_size}")

# Compare tokenization
text = "12+34=46"
char_tokens = char_tokenizer.encode(text)  # [1,2,+,3,4,=,4,6] - 8 tokens
num_tokens = num_tokenizer.encode(text)    # [12,+,34,=,46] - 5 tokens

# Load from dataset with mode selection
tokenizer = CalcGPTTokenizer.from_dataset(mode='number')
info = tokenizer.get_vocab_info()
print(f"Mode: {info['mode']}, Numbers: {info['numbers_count']}")
```

#### Model Training
```python
from lib import CalcGPTTrainer, TrainingConfig
from pathlib import Path

# Training configuration
config = TrainingConfig(
    epochs=20,
    batch_size=8,
    learning_rate=1e-3,
    embedding_dim=128,
    num_layers=6,
    num_heads=8,
    test_split=0.2,
    verbose=True
)

# Train model
trainer = CalcGPTTrainer(
    config=config,
    dataset_path="datasets/my_dataset.txt",
    output_dir=Path("models/my_calcgpt"),
    verbose=True
)

results = trainer.train()
print(f"Final loss: {results['training_loss']:.4f}")
print(f"Model parameters: {results['model_params']:,}")
```

#### Model Evaluation
```python
from lib import CalcGPTEvaluator, EvaluationConfig

# Evaluation configuration
config = EvaluationConfig(
    sample_size=500,
    max_tokens=15,
    verbose=True
)

# Evaluate model
evaluator = CalcGPTEvaluator(
    config=config,
    model_path="models/my_calcgpt",
    dataset_path="datasets/test_set.txt"
)

results = evaluator.evaluate()
print(f"Overall accuracy: {results['accuracy_stats']['overall']:.1%}")
print(f"Arithmetic correctness: {results['accuracy_stats']['arithmetic']:.1%}")
```

#### Model Inference
```python
from lib import CalcGPT, InferenceConfig

# Inference configuration
config = InferenceConfig(
    temperature=0.0,
    max_tokens=10,
    verbose=False
)

# Load model
model = CalcGPT(
    config=config,
    model_path="models/my_calcgpt"
)

# Generate predictions
result = model.generate("25+25=")
print(f"Prediction: {result['completion']}")

# Batch processing
problems = ["10+5=", "20-7=", "99+1="]
for problem in problems:
    result = model.generate(problem)
    print(f"{problem} -> {result['completion']}")
```

#### Comprehensive Logging
```python
from lib.logger import setup_logging, get_logger, log_step, log_metric, log_performance

# Setup logging system
setup_logging(
    logs_dir="logs",
    console_level="INFO",  # Console output level
    file_level="DEBUG"     # File output level (more detailed)
)

# Get component-specific loggers
train_logger = get_logger('train')
inference_logger = get_logger('inference')

# Basic logging
train_logger.info("Starting training process")
inference_logger.warning("Model accuracy below threshold")

# Structured logging with convenience functions
log_step("Epoch 1 completed", 'train')
log_metric("accuracy", 0.95, 'train')

# Performance monitoring with decorators
@log_performance('model_training', 'train')
def train_model():
    # Training code here
    return {"loss": 0.25}

# Function tracing
@log_function('inference', log_args=True, log_result=True)
def predict(input_data):
    return f"prediction for {input_data}"

# Automatic component-specific log files:
# - logs/calcgpt.log (main log)
# - logs/train.log (training-specific)
# - logs/inference.log (inference-specific)
```

## 🏗️ Architecture Overview

### Project Structure
```
calcgpt/
├── lib/                           # Core library package
│   ├── __init__.py               # Unified exports
│   ├── datagen.py               # Dataset generation
│   ├── tokenizer.py             # Dual-mode tokenization system
│   ├── train.py                 # Model training
│   ├── inference.py             # Model inference
│   ├── evaluation.py            # Model evaluation
│   ├── logger.py                # Comprehensive logging system
│   └── README.md                # Library documentation
├── examples/                    # Example scripts
│   └── complete_workflow.py    # Complete end-to-end example
├── calcgpt_dategen.py           # Dataset generation CLI
├── calcgpt_train.py             # Model training CLI
├── calcgpt_eval.py              # Model evaluation CLI
├── calcgpt.py                   # Interactive inference CLI
├── calcgpt.ipynb               # Comprehensive tutorial notebook
├── datasets/                   # Generated datasets
├── models/                     # Trained models
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### Core Components

#### 🎯 **DatasetGenerator**
- Generates systematic arithmetic datasets
- Supports multiple operations (addition, subtraction)
- Intelligent filename encoding with parameters
- Built-in data augmentation (commutative property)
- Comprehensive dataset analysis

#### 🔤 **CalcGPTTokenizer**
- Dual tokenization modes: character-level and number-level (0-99)
- Character mode: Individual characters as tokens (efficient vocab)
- Number mode: Whole numbers as tokens (semantic understanding)
- Automatic mode selection and vocabulary optimization
- Simplified, focused API for arithmetic expressions

#### 🏋️ **CalcGPTTrainer**
- Advanced transformer model training
- Automatic architecture optimization
- Intelligent model naming based on configuration
- Built-in validation and checkpointing
- Comprehensive training metrics and testing

#### 🔍 **CalcGPTEvaluator**
- Multi-dimensional model assessment
- Three test types: first_operand, expression_complete, answer_complete
- Format validation and arithmetic correctness checking
- Performance timing analysis
- Detailed statistical reporting

#### 🚀 **CalcGPT**
- High-performance model inference
- Temperature-controlled generation
- Batch processing capabilities
- Multiple output formats
- Production-ready error handling

#### 📝 **CalcGPTLogger**
- Comprehensive logging system with high traceability
- Component-specific log files (train.log, inference.log, etc.)
- Colored console output with different levels
- Detailed file logging with timestamps, thread IDs, and module info
- Performance monitoring decorators and convenience functions
- Automatic log rotation and configurable levels

## 📊 Examples & Tutorials

### 🎓 **Interactive Tutorial**
The `calcgpt.ipynb` notebook provides a comprehensive, step-by-step tutorial covering:

- **Transformer Architecture**: Understanding GPT-2 models and attention mechanisms
- **Dataset Engineering**: Creating and analyzing training datasets
- **Model Training**: From tiny models (38K params) to production (1.2M+ params)
- **Evaluation Methodologies**: Comprehensive model assessment
- **Production Deployment**: Real-world inference and usage patterns
- **Library Integration**: Using both programmatic and CLI interfaces

```bash
# Launch the tutorial
jupyter notebook calcgpt.ipynb
```

### 💡 **Example Scripts**
Explore the `examples/` directory for practical usage demonstrations:

```bash
# Run complete end-to-end workflow example
python examples/complete_workflow.py
```

This comprehensive example demonstrates:
- Dataset generation with custom configurations
- Model training with validation
- Model evaluation with detailed metrics  
- Interactive inference and testing
- Complete workflow from data to deployment

### 🎮 **End-to-End Workflow**
```python
from lib import *
from pathlib import Path

# 0. Setup logging (optional but recommended)
setup_logging(console_level="INFO", file_level="DEBUG")

# 1. Generate dataset
dataset_config = DatagenConfig(max_value=50, max_expressions=5000)
generator = DatasetGenerator(dataset_config)
dataset_path = generator.generate()

# 2. Train model with number-level tokenization
train_config = TrainingConfig(epochs=20, embedding_dim=128, num_layers=4)
trainer = CalcGPTTrainer(train_config, dataset_path, Path("models/demo"))
results = trainer.train()

# 3. Evaluate model
eval_config = EvaluationConfig(sample_size=200)
evaluator = CalcGPTEvaluator(eval_config, "models/demo", dataset_path)
eval_results = evaluator.evaluate()

# 4. Use for inference
inference_config = InferenceConfig(temperature=0.0)
model = CalcGPT(inference_config, "models/demo")
prediction = model.generate("25+25=")
print(f"25+25 = {prediction['completion']}")

# 5. Check logs for detailed traceability
# See logs/calcgpt.log, logs/train.log, logs/inference.log
```

## 🔧 Advanced Configuration

### Tokenization Mode Selection
```python
# Character-level tokenization (default) - smaller vocab, longer sequences
CalcGPTTokenizer(examples, mode='char')     # ~15 tokens vocab
CalcGPTTokenizer.from_dataset(mode='char')

# Number-level tokenization - larger vocab, shorter sequences  
CalcGPTTokenizer(examples, mode='number')   # ~105 tokens vocab
CalcGPTTokenizer.from_dataset(mode='number')

# Performance comparison for "12+34=46":
# Character mode: 8 tokens [1,2,+,3,4,=,4,6] 
# Number mode:   5 tokens [12,+,34,=,46]
```

### Model Architecture Options
```python
TrainingConfig(
    embedding_dim=256,      # Embedding dimension [32, 64, 128, 256, 512]
    num_layers=8,           # Number of transformer layers [1-12]
    num_heads=16,           # Number of attention heads [1-16]
    feedforward_dim=1024,   # Feedforward network dimension
    # embedding_dim must be divisible by num_heads
)
```

### Training Hyperparameters
```python
TrainingConfig(
    epochs=50,              # Training epochs
    batch_size=16,          # Training batch size
    learning_rate=1e-4,     # Learning rate
    weight_decay=0.01,      # L2 regularization
    warmup_steps=100,       # Learning rate warmup
    test_split=0.2,         # Validation split ratio
    save_steps=1000,        # Checkpoint frequency
)
```

### Dataset Configuration
```python
DatagenConfig(
    min_value=0,                              # Minimum operand value
    max_value=100,                            # Maximum operand value
    operations=['addition', 'subtraction'],   # Operations to include
    max_expressions=10000,                    # Maximum number of expressions
    allowed_digits='all',                     # Digit constraints
    verbose=True                              # Progress reporting
)
```

### Logging Configuration
```python
# Basic setup
setup_logging()  # Uses defaults: INFO console, DEBUG file

# Custom setup
setup_logging(
    logs_dir="custom_logs",           # Log directory
    console_level="DEBUG",            # Console verbosity
    file_level="DEBUG"                # File verbosity
)

# Component-specific logging
train_logger = get_logger('train')      # Creates logs/train.log
inference_logger = get_logger('inference')  # Creates logs/inference.log

# Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
# File features: automatic rotation (10MB), 5 backups, UTF-8 encoding
# Console features: colored output, timestamps, function:line info
```

## 📈 Performance & Benchmarks

### Model Performance by Architecture

| Architecture | Parameters | Training Time | Accuracy | Use Case |
|-------------|------------|---------------|----------|----------|
| Tiny (32d, 1L, 2H) | 38K | 30 seconds | 60-80% | Learning & prototyping |
| Small (64d, 3L, 4H) | 180K | 2 minutes | 80-90% | Development & testing |
| Medium (128d, 6L, 8H) | 1.2M | 10 minutes | 90-95% | Production ready |
| Large (256d, 8L, 16H) | 4.8M | 30 minutes | 95-98% | High accuracy needs |

### Evaluation Metrics

- **Format Validity**: Does the output follow `number+number=result` format?
- **Arithmetic Correctness**: Is the mathematical result correct?
- **Complete Expressions**: Does the model generate complete, valid expressions?
- **Inference Speed**: Average time per prediction (typically 10-50ms)
- **Tokenization Efficiency**: Character vs number mode sequence length impact

### Scaling Guidelines

- **For learning**: Start with tiny models (38K parameters) + character tokenization
- **For development**: Use small to medium models (180K-1.2M parameters)
- **For production**: Medium to large models (1.2M-4.8M parameters) + number tokenization
- **For research**: Large models with custom architectures + experiment with tokenization modes

### Tokenization Mode Guidelines

- **Character mode**: Better for learning transformer mechanics, smaller vocabulary
- **Number mode**: Better for arithmetic understanding, more efficient sequences
- **Experimentation**: Compare both modes for your specific use case and data range

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/yourusername/calcgpt.git
cd calcgpt
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Additional dev dependencies
```

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints for all functions
- Include comprehensive docstrings
- Write unit tests for new functionality
- Update documentation for API changes

### Testing
```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/ --integration

# Test CLI tools
python tests/test_cli.py
```

### Pull Request Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HuggingFace Transformers**: For the excellent transformer library
- **PyTorch**: For the deep learning framework
- **OpenAI**: For the original GPT architecture inspiration
- **The Open Source Community**: For continuous inspiration and support

## 📚 Citation

If you use CalcGPT in your research, please cite:

```bibtex
@software{calcgpt2024,
  title={CalcGPT: Transformer-Based Arithmetic Language Models},
  author={Mihai NADAS},
  year={2025},
  url={https://github.com/mihainadas/calcgpt}
}
```

## 🔗 Related Projects

- [GPT-2](https://github.com/openai/gpt-2) - Original GPT-2 implementation
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - Transformer library
- [PyTorch](https://github.com/pytorch/pytorch) - Deep learning framework

---

**Built with ❤️ for the AI/ML community**

*For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/yourusername/calcgpt) or open an issue.* 
