"""
CalcGPT Training Library

Core training functionality for CalcGPT models.
"""

from .train import (
    CalcGPTTrainer,
    detect_device,
    load_dataset,
    augment_data,
    create_vocab,
    OptimizedDataset,
    create_model_config,
    print_model_info
)

__version__ = "1.0.0"
__all__ = [
    "CalcGPTTrainer",
    "detect_device", 
    "load_dataset",
    "augment_data",
    "create_vocab",
    "OptimizedDataset",
    "create_model_config",
    "print_model_info"
] 