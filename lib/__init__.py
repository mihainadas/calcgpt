"""
CalcGPT Training Library

Core training functionality for CalcGPT models.
"""

from .train import (
    CalcGPTTrainer,
    TrainingConfig,
    detect_device,
    load_dataset,
    augment_data,
    create_vocab,
    OptimizedDataset,
    create_model_config,
    print_model_info
)

from .dategen import (
    DatasetGenerator,
    DatagenConfig,
    contains_only_allowed_digits,
    generate_valid_numbers,
    generate_expressions,
    write_expressions_to_file,
    parse_digit_set,
    generate_output_filename,
    parse_filename_parameters,
    get_file_stats
)

__version__ = "1.0.0"
__all__ = [
    # Training exports
    "CalcGPTTrainer",
    "TrainingConfig",
    "detect_device", 
    "load_dataset",
    "augment_data",
    "create_vocab",
    "OptimizedDataset",
    "create_model_config",
    "print_model_info",
    # Dataset generation exports
    "DatasetGenerator",
    "DatagenConfig",
    "contains_only_allowed_digits",
    "generate_valid_numbers",
    "generate_expressions",
    "write_expressions_to_file",
    "parse_digit_set",
    "generate_output_filename",
    "parse_filename_parameters",
    "get_file_stats"
] 