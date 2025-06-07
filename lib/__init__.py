"""
CalcGPT Library

Core functionality for CalcGPT models including training, dataset generation, inference, and evaluation.
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

from .inference import (
    CalcGPT,
    InferenceConfig,
    get_device,
    find_latest_model,
    get_model_path,
    load_vocabulary_from_dataset,
    validate_simple_arithmetic
)

from .evaluation import (
    CalcGPTEvaluator,
    EvaluationConfig,
    load_evaluation_dataset,
    create_test_cases,
    validate_completion,
    calculate_metrics
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
    "get_file_stats",
    # Inference exports
    "CalcGPT",
    "InferenceConfig",
    "get_device",
    "find_latest_model",
    "get_model_path",
    "load_vocabulary_from_dataset",
    "validate_simple_arithmetic",
    # Evaluation exports
    "CalcGPTEvaluator",
    "EvaluationConfig",
    "load_evaluation_dataset",
    "create_test_cases",
    "validate_completion",
    "calculate_metrics"
] 