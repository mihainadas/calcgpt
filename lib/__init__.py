"""Public CalcGPT API with lazy imports for optional ML dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

from .version import __version__

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "CalcGPTTrainer": (".train", "CalcGPTTrainer"),
    "TrainingConfig": (".train", "TrainingConfig"),
    "detect_device": (".train", "detect_device"),
    "load_dataset": (".train", "load_dataset"),
    "augment_data": (".train", "augment_data"),
    "OptimizedDataset": (".train", "OptimizedDataset"),
    "create_model_config": (".train", "create_model_config"),
    "print_model_info": (".train", "print_model_info"),
    "DatasetGenerator": (".dategen", "DatasetGenerator"),
    "DatagenConfig": (".dategen", "DatagenConfig"),
    "contains_only_allowed_digits": (".dategen", "contains_only_allowed_digits"),
    "generate_valid_numbers": (".dategen", "generate_valid_numbers"),
    "generate_expressions": (".dategen", "generate_expressions"),
    "write_expressions_to_file": (".dategen", "write_expressions_to_file"),
    "parse_digit_set": (".dategen", "parse_digit_set"),
    "generate_output_filename": (".dategen", "generate_output_filename"),
    "parse_filename_parameters": (".dategen", "parse_filename_parameters"),
    "get_file_stats": (".dategen", "get_file_stats"),
    "CalcGPTTokenizer": (".tokenizer", "CalcGPTTokenizer"),
    "CalcGPT": (".inference", "CalcGPT"),
    "InferenceConfig": (".inference", "InferenceConfig"),
    "get_device": (".inference", "get_device"),
    "find_latest_model": (".inference", "find_latest_model"),
    "get_model_path": (".inference", "get_model_path"),
    "validate_simple_arithmetic": (".inference", "validate_simple_arithmetic"),
    "validate_tokenizer_compatibility": (".inference", "validate_tokenizer_compatibility"),
    "CalcGPTEvaluator": (".evaluation", "CalcGPTEvaluator"),
    "EvaluationConfig": (".evaluation", "EvaluationConfig"),
    "load_evaluation_dataset": (".evaluation", "load_evaluation_dataset"),
    "create_test_cases": (".evaluation", "create_test_cases"),
    "validate_completion": (".evaluation", "validate_completion"),
    "calculate_metrics": (".evaluation", "calculate_metrics"),
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str):
    """Import an exported symbol only when it is first requested."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
