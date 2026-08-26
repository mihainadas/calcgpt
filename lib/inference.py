"""Core inference functionality and model-artifact validation for CalcGPT."""

from __future__ import annotations

import json
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import GPT2LMHeadModel

from .representation import REPRESENTATION_SCHEMA, RepresentationSpec
from .tokenizer import TOKENIZER_FILENAME, CalcGPTTokenizer

PathLike = Union[str, Path]
TASK_SPEC_FILENAME = "task_spec.json"


@dataclass
class InferenceConfig:
    """Configuration for inference parameters."""

    temperature: float = 0.0
    max_tokens: int = 10
    device: str = "auto"
    show_tokens: bool = False

    def validate(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


def get_device(device_spec: str = "auto") -> torch.device:
    """Return the requested device, or the best available device for ``auto``."""

    if device_spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_spec)


def find_latest_model(models_dir: PathLike = "models") -> Optional[str]:
    """Find the most recently modified CalcGPT model directory."""

    models_path = Path(models_dir)
    if not models_path.is_dir():
        return None
    model_dirs = [
        path for path in models_path.iterdir()
        if path.is_dir() and path.name.startswith("calcgpt")
    ]
    if not model_dirs:
        return None
    return str(max(model_dirs, key=lambda path: path.stat().st_mtime))


def get_model_path(specified_path: Optional[str]) -> str:
    """Resolve an explicit model path or auto-detect the latest local model."""

    if specified_path and specified_path != "auto":
        return specified_path
    latest_model = find_latest_model()
    if latest_model:
        return latest_model
    if Path("out").is_dir():
        return "out"
    raise FileNotFoundError(
        "No trained models found. Please:\n"
        "  1. Train a model using: python calcgpt_train.py\n"
        "  2. Or specify a model path with: -m /path/to/model"
    )


def _has_model_weights(path: Path) -> bool:
    return any(path.glob("*.bin")) or any(path.glob("*.safetensors")) or any(
        path.glob("*.index.json")
    )


def _resolve_model_artifact(model_path: Path) -> Path:
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    if _has_model_weights(model_path):
        return model_path

    numbered_checkpoints = []
    for path in model_path.iterdir():
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and match and _has_model_weights(path):
            numbered_checkpoints.append((int(match.group(1)), path))
    if not numbered_checkpoints:
        raise FileNotFoundError(f"No model weights found in directory: {model_path}")
    return max(numbered_checkpoints, key=lambda item: item[0])[1]


def _load_artifact_tokenizer(
    model_path: Path,
    loaded_model_path: Path,
    tokenizer_path: Optional[PathLike],
    legacy_dataset_path: Optional[PathLike],
) -> CalcGPTTokenizer:
    """Load a saved tokenizer, with an explicit and warned legacy fallback."""

    if tokenizer_path is not None:
        return CalcGPTTokenizer.from_pretrained(tokenizer_path)

    candidates = [loaded_model_path]
    if loaded_model_path != model_path:
        candidates.append(model_path)
    for candidate in candidates:
        if (candidate / TOKENIZER_FILENAME).is_file():
            return CalcGPTTokenizer.from_pretrained(candidate)

    searched = ", ".join(str(path / TOKENIZER_FILENAME) for path in candidates)
    if legacy_dataset_path is None:
        raise FileNotFoundError(
            f"Model artifact has no {TOKENIZER_FILENAME}; searched {searched}. "
            "This model is not self-contained. For a legacy model only, pass "
            "legacy_dataset_path explicitly, then re-save the tokenizer beside the model."
        )

    warnings.warn(
        f"Legacy model has no {TOKENIZER_FILENAME}; rebuilding its tokenizer from "
        f"{legacy_dataset_path}. Token IDs are correct only if this is the exact training "
        "dataset. Save the tokenizer with tokenizer.save_pretrained(model_path).",
        RuntimeWarning,
        stacklevel=3,
    )
    return CalcGPTTokenizer.from_dataset(legacy_dataset_path)


def _load_task_spec(model_path: Path, loaded_model_path: Path) -> Dict[str, Any]:
    """Load and validate the representation contract saved with a model."""

    candidates = [loaded_model_path]
    if loaded_model_path != model_path:
        candidates.append(model_path)
    artifact_path = next(
        (path / TASK_SPEC_FILENAME for path in candidates if (path / TASK_SPEC_FILENAME).is_file()),
        None,
    )
    if artifact_path is None:
        warnings.warn(
            f"Model artifact has no {TASK_SPEC_FILENAME}; assuming legacy plain format.",
            RuntimeWarning,
            stacklevel=3,
        )
        return {
            "schema_version": 0,
            "format": "plain",
            "operand_width": None,
            "answer_width": None,
            "answer_order": "normal",
            "operators": ["+", "-"],
            "negative_results": False,
        }

    try:
        task_spec = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid task specification {artifact_path}: {exc}") from exc
    if not isinstance(task_spec, dict):
        raise ValueError(f"Invalid task specification {artifact_path}: expected an object")
    if task_spec.get("schema") == REPRESENTATION_SCHEMA:
        try:
            RepresentationSpec.from_dict(task_spec)
        except ValueError as exc:
            raise ValueError(f"Invalid task specification {artifact_path}: {exc}") from exc
        roster_hash = task_spec.get("task_roster_sha256")
        if roster_hash is not None and (
            not isinstance(roster_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", roster_hash) is None
        ):
            raise ValueError(
                f"Invalid task specification {artifact_path}: bad task_roster_sha256"
            )
        return task_spec

    # Explicit compatibility path for artifacts created before the orthogonal
    # representation schema. New training runs never write this shape.
    if task_spec.get("schema_version") not in {0, 1}:
        raise ValueError(
            f"Invalid task specification {artifact_path}: unsupported schema version"
        )
    task_format = task_spec.get("format")
    if task_format not in {"plain", "padded-reversed"}:
        raise ValueError(
            f"Invalid task specification {artifact_path}: unsupported format {task_format!r}"
        )
    if task_spec.get("answer_order") not in {"normal", "reversed"}:
        raise ValueError(f"Invalid task specification {artifact_path}: bad answer_order")
    operators = task_spec.get("operators")
    if (
        not isinstance(operators, list)
        or not operators
        or any(operator not in {"+", "-"} for operator in operators)
        or len(set(operators)) != len(operators)
    ):
        raise ValueError(f"Invalid task specification {artifact_path}: bad operators")
    if not isinstance(task_spec.get("negative_results"), bool):
        raise ValueError(f"Invalid task specification {artifact_path}: bad negative_results")
    if task_spec["negative_results"]:
        raise ValueError(f"Invalid task specification {artifact_path}: negative results")
    if task_format == "padded-reversed":
        width = task_spec.get("operand_width")
        answer_width = task_spec.get("answer_width")
        if not isinstance(width, int) or width < 1:
            raise ValueError(f"Invalid task specification {artifact_path}: bad operand_width")
        if answer_width != width + 1:
            raise ValueError(f"Invalid task specification {artifact_path}: bad answer_width")
        if task_spec.get("answer_order") != "reversed":
            raise ValueError(f"Invalid task specification {artifact_path}: conflicting format")
    elif (
        task_spec.get("answer_order") != "normal"
        or task_spec.get("operand_width") is not None
        or task_spec.get("answer_width") is not None
    ):
        raise ValueError(f"Invalid task specification {artifact_path}: conflicting format")
    return task_spec


def _representation_from_task_spec(
    task_spec: Dict[str, Any],
) -> Optional[RepresentationSpec]:
    """Return the representation contract, retaining legacy plain compatibility."""
    if task_spec.get("schema") == REPRESENTATION_SCHEMA:
        return RepresentationSpec.from_dict(task_spec)
    if task_spec.get("format") == "padded-reversed":
        return RepresentationSpec.from_name(
            "padded-reversed", task_spec["operand_width"]
        )
    return None


def validate_tokenizer_compatibility(tokenizer: CalcGPTTokenizer, model: Any) -> None:
    """Validate tokenizer IDs and context metadata against a loaded model."""

    config = model.config
    expected_vocab_size = getattr(config, "vocab_size", None)
    if expected_vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: tokenizer has {tokenizer.vocab_size} "
            f"tokens but model config expects {expected_vocab_size}"
        )

    for config_name, actual in (
        ("pad_token_id", tokenizer.pad_token_id),
        ("eos_token_id", tokenizer.eos_token_id),
    ):
        expected = getattr(config, config_name, None)
        if expected != actual:
            raise ValueError(
                f"Tokenizer/model {config_name} mismatch: tokenizer uses {actual}, "
                f"model config uses {expected}"
            )

    context_length = getattr(config, "n_positions", None)
    if context_length is None:
        context_length = getattr(config, "max_position_embeddings", None)
    if context_length is not None and tokenizer.max_length > context_length:
        raise ValueError(
            f"Tokenizer training length {tokenizer.max_length} exceeds model context "
            f"length {context_length}"
        )


def validate_simple_arithmetic(problem: str, answer: str) -> bool:
    """Validate a direct answer for one non-negative integer addition/subtraction."""

    match = re.fullmatch(r"\s*(\d+)\s*([+-])\s*(\d+)\s*=?\s*", problem)
    if not match:
        return False
    left, operator, right = match.groups()
    expected = int(left) + int(right) if operator == "+" else int(left) - int(right)
    return str(expected) == answer.strip()


class CalcGPT:
    """CalcGPT inference engine."""

    def __init__(
        self,
        model_path: PathLike,
        config: Optional[InferenceConfig] = None,
        verbose: bool = False,
        *,
        tokenizer_path: Optional[PathLike] = None,
        legacy_dataset_path: Optional[PathLike] = None,
    ):
        """Load a model and its tokenizer.

        ``legacy_dataset_path`` is deliberately opt-in because rebuilding a
        tokenizer from the wrong dataset can silently invalidate every prediction.
        """

        self.model_path = Path(model_path)
        self.config = config or InferenceConfig()
        self.config.validate()
        self.verbose = verbose
        self.tokenizer_path = tokenizer_path
        self.legacy_dataset_path = legacy_dataset_path
        self.device = get_device(self.config.device)
        self.model = None
        self.tokenizer = None
        self.task_spec: Dict[str, Any] = {}
        self.representation_spec: Optional[RepresentationSpec] = None
        self.loaded_model_path = self.model_path

        self._load_model()
        self._load_tokenizer()
        self.task_spec = _load_task_spec(self.model_path, self.loaded_model_path)
        self.representation_spec = _representation_from_task_spec(self.task_spec)

    def log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _load_model(self) -> None:
        try:
            self.loaded_model_path = _resolve_model_artifact(self.model_path)
            self.log(f"Loading model from: {self.loaded_model_path}")
            self.model = GPT2LMHeadModel.from_pretrained(str(self.loaded_model_path))
            self.model.to(self.device)
            self.model.eval()
            if self.verbose:
                total_params = sum(parameter.numel() for parameter in self.model.parameters())
                self.log(f"Model loaded: {total_params:,} parameters on {self.device}")
        except Exception as exc:
            raise RuntimeError(f"Error loading model: {exc}") from exc

    def _load_tokenizer(self) -> None:
        try:
            self.tokenizer = _load_artifact_tokenizer(
                self.model_path,
                self.loaded_model_path,
                self.tokenizer_path,
                self.legacy_dataset_path,
            )
            validate_tokenizer_compatibility(self.tokenizer, self.model)
            self.log(
                f"Tokenizer loaded: {self.tokenizer.vocab_size} tokens, "
                f"training length {self.tokenizer.max_length}"
            )
        except Exception as exc:
            raise RuntimeError(f"Error loading tokenizer: {exc}") from exc

    def _validate_context(self, input_length: int, max_new_tokens: int) -> None:
        context_length = getattr(self.model.config, "n_positions", None)
        if context_length is None:
            context_length = getattr(self.model.config, "max_position_embeddings", None)
        requested_length = input_length + max_new_tokens
        if context_length is not None and requested_length > context_length:
            raise ValueError(
                f"Input plus max_tokens requires {requested_length} positions, but the "
                f"model context length is {context_length}"
            )

    def solve(self, problem: str) -> Dict[str, Any]:
        """Solve one arithmetic problem and return structured inference details."""

        start_time = time.perf_counter()
        problem = problem.strip()

        try:
            match = re.fullmatch(r"(\d+)([+-])(\d+)=?", problem)
            if match is None:
                raise ValueError(
                    "Problem must use the form nonnegative_integer+nonnegative_integer "
                    "or nonnegative_integer-nonnegative_integer"
                )
            left_text, operator, right_text = match.groups()
            left = int(left_text)
            right = int(right_text)
            display_problem = f"{left_text}{operator}{right_text}="

            if operator not in self.task_spec["operators"]:
                supported = ", ".join(self.task_spec["operators"])
                raise ValueError(
                    f"Operator {operator!r} is not supported by this model; "
                    f"supported operators: {supported}"
                )
            if operator == "-" and left < right and not self.task_spec["negative_results"]:
                raise ValueError("This model does not support negative subtraction results")

            if self.representation_spec is not None:
                encoded_equation = self.representation_spec.format_task(
                    (left, operator, right)
                )
                model_prompt = encoded_equation.split("=", 1)[0] + "="
                max_new_tokens = min(
                    self.config.max_tokens,
                    self.representation_spec.answer_width or self.config.max_tokens,
                )
            else:
                # Legacy plain artifacts did not record a domain width.
                model_prompt = display_problem
                max_new_tokens = self.config.max_tokens

            input_tokens = self.tokenizer.encode(model_prompt, add_eos=False)
            if not input_tokens:
                raise ValueError("Problem produced no input tokens")
            remaining_training_length = self.tokenizer.max_length - len(input_tokens)
            if remaining_training_length < 1:
                raise ValueError("Problem leaves no room for a generated answer")
            max_new_tokens = min(max_new_tokens, remaining_training_length)
            self._validate_context(len(input_tokens), max_new_tokens)
            input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self.device)

            generation_args = {
                "max_new_tokens": max_new_tokens,
                "do_sample": self.config.temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "bad_words_ids": [[self.tokenizer.pad_token_id]],
                "num_return_sequences": 1,
            }
            if self.config.temperature > 0:
                generation_args["temperature"] = self.config.temperature

            with torch.no_grad():
                generated = self.model.generate(input_ids, **generation_args)

            result_tokens = generated[0].tolist()
            new_tokens = result_tokens[len(input_tokens):]
            full_result = self.tokenizer.decode(result_tokens)
            answer_part = self.tokenizer.decode(new_tokens) if new_tokens else ""
            encoded_answer = (
                full_result.split("=", 1)[1].strip() if "=" in full_result else answer_part
            )
            numerical_answer = encoded_answer
            answer_order = (
                self.representation_spec.answer_order
                if self.representation_spec is not None
                else self.task_spec["answer_order"]
            )
            if not encoded_answer.isdigit():
                raise ValueError("Model emitted a non-numeric answer")
            if answer_order == "reversed":
                numerical_answer = str(int(encoded_answer[::-1]))
            elif (
                self.representation_spec is not None
                and self.representation_spec.layout == "fixed"
            ):
                numerical_answer = str(int(encoded_answer))
            return {
                "problem": display_problem,
                "model_prompt": model_prompt,
                "full_result": full_result,
                "answer": numerical_answer,
                "input_tokens": input_tokens if self.config.show_tokens else None,
                "output_tokens": new_tokens if self.config.show_tokens else None,
                "inference_time": time.perf_counter() - start_time,
                "is_correct": validate_simple_arithmetic(display_problem, numerical_answer),
                "model_path": str(self.loaded_model_path),
                "device": str(self.device),
                "task_format": (
                    self.representation_spec.name
                    if self.representation_spec is not None
                    else self.task_spec["format"]
                ),
            }
        except Exception as exc:
            return {
                "problem": problem,
                "error": str(exc),
                "inference_time": time.perf_counter() - start_time,
            }

    def solve_batch(self, problems: List[str]) -> List[Dict[str, Any]]:
        return [self.solve(problem) for problem in problems]

    def get_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {}
        total_params = sum(parameter.numel() for parameter in self.model.parameters())
        trainable_params = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
        return {
            "model_path": str(self.loaded_model_path),
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "vocab_size": self.tokenizer.vocab_size,
            "max_length": self.tokenizer.max_length,
            "config": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "show_tokens": self.config.show_tokens,
            },
            "task_spec": self.task_spec,
        }
