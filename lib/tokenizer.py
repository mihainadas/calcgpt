"""Tokenizer and tokenizer-artifact support for CalcGPT models."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Literal, Mapping, Union

PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
TOKENIZER_FILENAME = "tokenizer.json"
TOKENIZER_FORMAT = "calcgpt-tokenizer"
TOKENIZER_FORMAT_VERSION = 1

TokenizationMode = Literal["char", "number"]
PathLike = Union[str, Path]


class CalcGPTTokenizer:
    """Tokenizer for arithmetic expressions with character or number tokens.

    Unknown tokens are rejected rather than skipped. Silently dropping a token can
    change an expression's meaning (for example, ``1*2`` becoming ``12``).
    """

    def __init__(self, examples: List[str], mode: TokenizationMode = "char"):
        if not examples:
            raise ValueError("Examples cannot be empty")
        if mode not in ("char", "number"):
            raise ValueError(f"Invalid mode: {mode}. Use 'char' or 'number'")

        self.mode = mode
        if mode == "char":
            self._build_char_vocab(examples)
        else:
            self._build_number_vocab(examples)

        self.maxlen = max(len(self.encode(example)) for example in examples)

    def _build_char_vocab(self, examples: List[str]) -> None:
        chars = sorted(set("".join(examples)))
        self._set_vocab({token: i for i, token in enumerate([PAD_TOKEN, EOS_TOKEN] + chars)})

    def _build_number_vocab(self, examples: List[str]) -> None:
        tokens = [PAD_TOKEN, EOS_TOKEN]
        tokens.extend(str(i) for i in range(100))
        tokens.extend(sorted(char for char in set("".join(examples)) if not char.isdigit()))
        self._set_vocab({token: i for i, token in enumerate(tokens)})

    def _set_vocab(self, vocab: Mapping[str, int]) -> None:
        self.vocab = dict(vocab)
        self.id2char = {token_id: token for token, token_id in self.vocab.items()}

    @classmethod
    def from_dataset(
        cls,
        dataset_path: PathLike = Path("datasets/ds-calcgpt.txt"),
        mode: TokenizationMode = "char",
    ) -> "CalcGPTTokenizer":
        """Build a tokenizer from a training dataset.

        This is intended for training and explicit migration of legacy artifacts.
        Inference should load :data:`TOKENIZER_FILENAME` from the model directory.
        """

        path = Path(dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with path.open("r", encoding="utf-8") as dataset_file:
            examples = [line.strip() for line in dataset_file if line.strip()]
        if not examples:
            raise ValueError(f"Dataset is empty: {path}")
        return cls(examples, mode)

    def save_pretrained(self, model_path: PathLike) -> Path:
        """Write a deterministic, versioned tokenizer artifact beside a model."""

        destination = Path(model_path)
        if destination.name == TOKENIZER_FILENAME:
            artifact_path = destination
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            artifact_path = destination / TOKENIZER_FILENAME

        payload = {
            "format": TOKENIZER_FORMAT,
            "format_version": TOKENIZER_FORMAT_VERSION,
            "max_length": self.max_length,
            "mode": self.mode,
            "special_tokens": {
                "eos_token": EOS_TOKEN,
                "eos_token_id": self.eos_token_id,
                "pad_token": PAD_TOKEN,
                "pad_token_id": self.pad_token_id,
            },
            "vocab": self.vocab,
        }
        artifact_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return artifact_path

    @classmethod
    def from_pretrained(cls, model_path: PathLike) -> "CalcGPTTokenizer":
        """Load and validate a tokenizer artifact from a model directory or file."""

        path = Path(model_path)
        artifact_path = path if path.name == TOKENIZER_FILENAME else path / TOKENIZER_FILENAME
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Tokenizer artifact not found: {artifact_path}")

        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid tokenizer JSON in {artifact_path}: {exc}") from exc

        cls._validate_artifact(payload, artifact_path)
        tokenizer = cls.__new__(cls)
        tokenizer.mode = payload["mode"]
        tokenizer._set_vocab(payload["vocab"])
        tokenizer.maxlen = payload["max_length"]
        return tokenizer

    @staticmethod
    def _validate_artifact(payload: object, artifact_path: Path) -> None:
        def invalid(message: str) -> ValueError:
            return ValueError(f"Invalid tokenizer artifact {artifact_path}: {message}")

        if not isinstance(payload, dict):
            raise invalid("top-level value must be an object")
        if payload.get("format") != TOKENIZER_FORMAT:
            raise invalid(f"unsupported format {payload.get('format')!r}")
        if payload.get("format_version") != TOKENIZER_FORMAT_VERSION:
            raise invalid(f"unsupported format version {payload.get('format_version')!r}")
        if payload.get("mode") not in ("char", "number"):
            raise invalid(f"unsupported mode {payload.get('mode')!r}")
        if not isinstance(payload.get("max_length"), int) or payload["max_length"] < 1:
            raise invalid("max_length must be a positive integer")

        vocab = payload.get("vocab")
        if not isinstance(vocab, dict) or not vocab:
            raise invalid("vocab must be a non-empty object")
        if any(not isinstance(token, str) for token in vocab):
            raise invalid("vocab tokens must be strings")
        if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in vocab.values()):
            raise invalid("vocab IDs must be integers")
        token_ids = list(vocab.values())
        if len(set(token_ids)) != len(token_ids):
            raise invalid("vocab IDs must be unique")
        if set(token_ids) != set(range(len(token_ids))):
            raise invalid("vocab IDs must be contiguous and start at zero")
        if vocab.get(PAD_TOKEN) != 0 or vocab.get(EOS_TOKEN) != 1:
            raise invalid("special token IDs must be <pad>=0 and <eos>=1")

        expected_special_tokens = {
            "eos_token": EOS_TOKEN,
            "eos_token_id": 1,
            "pad_token": PAD_TOKEN,
            "pad_token_id": 0,
        }
        if payload.get("special_tokens") != expected_special_tokens:
            raise invalid("special_tokens metadata does not match the vocabulary")

    def _parse_tokens(self, text: str) -> List[str]:
        if self.mode == "char":
            return list(text)
        return re.findall(r"\d+|[^\d]", text)

    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        """Encode text, raising when any token is outside the saved vocabulary."""

        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        encoded: List[int] = []
        for position, token in enumerate(self._parse_tokens(text)):
            try:
                encoded.append(self.vocab[token])
            except KeyError as exc:
                raise ValueError(
                    f"Unknown token {token!r} at token position {position} "
                    f"for {self.mode!r} tokenizer"
                ) from exc
        if add_eos:
            encoded.append(self.eos_token_id)
        return encoded

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs, rejecting IDs outside the saved vocabulary."""

        decoded: List[str] = []
        special_ids = {self.pad_token_id, self.eos_token_id}
        for position, token_id in enumerate(token_ids):
            if token_id not in self.id2char:
                raise ValueError(f"Unknown token ID {token_id!r} at position {position}")
            if token_id not in special_ids:
                decoded.append(self.id2char[token_id])
        return "".join(decoded)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab[PAD_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.vocab[EOS_TOKEN]

    @property
    def max_length(self) -> int:
        return self.maxlen
