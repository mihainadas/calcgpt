"""Arithmetic representation contracts shared by research data and benchmarks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence, Tuple

Task = Tuple[int, str, int]
Layout = Literal["minimal", "fixed"]
AnswerOrder = Literal["normal", "reversed"]

REPRESENTATION_SCHEMA = "calcgpt-representation"
REPRESENTATION_SCHEMA_VERSION = 1
REPRESENTATION_NAMES = ("plain", "reversed", "padded", "padded-reversed")
_EQUATION_RE = re.compile(r"^(\d+)([+-])(\d+)=(\d+)$")

_NAME_TO_AXES: dict[str, tuple[Layout, AnswerOrder]] = {
    "plain": ("minimal", "normal"),
    "reversed": ("minimal", "reversed"),
    "padded": ("fixed", "normal"),
    "padded-reversed": ("fixed", "reversed"),
}


def validate_task(task: Task, operand_width: int) -> None:
    """Validate one task against the nonnegative fixed-domain task universe."""
    if not isinstance(operand_width, int) or isinstance(operand_width, bool) or operand_width < 1:
        raise ValueError("operand_width must be a positive integer")
    if not isinstance(task, tuple) or len(task) != 3:
        raise ValueError("task must be a three-item tuple (left, operator, right)")
    left, operator, right = task
    if (
        not isinstance(left, int)
        or isinstance(left, bool)
        or not isinstance(right, int)
        or isinstance(right, bool)
    ):
        raise ValueError("operands must be integers")
    if operator not in {"+", "-"}:
        raise ValueError("operator must be '+' or '-'")
    limit = 10**operand_width
    if not (0 <= left < limit and 0 <= right < limit):
        raise ValueError(f"operands must be in [0, {limit - 1}]")
    if operator == "-" and left < right:
        raise ValueError("subtraction tasks must have a nonnegative result")


def normalized_task_bytes(tasks: Iterable[Task]) -> bytes:
    """Serialize an ordered task roster independently of textual representation."""
    return "".join(f"{left}\t{operator}\t{right}\n" for left, operator, right in tasks).encode(
        "ascii"
    )


def task_roster_sha256(tasks: Iterable[Task]) -> str:
    """Hash an ordered normalized task roster."""
    return hashlib.sha256(normalized_task_bytes(tasks)).hexdigest()


@dataclass(frozen=True)
class RepresentationSpec:
    """Orthogonal operand/answer layout and answer-generation order."""

    layout: Layout
    answer_order: AnswerOrder
    operand_width: int

    def __post_init__(self) -> None:
        if self.layout not in {"minimal", "fixed"}:
            raise ValueError("layout must be 'minimal' or 'fixed'")
        if self.answer_order not in {"normal", "reversed"}:
            raise ValueError("answer_order must be 'normal' or 'reversed'")
        if (
            not isinstance(self.operand_width, int)
            or isinstance(self.operand_width, bool)
            or self.operand_width < 1
        ):
            raise ValueError("operand_width must be a positive integer")

    @property
    def name(self) -> str:
        for name, axes in _NAME_TO_AXES.items():
            if axes == (self.layout, self.answer_order):
                return name
        raise RuntimeError("representation axes have no canonical name")

    @property
    def answer_width(self) -> int | None:
        return self.operand_width + 1 if self.layout == "fixed" else None

    @classmethod
    def from_name(cls, name: str, operand_width: int) -> "RepresentationSpec":
        """Build one of the four named representations.

        ``plain`` and ``padded-reversed`` retain their historical meanings.
        """
        try:
            layout, answer_order = _NAME_TO_AXES[name]
        except KeyError as exc:
            choices = ", ".join(REPRESENTATION_NAMES)
            raise ValueError(f"unknown representation {name!r}; choose from {choices}") from exc
        return cls(layout=layout, answer_order=answer_order, operand_width=operand_width)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPRESENTATION_SCHEMA,
            "schema_version": REPRESENTATION_SCHEMA_VERSION,
            "name": self.name,
            "layout": self.layout,
            "answer_order": self.answer_order,
            "operand_width": self.operand_width,
            "answer_width": self.answer_width,
            "operators": ["+", "-"],
            "negative_results": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RepresentationSpec":
        if payload.get("schema") != REPRESENTATION_SCHEMA:
            raise ValueError("unsupported representation schema")
        if payload.get("schema_version") != REPRESENTATION_SCHEMA_VERSION:
            raise ValueError("unsupported representation schema version")
        name = payload.get("name")
        width = payload.get("operand_width")
        if not isinstance(name, str):
            raise ValueError("representation name must be a string")
        if not isinstance(width, int) or isinstance(width, bool):
            raise ValueError("operand_width must be an integer")
        spec = cls.from_name(name, width)
        if payload.get("layout") != spec.layout:
            raise ValueError("representation layout conflicts with its name")
        if payload.get("answer_order") != spec.answer_order:
            raise ValueError("answer_order conflicts with representation name")
        if payload.get("answer_width") != spec.answer_width:
            raise ValueError("answer_width conflicts with representation layout")
        if payload.get("operators") != ["+", "-"]:
            raise ValueError("operators must be ['+', '-']")
        if payload.get("negative_results") is not False:
            raise ValueError("negative_results must be false")
        return spec

    def format_task(self, task: Task) -> str:
        validate_task(task, self.operand_width)
        left, operator, right = task
        result = left + right if operator == "+" else left - right

        if self.layout == "fixed":
            left_text = str(left).zfill(self.operand_width)
            right_text = str(right).zfill(self.operand_width)
            answer_text = str(result).zfill(self.operand_width + 1)
        else:
            left_text = str(left)
            right_text = str(right)
            answer_text = str(result)
        if self.answer_order == "reversed":
            answer_text = answer_text[::-1]
        return f"{left_text}{operator}{right_text}={answer_text}"

    def decode_example(self, example: str) -> tuple[Task, int]:
        """Decode one structurally valid equation without requiring a correct answer.

        This keeps representation validity separate from numerical correctness during
        evaluation. In particular, fixed-width outputs must retain their full width.
        """
        stripped = example.strip()
        match = _EQUATION_RE.fullmatch(stripped)
        if match is None:
            raise ValueError(f"invalid arithmetic example: {example!r}")
        left_text, operator, right_text, answer_text = match.groups()
        if self.layout == "fixed":
            if len(left_text) != self.operand_width or len(right_text) != self.operand_width:
                raise ValueError("fixed operands have the wrong width")
            if len(answer_text) != self.operand_width + 1:
                raise ValueError("fixed answer has the wrong width")
        else:
            if str(int(left_text)) != left_text or str(int(right_text)) != right_text:
                raise ValueError("minimal operands must not contain leading zeroes")

        decoded_answer_text = (
            answer_text[::-1] if self.answer_order == "reversed" else answer_text
        )
        if self.layout == "minimal" and str(int(decoded_answer_text)) != decoded_answer_text:
            raise ValueError("minimal answer must not contain redundant zeroes")
        task = (int(left_text), operator, int(right_text))
        validate_task(task, self.operand_width)
        return task, int(decoded_answer_text)

    def parse_example(self, example: str) -> Task:
        """Parse and strictly validate one fully encoded equation."""
        task, decoded_answer = self.decode_example(example)
        left, operator, right = task
        expected_answer = left + right if operator == "+" else left - right
        if decoded_answer != expected_answer:
            raise ValueError(
                f"example has incorrect arithmetic; expected result {expected_answer}"
            )
        expected = self.format_task(task)
        if example.strip() != expected:
            raise ValueError(
                f"example does not match {self.name!r} representation; expected {expected!r}"
            )
        return task

    def validate_dataset(self, examples: Sequence[str]) -> list[Task]:
        """Validate a nonempty dataset and return its ordered normalized roster."""
        if not examples:
            raise ValueError("dataset cannot be empty")
        tasks = [self.parse_example(example) for example in examples]
        if len(set(tasks)) != len(tasks):
            raise ValueError("dataset contains duplicate normalized tasks")
        return tasks

    def render_dataset(self, tasks: Sequence[Task]) -> list[str]:
        """Render one ordered roster without changing task membership or order."""
        rendered = [self.format_task(task) for task in tasks]
        if len(set(tasks)) != len(tasks):
            raise ValueError("task roster contains duplicates")
        return rendered

    def render_dataset_bytes(self, tasks: Sequence[Task]) -> bytes:
        return ("\n".join(self.render_dataset(tasks)) + "\n").encode("utf-8")
