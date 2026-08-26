#!/usr/bin/env python3
"""
CalcGPT Live Demo — a small transformer evaluated on held-out arithmetic
tasks within its configured fixed-width operand range.

Trick under study: zero-pad operands to a fixed width and reverse the answer
("0000007+0000008=51000000"). Every digit then lives at a fixed position,
and the decoder emits units first, matching the natural carry direction.
Whether that representation produces reusable arithmetic behavior is an
empirical question for the held-out benchmark, not an assumption of the demo.

Stages:
  1. Banner + architecture
  2. Token-by-token live generation
  3. Accuracy by magnitude on tasks absent from the training dataset
  4. Held-out task verification
  5. Head-to-head against the old in-distribution-only model
  6. Top-k probabilities for one step
  7. Interactive REPL
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import torch
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from transformers import GPT2LMHeadModel

from lib.benchmark import (
    load_examples,
    sample_heldout_by_magnitude,
    task_space_size,
)
from lib.inference import validate_tokenizer_compatibility
from lib.representation import RepresentationSpec
from lib.tokenizer import CalcGPTTokenizer

console = Console()

BANNER = r"""
   ____      _      ____ ____ _____
  / ___|__ _| | ___/ ___|  _ \_   _|
 | |   / _` | |/ __| |  _| |_) || |
 | |__| (_| | | (__| |_| |  __/ | |
  \____\__,_|_|\___|\____|_|    |_|

  a transformer evaluated on explicitly held-out arithmetic tasks
"""

V2_MODEL = Path("models/calcgpt-padded")
V2_DATASET = Path("datasets/ds-calcgpt-padded.txt")
V1_MODEL = Path("models/calcgpt-demo")
V1_DATASET = Path("datasets/ds-calcgpt.txt")

# Must match the operand-width used by scripts/gen_padded.py
OPERAND_WIDTH = 3
ANSWER_WIDTH = OPERAND_WIDTH + 1
CANONICAL_BENCHMARK_SEED = 42


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_v2_contract() -> tuple[dict[str, Any], RepresentationSpec]:
    """Require model metadata that proves which dataset the demo excludes."""
    manifest = _load_json_object(V2_MODEL / "training_manifest.json")
    task_spec = _load_json_object(V2_MODEL / "task_spec.json")
    representation = RepresentationSpec.from_dict(task_spec)
    if representation.name != "padded-reversed":
        raise ValueError(
            "demo requires a padded-reversed artifact, "
            f"not {representation.name!r}"
        )
    if representation.operand_width != OPERAND_WIDTH:
        raise ValueError(
            f"demo expects operand width {OPERAND_WIDTH}, but the artifact records "
            f"{representation.operand_width}"
        )

    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("training manifest has no dataset record")
    expected_hash = dataset.get("sha256")
    actual_hash = _sha256_file(V2_DATASET)
    if expected_hash != actual_hash:
        raise ValueError(
            "the demo exclusion dataset does not match the model training manifest: "
            f"expected SHA-256 {expected_hash}, got {actual_hash}"
        )
    return manifest, representation


def banner_panel() -> Panel:
    return Panel(
        Align.center(Text(BANNER, style="bold cyan")),
        border_style="cyan",
        padding=(0, 2),
    )


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path: Path, device: torch.device) -> GPT2LMHeadModel:
    m = GPT2LMHeadModel.from_pretrained(str(model_path))
    m.to(device)
    m.eval()
    return m


def architecture_panel(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    model_path: Path,
    training_manifest: dict[str, Any],
    representation: RepresentationSpec,
) -> Panel:
    cfg = model.config
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / 1024 / 1024

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(justify="right", style="dim")
    table.add_column(style="bold")
    table.add_row("model", str(model_path.name))
    table.add_row("device", str(device))
    table.add_row("embedding dim", str(cfg.n_embd))
    table.add_row("layers", str(cfg.n_layer))
    table.add_row("attention heads", str(cfg.n_head))
    table.add_row("vocabulary", f"{tokenizer.vocab_size} tokens")
    table.add_row("context length", f"{cfg.n_positions} positions")
    table.add_row("parameters", f"{total_params:,}  ({size_mb:.2f} MB)")
    table.add_row(
        "training",
        f"{training_manifest['dataset']['examples']:,} sampled tasks from "
        f"{task_space_size(representation.operand_width):,}, "
        f"{representation.name}",
    )
    return Panel(table, title="[bold]model architecture[/]", border_style="cyan")


def pad_prompt(a: int, op: str, b: int) -> str:
    return f"{a:0{OPERAND_WIDTH}d}{op}{b:0{OPERAND_WIDTH}d}="


def unpad_answer(raw: str) -> str:
    """Reverse + strip leading zeros + handle '0' edge case."""
    return raw[::-1].lstrip("0") or "0"


def greedy_generate_stream(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
) -> Iterator[Tuple[str, float, float]]:
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_eos=False)], dtype=torch.long
    ).to(device)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    for _ in range(max_new_tokens):
        start = time.perf_counter()
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
        logits[:, pad_id] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = int(torch.argmax(probs, dim=-1).item())
        conf = float(probs[0, nxt].item())
        latency = time.perf_counter() - start

        if nxt == eos_id:
            return
        yield tokenizer.id2char[nxt], conf, latency
        input_ids = torch.cat(
            [input_ids, torch.tensor([[nxt]], device=device)], dim=1
        )


def greedy_predict(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
) -> Tuple[str, float]:
    raw = ""
    total = 0.0
    for token, _, latency in greedy_generate_stream(
        model, tokenizer, device, prompt, max_new_tokens
    ):
        raw += token
        total += latency
    return raw, total


def truth(a: int, op: str, b: int) -> int:
    return a + b if op == "+" else a - b


def stage_load() -> Tuple[GPT2LMHeadModel, CalcGPTTokenizer, torch.device, Optional[GPT2LMHeadModel], Optional[CalcGPTTokenizer]]:
    device = detect_device()
    training_manifest, representation = load_verified_v2_contract()
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[cyan]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("Loading model…", total=None)
        tokenizer = CalcGPTTokenizer.from_pretrained(V2_MODEL)
        model = load_model(V2_MODEL, device)
        validate_tokenizer_compatibility(tokenizer, model)
        if V1_MODEL.exists() and V1_DATASET.exists():
            tokenizer_v1 = CalcGPTTokenizer.from_dataset(V1_DATASET)
            model_v1 = load_model(V1_MODEL, device)
        else:
            tokenizer_v1 = None
            model_v1 = None
    console.print(
        architecture_panel(
            model,
            tokenizer,
            device,
            V2_MODEL,
            training_manifest,
            representation,
        )
    )
    return model, tokenizer, device, model_v1, tokenizer_v1


def stage_live_generation(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    pairs: List[Tuple[int, str, int]],
) -> None:
    console.rule("[bold cyan]live generation", style="cyan")
    console.print(
        f"[dim]Operands are zero-padded to {OPERAND_WIDTH} digits.  "
        f"The model emits the answer [bold]least-significant digit "
        f"first[/bold] (the carry direction).  Reverse the bold part "
        f"to read the answer.[/dim]\n"
    )

    for a, op, b in pairs:
        prompt = pad_prompt(a, op, b)
        rendered = Text()
        rendered.append(prompt, style="bold white")
        with Live(rendered, console=console, refresh_per_second=20) as live:
            total_latency = 0.0
            raw = ""
            for token, conf, latency in greedy_generate_stream(
                model, tokenizer, device, prompt, ANSWER_WIDTH
            ):
                total_latency += latency
                color = "green" if conf > 0.9 else "yellow" if conf > 0.5 else "red"
                rendered.append(token, style=f"bold {color}")
                raw += token
                live.update(rendered)
                time.sleep(0.10)
            answer = unpad_answer(raw)
            expected = truth(a, op, b)
            ok = answer == str(expected)
            rendered.append("    reads as ", style="dim")
            rendered.append(answer, style="bold cyan")
            rendered.append("   ")
            rendered.append("✓" if ok else "✗", style="bold green" if ok else "bold red")
            rendered.append(
                f"  expected {expected}   ({total_latency*1000:.1f} ms total)",
                style="dim",
            )
            live.update(rendered)
        console.print()


def sample_pairs_by_magnitude(
    per_bucket: int,
    max_digits: int,
    excluded_examples: List[str],
    seed: int = 0,
) -> List[Tuple[int, int, int, str]]:
    """Return unique tasks that are exactly absent from the provided dataset."""
    return sample_heldout_by_magnitude(
        per_bucket, max_digits, excluded_examples, seed
    )


def stage_scaling(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    per_bucket: int = 100,
) -> None:
    console.rule(
        "[bold cyan]accuracy by operand magnitude",
        style="cyan",
    )
    console.print(
        f"[dim]{per_bucket} unique held-out tasks per digit-count bucket. "
        f"Every task is checked against {V2_DATASET.name} and included only "
        f"when absent.[/dim]\n"
    )

    table = Table(box=None, padding=(0, 2))
    table.add_column("operand range", style="bold", width=18)
    table.add_column("accuracy", justify="right", width=9)
    table.add_column("bar", width=26)
    table.add_column("avg ms", justify="right", style="dim", width=7)
    table.add_column("samples", justify="right", style="dim", width=8)

    excluded_examples = load_examples(V2_DATASET)
    problems = sample_pairs_by_magnitude(
        per_bucket,
        OPERAND_WIDTH,
        excluded_examples,
        seed=CANONICAL_BENCHMARK_SEED,
    )

    with Live(table, console=console, refresh_per_second=8) as live:
        for d in range(1, OPERAND_WIDTH + 1):
            lo = 0 if d == 1 else 10 ** (d - 1)
            hi = 10 ** d - 1
            bucket_problems = [p for p in problems if p[0] == d]
            correct = 0
            total_latency = 0.0
            for _, a, b, op in bucket_problems:
                prompt = pad_prompt(a, op, b)
                raw, latency = greedy_predict(
                    model, tokenizer, device, prompt, ANSWER_WIDTH
                )
                total_latency += latency
                if unpad_answer(raw) == str(truth(a, op, b)):
                    correct += 1
            pct = correct / per_bucket * 100
            avg_ms = total_latency / per_bucket * 1000
            bar_len = int(pct / 100 * 25)
            bar = Text("█" * bar_len + "░" * (25 - bar_len))
            bar.stylize("green" if pct >= 95 else "yellow" if pct >= 50 else "red")
            table.add_row(
                f"{lo:,}–{hi:,}",
                f"{pct:5.1f}%",
                bar,
                f"{avg_ms:5.1f}",
                f"{correct}/{per_bucket}",
            )
            live.update(table)
    console.print()


def stage_v1_vs_v2(
    v2: GPT2LMHeadModel,
    tokenizer_v2: CalcGPTTokenizer,
    v1: Optional[GPT2LMHeadModel],
    tokenizer_v1: Optional[CalcGPTTokenizer],
    device: torch.device,
) -> None:
    if v1 is None or tokenizer_v1 is None:
        return
    console.rule("[bold cyan]new model vs old", style="cyan")
    console.print(
        "[dim]The old model was trained on operands 0–100 with plain "
        "left-to-right answers.  It memorized that table and falls off "
        "a cliff above 100.  The new model uses zero-padded operands and "
        "a reversed answer, and is tested on explicit held-out tasks.[/dim]\n"
    )

    problems = [
        (7, "+", 8),
        (67, "+", 33),
        (123, "+", 456),
        (999, "+", 1),
        (456, "-", 123),
        (700, "+", 299),
    ]

    table = Table(box=None, padding=(0, 2))
    table.add_column("problem", style="bold", width=22)
    table.add_column("old (0–100)", width=14)
    table.add_column(f"new ({OPERAND_WIDTH}-digit)", width=14)
    table.add_column("truth", style="dim", width=14)

    for a, op, b in problems:
        plain = f"{a}{op}{b}="
        # v1 input is the plain form
        v1_raw, _ = greedy_predict(v1, tokenizer_v1, device, plain, max_new_tokens=8)
        v1_pred = v1_raw.split("=", 1)[-1] if "=" in v1_raw else v1_raw

        # v2 input is the padded form
        padded = pad_prompt(a, op, b)
        v2_raw, _ = greedy_predict(
            v2, tokenizer_v2, device, padded, max_new_tokens=ANSWER_WIDTH
        )
        v2_pred = unpad_answer(v2_raw)

        t = truth(a, op, b)
        v1_ok = v1_pred == str(t)
        v2_ok = v2_pred == str(t)
        table.add_row(
            plain,
            Text(v1_pred or "—", style="green" if v1_ok else "red"),
            Text(v2_pred or "—", style="green" if v2_ok else "red"),
            f"{t:,}",
        )
    console.print(table)
    console.print()


def stage_topk(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    a: int = 478,
    b: int = 365,
) -> None:
    prompt = pad_prompt(a, "+", b)
    expected = a + b
    rev = str(expected).zfill(ANSWER_WIDTH)
    expected_units = rev[-1]
    console.rule("[bold cyan]what is the model thinking?", style="cyan")
    console.print(
        f"[dim]Top-5 next-token probabilities for [bold]{prompt}[/bold]. "
        f"{a}+{b}={expected}; first emitted digit should be the units, "
        f"i.e. [bold]{expected_units}[/].[/dim]\n"
    )
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_eos=False)], dtype=torch.long
    ).to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[:, -1, :]
    logits[:, tokenizer.pad_token_id] = float("-inf")
    probs = torch.softmax(logits, dim=-1)[0]
    topk = torch.topk(probs, k=5)

    table = Table(box=None, padding=(0, 2))
    table.add_column("token", style="bold", width=8)
    table.add_column("probability", width=40)
    table.add_column("value", style="dim", justify="right")
    for prob, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        token = tokenizer.id2char[idx]
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        table.add_row(repr(token), Text(bar, style="cyan"), f"{prob*100:5.1f}%")
    console.print(table)
    console.print()


def parse_user(text: str) -> Optional[Tuple[int, str, int]]:
    text = text.strip().rstrip("=").replace(" ", "")
    for op in ["+", "-"]:
        if op in text[1:]:  # skip leading sign
            idx = text.rindex(op)
            try:
                a = int(text[:idx])
                b = int(text[idx + 1 :])
            except ValueError:
                return None
            if a < 0 or b < 0:
                return None
            if op == "-" and a < b:
                return None
            return a, op, b
    return None


def stage_interactive(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
) -> None:
    if not sys.stdin.isatty():
        console.print(
            "[dim]Skipping interactive REPL (no TTY). Run `python demo.py` "
            "in a terminal to try the model yourself.[/dim]"
        )
        return
    max_v = 10 ** OPERAND_WIDTH - 1
    console.rule("[bold cyan]your turn", style="cyan")
    console.print(
        f"[dim]Type any arithmetic problem with operands up to "
        f"{max_v:,} (e.g. [bold]123+456[/bold]). "
        f"Empty input or 'q' to quit.[/dim]\n"
    )
    while True:
        try:
            user = console.input("[bold cyan]you ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not user or user.lower() in {"q", "quit", "exit"}:
            break
        parsed = parse_user(user)
        if parsed is None:
            console.print(
                "[yellow]Use the format a+b or a-b with non-negative result, "
                f"operands ≤ {max_v:,}.[/yellow]"
            )
            continue
        a, op, b = parsed
        if a > max_v or b > max_v:
            console.print(f"[yellow]Operands must be ≤ {max_v:,}.[/yellow]")
            continue
        prompt = pad_prompt(a, op, b)
        rendered = Text(prompt, style="bold white")
        with Live(rendered, console=console, refresh_per_second=20) as live:
            raw = ""
            for token, conf, _ in greedy_generate_stream(
                model, tokenizer, device, prompt, ANSWER_WIDTH
            ):
                color = "green" if conf > 0.9 else "yellow" if conf > 0.5 else "red"
                rendered.append(token, style=f"bold {color}")
                raw += token
                live.update(rendered)
                time.sleep(0.07)
            answer = unpad_answer(raw)
            expected = truth(a, op, b)
            rendered.append("    reads as ", style="dim")
            rendered.append(answer, style="bold cyan")
            rendered.append("   ")
            if answer == str(expected):
                rendered.append("correct", style="green")
            else:
                rendered.append(f"wrong (expected {expected:,})", style="red")
            live.update(rendered)
        console.print()


def main() -> int:
    console.print(banner_panel())
    if not V2_MODEL.exists() or not V2_DATASET.exists():
        console.print(
            f"[red]No model found at {V2_MODEL}.[/red]\n\n"
            "Generate the dataset and train the model first:\n"
            f"  [bold]python scripts/gen_padded.py -w {OPERAND_WIDTH}"
            "[/bold]\n"
            "  [bold]python calcgpt_train.py "
            "-d datasets/ds-calcgpt-padded.txt -o models/calcgpt-padded "
            "--epochs 30 --batch-size 64 --embedding-dim 128 "
            "--num-layers 4 --num-heads 8 --feedforward-dim 256 "
            "--learning-rate 1e-3 --warmup-steps 100 --n-positions 20 "
            "--save-steps 2000 --no-augmentation --task-format padded-reversed "
            "--operand-width 3 --split-seed 42 --loss-scope answer-only[/bold]"
        )
        return 1
    model, tokenizer, device, model_v1, tokenizer_v1 = stage_load()
    console.print()

    showcase = [
        (7, "+", 8),
        (47, "+", 25),
        (234, "+", 567),
        (999, "+", 1),
        (728, "-", 134),
        (123, "+", 456),
    ]
    stage_live_generation(model, tokenizer, device, showcase)

    stage_scaling(model, tokenizer, device)

    stage_v1_vs_v2(model, tokenizer, model_v1, tokenizer_v1, device)

    stage_topk(model, tokenizer, device)

    stage_interactive(model, tokenizer, device)

    console.rule(style="cyan")
    console.print("[bold cyan]thanks for watching.[/]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
