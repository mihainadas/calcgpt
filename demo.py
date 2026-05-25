#!/usr/bin/env python3
"""
CalcGPT Live Demo — a walkthrough of a small transformer that learned to add
and subtract, and crucially generalizes far past its training range.

The model in `models/calcgpt-v2` was trained on randomly sampled 1-to-6-digit
operands with the answer written in REVERSE.  Reversing the answer lets the
decoder emit units, then tens, then hundreds, which is the natural carry
direction; it turns a memorization task into an algorithmic one.

Stages:
  1. Banner + environment summary
  2. Model architecture
  3. Token-by-token live generation (with answers un-reversed for the human)
  4. Accuracy curve by digit count, INCLUDING extrapolation past training
  5. Head-to-head against the old, in-distribution-only model
  6. Top-k probabilities for one step
  7. Interactive REPL
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
from rich.align import Align
from rich.console import Console, Group
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

from lib.tokenizer import CalcGPTTokenizer

console = Console()

BANNER = r"""
   ____      _      ____ ____ _____
  / ___|__ _| | ___/ ___|  _ \_   _|
 | |   / _` | |/ __| |  _| |_) || |
 | |__| (_| | | (__| |_| |  __/ | |
  \____\__,_|_|\___|\____|_|    |_|

  a transformer that learned arithmetic — and generalizes
"""

V2_MODEL = Path("models/calcgpt-v2")
V1_MODEL = Path("models/calcgpt-demo")
V2_DATASET = Path("datasets/ds-calcgpt-v2.txt")


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


def architecture_table(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    model_path: Path,
) -> Table:
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
    table.add_row("trained on", "1- to 6-digit operands, answers reversed")
    return table


def stage_load() -> Tuple[GPT2LMHeadModel, CalcGPTTokenizer, torch.device, Optional[GPT2LMHeadModel]]:
    device = detect_device()
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[cyan]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("Loading model…", total=None)
        # tokenizer must be the one v2 was trained with (max_length is bigger)
        tokenizer = CalcGPTTokenizer.from_dataset(V2_DATASET)
        model = load_model(V2_MODEL, device)
        v1 = load_model(V1_MODEL, device) if V1_MODEL.exists() else None
    console.print(
        Panel(
            architecture_table(model, tokenizer, device, V2_MODEL),
            title="[bold]model architecture[/]",
            border_style="cyan",
        )
    )
    return model, tokenizer, device, v1


def greedy_generate_stream(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 16,
) -> Iterator[Tuple[str, float, float]]:
    """Yield (token_str, confidence, step_latency_s) one decoded token at a time."""
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
        next_id = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0, next_id].item())
        latency = time.perf_counter() - start

        if next_id == eos_id:
            return

        token_str = tokenizer.id2char[next_id]
        yield token_str, confidence, latency
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device)], dim=1
        )


def greedy_predict(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 16,
) -> Tuple[str, float]:
    """Return the raw model emission (still reversed) and inference latency."""
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_eos=False)], dtype=torch.long
    ).to(device)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    out: List[int] = []

    start = time.perf_counter()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
        logits[:, pad_id] = float("-inf")
        nxt = int(torch.argmax(logits, dim=-1).item())
        if nxt == eos_id:
            break
        out.append(nxt)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[nxt]], device=device)], dim=1
        )
    return tokenizer.decode(out), time.perf_counter() - start


def truth(prompt: str) -> Optional[int]:
    text = prompt.rstrip("=")
    try:
        if "+" in text:
            a, b = text.split("+")
            return int(a) + int(b)
        if "-" in text:
            a, b = text.split("-")
            return int(a) - int(b)
    except ValueError:
        return None
    return None


def stage_live_generation(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    examples: List[str],
) -> None:
    console.rule("[bold cyan]live generation", style="cyan")
    console.print(
        "[dim]Watch the model emit the answer digit by digit, "
        "[bold]least-significant first[/bold]. "
        "Reverse the bold part to read the actual answer.[/dim]\n"
    )

    for prompt in examples:
        rendered = Text()
        rendered.append(prompt, style="bold white")
        with Live(rendered, console=console, refresh_per_second=20) as live:
            total_latency = 0.0
            raw_answer = ""
            for token, conf, latency in greedy_generate_stream(
                model, tokenizer, device, prompt
            ):
                total_latency += latency
                color = "green" if conf > 0.9 else "yellow" if conf > 0.5 else "red"
                rendered.append(token, style=f"bold {color}")
                raw_answer += token
                live.update(rendered)
                time.sleep(0.10)
            answer = raw_answer[::-1].lstrip("0") or "0"
            expected = truth(prompt)
            ok = (expected is not None) and (answer == str(expected))
            rendered.append("    reads as ", style="dim")
            rendered.append(f"{answer}", style="bold cyan")
            rendered.append("   ", style="dim")
            rendered.append("✓" if ok else "✗", style="bold green" if ok else "bold red")
            rendered.append(
                f"  expected {expected}   ({total_latency*1000:.1f} ms total)",
                style="dim",
            )
            live.update(rendered)
        console.print()


def sample_problems(n: int, digits: int, seed: int = 0) -> List[str]:
    rng = random.Random(seed * 1000 + digits)
    lo = 0 if digits == 1 else 10 ** (digits - 1)
    hi = 10 ** digits - 1
    out = []
    while len(out) < n:
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        op = rng.choice(["+", "-"])
        if op == "-" and a < b:
            a, b = b, a
        out.append(f"{a}{op}{b}=")
    return out


def stage_scaling(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    per_digit: int = 60,
    training_max_digits: int = 6,
    test_max_digits: int = 9,
) -> None:
    console.rule("[bold cyan]accuracy by digit count", style="cyan")
    console.print(
        f"[dim]The training set only contained operands up to "
        f"[bold]{training_max_digits} digits[/bold]. "
        f"Digit counts [bold]{training_max_digits+1}+[/bold] are pure extrapolation.[/dim]\n"
    )

    table = Table(box=None, padding=(0, 2))
    table.add_column("digits", justify="right", style="bold", width=8)
    table.add_column("region", style="dim", width=14)
    table.add_column("accuracy", justify="right", width=10)
    table.add_column("bar", width=30)
    table.add_column("avg ms", justify="right", style="dim", width=8)
    table.add_column("samples", justify="right", style="dim", width=8)

    def render(rows_done: int) -> Table:
        return table

    with Live(render(0), console=console, refresh_per_second=8) as live:
        for d in range(1, test_max_digits + 1):
            problems = sample_problems(per_digit, d, seed=d)
            correct = 0
            total_latency = 0.0
            for p in problems:
                raw, latency = greedy_predict(model, tokenizer, device, p)
                total_latency += latency
                pred = raw[::-1].lstrip("0") or "0"
                if pred == str(truth(p)):
                    correct += 1
            pct = correct / per_digit * 100
            avg_ms = total_latency / per_digit * 1000
            region = (
                "training" if d <= training_max_digits else "EXTRAPOLATION"
            )
            region_style = "dim" if d <= training_max_digits else "bold yellow"
            bar_len = int(pct / 100 * 25)
            bar = Text("█" * bar_len + "░" * (25 - bar_len))
            bar.stylize("green" if pct >= 95 else "yellow" if pct >= 50 else "red")
            table.add_row(
                str(d),
                Text(region, style=region_style),
                f"{pct:5.1f}%",
                bar,
                f"{avg_ms:5.1f}",
                f"{correct}/{per_digit}",
            )
            live.update(render(d))
    console.print()


def stage_v1_vs_v2(
    v2: GPT2LMHeadModel,
    tokenizer_v2: CalcGPTTokenizer,
    v1: Optional[GPT2LMHeadModel],
    device: torch.device,
) -> None:
    if v1 is None:
        return
    console.rule("[bold cyan]the new model vs the old", style="cyan")
    console.print(
        "[dim]Same problems, two models.  The old model was trained on "
        "operands 0–100 with answers written normally.  It memorized the "
        "table.  Watch what happens past its training range:[/dim]\n"
    )

    # The old model uses its own tokenizer — char-level, vocab is the same
    # subset (digits, ops, =) so we can reuse the v2 tokenizer for it.
    tokenizer_v1 = tokenizer_v2

    problems = ["7+8=", "67+33=", "123+456=", "999+1=", "1234+5678=", "9999-1234="]

    table = Table(box=None, padding=(0, 2))
    table.add_column("problem", style="bold", width=14)
    table.add_column("old (0–100)", width=18)
    table.add_column("new (1–6 digits)", width=18)
    table.add_column("truth", style="dim", width=10)

    for p in problems:
        v1_raw, _ = greedy_predict(v1, tokenizer_v1, device, p, max_new_tokens=8)
        v2_raw, _ = greedy_predict(v2, tokenizer_v2, device, p, max_new_tokens=16)
        v2_pred = v2_raw[::-1].lstrip("0") or "0"
        v1_pred = v1_raw.split("=", 1)[-1] if "=" in v1_raw else v1_raw
        t = truth(p)
        v1_ok = v1_pred == str(t)
        v2_ok = v2_pred == str(t)
        table.add_row(
            p,
            Text(v1_pred or "—", style="green" if v1_ok else "red"),
            Text(v2_pred or "—", style="green" if v2_ok else "red"),
            str(t),
        )
    console.print(table)
    console.print()


def stage_topk(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    prompt: str = "12345+67890=",
) -> None:
    console.rule("[bold cyan]what is the model thinking?", style="cyan")
    console.print(
        f"[dim]Top-5 next-token probabilities for [bold]{prompt}[/bold] "
        f"(should be [bold]5[/], the units digit of 80235):[/dim]\n"
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


def stage_interactive(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
) -> None:
    if not sys.stdin.isatty():
        console.print(
            "[dim]Skipping interactive REPL (no TTY). Run `python demo.py` "
            "in a terminal to chat with the model.[/dim]"
        )
        return

    console.rule("[bold cyan]your turn", style="cyan")
    console.print(
        "[dim]Type any arithmetic problem (e.g. [bold]123456+789012[/bold]).  "
        "Empty input or 'q' to quit.[/dim]\n"
    )
    while True:
        try:
            user = console.input("[bold cyan]you ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not user or user.lower() in {"q", "quit", "exit"}:
            break
        if not user.endswith("="):
            user += "="
        rendered = Text(user, style="bold white")
        with Live(rendered, console=console, refresh_per_second=20) as live:
            raw = ""
            for token, conf, _ in greedy_generate_stream(
                model, tokenizer, device, user
            ):
                color = "green" if conf > 0.9 else "yellow" if conf > 0.5 else "red"
                rendered.append(token, style=f"bold {color}")
                raw += token
                live.update(rendered)
                time.sleep(0.07)
            answer = raw[::-1].lstrip("0") or "0"
            expected = truth(user)
            rendered.append("    reads as ", style="dim")
            rendered.append(answer, style="bold cyan")
            if expected is not None:
                rendered.append("   ")
                if answer == str(expected):
                    rendered.append("correct", style="green")
                else:
                    rendered.append(f"wrong (expected {expected})", style="red")
            live.update(rendered)
        console.print()


def main() -> int:
    console.print(banner_panel())
    if not V2_MODEL.exists() or not V2_DATASET.exists():
        console.print(
            f"[red]No v2 model found at {V2_MODEL}.[/red]\n"
            "Generate the dataset and train the model first:\n"
            "  [bold]python scripts/gen_extended.py[/bold]\n"
            "  [bold]python calcgpt_train.py -d datasets/ds-calcgpt-v2.txt "
            "-o models/calcgpt-v2 --epochs 12 --batch-size 128 "
            "--embedding-dim 192 --num-layers 6 --num-heads 6 "
            "--feedforward-dim 384 --learning-rate 5e-4 --warmup-steps 300 "
            "--n-positions 48 --no-augmentation[/bold]"
        )
        return 1
    model, tokenizer, device, v1 = stage_load()
    console.print()

    showcase = ["7+8=", "234+567=", "9876+1234=", "100000-1=", "987654+12346="]
    stage_live_generation(model, tokenizer, device, showcase)

    stage_scaling(model, tokenizer, device)

    stage_v1_vs_v2(model, tokenizer, v1, device)

    stage_topk(model, tokenizer, device)

    stage_interactive(model, tokenizer, device)

    console.rule(style="cyan")
    console.print("[bold cyan]thanks for watching.[/]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
