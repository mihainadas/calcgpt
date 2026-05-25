#!/usr/bin/env python3
"""
CalcGPT Live Demo — a polished walkthrough of the trained model.

Stages:
  1. Banner + environment summary
  2. Model architecture + load
  3. Token-by-token live generation
  4. Stress test on random problems with a live results table
  5. Confidence inspection — show top-k probabilities for a single step
  6. Interactive REPL
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from transformers import GPT2LMHeadModel

from lib.inference import find_latest_model
from lib.tokenizer import CalcGPTTokenizer

console = Console()

BANNER = r"""
   ____      _      ____ ____ _____
  / ___|__ _| | ___/ ___|  _ \_   _|
 | |   / _` | |/ __| |  _| |_) || |
 | |__| (_| | | (__| |_| |  __/ | |
  \____\__,_|_|\___|\____|_|    |_|

         a transformer that learned arithmetic
"""


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


def load_model_and_tokenizer(
    model_path: Path, device: torch.device
) -> Tuple[GPT2LMHeadModel, CalcGPTTokenizer]:
    tokenizer = CalcGPTTokenizer.from_dataset()
    model = GPT2LMHeadModel.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    return model, tokenizer


def architecture_table(model: GPT2LMHeadModel, tokenizer: CalcGPTTokenizer, device: torch.device, model_path: Path) -> Table:
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
    return table


def stage_load(model_path: Path) -> Tuple[GPT2LMHeadModel, CalcGPTTokenizer, torch.device]:
    device = detect_device()
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[cyan]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Loading model…", total=None)
        model, tokenizer = load_model_and_tokenizer(model_path, device)
        progress.update(task, description="Loaded")
    console.print(
        Panel(
            architecture_table(model, tokenizer, device, model_path),
            title="[bold]model architecture[/]",
            border_style="cyan",
        )
    )
    return model, tokenizer, device


def greedy_generate_stream(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 8,
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
        # Mask out pad so the model can't generate it
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


def stage_live_generation(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    examples: List[str],
) -> None:
    console.rule("[bold cyan]live generation", style="cyan")
    console.print(
        "[dim]Watch the model produce each token one step at a time."
        " The colored fraction is the softmax confidence.[/dim]\n"
    )

    for prompt in examples:
        rendered = Text()
        rendered.append(prompt, style="bold white")
        with Live(rendered, console=console, refresh_per_second=20) as live:
            total_latency = 0.0
            for token, conf, latency in greedy_generate_stream(
                model, tokenizer, device, prompt
            ):
                total_latency += latency
                color = "green" if conf > 0.9 else "yellow" if conf > 0.5 else "red"
                rendered.append(token, style=f"bold {color}")
                live.update(rendered)
                time.sleep(0.12)  # so the human can see it
            answer = rendered.plain.split("=", 1)[1] if "=" in rendered.plain else "?"
            expected = ground_truth(prompt)
            ok = answer.strip() == str(expected)
            rendered.append("    ")
            rendered.append("✓" if ok else "✗", style="bold green" if ok else "bold red")
            rendered.append(
                f"  expected {expected}   ({total_latency*1000:.1f} ms total)",
                style="dim",
            )
            live.update(rendered)
        console.print()


def ground_truth(prompt: str) -> int | None:
    """Compute the correct answer for a normalized 'a+b=' or 'a-b=' prompt."""
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


def greedy_answer(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 8,
) -> Tuple[str, float]:
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_eos=False)], dtype=torch.long
    ).to(device)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    out_tokens: List[int] = []

    start = time.perf_counter()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
        logits[:, pad_id] = float("-inf")
        next_id = int(torch.argmax(logits, dim=-1).item())
        if next_id == eos_id:
            break
        out_tokens.append(next_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device)], dim=1
        )
    latency = time.perf_counter() - start
    return tokenizer.decode(out_tokens), latency


def sample_problems(n: int) -> List[str]:
    rng = random.Random(7)
    problems = []
    while len(problems) < n:
        a = rng.randint(0, 100)
        b = rng.randint(0, 100)
        if rng.random() < 0.5:
            problems.append(f"{a}+{b}=")
        elif a >= b:
            problems.append(f"{a}-{b}=")
    return problems


def stage_stress_test(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    n: int = 40,
) -> None:
    console.rule("[bold cyan]stress test", style="cyan")
    console.print(
        f"[dim]Sampling {n} random problems and grading the model in real time.[/dim]\n"
    )

    problems = sample_problems(n)

    table = Table(box=None, padding=(0, 2))
    table.add_column("#", style="dim", justify="right", width=3)
    table.add_column("problem", style="bold", width=12)
    table.add_column("model", width=10)
    table.add_column("truth", style="dim", width=8)
    table.add_column("ms", justify="right", style="dim", width=6)
    table.add_column("ok", justify="center", width=3)

    correct = 0
    total_latency = 0.0

    def render(rows_done: int) -> Group:
        acc = correct / rows_done * 100 if rows_done else 0.0
        avg = total_latency / rows_done * 1000 if rows_done else 0.0
        summary = Text()
        summary.append(f"  {rows_done}/{n}  ", style="bold")
        summary.append(f"accuracy {acc:5.1f}%  ", style="green" if acc >= 90 else "yellow")
        summary.append(f"avg {avg:5.1f} ms/sample", style="dim")
        return Group(table, summary)

    with Live(render(0), console=console, refresh_per_second=15) as live:
        for i, problem in enumerate(problems, start=1):
            ans, latency = greedy_answer(model, tokenizer, device, problem)
            total_latency += latency
            # The model echoes the prompt; isolate the predicted RHS
            predicted = ans.split("=", 1)[1].strip() if "=" in ans else ans.strip()
            expected = ground_truth(problem)
            ok = predicted == str(expected)
            if ok:
                correct += 1
            table.add_row(
                str(i),
                problem,
                Text(predicted or "—", style="green" if ok else "red"),
                str(expected),
                f"{latency*1000:.1f}",
                "[green]✓[/]" if ok else "[red]✗[/]",
            )
            live.update(render(i))


def stage_topk(
    model: GPT2LMHeadModel,
    tokenizer: CalcGPTTokenizer,
    device: torch.device,
    prompt: str = "47+25=",
) -> None:
    console.rule("[bold cyan]what is the model thinking?", style="cyan")
    console.print(
        f"[dim]Top-5 next-token probabilities for [bold]{prompt}[/bold]:[/dim]\n"
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
            "[dim]Skipping interactive REPL (no TTY). Run `python demo.py` in a terminal to chat with the model.[/dim]"
        )
        return

    console.rule("[bold cyan]your turn", style="cyan")
    console.print(
        "[dim]Type an arithmetic problem (e.g. [bold]23+58[/bold]).  Empty input or 'q' to quit.[/dim]\n"
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
            for token, conf, _ in greedy_generate_stream(
                model, tokenizer, device, user
            ):
                color = "green" if conf > 0.9 else "yellow" if conf > 0.5 else "red"
                rendered.append(token, style=f"bold {color}")
                live.update(rendered)
                time.sleep(0.08)
            predicted = rendered.plain.split("=", 1)[1] if "=" in rendered.plain else "?"
            expected = ground_truth(user)
            if expected is not None:
                ok = predicted.strip() == str(expected)
                rendered.append("   ")
                if ok:
                    rendered.append("correct", style="green")
                else:
                    rendered.append(f"wrong (expected {expected})", style="red")
                live.update(rendered)
        console.print()


def main() -> int:
    console.print(banner_panel())

    model_path_str = find_latest_model()
    if not model_path_str:
        console.print(
            "[red]No trained model found.[/red] Train one first:\n"
            "  [bold]python calcgpt_train.py --epochs 30 --batch-size 64 "
            "--embedding-dim 128 --num-layers 4 --num-heads 8 -o models/calcgpt-demo[/bold]"
        )
        return 1
    model_path = Path(model_path_str)

    model, tokenizer, device = stage_load(model_path)
    console.print()

    showcase = ["7+8=", "23+58=", "99-50=", "100-1=", "42+42="]
    stage_live_generation(model, tokenizer, device, showcase)

    stage_stress_test(model, tokenizer, device, n=40)
    console.print()

    stage_topk(model, tokenizer, device)

    stage_interactive(model, tokenizer, device)

    console.rule(style="cyan")
    console.print("[bold cyan]thanks for watching.[/]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
