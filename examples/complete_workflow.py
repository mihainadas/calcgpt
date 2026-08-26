#!/usr/bin/env python3
# ruff: noqa: E402
"""Small, executable end-to-end CalcGPT workflow.

This example intentionally uses a tiny dataset and model. It validates the public
APIs and artifact round-trip; it is not a meaningful accuracy benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.dategen import DatagenConfig, DatasetGenerator
from lib.evaluation import CalcGPTEvaluator, EvaluationConfig
from lib.inference import CalcGPT, InferenceConfig
from lib.train import CalcGPTTrainer, TrainingConfig


def main() -> int:
    dataset_path = PROJECT_ROOT / "datasets" / "example-workflow.txt"
    model_dir = PROJECT_ROOT / "models" / "example-workflow"

    print("1/4 Generating a tiny arithmetic dataset")
    generation = DatasetGenerator(
        DatagenConfig(max_value=15, max_expressions=100), verbose=False
    ).generate_dataset(dataset_path)
    print(f"    {generation['expressions_generated']} rows -> {dataset_path}")

    print("2/4 Training a tiny smoke-test model")
    trainer = CalcGPTTrainer(
        config=TrainingConfig(
            epochs=1,
            batch_size=8,
            embedding_dim=32,
            num_layers=1,
            num_heads=4,
            feedforward_dim=64,
            save_steps=1000,
            seed=42,
        ),
        dataset_path=dataset_path,
        output_dir=model_dir,
        verbose=False,
    )
    training = trainer.train()
    print(f"    loss={training['training_loss']:.4f} -> {model_dir}")

    print("3/4 Evaluating a deterministic sample")
    evaluator = CalcGPTEvaluator(
        str(model_dir),
        EvaluationConfig(sample_size=30, max_tokens=10, sample_seed=42),
        verbose=False,
    )
    _, metrics = evaluator.evaluate_dataset(str(dataset_path))
    print(f"    arithmetic accuracy={metrics['correct_arithmetic_pct']:.1f}%")

    print("4/4 Reloading the self-contained model artifact")
    model = CalcGPT(
        str(model_dir),
        InferenceConfig(temperature=0.0, max_tokens=10),
        verbose=False,
    )
    for problem in ("5+3", "10-4", "7+8"):
        result = model.solve(problem)
        print(f"    {problem} -> {result.get('answer', result.get('error'))}")

    print("Workflow complete. This was a plumbing smoke test, not a benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
