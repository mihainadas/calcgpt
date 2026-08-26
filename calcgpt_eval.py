#!/usr/bin/env python3
"""CalcGPT Evaluation Tool - CLI Interface.

A command-line interface for evaluating CalcGPT models on arithmetic tasks.
Provides detailed accuracy metrics, completion analysis, and performance benchmarks.

Author: Mihai NADAS
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from lib.benchmark import holdout_manifest_sha256
from lib.version import __version__

if TYPE_CHECKING:
    from lib.evaluation import CalcGPTEvaluator

# ANSI color codes for beautiful output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_banner():
    """Print the evaluation tool banner"""
    banner = f"""
{Colors.BLUE}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════╗
║                        CalcGPT Eval                           ║
║                   Model Evaluation Tool                       ║
║{f'v{__version__}':^63}║
╚═══════════════════════════════════════════════════════════════╝
{Colors.ENDC}"""
    print(banner)

def create_config_from_args(args):
    """Create EvaluationConfig from command line arguments"""
    from lib.evaluation import EvaluationConfig

    return EvaluationConfig(
        max_tokens=args.max_tokens,
        device=args.device,
        sample_size=args.sample,
        verbose=args.verbose,
        sample_seed=args.seed,
    )

def run_evaluation_with_progress(evaluator: CalcGPTEvaluator, test_cases: List[Dict[str, str]], 
                                verbose: bool) -> List[Dict[str, Any]]:
    """Run the evaluation with progress display"""
    total = len(test_cases)
    print(f"\n{Colors.GREEN}🧪 Running evaluation on {total} test cases{Colors.ENDC}")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        if not verbose:
            print(f"\r{Colors.CYAN}Progress: {i}/{total} ({i/total*100:.1f}%){Colors.ENDC}", end='', flush=True)
        
        # Get model completion
        completion_result = evaluator.complete_expression(test_case['input'])
        
        # Validate the completion
        from lib.evaluation import validate_completion
        validation = validate_completion(
            test_case,
            completion_result['completion'],
            getattr(evaluator, 'representation_spec', None),
        )
        
        result = {
            'test_case': test_case,
            'completion_result': completion_result,
            'validation': validation
        }
        
        results.append(result)
        
        if verbose:
            status = "✅" if validation['correct_arithmetic'] else "❌"
            print(f"{status} '{test_case['input']}' → '{completion_result['completion']}' [{test_case['type']}]")
    
    if not verbose:
        print()  # New line after progress
    
    return results

def print_results(metrics: Dict[str, Any], verbose: bool):
    """Print evaluation results in a beautiful format"""
    print(f"\n{Colors.BOLD}📊 EVALUATION RESULTS{Colors.ENDC}")
    print("=" * 60)
    
    # Overall metrics
    print(f"\n{Colors.CYAN}Primary task-answer performance:{Colors.ENDC}")
    print(f"  Prompt type: {metrics['primary_task_type']}")
    print(f"  Determined test cases: {metrics['total_tests']}")
    print(f"  Successful completions: {metrics['successful_completions']} ({metrics['successful_completions_pct']:.1f}%)")
    print(f"  Valid format: {metrics['valid_format']} ({metrics['valid_format_pct']:.1f}%)")
    print(f"  Correct arithmetic: {Colors.GREEN}{metrics['correct_arithmetic']}{Colors.ENDC} ({Colors.GREEN}{metrics['correct_arithmetic_pct']:.1f}%{Colors.ENDC})")
    print(f"  Complete expressions: {metrics['complete_expressions']} ({metrics['complete_expressions_pct']:.1f}%)")
    print(f"  Exact matches: {metrics['exact_matches']} ({metrics['exact_matches_pct']:.1f}%)")
    
    diagnostics = metrics['diagnostic_all_prompts']
    print(f"  Diagnostic prompts (all types): {diagnostics['total_tests']}")

    # Performance by prompt type
    if 'by_type' in metrics:
        print(f"\n{Colors.CYAN}Performance by Test Type:{Colors.ENDC}")
        for test_type, stats in metrics['by_type'].items():
            accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            format_acc = (stats['valid_format'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"  {test_type.replace('_', ' ').title()}:")
            print(f"    Arithmetic accuracy: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
            print(f"    Format accuracy: {stats['valid_format']}/{stats['total']} ({format_acc:.1f}%)")
    
    # Timing statistics
    if 'timing' in metrics:
        timing = metrics['timing']
        print(f"\n{Colors.CYAN}Performance Timing:{Colors.ENDC}")
        if timing['count']:
            print(f"  Mean: {timing['mean_ms']:.1f}ms")
            if 'median_ms' in timing:
                print(f"  Median: {timing['median_ms']:.1f}ms")
            print(f"  Range: {timing['min_ms']:.1f}ms - {timing['max_ms']:.1f}ms")
            if 'std_ms' in timing:
                print(f"  Std Dev: {timing['std_ms']:.1f}ms")
        else:
            print("  No successful completions to time")

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _artifact_path(evaluator: Any, filename: str) -> Path | None:
    for directory in (Path(evaluator.loaded_model_path), Path(evaluator.model_path)):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def _model_hash(evaluator: Any) -> Dict[str, Any]:
    directory = Path(evaluator.loaded_model_path)
    weight_files = sorted(
        {
            *directory.glob('*.safetensors'),
            *directory.glob('*.bin'),
        },
        key=lambda path: path.name,
    )
    file_hashes = {path.name: _sha256_file(path) for path in weight_files}
    digest = hashlib.sha256()
    for filename, file_hash in file_hashes.items():
        digest.update(filename.encode('utf-8'))
        digest.update(b'\0')
        digest.update(file_hash.encode('ascii'))
        digest.update(b'\n')
    return {
        'sha256': digest.hexdigest() if file_hashes else None,
        'files': file_hashes,
    }


def _json_artifact(evaluator: Any, filename: str) -> Dict[str, Any] | None:
    path = _artifact_path(evaluator, filename)
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    return payload if isinstance(payload, dict) else None


def _git_provenance() -> Dict[str, Any]:
    source_root = Path(__file__).resolve().parent
    if not (source_root / '.git').exists():
        return {'commit': None, 'dirty': None}
    try:
        commit = subprocess.run(
            ['git', '-C', str(source_root), 'rev-parse', 'HEAD'],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ['git', '-C', str(source_root), 'status', '--porcelain'],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {'commit': commit, 'dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'dirty': None}


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def build_evaluation_report(
    evaluator: Any,
    config: Any,
    dataset_path: str,
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    benchmark_manifest_path: str | None = None,
    ablation_plan_path: str | None = None,
) -> Dict[str, Any]:
    """Build the versioned, auditable JSON evaluation report."""
    dataset = Path(dataset_path)
    config_payload = asdict(config)
    config_bytes = json.dumps(
        config_payload, sort_keys=True, separators=(',', ':')
    ).encode('utf-8')

    artifacts = {}
    for key, filename in (
        ('model_config', 'config.json'),
        ('tokenizer', 'tokenizer.json'),
        ('task_spec', 'task_spec.json'),
        ('training_manifest', 'training_manifest.json'),
    ):
        path = _artifact_path(evaluator, filename)
        artifacts[key] = {
            'path': str(path) if path else None,
            'sha256': _sha256_file(path) if path else None,
            'content': _json_artifact(evaluator, filename),
        }

    task_spec = getattr(evaluator, 'task_spec', None) or _json_artifact(
        evaluator, 'task_spec.json'
    )
    training_manifest = _json_artifact(evaluator, 'training_manifest.json') or {}
    model_config = _json_artifact(evaluator, 'config.json') or {}
    training_config = training_manifest.get('training_config', {})
    if not isinstance(training_config, dict):
        training_config = {}
    representation = None
    task_roster_hash = None
    if isinstance(task_spec, dict):
        representation = task_spec.get('name', task_spec.get('format'))
        task_roster_hash = task_spec.get('task_roster_sha256')

    training_dataset = training_manifest.get('dataset', {})
    if not isinstance(training_dataset, dict):
        training_dataset = {}
    training_splits = training_manifest.get('splits', {})
    if not isinstance(training_splits, dict):
        training_splits = {}
    model_controls = {
        'embedding_dim': model_config.get('n_embd'),
        'num_layers': model_config.get('n_layer'),
        'num_heads': model_config.get('n_head'),
        'feedforward_dim': model_config.get('n_inner'),
        'n_positions': model_config.get('n_positions'),
    }
    training_controls = {
        'epochs': training_config.get('epochs'),
        'batch_size': training_config.get('batch_size'),
        'learning_rate': training_config.get('learning_rate'),
        'warmup_steps': training_config.get('warmup_steps'),
        'weight_decay': training_config.get('weight_decay'),
        'save_steps': training_config.get('save_steps'),
        'validation_fraction': training_config.get('test_split'),
        'augmentation': (
            not training_config['no_augmentation']
            if isinstance(training_config.get('no_augmentation'), bool)
            else None
        ),
        'loss_scope': training_config.get('loss_scope'),
    }

    benchmark = {
        'manifest_path': None,
        'manifest_sha256': None,
        'task_roster_sha256': None,
        'manifest': None,
    }
    if benchmark_manifest_path is not None:
        manifest_path = Path(benchmark_manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not isinstance(manifest, dict):
            raise ValueError("benchmark manifest must contain a JSON object")
        manifest_roster_hash = manifest.get('task_roster_sha256')
        evaluation_roster_hash = getattr(
            evaluator, 'evaluation_task_roster_sha256', None
        )
        if manifest_roster_hash != evaluation_roster_hash:
            raise ValueError(
                "benchmark manifest task roster does not match evaluated tasks"
            )
        benchmark = {
            'manifest_path': str(manifest_path),
            'manifest_sha256': holdout_manifest_sha256(manifest),
            'task_roster_sha256': manifest_roster_hash,
            'manifest': manifest,
        }

    plan = {'path': None, 'sha256': None}
    if ablation_plan_path is not None:
        plan_path = Path(ablation_plan_path)
        # Parsing and exact plan validation belong to the ablation summarizer. The
        # evaluation record binds this run to the bytes selected at execution time.
        plan = {
            'path': str(plan_path.resolve()),
            'sha256': _sha256_file(plan_path),
        }

    return {
        'schema': 'calcgpt-evaluation-report',
        'schema_version': 1,
        'report_type': 'calcgpt-evaluation',
        'status': 'completed',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'calcgpt_version': __version__,
        'git': _git_provenance(),
        'dataset': {
            'path': str(dataset),
            'sha256': _sha256_file(dataset),
            'source_equations': sum(
                1 for line in dataset.read_text(encoding='utf-8').splitlines() if line.strip()
            ),
            'training_task_roster_sha256': task_roster_hash,
            'evaluation_task_roster_sha256': getattr(
                evaluator, 'evaluation_task_roster_sha256', None
            ),
        },
        'training': {
            'dataset_path': training_dataset.get('path'),
            'dataset_sha256': training_dataset.get('sha256'),
            'full_task_roster_sha256': training_dataset.get(
                'task_roster_sha256', task_roster_hash
            ),
            'train_task_roster_sha256': training_splits.get(
                'train_task_roster_sha256'
            ),
            'validation_task_roster_sha256': training_splits.get(
                'validation_task_roster_sha256'
            ),
            'target_tokens': training_manifest.get('target_tokens'),
            'model_controls': model_controls,
            'training_controls': training_controls,
        },
        'benchmark': benchmark,
        'ablation_plan': plan,
        'model': {
            'requested_path': str(evaluator.model_path),
            'loaded_path': str(evaluator.loaded_model_path),
            **_model_hash(evaluator),
        },
        'artifacts': artifacts,
        'evaluation': {
            'config': config_payload,
            'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
            'primary_task_type': metrics['primary_task_type'],
            'representation': representation,
            'model_seed': training_config.get('seed'),
            'split_seed': training_config.get('split_seed'),
            'loss_scope': training_config.get('loss_scope'),
            'representation_spec': task_spec,
            'model_controls': model_controls,
            'training_controls': training_controls,
            'plan_sha256': plan['sha256'],
        },
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'torch': _package_version('torch'),
            'transformers': _package_version('transformers'),
            'accelerate': _package_version('accelerate'),
            'device': str(evaluator.device),
        },
        'metrics': metrics,
        'detailed_results': results,
    }


def save_results(report: Dict[str, Any], output_file: str) -> None:
    """Save a report, allowing write errors to reach the CLI exit boundary."""
    Path(output_file).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )

def main():
    parser = argparse.ArgumentParser(
        description="CalcGPT Evaluation Tool - Comprehensive model assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Evaluate default model on default dataset
  %(prog)s -m ./custom_model                  # Evaluate custom model
  %(prog)s -d datasets/custom.txt             # Use custom dataset
  %(prog)s --max-tokens 20 --verbose          # Detailed output with longer generations
  %(prog)s -o evaluation_results.json         # Save detailed results to JSON
  %(prog)s --sample 100                       # Evaluate on random sample of 100 cases
        """
    )
    
    # Model options
    parser.add_argument('-m', '--model', default='auto',
                       help='Path to model directory (default: auto-detect latest model)')
    parser.add_argument(
        '--legacy-dataset',
        help='Exact training dataset used to migrate a legacy model without tokenizer.json',
    )
    parser.add_argument('--device', default='auto', 
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use for inference (default: auto)')
    
    # Dataset options
    parser.add_argument('-d', '--dataset', default='datasets/ds-calcgpt.txt',
                       help='Path to evaluation dataset (default: datasets/ds-calcgpt.txt)')
    parser.add_argument('--sample', type=int,
                       help='Evaluate a deterministic sample of N source equations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Deterministic evaluation sample seed (default: 42)')
    parser.add_argument(
        '--benchmark-manifest',
        help='Canonical semantic benchmark manifest used to produce the dataset',
    )
    parser.add_argument(
        '--ablation-plan',
        help='Ablation TOML whose exact bytes this evaluation run implements',
    )
    
    # Generation parameters
    parser.add_argument('--max-tokens', type=int, default=15,
                       help='Maximum tokens to generate (default: 15)')
    
    # Output options
    output_modes = parser.add_mutually_exclusive_group()
    output_modes.add_argument('-o', '--output', type=str,
                              help='Save the versioned JSON report to a file')
    output_modes.add_argument('--json', action='store_true',
                              help='Write only the versioned JSON report to stdout')
    parser.add_argument('--summary-only', action='store_true',
                       help='Show only summary metrics (no detailed output)')
    parser.add_argument(
        '--fail-under',
        type=float,
        help='Return nonzero when primary arithmetic accuracy is below this percentage',
    )
    
    # Utility options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output with individual test results')
    parser.add_argument('--version', action='version', version=f'CalcGPT Eval {__version__}')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress banner and verbose output')
    
    args = parser.parse_args()

    if args.sample is not None and args.sample < 1:
        parser.error("sample must be positive")
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    if args.fail_under is not None and not 0 <= args.fail_under <= 100:
        parser.error("fail-under must be between 0 and 100")

    try:
        from lib.evaluation import CalcGPTEvaluator
        from lib.inference import get_model_path
    except ModuleNotFoundError as exc:
        print(
            f"CalcGPT evaluation dependencies are unavailable ({exc}). "
            'Install them with: python -m pip install ".[train]"',
            file=sys.stderr,
        )
        return 1

    machine_output = args.json
    quiet = args.quiet or machine_output
    
    # Print banner unless suppressed
    if not quiet:
        print_banner()
    
    # Get model path
    try:
        model_path = get_model_path(args.model)
        if not quiet:
            if args.model == 'auto':
                print(f"{Colors.GREEN}🎯 Auto-detected model: {Colors.CYAN}{Path(model_path).name}{Colors.ENDC}")
            else:
                print(f"{Colors.CYAN}📁 Using model: {model_path}{Colors.ENDC}")
    except FileNotFoundError as e:
        print(f"{Colors.FAIL}❌ {e}{Colors.ENDC}", file=sys.stderr)
        return 1
    
    # Create configuration
    config = create_config_from_args(args)
    config.verbose = args.verbose and not quiet
    
    # Initialize evaluator
    if not quiet:
        print(f"{Colors.CYAN}Initializing CalcGPT evaluator...{Colors.ENDC}")
    
    try:
        evaluator = CalcGPTEvaluator(
            model_path,
            config,
            verbose=not quiet,
            legacy_dataset_path=args.legacy_dataset,
        )
    except Exception as e:
        print(
            f"{Colors.FAIL}❌ Error initializing evaluator: {e}{Colors.ENDC}",
            file=sys.stderr,
        )
        return 1
    
    # Load evaluation dataset and run evaluation
    if not quiet:
        print(f"{Colors.CYAN}Loading evaluation dataset: {args.dataset}{Colors.ENDC}")
    
    try:
        # Use the evaluator's built-in dataset evaluation method
        results, metrics = evaluator.evaluate_dataset(args.dataset)
        
        if not quiet:
            equations_count = len(set(r['test_case']['expected'] for r in results))
            print(f"{Colors.GREEN}✅ Loaded {equations_count} equations from dataset{Colors.ENDC}")
            diagnostic_total = metrics['diagnostic_all_prompts']['total_tests']
            print(f"{Colors.GREEN}✅ Generated {diagnostic_total} prompt cases{Colors.ENDC}")
            
            if config.sample_size:
                print(
                    f"{Colors.WARNING}📝 Sampled {equations_count} source equations "
                    f"with seed {config.sample_seed}{Colors.ENDC}"
                )
        
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error during evaluation: {e}{Colors.ENDC}", file=sys.stderr)
        return 1
    
    try:
        report = build_evaluation_report(
            evaluator,
            config,
            args.dataset,
            results,
            metrics,
            args.benchmark_manifest,
            args.ablation_plan,
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        else:
            print_results(metrics, args.verbose and not args.summary_only and not quiet)
            if args.output:
                save_results(report, args.output)
                if not quiet:
                    print(
                        f"\n{Colors.GREEN}📄 Evaluation report saved to: "
                        f"{args.output}{Colors.ENDC}"
                    )
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        print(
            f"{Colors.FAIL}❌ Error creating or writing evaluation report: "
            f"{exc}{Colors.ENDC}",
            file=sys.stderr,
        )
        return 1

    primary_accuracy = metrics['correct_arithmetic_pct']
    if args.fail_under is not None and primary_accuracy < args.fail_under:
        if not quiet:
            print(
                f"\n{Colors.WARNING}⚠️ Primary accuracy {primary_accuracy:.1f}% is below "
                f"--fail-under {args.fail_under:.1f}%{Colors.ENDC}"
            )
        return 1
    if not quiet:
        print(f"\n{Colors.GREEN}🎉 Evaluation completed successfully{Colors.ENDC}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
