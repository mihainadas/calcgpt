# Contributing to CalcGPT

CalcGPT is an educational research project. Contributions should keep the code approachable and make experimental claims reproducible.

## Development setup

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train,demo,dev]"
```

Before submitting a change, run:

```bash
python -m compileall -q calcgpt.py calcgpt_train.py calcgpt_eval.py calcgpt_dategen.py demo.py lib scripts examples
python -m unittest discover -s tests -p "test_*.py" -v
```

Do not run the full benchmark suite merely to validate a documentation or unit-level change. The scheduled and manually dispatched reproducibility workflow handles the canonical dataset check.

## Change guidelines

- Keep pull requests focused and explain user-visible behavior.
- Add or update tests for behavior changes and regressions.
- Preserve command-line compatibility unless the pull request explicitly proposes a breaking change.
- Use fixed seeds, stable ordering, and disjoint evaluation data for empirical work.
- Record the dataset checksum, Git revision, complete experiment configuration, dependency versions, and hardware with benchmark results.
- Treat reported accuracy as an experimental result, not a universal model guarantee.

## Data and model artifacts

Small canonical datasets and test fixtures may be tracked. Generated datasets belong under `datasets/generated/`; model weights, optimizer state, checkpoints, logs, and evaluation outputs must not be committed. See [DATA_CARD.md](DATA_CARD.md) and [ARTIFACTS.md](ARTIFACTS.md).

## Pull requests

Complete the pull-request checklist and ensure CI passes on all supported Python versions. Never put credentials, private data, or embargoed security information in a pull request.

GitHub Actions currently use stable major-version tags because immutable upstream commit SHAs were not verified when the workflows were introduced. Dependabot monitors those references. A maintainer should replace them with verified full SHAs when establishing a formal release process.

## Reporting problems

Use the issue templates for bugs and feature requests. Follow [SECURITY.md](SECURITY.md) for suspected vulnerabilities.
