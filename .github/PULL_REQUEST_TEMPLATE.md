## Summary

Describe the problem and the chosen solution.

## Validation

- [ ] `python -m compileall -q calcgpt.py calcgpt_train.py calcgpt_eval.py calcgpt_dategen.py demo.py lib scripts examples`
- [ ] `python -m unittest discover -s tests -p "test_*.py" -v`
- [ ] Generated data or benchmark changes include their seed, configuration, and checksums.
- [ ] No model weights, checkpoints, credentials, or private data are included.

## Reproducibility and documentation

List any configuration, dataset, model-card, or documentation changes needed to reproduce the result.
