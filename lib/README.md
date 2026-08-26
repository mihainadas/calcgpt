# CalcGPT library modules

The `lib` package remains for backwards compatibility with the original project.
Its public exports are lazy, so importing dataset or tokenizer helpers does not
require PyTorch.

Prefer explicit imports:

```python
from lib.dategen import DatagenConfig, DatasetGenerator
from lib.tokenizer import CalcGPTTokenizer
from lib.train import CalcGPTTrainer, TrainingConfig
from lib.inference import CalcGPT, InferenceConfig
from lib.evaluation import CalcGPTEvaluator, EvaluationConfig
```

New model artifacts are self-contained. Training writes:

- Hugging Face model configuration and Safetensors weights;
- `tokenizer.json`, containing the exact ordered vocabulary;
- `task_spec.json`, containing the input/output representation contract;
- `training_manifest.json`, containing the dataset checksum, grouped split
  counts, seed, task format, package versions, metrics, and Git revision.

Inference rejects a model that lacks its tokenizer. To migrate a legacy model,
pass the exact original dataset through the CLI's `--legacy-dataset` option,
then save the reconstructed tokenizer beside the model.

The stable public commands and reproducibility protocol are documented in the
repository's top-level `README.md`, `ARTIFACTS.md`, and `DATA_CARD.md`.
