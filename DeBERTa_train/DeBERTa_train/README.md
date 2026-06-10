# DeBERTa Paragraph Scoring Module

This module fine-tunes a DeBERTa-v3-base regression model for paragraph-level value scoring in the photoresist knowledge graph pipeline.

The model predicts four continuous dimensions for each paragraph:

- `relevance`: topical relevance to photoresist and related lithography materials.
- `info_density`: density of technical details, numerical values, and extractable facts.
- `structure`: clarity of causal, parameter-value, or relation-like structure.
- `factual`: degree to which the paragraph states verifiable facts rather than unsupported opinions.

Each score is expected to fall in the range `[0, 1]`. The downstream overall score is computed as a weighted average using the weights configured in `configs/base.yaml`.

## Directory Layout

```text
.
|-- README.md
|-- configs/
|   `-- base.yaml
|-- requirements.txt
`-- src/
    |-- cli/
    |   |-- predict.py
    |   `-- run_predict.py
    |-- data/
    |   `-- dataset.py
    |-- evaluation/
    |   `-- evaluate1.py
    |-- training/
    |   `-- train.py
    `-- utils/
```

## Input Data Format

Training, validation, and test files are JSON arrays. Each record should contain a paragraph and four continuous labels.

```json
[
  {
    "paragraph": "Positive photoresist BP212 was spin coated at 5000 r/min ...",
    "labels": {
      "relevance": 0.85,
      "info_density": 0.75,
      "structure": 0.70,
      "factual": 0.90
    }
  }
]
```

The text field name and label order are configured in `configs/base.yaml`.

## Installation

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU training requires a PyTorch build compatible with the local CUDA runtime.

## Configuration

Edit `configs/base.yaml` before training:

- `model_name`: Hugging Face model name or local checkpoint path.
- `train_file`, `val_file`, `test_file`: paths to supervised JSON files.
- `training.output_dir`: directory for checkpoints and metrics.
- `training.max_length`: maximum token length per paragraph.
- `label_fields`: label names and weights used for aggregate metrics and prediction totals.

The current training entry point uses Hugging Face's standard regression loss through `AutoModelForSequenceClassification(problem_type="regression")`. The weights in `label_fields` are used for aggregate metrics and prediction totals, not as custom loss weights. Custom task-loss weighting, focal loss, data augmentation, and early stopping are not enabled in this released training script.

## Training

```bash
python -m src.training.train --config configs/base.yaml
```

The training script saves:

- Hugging Face checkpoints under `training.output_dir`.
- The final model and trainer state.
- A `best/` directory copied from the best checkpoint when available.
- Metric logs and training curves.

## Prediction

Use `src.cli.run_predict` for batch inference over paragraph JSON files:

```bash
python -m src.cli.run_predict \
  --config configs/base.yaml \
  --checkpoint outputs/base_retrain/best \
  --input path/to/paragraph_json \
  --output outputs/predictions
```

Prediction output contains per-label scores, a weighted `total_score`, and a binary `classification` flag based on a threshold of `0.5`.

## Evaluation

The evaluation module computes:

- Dimension-wise MAE, RMSE, and R2.
- Weighted total-score metrics.
- Binary classification metrics after thresholding the weighted total score.
- Detailed prediction CSV/JSON files.

Before using `src.evaluation.evaluate1`, set its local `config_path`, `checkpoint_path`, and `output_path` variables, or refactor it to command-line arguments for public release.

## Notes for Public Release

Do not commit supervised training data, large checkpoints, generated prediction files, or `__pycache__` directories. If checkpoints are released, host them separately and document the exact path or download command here.
