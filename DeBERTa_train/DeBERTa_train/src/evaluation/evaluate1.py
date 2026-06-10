from __future__ import annotations

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                            confusion_matrix, precision_score, recall_score, f1_score,
                            classification_report)

from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data.dataset import load_datasets
from src.utils.config import load_config
from src.utils.device import limit_visible_gpus
from src.utils.metrics import compute_regression_metrics
from src.utils.random import set_global_seed

LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool = True) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _prepare_dataset(
    *,
    config: Dict,
    tokenizer,
    label_names: List[str],
    split: str,
):
    text_field = config["text_field"]
    datasets_dict = load_datasets(
        train_file=config.get("train_file"),
        val_file=config.get("val_file"),
        test_file=config.get("test_file"),
        text_field=text_field,
        label_order=label_names,
    )
    dataset_dict = DatasetDict(datasets_dict)
    if split not in dataset_dict:
        raise ValueError(f"Requested split '{split}' is not available")
    max_length = config["training"]["max_length"]

    def preprocess(batch):
        tokenized = tokenizer(
            batch[text_field],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokenized["labels"] = batch["labels"]
        return tokenized

    processed_dataset = dataset_dict[split].map(
        preprocess,
        batched=True,
        remove_columns=[text_field],
        desc=f"Tokenising {split} split",
    )
    return processed_dataset, dataset_dict[split]


def calculate_total_score(scores: np.ndarray, label_weights: Dict[str, float], label_names: List[str]) -> np.ndarray:
    weights = np.array([label_weights[name] for name in label_names])
    weights = weights / weights.sum()
    total_scores = np.dot(scores, weights)
    return total_scores


def classify_paragraphs(total_scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (total_scores >= threshold).astype(int)


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    accuracy = np.mean(y_true == y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
 
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        if len(np.unique(y_true)) == 1:
            if np.unique(y_true)[0] == 0:
                tn, fp, fn, tp = len(y_true), 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, len(y_true)
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_positive': int(tp),
        'specificity': specificity,
        'false_positive_rate': false_positive_rate
    }


def evaluate_split(
    *,
    trainer: Trainer,
    dataset,
    original_dataset,
    label_names: List[str],
    label_weights: Dict[str, float],
    output_dir: Path,
    split_name: str,
    text_field: str = "text"
) -> Dict[str, float]:
    LOGGER.info(f"Evaluating {split_name} split, samples: {len(dataset)}")

    predictions_output = trainer.predict(dataset)
    predictions = predictions_output.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    classification_threshold = 0.5
    
    predicted_total_scores = calculate_total_score(predictions, label_weights, label_names)
    predicted_classifications = classify_paragraphs(predicted_total_scores, threshold=classification_threshold)

    original_scores = np.array(original_dataset["labels"])
    original_total_scores = calculate_total_score(original_scores, label_weights, label_names)
    original_classifications = classify_paragraphs(original_total_scores, threshold=classification_threshold)
    predicted_total_scores = calculate_total_score(predictions, label_weights, label_names)
    predicted_classifications = classify_paragraphs(predicted_total_scores)

    evaluation_results = []
    for i in range(len(dataset)):
        result = {
            'text': original_dataset[i][text_field],
            'predicted_scores': {name: float(predictions[i][j]) for j, name in enumerate(label_names)},
            'predicted_total_score': float(predicted_total_scores[i]),
            'predicted_classification': int(predicted_classifications[i]),
            'original_scores': {name: float(original_scores[i][j]) for j, name in enumerate(label_names)},
            'original_total_score': float(original_total_scores[i]),
            'original_classification': int(original_classifications[i]),
        }
        evaluation_results.append(result)

    save_detailed_predictions(
        evaluation_results=evaluation_results,
        output_dir=output_dir,
        split_name=split_name,
        label_names=label_names
    )
    save_model_predictions(
        evaluation_results=evaluation_results,
        output_dir=output_dir,
        split_name=split_name,
        label_names=label_names
    )

    metrics = calculate_evaluation_metrics(evaluation_results, label_names)
    metrics_file = output_dir / f"{split_name}_evaluation_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    save_classification_report(
        original_classifications,
        predicted_classifications,
        output_dir,
        split_name
    )

    LOGGER.info(f"{split_name} split evaluation completed")
    LOGGER.info(f"Total score - R2: {metrics['total_score_r2']:.4f}, MAE: {metrics['total_score_mae']:.4f}, RMSE: {metrics['total_score_rmse']:.4f}")
    LOGGER.info(f"Classification - Accuracy: {metrics['classification_accuracy']:.4f}, Precision: {metrics['classification_precision']:.4f}, Recall: {metrics['classification_recall']:.4f}, F1: {metrics['classification_f1']:.4f}")
    
    cm = metrics['classification_confusion_matrix']
    LOGGER.info(f"Confusion matrix:")
    LOGGER.info(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
    LOGGER.info(f"FN: {cm[1][0]}, TP: {cm[1][1]}")

    return metrics


def calculate_evaluation_metrics(evaluation_results: List[Dict[str, Any]], label_names: List[str]) -> Dict[str, float]:
    if not evaluation_results:
        return {}

    predicted_total_scores = np.array([r['predicted_total_score'] for r in evaluation_results])
    original_total_scores = np.array([r['original_total_score'] for r in evaluation_results])
    predicted_classifications = np.array([r['predicted_classification'] for r in evaluation_results])
    original_classifications = np.array([r['original_classification'] for r in evaluation_results])

    total_score_r2 = r2_score(original_total_scores, predicted_total_scores)
    total_score_mae = mean_absolute_error(original_total_scores, predicted_total_scores)
    total_score_rmse = np.sqrt(mean_squared_error(original_total_scores, predicted_total_scores))
    
    classification_metrics = calculate_classification_metrics(original_classifications, predicted_classifications)

    dimension_metrics = {}
    for name in label_names:
        original_dim_scores = np.array([r['original_scores'][name] for r in evaluation_results])
        predicted_dim_scores = np.array([r['predicted_scores'][name] for r in evaluation_results])
        dimension_metrics[f'{name}_r2'] = r2_score(original_dim_scores, predicted_dim_scores)
        dimension_metrics[f'{name}_mae'] = mean_absolute_error(original_dim_scores, predicted_dim_scores)
        dimension_metrics[f'{name}_rmse'] = np.sqrt(mean_squared_error(original_dim_scores, predicted_dim_scores))

    return {
        'total_score_r2': total_score_r2,
        'total_score_mae': total_score_mae,
        'total_score_rmse': total_score_rmse,
        'classification_accuracy': classification_metrics['accuracy'],
        'classification_precision': classification_metrics['precision'],
        'classification_recall': classification_metrics['recall'],
        'classification_f1': classification_metrics['f1_score'],
        'classification_confusion_matrix': classification_metrics['confusion_matrix'],
        'classification_true_negative': classification_metrics['true_negative'],
        'classification_false_positive': classification_metrics['false_positive'],
        'classification_false_negative': classification_metrics['false_negative'],
        'classification_true_positive': classification_metrics['true_positive'],
        'classification_specificity': classification_metrics['specificity'],
        'classification_false_positive_rate': classification_metrics['false_positive_rate'],
        'num_samples': len(evaluation_results),
        **dimension_metrics
    }


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path, split_name: str) -> None:
    report = classification_report(y_true, y_pred, target_names=['unusable', 'usable'], output_dict=True)
    
    report_file = output_dir / f"{split_name}_classification_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
  
    text_report = classification_report(y_true, y_pred, target_names=['unusable', 'usable'])
    text_file = output_dir / f"{split_name}_classification_report.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("Classification Report\n")
        f.write("====================\n\n")
        f.write(text_report)
        f.write(f"\n\nTotal samples: {len(y_true)}")
        f.write(f"\nPositive samples (usable): {np.sum(y_true)}")
        f.write(f"\nNegative samples (unusable): {len(y_true) - np.sum(y_true)}")
    
    LOGGER.info(f"Saved classification report to: {report_file} and {text_file}")


def save_detailed_predictions(
    evaluation_results: List[Dict[str, Any]],
    output_dir: Path,
    split_name: str,
    label_names: List[str]
) -> None:
    if not evaluation_results:
        return
    csv_data = []
    for result in evaluation_results:
        row = {
            'text': result['text'],
            'predicted_total_score': result['predicted_total_score'],
            'original_total_score': result['original_total_score'],
            'predicted_classification': result['predicted_classification'],
            'original_classification': result['original_classification'],
            'total_score_diff': abs(result['predicted_total_score'] - result['original_total_score']),
            'classification_match': int(result['predicted_classification'] == result['original_classification'])
        }
        for name in label_names:
            row[f'predicted_{name}'] = result['predicted_scores'][name]
            row[f'original_{name}'] = result['original_scores'][name]
            row[f'{name}_diff'] = abs(result['predicted_scores'][name] - result['original_scores'][name])
        csv_data.append(row)
    df = pd.DataFrame(csv_data)
    output_file = output_dir / f"{split_name}_detailed_predictions.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    LOGGER.info(f"Saved detailed predictions to: {output_file}")


def save_model_predictions(
    evaluation_results: List[Dict[str, Any]],
    output_dir: Path,
    split_name: str,
    label_names: List[str]
) -> None:
    if not evaluation_results:
        return
    output_data = []
    for result in evaluation_results:
        item = {
            'text': result['text'],
            'predicted_total_score': result['predicted_total_score'],
            'predicted_classification': result['predicted_classification'],
        }
        for name in label_names:
            item[f'predicted_{name}'] = result['predicted_scores'][name]
        output_data.append(item)
    output_file = output_dir / f"{split_name}_model_predictions.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    df = pd.DataFrame(output_data)
    csv_file = output_dir / f"{split_name}_model_predictions.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')
    LOGGER.info(f"Saved model predictions to: {output_file} and {csv_file}")


def _make_compute_metrics(label_names: List[str], weights: Dict[str, float]):
    ordered = list(label_names)
    weight_map = dict(weights)

    def compute(eval_prediction):
        predictions, references = eval_prediction.predictions, eval_prediction.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        return compute_regression_metrics(
            predictions,
            references,
            label_names=ordered,
            weights=weight_map,
        )

    return compute


def main() -> None:
    config_path = ""
    checkpoint_path = ""
    output_path = ""
    
    splits_to_evaluate = ["train", "validation", "test"]

    _configure_logging(verbose=True)
    config = load_config(config_path)
    limit_visible_gpus(config, logger=LOGGER)

    label_weights = config.get("label_fields")
    if not label_weights:
        raise ValueError("Config must define 'label_fields' with label weights")
    label_names = list(label_weights.keys())

    set_global_seed(config.get("seed"))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)

    pad_multiple = 8 if (config["training"].get("mixed_precision", "").lower() in {"fp16", "bf16"}) else None
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=pad_multiple)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        num_labels=len(label_names),
        problem_type="regression",
        use_safetensors=True,
    )

    training_cfg = config.get("training", {})
    eval_args = TrainingArguments(
        output_dir=training_cfg.get("output_dir", "outputs/eval"),
        per_device_eval_batch_size=training_cfg.get("eval_batch_size", 8),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 0),
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_make_compute_metrics(label_names, label_weights),
    )

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for split in splits_to_evaluate:
        try:
            tokenized_dataset, original_dataset = _prepare_dataset(
                config=config,
                tokenizer=tokenizer,
                label_names=label_names,
                split=split,
            )
            LOGGER.info("Evaluating %s on %s split (%s examples)", checkpoint_path, split, len(tokenized_dataset))

            metrics = evaluate_split(
                trainer=trainer,
                dataset=tokenized_dataset,
                original_dataset=original_dataset,
                label_names=label_names,
                label_weights=label_weights,
                output_dir=output_dir,
                split_name=split,
                text_field=config["text_field"]
            )
            all_metrics[split] = metrics

            LOGGER.info("=== Results for %s split ===", split)
            for key, value in metrics.items():
                if isinstance(value, float):
                    LOGGER.info("%s = %.6f", key, value)
                else:
                    LOGGER.info("%s = %s", key, value)
            LOGGER.info("=== End of %s split results ===", split)

        except ValueError as e:
            LOGGER.warning("Skipping split '%s': %s", split, e)
            continue

    summary_file = output_dir / "evaluation_summary.json"
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)
    LOGGER.info("Saved all metrics to %s", summary_file)


if __name__ == "__main__":
    main()