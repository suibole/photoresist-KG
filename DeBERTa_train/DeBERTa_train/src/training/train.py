"""CLI entry point for fine-tuning DeBERTa on the paragraph scoring task."""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import load_datasets
from src.utils.config import ensure_dir, load_config
from src.utils.device import limit_visible_gpus
from src.utils.metrics import compute_regression_metrics
from src.utils.random import set_global_seed

LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _prepare_datasets(
    *,
    config: Dict,
    tokenizer,
    label_names: List[str],
) -> DatasetDict:
    text_field = config["text_field"]
    datasets_dict = load_datasets(
        train_file=config.get("train_file"),
        val_file=config.get("val_file"),
        test_file=config.get("test_file"),
        text_field=text_field,
        label_order=label_names,
    )
    dataset_dict = DatasetDict(datasets_dict)
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

    remove_columns = [text_field]
    for split_name, split_dataset in dataset_dict.items():
        dataset_dict[split_name] = split_dataset.map(
            preprocess,
            batched=True,
            remove_columns=remove_columns,
            desc=f"Tokenising {split_name} split",
        )
    return dataset_dict


def _build_training_args(config: Dict) -> TrainingArguments:
    training_cfg = config["training"]
    checkpoint_cfg = config.get("checkpointing", {})
    ensure_dir(training_cfg["output_dir"])
    precision = (training_cfg.get("mixed_precision") or "").lower()
    
    metric_for_best = training_cfg.get("metric_for_best_model", "overall_rmse")
    if not metric_for_best.startswith("eval_"):
        metric_for_best = f"eval_{metric_for_best}"
    
    args = TrainingArguments(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg["num_epochs"],
        per_device_train_batch_size=training_cfg["train_batch_size"],
        per_device_eval_batch_size=training_cfg["eval_batch_size"],
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg.get("weight_decay", 0.0),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.0),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        logging_steps=training_cfg.get("logging_steps", 10),
        save_strategy=checkpoint_cfg.get("save_strategy", "epoch"),
        save_steps=checkpoint_cfg.get("save_steps"),
        save_total_limit=training_cfg.get("save_total_limit"),
        eval_strategy=checkpoint_cfg.get("eval_strategy", "epoch"),
        eval_steps=training_cfg.get("eval_steps"),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "linear"),
        load_best_model_at_end=training_cfg.get("load_best_model_at_end", False),
        metric_for_best_model=metric_for_best,
        greater_is_better=training_cfg.get("greater_is_better"),
        seed=config.get("seed"),
        report_to=checkpoint_cfg.get("report_to", []),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 0),
        disable_tqdm=False,
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
    )
    return args


class MetricsRecorderCallback(TrainerCallback):
    def __init__(self) -> None:
        self.records: List[Dict[str, float]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = int(logs.get("step", state.global_step))
        entry: Dict[str, float] = {"step": float(step)}
        if any(key.startswith("eval_") for key in logs):
            entry["split"] = "eval"
        else:
            entry["split"] = "train"
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                entry[key] = float(value)
        self.records.append(entry)


def _export_training_curves(
    *,
    records: List[Dict[str, float]],
    label_names: List[str],
    output_dir: Path,
) -> None:
    if not records:
        LOGGER.warning("No metric records captured; skipping export")
        return

    per_step: Dict[int, Dict[str, float]] = defaultdict(lambda: {})
    for entry in records:
        step = int(entry.get("step", 0))
        record = per_step[step]
        record.setdefault("step", float(step))
        split = entry.get("split")
        if split == "train" and "loss" in entry:
            record["train_loss"] = entry["loss"]
        if split == "eval":
            if "eval_loss" in entry:
                record["eval_loss"] = entry["eval_loss"]
            for key, value in entry.items():
                if key.startswith("eval_"):
                    record[key] = value

    if not per_step:
        LOGGER.warning("Metric records did not include any step-aligned data")
        return

    ordered_steps = sorted(per_step)
    rows = [per_step[step] for step in ordered_steps]
    metric_keys = sorted({key for row in rows for key in row.keys() if key != "step"})
    csv_path = output_dir / "metrics_history.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["step"] + metric_keys
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    LOGGER.info("Saved training log metrics to %s", csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    for idx, label in enumerate(label_names):
        ax = axes[idx]
        metric_key = f"eval_{label}_rmse"
        steps = [row["step"] for row in rows if metric_key in row]
        values = [row[metric_key] for row in rows if metric_key in row]
        if not steps:
            ax.set_visible(False)
            continue
        ax.plot(steps, values, marker="o", linewidth=1.5)
        ax.set_title(f"{label} RMSE")
        ax.set_xlabel("Step")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.3)

    for ax in axes[len(label_names) :]:
        ax.set_visible(False)

    fig.tight_layout()
    fig_path = output_dir / "metrics_history.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved training metric plot to %s", fig_path)


def _make_compute_metrics(label_names: Iterable[str], weights: Dict[str, float]):
    ordered_labels = list(label_names)
    weight_map = dict(weights)

    def compute(eval_prediction):
        predictions, references = eval_prediction.predictions, eval_prediction.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        metrics = compute_regression_metrics(
            predictions,
            references,
            label_names=ordered_labels,
            weights=weight_map,
        )
        return metrics

    return compute


def _save_best_model(trainer, output_dir: Path) -> None:
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    
    if trainer.state.best_model_checkpoint:
        LOGGER.info(f"Found best model checkpoint: {trainer.state.best_model_checkpoint}")
        checkpoint_dir = Path(trainer.state.best_model_checkpoint)
        
        for file_path in checkpoint_dir.iterdir():
            if file_path.is_file():
                shutil.copy2(file_path, best_dir / file_path.name)
    else:
        LOGGER.warning("No best model checkpoint found, using final model")
        
        model_files = [
            "config.json", "model.safetensors", "special_tokens_map.json",
            "tokenizer_config.json", "tokenizer.json", "training_args.bin"
        ]
        
        for file_name in model_files:
            source_file = output_dir / file_name
            if source_file.exists():
                shutil.copy2(source_file, best_dir / file_name)
    
    tokenizer_files = ["spm.model", "tokenizer.model"]
    for file_name in tokenizer_files:
        source_file = output_dir / file_name
        if source_file.exists():
            shutil.copy2(source_file, best_dir / file_name)
    
    LOGGER.info(f"Best model saved to: {best_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Optional path to a checkpoint directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_logging(args.verbose)
    config = load_config(args.config)
    limit_visible_gpus(config, logger=LOGGER)
    label_weights = config.get("label_fields")
    if not label_weights:
        raise ValueError("Config must define 'label_fields' with label weights")
    label_names = list(label_weights.keys())
    set_global_seed(config.get("seed"))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    datasets = _prepare_datasets(config=config, tokenizer=tokenizer, label_names=label_names)
    pad_multiple = 8 if (config["training"].get("mixed_precision", "").lower() in {"fp16", "bf16"}) else None
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=pad_multiple)
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=len(label_names),
        problem_type="regression",
        use_safetensors=True,
    )
    training_args = _build_training_args(config)
    compute_metrics = _make_compute_metrics(label_names, label_weights)

    metrics_callback = MetricsRecorderCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets.get("train"),
        eval_dataset=datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback],
    )

    if datasets.get("train") is None:
        raise ValueError("Training split is required for fine-tuning")

    LOGGER.info("Starting training with %s examples", len(datasets["train"]))
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    if datasets.get("validation") is not None:
        LOGGER.info(
            "Running evaluation on validation split (%s examples)",
            len(datasets["validation"]),
        )
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    _save_best_model(trainer, Path(training_args.output_dir))

    _export_training_curves(
        records=metrics_callback.records,
        label_names=label_names,
        output_dir=Path(training_args.output_dir),
    )


if __name__ == "__main__":
    main()