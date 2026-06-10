"""Generate predictions for paragraphs using a fine-tuned checkpoint."""
from __future__ import annotations

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Set

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.utils.config import load_config
from src.utils.device import limit_visible_gpus
from src.utils.random import set_global_seed

LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def _get_processed_samples(progress_file: Path) -> Set[str]:
    if not progress_file.exists():
        return set()
    try:
        with progress_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("processed_samples", []))
    except Exception as e:
        LOGGER.warning(f"Failed to load progress file {progress_file}: {e}")
        return set()

def _save_progress(progress_file: Path, processed_samples: Set[str]) -> None:
    try:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with progress_file.open("w", encoding="utf-8") as f:
            json.dump({"processed_samples": list(processed_samples)}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        LOGGER.warning(f"Failed to save progress file {progress_file}: {e}")

def _load_records(input_dir: Path, text_field: str, progress_file: Path) -> List[Dict[str, object]]:
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {input_dir}")
    processed_samples = _get_processed_samples(progress_file)
    records = []
    for jf in json_files:
        try:
            with jf.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            metadata = data.get("metadata", {})
            doi = metadata.get("doi", jf.stem)
            for para in data.get("paragraphs", []):
                if "content" not in para:
                    continue
                sid = f"{doi}_para_{para.get('index', 0)}"
                if sid in processed_samples:
                    continue
                records.append({
                    text_field: para["content"],
                    "sample_id": sid,
                    "file_path": str(jf),
                    "metadata": metadata,
                    "paragraph_index": para.get("index", 0),
                    "doi": doi,
                })
        except Exception as e:
            LOGGER.warning(f"Skip {jf}: {e}")
    LOGGER.info(f"Loaded {len(records)} new paragraphs from {len(json_files)} files")
    return records

def _build_dataset(records: List[Dict[str, object]]) -> Dataset:
    return Dataset.from_list(records)

def _preprocess_dataset(dataset: Dataset, *, tokenizer, text_field: str, max_length: int) -> Dataset:
    def preprocess(batch):
        tokenized = tokenizer(
            batch[text_field],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return tokenized
    return dataset.map(preprocess, batched=True, remove_columns=[], desc="Tokenizing")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--input", required=True, help="JSON dir")
    parser.add_argument("--output", required=True, help="Output dir")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    _configure_logging(args.verbose)
    config = load_config(args.config)
    limit_visible_gpus(config, logger=LOGGER)
    label_weights = config.get("label_fields")
    if not label_weights:
        raise ValueError("Config must define 'label_fields'")
    label_names = list(label_weights.keys())
    weights = np.array([label_weights[k] for k in label_names], dtype=float)
    weights /= weights.sum()
    set_global_seed(config.get("seed"))

    text_field = config["text_field"]
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_file = output_dir / "prediction_progress.json"

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    max_length = config["training"]["max_length"]
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if config["training"].get("mixed_precision") in {"fp16", "bf16"} else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=len(label_names),
        problem_type="regression",
        use_safetensors=True,
    )
    batch_size = args.batch_size or config["training"].get("eval_batch_size", 8)
    eval_args = TrainingArguments(
        output_dir="outputs/predict",
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=config["training"].get("dataloader_num_workers", 0),
        disable_tqdm=False,
    )
    trainer = Trainer(model=model, args=eval_args, tokenizer=tokenizer, data_collator=data_collator)

    records = _load_records(input_dir, text_field, progress_file)
    if not records:
        LOGGER.info("No new records to predict")
        return
    dataset = _build_dataset(records)
    dataset = _preprocess_dataset(dataset, tokenizer=tokenizer, text_field=text_field, max_length=max_length)

    from collections import defaultdict
    file2idx = defaultdict(list)
    for i, r in enumerate(records):
        file2idx[r["file_path"]].append(i)

    processed_samples = _get_processed_samples(progress_file)
    for file_path, idxs in file2idx.items():
        sub_ds = dataset.select(idxs)
        sub_rec = [records[i] for i in idxs]
        logits = trainer.predict(sub_ds).predictions
        if isinstance(logits, tuple):
            logits = logits[0]

        out_paras = []
        for idx, scores in enumerate(logits):
            base = sub_rec[idx]
            score_map = {label_names[li]: float(scores[li]) for li in range(len(label_names))}
            total_score = float(np.dot(scores, weights))
            processed_samples.add(base["sample_id"])
            out_paras.append({
                "index": base["paragraph_index"],
                "content": base[text_field],
                "predictions": score_map,
                "total_score": round(total_score, 6),
                "classification": 1 if total_score >= 0.5 else 0,
            })

        original_name = Path(file_path).stem
        out_file = output_dir / f"{original_name}_predictions.json"
        out_data = {
            "doi": sub_rec[0]["doi"],
            "metadata": sub_rec[0]["metadata"],
            "paragraphs": out_paras,
        }
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

        LOGGER.info("Processed %s: %d paragraphs", original_name, len(out_paras))

        _save_progress(progress_file, processed_samples)

    LOGGER.info("All predictions done -> %s", output_dir)


if __name__ == "__main__":
    main()