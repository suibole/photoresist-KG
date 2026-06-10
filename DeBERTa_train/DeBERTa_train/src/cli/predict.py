"""Common utilities for both training and prediction."""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml
from datasets import Dataset
from transformers import AutoTokenizer

def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int | None = None) -> None:
    if seed is not None:
        random.seed(seed)
        import numpy as np
        import torch
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def limit_visible_gpus(config: Dict[str, Any], logger: logging.Logger | None = None) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)
    gpu_ids = config.get("gpu_ids")
    if gpu_ids is not None:
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        visible = ",".join(str(gid) for gid in gpu_ids)
        logger.info("Set CUDA_VISIBLE_DEVICES=%s", visible)
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = visible

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def _preprocess_dataset(
    dataset: Dataset,
    *,
    tokenizer: AutoTokenizer,
    text_field: str,
    max_length: int,
) -> Dataset:
    def tokenize(batch):
        return tokenizer(
            batch[text_field],
            truncation=True,
            max_length=max_length,
            padding=False,  
        )

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=[],  
        desc="Tokenising",
    )