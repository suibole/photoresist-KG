
"""Utilities for building datasets from project JSON files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from datasets import Dataset


def _json_to_records(path: Path) -> List[Dict]:
    """Read a JSON array file into a list of dictionary records."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data)!r}")
    return data


def load_split(
    file_path: Union[str, Path],
    *,
    text_field: str,
    label_order: Iterable[str],
) -> Dataset:
    """Return a HuggingFace Dataset with text and ordered label vectors."""
    path = Path(file_path)
    label_order = list(label_order)
    records = _json_to_records(path)
    processed: List[Dict[str, object]] = []
    for idx, example in enumerate(records):
        if text_field not in example:
            raise KeyError(f"Missing text field '{text_field}' in record {idx} of {path}")
        labels = example.get("labels", {}) or {}
        vector: List[float] = []
        for name in label_order:
            if name not in labels:
                raise KeyError(f"Missing label '{name}' in record {idx} of {path}")
            vector.append(float(labels[name]))
        processed.append({text_field: example[text_field], "labels": vector})
    return Dataset.from_list(processed)


def load_datasets(
    *,
    train_file: Optional[Union[str, Path]] = None,
    val_file: Optional[Union[str, Path]] = None,
    test_file: Optional[Union[str, Path]] = None,
    text_field: str,
    label_order: Iterable[str],
) -> Dict[str, Dataset]:
    """Load the available dataset splits into a dictionary keyed by split name."""
    splits = {}
    for split_name, file_path in (
        ("train", train_file),
        ("validation", val_file),
        ("test", test_file),
    ):
        if file_path:
            splits[split_name] = load_split(
                file_path,
                text_field=text_field,
                label_order=label_order,
            )
    if not splits:
        raise ValueError("At least one dataset split must be provided")
    return splits
