"""GPU visibility utilities."""
from __future__ import annotations

import logging
import os
from typing import Mapping, MutableMapping


def limit_visible_gpus(
    config: Mapping[str, object],
    *,
    logger: logging.Logger,
    env: MutableMapping[str, str] | None = None,
) -> None:
    """Restrict execution to a single GPU unless explicitly overridden.

    Parameters
    ----------
    config:
        Training configuration dictionary; expects optional ``training.device_id``.
    logger:
        Logger used to emit status updates.
    env:
        Environment mapping to mutate, defaults to :data:`os.environ`.
    """

    env = os.environ if env is None else env
    if env.get("CUDA_VISIBLE_DEVICES"):
        logger.info(
            "Respecting existing CUDA_VISIBLE_DEVICES=%s",
            env["CUDA_VISIBLE_DEVICES"],
        )
        return

    training_cfg = config.get("training", {})
    if isinstance(training_cfg, Mapping):
        device_index = training_cfg.get("device_id", 0)
    else:
        device_index = 0
    env["CUDA_VISIBLE_DEVICES"] = str(device_index)
    logger.info("Restricting execution to GPU index %s", device_index)

