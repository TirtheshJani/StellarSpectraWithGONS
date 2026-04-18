#!/usr/bin/env python
"""Train the LightGBM MK classifier from a features.npz payload."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.interpret.classifier import (
    evaluate,
    save_model,
    train,
    write_metrics,
)
from src.interpret.lines import ALLOWED_MK_CLASSES


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--model-out", required=True, type=Path)
    p.add_argument("--metrics-out", required=True, type=Path)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    payload = np.load(args.features, allow_pickle=False)
    X = payload["X"]
    y = payload["y"].astype(np.int64)
    train_idx = payload["train_idx"]
    val_idx = payload["val_idx"]
    test_idx = payload["test_idx"]

    present = sorted(np.unique(y).tolist())
    class_labels = [ALLOWED_MK_CLASSES[i] for i in present]
    num_class = len(present)
    logger = logging.getLogger(__name__)
    logger.info(
        "classes present: %s  train/val/test = %d/%d/%d  n_bins=%d",
        class_labels, len(train_idx), len(val_idx), len(test_idx), X.shape[1],
    )

    model = train(
        X_train=X[train_idx], y_train=y[train_idx],
        X_val=X[val_idx], y_val=y[val_idx],
        num_class=num_class,
    )
    metrics = evaluate(
        model,
        X[test_idx], y[test_idx],
        class_labels=class_labels,
        n_train=len(train_idx),
        n_val=len(val_idx),
    )
    save_model(model, args.model_out)
    write_metrics(metrics, args.metrics_out)
    print(f"accuracy={metrics.accuracy:.4f}  macro_f1={metrics.macro_f1:.4f}")
    print(f"per-class F1: {metrics.per_class_f1}")


if __name__ == "__main__":
    main()
