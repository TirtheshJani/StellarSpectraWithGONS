"""LightGBM multiclass wrapper for the MK interpretability pipeline.

Hyper-parameters follow the locked-in plan: balanced class weights, shallow
enough trees to keep SHAP tractable, early stopping on validation loss.

The wrapper exposes the sklearn API (``fit``/``predict``) so that
``sklearn.inspection.permutation_importance`` and ``shap.TreeExplainer``
both work without glue code.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_HPARAMS: Final[dict[str, Any]] = {
    "objective": "multiclass",
    "class_weight": "balanced",
    "max_depth": 8,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

DEFAULT_EARLY_STOPPING: Final[int] = 50


@dataclass
class ClassifierMetrics:
    accuracy: float
    macro_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    class_labels: list[str]
    n_train: int
    n_val: int
    n_test: int


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_class: int,
    hparams: dict[str, Any] | None = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING,
) -> Any:
    """Train a ``lightgbm.LGBMClassifier`` with validation-based early stopping.

    Returns the fitted estimator (already has ``best_iteration_``).
    """
    import lightgbm as lgb

    params = dict(DEFAULT_HPARAMS)
    params["num_class"] = int(num_class)
    if hparams:
        params.update(hparams)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    logger.info(
        "training complete; best_iteration=%s",
        getattr(model, "best_iteration_", None),
    )
    return model


def evaluate(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_labels: list[str],
    n_train: int = 0,
    n_val: int = 0,
) -> ClassifierMetrics:
    """Compute accuracy, macro-F1, per-class precision/recall/F1, confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    present = np.arange(len(class_labels))
    p, r, f, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=present, average=None, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=present).tolist()
    return ClassifierMetrics(
        accuracy=acc,
        macro_f1=macro_f1,
        per_class_precision={class_labels[i]: float(p[i]) for i in range(len(class_labels))},
        per_class_recall={class_labels[i]: float(r[i]) for i in range(len(class_labels))},
        per_class_f1={class_labels[i]: float(f[i]) for i in range(len(class_labels))},
        confusion_matrix=cm,
        class_labels=list(class_labels),
        n_train=int(n_train),
        n_val=int(n_val),
        n_test=int(len(y_test)),
    )


def save_model(model: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)
    logger.info("saved model -> %s", path)


def load_model(path: Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def write_metrics(metrics: ClassifierMetrics, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(asdict(metrics), f, indent=2)
    logger.info("wrote metrics -> %s", path)
