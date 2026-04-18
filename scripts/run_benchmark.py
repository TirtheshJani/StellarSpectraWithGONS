#!/usr/bin/env python
"""Run the Pickles template-matching benchmark against the LightGBM classifier."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.interpret.benchmark import (
    benchmark_report,
    best_template_per_spectrum,
    load_pickles_library,
    save_report,
)
from src.interpret.classifier import load_model
from src.interpret.lines import ALLOWED_MK_CLASSES


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--pickles-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    payload = np.load(args.features, allow_pickle=False)
    X = payload["X"]
    y = payload["y"].astype(np.int64)
    wc = payload["wave_centers"]
    test_idx = payload["test_idx"]

    templates = load_pickles_library(args.pickles_dir, wc)
    mk_per_template = np.array([t.mk_class for t in templates])

    best_t = best_template_per_spectrum(X[test_idx], templates)
    y_pickles = mk_per_template[best_t]

    model = load_model(args.model)
    present = sorted(np.unique(y).tolist())
    class_labels = [ALLOWED_MK_CLASSES[i] for i in present]
    y_model = np.array([class_labels[i] for i in model.predict(X[test_idx])])

    report = benchmark_report(y_pred_model=y_model, y_pickles_mk=y_pickles,
                              class_labels=class_labels)
    save_report(args.out_dir, report)
    print(f"agreement with Pickles: {report['agreement_rate']:.3f}")
    print(f"macro-F1 on FGK:        {report['macro_f1_fgk']:.3f}")


if __name__ == "__main__":
    main()
