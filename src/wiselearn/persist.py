"""
wl.save() / wl.load_model() — persist a model + its prep together.

Beginners constantly lose their encoders/scaler when saving models.
wiselearn saves them as one bundle so it's impossible to lose.
"""
from pathlib import Path
from typing import Any
import joblib

from wiselearn.core.reporter import reporter
from wiselearn.core.prep_object import Prep


def save(model: Any, prep: Prep, path: str | Path) -> None:
    """Save a model and its Prep object together.

    Parameters
    ----------
    model : fitted scikit-learn model
    prep : Prep
    path : str or Path
        Where to write the file. Convention: use .wl extension.
    """
    p = Path(path)
    bundle = {
        "model": model,
        "prep": prep,
        "wiselearn_version": "0.1.0",
    }
    joblib.dump(bundle, p)
    reporter.section("💾", "Saved model")
    reporter.info(f"File: {p}")
    reporter.info(f"Size: {p.stat().st_size / 1024:.1f} KB")
    reporter.success(
        "Model + prep saved together. You won't lose your encoders this time."
    )


def load_model(path: str | Path) -> tuple[Any, Prep]:
    """Load a saved model + prep bundle.

    Returns
    -------
    (model, prep) tuple
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    bundle = joblib.load(p)
    reporter.section("📂", "Loaded model")
    reporter.info(f"From: {p}")
    reporter.info(f"Task: {bundle['prep'].task}")
    reporter.info(f"Target: {bundle['prep'].target}")
    reporter.success("Ready to predict.")
    return bundle["model"], bundle["prep"]
