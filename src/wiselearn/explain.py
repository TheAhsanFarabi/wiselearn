"""
wl.explain() — understand what your model learned.
"""
import numpy as np
import pandas as pd

from wiselearn.core.reporter import reporter
from wiselearn.core.prep_object import Prep


def explain(model, prep: Prep | None = None, top_n: int = 10) -> dict | None:
    """Print global feature importances with interpretation.

    Parameters
    ----------
    model : fitted scikit-learn model
    prep : Prep, optional
        The Prep object used to train this model. Helps with feature names.
    top_n : int, default 10
        How many top features to show.

    Returns
    -------
    dict or None
        Feature importances if available.
    """
    reporter.section("🧠", "What your model learned")

    feature_names = (
        prep.feature_columns if prep is not None
        else [f"feature_{i}" for i in range(_n_features(model))]
    )

    importances = _get_importances(model)
    if importances is None:
        reporter.warn(
            "This model doesn't expose feature importances. "
            "(Tree-based and linear models work; some others don't.)"
        )
        return None

    # Combine into a DataFrame
    fi = pd.DataFrame({
        "feature": feature_names[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    total = fi["importance"].abs().sum()
    if total == 0:
        reporter.warn("All importances are zero — model didn't learn anything meaningful.")
        return None

    fi["pct"] = (fi["importance"].abs() / total) * 100

    reporter.info(f"Top {min(top_n, len(fi))} features driving the predictions:")

    for i, row in fi.head(top_n).iterrows():
        bar_len = int(row["pct"] / 2)  # max ~50 chars
        bar = "█" * bar_len
        reporter.console.print(
            f"   [bold]{i+1:>2}.[/bold] {row['feature']:<25} "
            f"[cyan]{bar}[/cyan] {row['pct']:.1f}%"
        )

    # Sanity check
    top_feature = fi.iloc[0]
    if top_feature["pct"] > 70:
        reporter.blank()
        reporter.warn(
            f"'{top_feature['feature']}' alone accounts for "
            f"{top_feature['pct']:.0f}% of decisions.\n"
            f"   Either this feature is genuinely dominant, or it's leakage."
        )

    return fi.to_dict(orient="records")


def _get_importances(model) -> np.ndarray | None:
    """Extract feature importances from various model types."""
    # Tree-based models
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    # Linear models
    if hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            # Multi-class: average absolute coefficients
            return np.abs(coef).mean(axis=0)
        return np.abs(coef)
    return None


def _n_features(model) -> int:
    """Best-effort guess at the number of features."""
    if hasattr(model, "n_features_in_"):
        return model.n_features_in_
    if hasattr(model, "feature_importances_"):
        return len(model.feature_importances_)
    if hasattr(model, "coef_"):
        coef = model.coef_
        return coef.shape[-1]
    return 0
