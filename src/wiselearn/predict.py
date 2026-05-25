"""
wl.predict() — make predictions on new data, applying the same prep transforms.

This is where beginners constantly mess up: they train with scaled/encoded
data, then predict on raw data. The Prep object solves this — wl.predict()
ALWAYS uses the same transformations that prep was built with.
"""
import numpy as np
import pandas as pd

from wiselearn.core.reporter import reporter
from wiselearn.core.prep_object import Prep


def predict(
    model,
    new_data: pd.DataFrame,
    prep: Prep,
    return_proba: bool = False,
) -> np.ndarray:
    """Predict on new data using the model and its prep pipeline.

    Parameters
    ----------
    model : fitted scikit-learn model
    new_data : pd.DataFrame
        New data with the same feature columns as the training data.
    prep : Prep
        The Prep object the model was trained on. Used to apply the
        same encoders/scaler so predictions are consistent.
    return_proba : bool, default False
        If True (and classification), return class probabilities.

    Returns
    -------
    np.ndarray
        Predictions (or probabilities if return_proba=True).
    """
    reporter.section("🔮", f"Predicting on {len(new_data):,} new rows")

    try:
        X = prep.transform(new_data)
    except ValueError as e:
        reporter.critical(
            f"Cannot transform the new data: {e}",
            fix="Make sure new data has the same columns as the training data.",
        )
        raise

    if return_proba:
        if not hasattr(model, "predict_proba"):
            reporter.warn("This model doesn't support predict_proba. Returning class predictions.")
            return model.predict(X)
        proba = model.predict_proba(X)
        reporter.success(f"Returned probabilities. Shape: {proba.shape}")
        return proba

    predictions = model.predict(X)

    # Decode classification labels if they were encoded
    if prep.task == "classification" and prep.classes is not None:
        predictions = np.array([prep.classes[int(p)] for p in predictions])

    reporter.success(f"Made {len(predictions):,} predictions.")
    if prep.task == "classification":
        unique, counts = np.unique(predictions, return_counts=True)
        dist = dict(zip(unique, counts))
        reporter.info(f"Predicted class distribution: {dist}")
    else:
        reporter.info(
            f"Range: {predictions.min():.2f} to {predictions.max():.2f} "
            f"(mean: {predictions.mean():.2f})"
        )

    return predictions
