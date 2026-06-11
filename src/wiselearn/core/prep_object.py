"""
The Prep object — bundles everything needed to transform raw data into
model-ready data: encoders, scaler, train/test split, target info.

This is wiselearn's solution to the #1 beginner mistake:
training on transformed data, then predicting on raw data.
"""
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
import numpy as np


@dataclass
class Prep:
    """Holds the fitted transformations and split data for a training task."""

    target: str
    task: str  # "classification" or "regression"
    feature_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)

    # Fitted transformers (fit on training data only — no leakage)
    encoders: dict[str, Any] = field(default_factory=dict)
    scaler: Any = None

    # Split data
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None

    # Class info (for classification)
    classes: list | None = None
    is_imbalanced: bool = False
    class_distribution: dict | None = None

    def transform(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Apply the same transformations to new data.

        Uses the encoders and scaler that were fit on training data,
        ensuring no leakage and consistent predictions.
        """
        df = new_data.copy()

        # Drop target if present in new data
        if self.target in df.columns:
            df = df.drop(columns=[self.target])

        # Keep only the columns the model was trained on
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"New data is missing columns the model was trained on: {missing}"
            )

        # Apply categorical encoders
        for col, encoder in self.encoders.items():
            if col in df.columns:
                # Handle unseen categories: map to a default value
                known = set(encoder.classes_)
                unseen = set(df[col].astype(str).unique()) - known
                if unseen:
                    import warnings
                    warnings.warn(
                        f"Column '{col}' contains {len(unseen)} unseen category value(s) "
                        f"not seen during training (e.g. {repr(next(iter(unseen)))}). "
                        f"These will be replaced with '{encoder.classes_[0]}'. "
                        f"Predictions for these rows may be unreliable.",
                        UserWarning,
                        stacklevel=3,
                    )
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else encoder.classes_[0]
                )
                df[col] = encoder.transform(df[col])

        # Reorder to match training column order
        df = df[self.feature_columns]

        # Apply scaler if one was fitted
        if self.scaler is not None and self.numeric_columns:
            df[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])

        return df

    def __repr__(self) -> str:
        train_n = len(self.X_train) if self.X_train is not None else 0
        test_n = len(self.X_test) if self.X_test is not None else 0
        return (
            f"Prep(target='{self.target}', task='{self.task}', "
            f"features={len(self.feature_columns)}, "
            f"train={train_n}, test={test_n})"
        )
