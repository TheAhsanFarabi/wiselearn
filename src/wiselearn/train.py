"""
wl.train() — fit a model on a prepared dataset.

This now does ONE thing: fit a model. Cleaning, encoding, splitting all
already happened in wl.prepare(). This function picks the right model,
explains why, fits it, and checks for overfitting after.
"""
from typing import Any
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from wiselearn.core.reporter import reporter
from wiselearn.core.prep_object import Prep
from wiselearn.rules import POST_TRAINING_RULES, run_rules


MODEL_REGISTRY = {
    "classification": {
        "random_forest": (RandomForestClassifier, "Robust default for tabular data"),
        "logistic": (LogisticRegression, "Simple, fast, and interpretable"),
        "decision_tree": (DecisionTreeClassifier, "Very interpretable, but overfits easily"),
    },
    "regression": {
        "random_forest": (RandomForestRegressor, "Robust default for tabular data"),
        "ridge": (Ridge, "Linear with regularization — robust to overfitting"),
        "linear": (LinearRegression, "Simplest possible baseline"),
        "decision_tree": (DecisionTreeRegressor, "Very interpretable, but overfits easily"),
    },
}


def train(
    prep: Prep,
    model: str = "auto",
    regularize: bool = False,
    **model_kwargs,
) -> Any:
    """Train a model on prepared data.

    Parameters
    ----------
    prep : Prep
        Output of wl.prepare().
    model : str, default "auto"
        Which model to use. Options:
        - "auto" — let wiselearn pick
        - classification: "random_forest", "logistic", "decision_tree"
        - regression: "random_forest", "ridge", "linear", "decision_tree"
    regularize : bool, default False
        If True, pick a model with regularization to combat overfitting.
    **model_kwargs
        Extra arguments passed directly to the underlying scikit-learn model.

    Returns
    -------
    The fitted scikit-learn model.
    """
    reporter.section("🤖", "Training your model")

    model_name = _pick_model(prep, model, regularize)
    ModelClass, why = MODEL_REGISTRY[prep.task][model_name]
    reporter.info(f"Model: {ModelClass.__name__}")
    reporter.explain(why)

    # Build default kwargs
    init_kwargs = _default_kwargs(ModelClass, prep)
    init_kwargs.update(model_kwargs)

    estimator = ModelClass(**init_kwargs)

    # Fit
    estimator.fit(prep.X_train, prep.y_train)
    reporter.success("Model trained.")

    # Score on train and test
    train_score = estimator.score(prep.X_train, prep.y_train)
    test_score = estimator.score(prep.X_test, prep.y_test)
    metric = "R²" if prep.task == "regression" else "accuracy"
    reporter.info(f"Train {metric}: {train_score:.3f}")
    reporter.info(f"Test  {metric}: {test_score:.3f}")

    # Post-training rules
    findings = run_rules(
        POST_TRAINING_RULES,
        train_score=train_score,
        test_score=test_score,
        task=prep.task,
    )
    for finding in findings:
        if finding.severity == "critical":
            reporter.critical(finding.message, fix=finding.fix)
        elif finding.severity == "warn":
            reporter.warn(finding.message, fix=finding.fix)

    reporter.next_step("wl.evaluate(model, prep)  or  wl.explain(model)")
    return estimator


def _pick_model(prep: Prep, model: str, regularize: bool) -> str:
    """Choose a model based on task, data size, and user preferences."""
    if model != "auto":
        if model not in MODEL_REGISTRY[prep.task]:
            valid = list(MODEL_REGISTRY[prep.task].keys())
            raise ValueError(
                f"Unknown model '{model}' for {prep.task}. Valid options: {valid}"
            )
        return model

    if regularize:
        if prep.task == "regression":
            reporter.explain("Picked Ridge because regularize=True (combats overfitting).")
            return "ridge"
        else:
            reporter.explain("Picked Logistic Regression because regularize=True.")
            return "logistic"

    # Default: random forest is a solid choice for most tabular data
    n_train = len(prep.X_train) if prep.X_train is not None else 0
    n_features = len(prep.feature_columns)
    reporter.explain(
        f"Picked Random Forest: {n_train:,} rows, {n_features} features, "
        f"mixed types. Robust default that rarely needs tuning."
    )
    return "random_forest"


def _default_kwargs(ModelClass, prep: Prep) -> dict:
    """Sensible defaults for each model type."""
    kwargs: dict = {}
    if ModelClass.__name__.startswith("RandomForest"):
        kwargs["n_estimators"] = 100
        kwargs["random_state"] = 42
        kwargs["n_jobs"] = -1
        if prep.task == "classification" and prep.is_imbalanced:
            kwargs["class_weight"] = "balanced"
            reporter.explain("class_weight='balanced' applied (imbalanced target).")
    elif ModelClass.__name__ == "LogisticRegression":
        kwargs["max_iter"] = 1000
        kwargs["random_state"] = 42
        if prep.is_imbalanced:
            kwargs["class_weight"] = "balanced"
    elif ModelClass.__name__ in ("DecisionTreeClassifier", "DecisionTreeRegressor"):
        kwargs["random_state"] = 42
        kwargs["max_depth"] = 10  # Prevent extreme overfitting
    elif ModelClass.__name__ == "Ridge":
        kwargs["random_state"] = 42
    return kwargs
