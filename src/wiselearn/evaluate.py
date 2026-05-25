"""
wl.evaluate() — print the RIGHT metrics for your task, with interpretation.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)

from wiselearn.core.reporter import reporter
from wiselearn.core.prep_object import Prep


def evaluate(model, prep: Prep) -> dict:
    """Evaluate the model with appropriate metrics for the task.

    For classification: picks Precision/Recall/F1 + PR-AUC if imbalanced,
    or Accuracy + ROC-AUC if balanced.
    For regression: R², MAE, RMSE.

    Parameters
    ----------
    model : fitted scikit-learn model
    prep : Prep object

    Returns
    -------
    dict
        The computed metrics, in case you want to use them programmatically.
    """
    reporter.section("📈", "Evaluating your model")

    if prep.task == "classification":
        metrics = _evaluate_classification(model, prep)
    else:
        metrics = _evaluate_regression(model, prep)

    reporter.next_step("wl.explain(model)  to see what features matter")
    return metrics


def _evaluate_regression(model, prep: Prep) -> dict:
    y_true = prep.y_test
    y_pred = model.predict(prep.X_test)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    reporter.info(f"R²:   {r2:.3f}")
    reporter.detail("(1.0 = perfect, 0 = no better than predicting the mean)")

    reporter.info(f"MAE:  {mae:,.2f}")
    reporter.detail(f"On average, predictions are off by ±{mae:,.2f}")

    reporter.info(f"RMSE: {rmse:,.2f}")
    reporter.detail("RMSE penalizes large errors more than MAE")

    # Interpretation
    if r2 < 0:
        reporter.warn(
            "R² is negative. Your model is worse than just predicting the mean."
        )
    elif r2 < 0.3:
        reporter.warn("R² is low. Your features may not predict the target well.")
    elif r2 > 0.95:
        reporter.warn(
            "R² is very high — sometimes too good to be true. Check for leakage."
        )
    else:
        reporter.success(f"R² of {r2:.3f} is reasonable for most real-world problems.")

    return {"r2": r2, "mae": mae, "rmse": rmse}


def _evaluate_classification(model, prep: Prep) -> dict:
    y_true = prep.y_test
    y_pred = model.predict(prep.X_test)
    n_classes = len(np.unique(prep.y_train))
    is_binary = n_classes == 2

    acc = accuracy_score(y_true, y_pred)

    metrics: dict = {"accuracy": acc}

    if prep.is_imbalanced:
        reporter.explain(
            "Your classes are imbalanced — accuracy can be misleading. "
            "Using Precision/Recall/F1 instead."
        )
        avg = "binary" if is_binary else "weighted"
        prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)

        reporter.info(f"Precision: {prec:.3f}")
        reporter.detail("Of the rows the model flagged positive, how many really are")
        reporter.info(f"Recall:    {rec:.3f}")
        reporter.detail("Of all the real positives, how many the model caught")
        reporter.info(f"F1:        {f1:.3f}")
        reporter.detail("Balance of precision and recall (higher is better)")

        metrics.update({"precision": prec, "recall": rec, "f1": f1})

        # PR-AUC if we can get probabilities
        if hasattr(model, "predict_proba") and is_binary:
            try:
                y_proba = model.predict_proba(prep.X_test)[:, 1]
                pr_auc = average_precision_score(y_true, y_proba)
                reporter.info(f"PR-AUC:    {pr_auc:.3f}")
                reporter.detail("Better than ROC-AUC for imbalanced problems")
                metrics["pr_auc"] = pr_auc
            except Exception:
                pass
    else:
        reporter.info(f"Accuracy: {acc:.3f}")
        reporter.detail(
            f"(Random guessing would score ~{1/n_classes:.2f} with {n_classes} classes)"
        )

        if hasattr(model, "predict_proba") and is_binary:
            try:
                y_proba = model.predict_proba(prep.X_test)[:, 1]
                roc = roc_auc_score(y_true, y_proba)
                reporter.info(f"ROC-AUC:  {roc:.3f}")
                reporter.detail("How well the model ranks positives above negatives (0.5 = random)")
                metrics["roc_auc"] = roc
            except Exception:
                pass

    # Confusion matrix for binary classification
    if is_binary:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        reporter.blank()
        reporter.info("Confusion matrix:")
        reporter.detail(f"  True Negatives:  {tn:5d}    False Positives: {fp:5d}")
        reporter.detail(f"  False Negatives: {fn:5d}    True Positives:  {tp:5d}")

    return metrics
