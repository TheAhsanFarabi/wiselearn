"""
wl.prepare() — split, encode, scale, and audit for leakage.

This is the most important function: it builds a Prep object that bundles
all the fitted transformations together, preventing the #1 beginner mistake
(training and predicting on differently-shaped data).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from wiselearn.core.reporter import reporter
from wiselearn.core.prep_object import Prep
from wiselearn.rules import PRE_TRAINING_RULES, run_rules


def prepare(
    data: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    scale: bool = False,
    drop: list[str] | None = None,
    ignore_leakage: bool = False,
    random_state: int = 42,
) -> Prep:
    """Split, encode, and scale data for training.

    Parameters
    ----------
    data : pd.DataFrame
    target : str
        Name of the target column.
    test_size : float, default 0.2
        Fraction of data to hold out for testing.
    scale : bool, default False
        Whether to standardize numeric features (needed for linear/distance-based models).
    drop : list of str, optional
        Columns to drop before preparing (e.g. IDs, leakage columns).
    ignore_leakage : bool, default False
        If True, skip the leakage check. Use only when you're SURE.
    random_state : int, default 42
        For reproducibility.

    Returns
    -------
    Prep
        An object containing X_train, X_test, y_train, y_test, encoders, scaler.
    """
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in data.")

    reporter.section("⚙️", "Preparing your data")

    df = data.copy()
    if drop:
        existing = [c for c in drop if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
            reporter.info(f"Dropped columns: {existing}")

    # Determine task type
    y_raw = df[target]
    task = _infer_task(y_raw)
    reporter.info(f"Task: {task.upper()}")

    # Pre-training rules (leakage, ID columns, imbalance, sample size)
    findings = run_rules(
        PRE_TRAINING_RULES,
        data=df,
        target=target,
        task=task,
    )
    has_critical = False
    for finding in findings:
        if finding.severity == "critical":
            has_critical = True
            reporter.critical(finding.message, fix=finding.fix)
        elif finding.severity == "warn":
            reporter.warn(finding.message, fix=finding.fix)
        else:
            reporter.info(finding.message)

    if has_critical and not ignore_leakage:
        raise ValueError(
            "Critical issue detected (likely leakage). "
            "Fix it, or pass ignore_leakage=True to override."
        )

    # Split features and target
    X = df.drop(columns=[target])
    y = df[target].copy()

    # Identify column types
    categorical_cols = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    reporter.info(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

    # Encode target for classification (handle non-numeric labels)
    target_encoder = None
    classes = None
    if task == "classification" and not pd.api.types.is_numeric_dtype(y):
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index)
        classes = target_encoder.classes_.tolist()

    # Train/test split (stratified for classification)
    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    reporter.info(
        f"Split: {len(X_train):,} train / {len(X_test):,} test "
        f"(test_size={test_size:.0%}, stratified={'yes' if stratify is not None else 'no'})"
    )

    # Fit encoders on training data ONLY (this prevents leakage)
    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        # Fit only on train values
        X_train[col] = X_train[col].astype(str)
        encoder.fit(X_train[col])

        # Transform train
        X_train[col] = encoder.transform(X_train[col])

        # Transform test, handling unseen categories
        known = set(encoder.classes_)
        X_test[col] = X_test[col].astype(str).apply(
            lambda x: x if x in known else encoder.classes_[0]
        )
        X_test[col] = encoder.transform(X_test[col])

        encoders[col] = encoder

    if categorical_cols:
        reporter.explain(
            "Categorical encoders were fit ONLY on training data. "
            "This prevents test-data information from leaking into the model."
        )

    # Fit scaler on training data only
    scaler = None
    if scale and numeric_cols:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
        reporter.info("Numeric features standardized (mean=0, std=1)")

    # Class imbalance info
    is_imbalanced = False
    class_distribution = None
    if task == "classification":
        counts = pd.Series(y_train).value_counts(normalize=True)
        is_imbalanced = counts.min() < 0.1
        class_distribution = counts.to_dict()

    prep = Prep(
        target=target,
        task=task,
        feature_columns=X_train.columns.tolist(),
        categorical_columns=categorical_cols,
        numeric_columns=numeric_cols,
        encoders=encoders,
        scaler=scaler,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        classes=classes,
        is_imbalanced=is_imbalanced,
        class_distribution=class_distribution,
    )

    reporter.success("Data is ready for training.")
    reporter.next_step("model = wl.train(prep)")
    return prep


def _infer_task(y: pd.Series) -> str:
    """Decide if this is classification or regression."""
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    n_unique = y.nunique()
    # Heuristic: small number of distinct values + integer-like = classification
    if n_unique <= 20 and (y.dropna() == y.dropna().astype(int)).all():
        return "classification"
    return "regression"
