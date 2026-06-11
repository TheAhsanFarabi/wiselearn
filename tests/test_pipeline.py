"""End-to-end tests for wiselearn's pipeline."""
import numpy as np
import pandas as pd
import pytest

import wiselearn as wl


@pytest.fixture
def regression_data(tmp_path):
    """Create a small synthetic regression dataset."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "sqft": np.random.randint(500, 4000, n),
        "bedrooms": np.random.randint(1, 6, n),
        "neighborhood": np.random.choice(["A", "B", "C", "D"], n),
        "year_built": np.random.randint(1950, 2024, n),
    })
    # Create a target that genuinely depends on features
    df["price"] = (
        df["sqft"] * 150
        + df["bedrooms"] * 10000
        + (df["year_built"] - 1950) * 500
        + np.random.normal(0, 60000, n)  # enough noise to be realistic
    )
    path = tmp_path / "houses.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def classification_data(tmp_path):
    """Create a small synthetic classification dataset."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "age": np.random.randint(18, 80, n),
        "income": np.random.randint(20000, 200000, n),
        "city": np.random.choice(["X", "Y", "Z"], n),
    })
    # Target: did they buy? (Higher income => more likely)
    proba = (df["income"] / 200000).clip(0, 1)
    df["bought"] = (np.random.random(n) < proba).astype(int)
    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    return path


def test_load_csv(regression_data):
    df = wl.load(regression_data)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
    assert "price" in df.columns


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        wl.load("does_not_exist.csv")


def test_explore_runs(regression_data):
    df = wl.load(regression_data)
    # Should not raise
    wl.explore(df, target="price")


def test_clean_removes_duplicates():
    df = pd.DataFrame({
        "a": [1, 1, 2, 3],
        "b": [4, 4, 5, 6],
    })
    cleaned = wl.clean(df)
    assert len(cleaned) == 3


def test_clean_fills_missing():
    df = pd.DataFrame({
        "a": [1.0, 2.0, None, 4.0],
        "b": ["x", "y", None, "x"],
    })
    cleaned = wl.clean(df)
    assert cleaned.isna().sum().sum() == 0


def test_clean_preview_doesnt_modify():
    df = pd.DataFrame({"a": [1, 1, 2]})
    cleaned = wl.clean(df, preview=True)
    assert len(cleaned) == 3  # Preview: nothing dropped


def test_full_regression_pipeline(regression_data):
    """The big one: full pipeline end-to-end."""
    data = wl.load(regression_data)
    data = wl.clean(data)
    prep = wl.prepare(data, target="price")

    assert prep.task == "regression"
    assert prep.X_train is not None
    assert prep.X_test is not None
    assert "neighborhood" in prep.categorical_columns

    model = wl.train(prep)
    metrics = wl.evaluate(model, prep)

    # Should have learned SOMETHING
    assert metrics["r2"] > 0.5

    # Explain should work
    importances = wl.explain(model, prep)
    assert importances is not None
    assert len(importances) > 0


def test_full_classification_pipeline(classification_data):
    data = wl.load(classification_data)
    data = wl.clean(data)
    prep = wl.prepare(data, target="bought")

    assert prep.task == "classification"

    model = wl.train(prep)
    metrics = wl.evaluate(model, prep)

    assert "accuracy" in metrics or "f1" in metrics


def test_leakage_detection_stops_training():
    """The killer feature — leakage should be caught before training."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "feature_a": np.random.normal(0, 1, n),
        "target": np.random.normal(0, 1, n),
    })
    # Create a column that's essentially a copy of the target (leakage!)
    df["sneaky_leak"] = df["target"] + np.random.normal(0, 0.01, n)

    with pytest.raises(ValueError, match="leakage|Critical"):
        wl.prepare(df, target="target")


def test_leakage_can_be_overridden():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "feature_a": np.random.normal(0, 1, n),
        "target": np.random.normal(0, 1, n),
    })
    df["sneaky_leak"] = df["target"] + np.random.normal(0, 0.01, n)

    # Should not raise when ignore_leakage=True
    prep = wl.prepare(df, target="target", ignore_leakage=True)
    assert prep.X_train is not None


def test_prep_transform_handles_new_data(regression_data):
    """prep.transform() should consistently apply the same encoders."""
    data = wl.load(regression_data)
    prep = wl.prepare(data, target="price")

    # Take a few new rows
    new_rows = data.drop(columns=["price"]).head(3)
    transformed = prep.transform(new_rows)

    assert len(transformed) == 3
    assert list(transformed.columns) == prep.feature_columns


def test_predict_on_new_data(regression_data):
    data = wl.load(regression_data)
    prep = wl.prepare(data, target="price")
    model = wl.train(prep)

    new_data = data.drop(columns=["price"]).head(5)
    predictions = wl.predict(model, new_data, prep=prep)

    assert len(predictions) == 5
    assert isinstance(predictions, np.ndarray)


def test_stratified_split_fallback_on_single_member_class():
    """prepare() should fall back to non-stratified split instead of crashing
    when a class has only one sample."""
    df = pd.DataFrame({
        "x": np.random.normal(0, 1, 50),
        "y": [0] * 49 + [1],  # class 1 has only 1 sample
    })
    # Should not raise — should warn and fall back
    prep = wl.prepare(df, target="y")
    assert prep.X_train is not None
    assert prep.X_test is not None


def test_id_column_rule_no_false_positive_on_common_words():
    """IDColumnRule should not flag columns like 'grid', 'fluid', 'acid'
    that end in 'id' but are not row identifiers."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "grid": np.arange(n),       # unique, ends in 'id' — old bug flagged this
        "fluid": np.arange(n),      # unique, ends in 'id' — old bug flagged this
        "price": np.random.normal(100, 10, n),
    })
    # Should prepare without any ID-column warning for 'grid' or 'fluid'
    import io
    from unittest.mock import patch
    warned_cols = []
    original_warn = wl.prepare.__wrapped__ if hasattr(wl.prepare, '__wrapped__') else None

    # Run prepare and check no ValueError / no ID warning causes wrong drops
    prep = wl.prepare(df, target="price")
    # grid and fluid should still be in features (not dropped)
    assert "grid" in prep.feature_columns
    assert "fluid" in prep.feature_columns


def test_unseen_category_in_predict_raises_warning():
    """predict() should emit a UserWarning when new data contains category
    values not seen during training, instead of silently replacing them."""
    import warnings
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "cat": np.random.choice(["A", "B", "C"], n),
        "x": np.random.normal(0, 1, n),
        "y": np.random.randint(0, 2, n),
    })
    prep = wl.prepare(df, target="y")
    model = wl.train(prep)

    new_data = pd.DataFrame({
        "cat": ["NEVER_SEEN_BEFORE", "A"],
        "x": [0.1, 0.2],
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wl.predict(model, new_data, prep=prep)

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "NEVER_SEEN_BEFORE" in str(user_warnings[0].message) or "unseen" in str(user_warnings[0].message).lower()


def test_save_and_load(regression_data, tmp_path):
    data = wl.load(regression_data)
    prep = wl.prepare(data, target="price")
    model = wl.train(prep)

    model_path = tmp_path / "model.wl"
    wl.save(model, prep, model_path)

    loaded_model, loaded_prep = wl.load_model(model_path)

    # Should make the same predictions
    new_data = data.drop(columns=["price"]).head(5)
    pred_original = wl.predict(model, new_data, prep=prep)
    pred_loaded = wl.predict(loaded_model, new_data, prep=loaded_prep)
    np.testing.assert_array_almost_equal(pred_original, pred_loaded)
