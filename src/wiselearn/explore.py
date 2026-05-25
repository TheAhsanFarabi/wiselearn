"""
wl.explore() — teaching-focused EDA that surfaces only what matters.

Unlike pandas-profiling which dumps 200 charts, this surfaces the
3-5 things you should actually act on.
"""
import pandas as pd
import numpy as np
from wiselearn.core.reporter import reporter


def explore(data: pd.DataFrame, target: str | None = None) -> None:
    """Print a focused EDA report for the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to explore.
    target : str, optional
        Target column name. If provided, includes target-aware insights.

    Examples
    --------
    >>> wl.explore(data, target="price")
    """
    reporter.section("🔍", "Exploring your data")

    _basic_overview(data)

    if target and target in data.columns:
        _target_overview(data, target)

    findings = []
    findings.extend(_check_high_cardinality(data, target))
    findings.extend(_check_skewed_columns(data, target))
    findings.extend(_check_missing(data))
    findings.extend(_check_constant_columns(data))
    findings.extend(_check_duplicates(data))

    if findings:
        reporter.blank()
        reporter.section("⚡", f"{len(findings)} things worth your attention")
        for i, finding in enumerate(findings, 1):
            reporter.console.print(f"   [bold]{i}.[/bold] {finding['msg']}")
            if finding.get("fix"):
                reporter.console.print(f"      [dim]→ {finding['fix']}[/dim]")
    else:
        reporter.success("Your data looks clean. No major issues spotted.")

    reporter.blank()
    reporter.next_step("wl.clean(data)  then  wl.prepare(data, target=...)")


def _basic_overview(df: pd.DataFrame) -> None:
    n_numeric = len(df.select_dtypes(include=[np.number]).columns)
    n_categorical = len(df.select_dtypes(include=["object", "string", "category"]).columns)
    n_datetime = len(df.select_dtypes(include=["datetime"]).columns)

    reporter.info(f"{len(df):,} rows × {len(df.columns)} columns")
    parts = []
    if n_numeric:
        parts.append(f"{n_numeric} numeric")
    if n_categorical:
        parts.append(f"{n_categorical} categorical")
    if n_datetime:
        parts.append(f"{n_datetime} datetime")
    if parts:
        reporter.info(f"Column types: {', '.join(parts)}")


def _target_overview(df: pd.DataFrame, target: str) -> None:
    y = df[target]
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
        reporter.info(f"Target '{target}' is numeric → REGRESSION task")
        reporter.detail(
            f"range: {y.min():.2f} to {y.max():.2f} | "
            f"mean: {y.mean():.2f} | median: {y.median():.2f}"
        )
    else:
        reporter.info(f"Target '{target}' has {y.nunique()} classes → CLASSIFICATION task")
        counts = y.value_counts()
        reporter.detail(f"class distribution: {dict(counts.head(5))}")


def _check_high_cardinality(df: pd.DataFrame, target: str | None) -> list[dict]:
    findings = []
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    n_rows = len(df)
    for col in cat_cols:
        if col == target:
            continue
        n_unique = df[col].nunique()
        if n_unique > 50 and n_unique > n_rows * 0.1:
            findings.append({
                "msg": (
                    f"'{col}' has {n_unique} unique values out of {n_rows} rows "
                    f"(high cardinality)"
                ),
                "fix": (
                    "One-hot encoding will explode your feature space. "
                    "wiselearn will use frequency encoding instead during prepare()."
                ),
            })
    return findings


def _check_skewed_columns(df: pd.DataFrame, target: str | None) -> list[dict]:
    findings = []
    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        skew = df[target].skew()
        if abs(skew) > 2:
            findings.append({
                "msg": f"Target '{target}' is highly skewed (skew={skew:.2f})",
                "fix": (
                    "For linear models, consider log-transforming: "
                    f"data['{target}'] = np.log1p(data['{target}'])"
                ),
            })
    return findings


def _check_missing(df: pd.DataFrame) -> list[dict]:
    findings = []
    missing_pct = df.isna().mean() * 100
    high_missing = missing_pct[missing_pct > 30]

    for col, pct in high_missing.items():
        findings.append({
            "msg": f"'{col}' is {pct:.0f}% missing",
            "fix": (
                "If structural (e.g. 'no garage'), fill with 0 or 'none'. "
                "If random, consider dropping the column."
            ),
        })
    return findings


def _check_constant_columns(df: pd.DataFrame) -> list[dict]:
    findings = []
    for col in df.columns:
        if df[col].nunique(dropna=False) == 1:
            findings.append({
                "msg": f"'{col}' has only one unique value — useless as a feature",
                "fix": f"Drop it: data = data.drop(columns=['{col}'])",
            })
    return findings


def _check_duplicates(df: pd.DataFrame) -> list[dict]:
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        return [{
            "msg": f"{n_dups} duplicate rows detected",
            "fix": "wl.clean(data) will remove these automatically.",
        }]
    return []
