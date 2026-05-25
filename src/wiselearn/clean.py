"""
wl.clean() — apply recommended fixes to data, with explanations.
"""
import pandas as pd
import numpy as np
from wiselearn.core.reporter import reporter


def clean(
    data: pd.DataFrame,
    fix: list[str] | None = None,
    preview: bool = False,
) -> pd.DataFrame:
    """Clean common data issues with full transparency.

    By default, fixes: missing values, duplicates, constant columns.

    Parameters
    ----------
    data : pd.DataFrame
    fix : list of str, optional
        Which fixes to apply. Default: ["missing", "duplicates", "constants"].
        Options: "missing", "duplicates", "constants".
    preview : bool, default False
        If True, just print what WOULD be done without changing anything.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.
    """
    if fix is None:
        fix = ["missing", "duplicates", "constants"]

    df = data.copy()
    actions: list[str] = []

    reporter.section("🧹", "Cleaning your data")

    if "duplicates" in fix:
        n_dups = df.duplicated().sum()
        if n_dups > 0:
            if not preview:
                df = df.drop_duplicates().reset_index(drop=True)
            actions.append(f"Removed {n_dups} duplicate rows")

    if "constants" in fix:
        const_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
        if const_cols:
            if not preview:
                df = df.drop(columns=const_cols)
            actions.append(f"Dropped constant columns: {const_cols}")

    if "missing" in fix:
        missing_actions = _handle_missing(df, preview)
        actions.extend(missing_actions)

    if not actions:
        reporter.success("Nothing to clean — your data is already tidy!")
    else:
        for action in actions:
            reporter.info(action)
        if preview:
            reporter.warn("This was a preview. No changes were made.")
            reporter.next_step("Run wl.clean(data) without preview=True to apply.")
        else:
            reporter.success(f"Done. Final shape: {df.shape}")
            reporter.next_step("wl.prepare(data, target=...)")

    return df


def _handle_missing(df: pd.DataFrame, preview: bool) -> list[str]:
    """Decide how to fill each column with missing values."""
    actions = []
    missing_pct = df.isna().mean() * 100

    for col in df.columns:
        pct = missing_pct[col]
        if pct == 0:
            continue

        n_missing = df[col].isna().sum()

        # Drop columns with more than 70% missing
        if pct > 70:
            if not preview:
                df.drop(columns=[col], inplace=True)
            actions.append(f"Dropped '{col}' (>{pct:.0f}% missing)")
            continue

        # Fill numeric with median
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
            if not preview:
                df[col] = df[col].fillna(fill_value)
            actions.append(
                f"'{col}': filled {n_missing} missing ({pct:.1f}%) "
                f"with median ({fill_value:.2f})"
            )
        # Fill categorical with mode (most common value)
        else:
            mode_vals = df[col].mode()
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else "unknown"
            if not preview:
                df[col] = df[col].fillna(fill_value)
            actions.append(
                f"'{col}': filled {n_missing} missing ({pct:.1f}%) "
                f"with mode ('{fill_value}')"
            )

    return actions
