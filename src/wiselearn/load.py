"""
wl.load() — load data from a file with auto-detection and a sanity check.
"""
from pathlib import Path
import pandas as pd
from wiselearn.core.reporter import reporter


def load(path: str | Path) -> pd.DataFrame:
    """Load a dataset from a CSV, Parquet, Excel, or JSON file.

    Auto-detects the file format from the extension and prints a quick
    sanity check (row count, column count, memory usage, any red flags).

    Parameters
    ----------
    path : str or Path
        Path to the data file.

    Returns
    -------
    pd.DataFrame
        The loaded data.

    Examples
    --------
    >>> data = wl.load("house_prices.csv")
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    elif suffix == ".json":
        df = pd.read_json(p)
    elif suffix in (".tsv", ".txt"):
        df = pd.read_csv(p, sep="\t")
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported: .csv, .parquet, .xlsx, .json, .tsv"
        )

    _report_load(df, p)
    return df


def _report_load(df: pd.DataFrame, path: Path) -> None:
    reporter.section("📂", f"Loaded '{path.name}'")
    reporter.info(f"{len(df):,} rows × {len(df.columns)} columns")

    mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    if mem_mb < 1:
        reporter.info(f"Memory: {mem_mb * 1024:.0f} KB")
    else:
        reporter.info(f"Memory: {mem_mb:.1f} MB")

    # Red flags
    all_null_cols = df.columns[df.isna().all()].tolist()
    if all_null_cols:
        reporter.warn(
            f"Columns with only missing values: {all_null_cols}",
            fix=f"Consider dropping: data = data.drop(columns={all_null_cols})",
        )

    if len(df) == 0:
        reporter.warn("The dataset is empty!")

    if len(df.columns) == 1:
        reporter.warn("Only one column — you'll need features AND a target")

    reporter.next_step("wl.explore(data, target='your_target_column')")
