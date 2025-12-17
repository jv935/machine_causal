"""
Data loading utilities for the Microsoft Azure Predictive Maintenance (PdM) dataset.

This module provides functions to:
- Load raw PdM CSV files
- Build a MultiIndex hourly panel with telemetry + events
- Infer data types (continuous vs discrete)
- Scale continuous columns
- Split panel by machine
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


@dataclass
class PdMConfig:
    """Configuration for PdM dataset file names and column prefixes."""
    telemetry_file: str = "PdM_telemetry.csv"
    errors_file: str = "PdM_errors.csv"
    maint_file: str = "PdM_maint.csv"
    failures_file: str = "PdM_failures.csv"
    machines_file: str = "PdM_machines.csv"

    # Prefixes for generated columns
    error_prefix: str = "err_"
    maint_prefix: str = "mnt_"
    fail_prefix: str = "fail_"
    model_prefix: str = "model_"


def _read_csv(path: Union[str, Path], parse_dates: Sequence[str]) -> pd.DataFrame:
    """Read CSV with date parsing."""
    return pd.read_csv(path, parse_dates=list(parse_dates))


def load_pdm_raw(data_dir: Union[str, Path], cfg: PdMConfig = PdMConfig()) -> Dict[str, pd.DataFrame]:
    """
    Load all 5 PdM CSVs with parsed datetimes.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing the PdM_*.csv files
    cfg : PdMConfig
        Configuration object with file names
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys: 'telemetry', 'errors', 'maint', 'failures', 'machines'
    """
    data_dir = Path(data_dir)
    telemetry = _read_csv(data_dir / cfg.telemetry_file, parse_dates=["datetime"])
    errors = _read_csv(data_dir / cfg.errors_file, parse_dates=["datetime"])
    maint = _read_csv(data_dir / cfg.maint_file, parse_dates=["datetime"])
    failures = _read_csv(data_dir / cfg.failures_file, parse_dates=["datetime"])
    machines = pd.read_csv(data_dir / cfg.machines_file)

    return {
        "telemetry": telemetry,
        "errors": errors,
        "maint": maint,
        "failures": failures,
        "machines": machines,
    }


def _filter_machines(
    machines_df: pd.DataFrame,
    model: Optional[str] = None,
    machine_ids: Optional[Sequence[int]] = None
) -> pd.DataFrame:
    """Filter machines dataframe by model and/or machine IDs."""
    out = machines_df.copy()
    if model is not None:
        out = out[out["model"] == model]
    if machine_ids is not None:
        out = out[out["machineID"].isin(list(machine_ids))]
    return out


def _hourly_pivot_counts(
    df: pd.DataFrame,
    value_col: str,
    cfg: PdMConfig,
    prefix: str
) -> pd.DataFrame:
    """
    Pivot event data to hourly counts.
    
    Input df columns: ['datetime', 'machineID', value_col]
    Output: indexed by ['datetime', 'machineID'] with columns like f"{prefix}{category}"
    """
    if df.empty:
        return pd.DataFrame(columns=["datetime", "machineID"]).set_index(["datetime", "machineID"])

    tmp = df.copy()
    # Dataset is already hourly; defensively drop exact duplicates
    tmp = tmp.drop_duplicates(subset=["datetime", "machineID", value_col])
    counts = (
        tmp.groupby(["datetime", "machineID", value_col])
           .size()
           .rename("count")
           .reset_index()
    )
    pivot = (
        counts.pivot_table(
            index=["datetime", "machineID"],
            columns=value_col,
            values="count",
            fill_value=0,
            aggfunc="sum"
        )
    )
    pivot.columns = [f"{prefix}{c}" for c in pivot.columns.astype(str)]
    pivot = pivot.sort_index()
    return pivot


def build_hourly_panel(
    raw: Dict[str, pd.DataFrame],
    *,
    model: Optional[str] = None,
    machine_ids: Optional[Sequence[int]] = None,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    cfg: PdMConfig = PdMConfig(),
    drop_constant_cols: bool = True,
    one_hot_model: bool = True,
    include_model: bool = True,
    include_age: bool = True,
) -> pd.DataFrame:
    """
    Build a MultiIndex (datetime, machineID) dataframe with all requested features.
    
    Parameters
    ----------
    raw : Dict[str, pd.DataFrame]
        Output from load_pdm_raw()
    model : str, optional
        Filter to specific machine model type
    machine_ids : Sequence[int], optional
        Filter to specific machine IDs
    start : str or pd.Timestamp, optional
        Start datetime for filtering
    end : str or pd.Timestamp, optional
        End datetime for filtering
    cfg : PdMConfig
        Configuration object
    drop_constant_cols : bool
        Whether to drop columns with only one unique value (default: True)
    one_hot_model : bool
        Whether to one-hot encode the model column (default: True)
    include_model : bool
        Whether to include model information (default: True)
    include_age : bool
        Whether to include machine age (default: True)
        
    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with (datetime, machineID) index
    """
    telemetry = raw["telemetry"].copy()
    errors = raw["errors"].copy()
    maint = raw["maint"].copy()
    failures = raw["failures"].copy()
    machines = raw["machines"].copy()

    # Machine filter
    machines_f = _filter_machines(machines, model=model, machine_ids=machine_ids)
    keep_ids = machines_f["machineID"].unique().tolist()

    # Time filter on telemetry (our base timeline)
    telemetry = telemetry[telemetry["machineID"].isin(keep_ids)].copy()
    if start is not None:
        start_ts = pd.to_datetime(start)
        telemetry = telemetry[telemetry["datetime"] >= start_ts]
    if end is not None:
        end_ts = pd.to_datetime(end)
        telemetry = telemetry[telemetry["datetime"] <= end_ts]

    # Base panel: telemetry indexed by (datetime, machineID)
    base = telemetry.set_index(["datetime", "machineID"]).sort_index()

    # Pivot event tables (already hourly) to aligned indicator/count columns
    err_p = _hourly_pivot_counts(
        errors[errors["machineID"].isin(keep_ids)], "errorID", cfg, cfg.error_prefix
    )
    mnt_p = _hourly_pivot_counts(
        maint[maint["machineID"].isin(keep_ids)], "comp", cfg, cfg.maint_prefix
    )
    fail_p = _hourly_pivot_counts(
        failures[failures["machineID"].isin(keep_ids)], "failure", cfg, cfg.fail_prefix
    )

    # Join onto telemetry timeline
    panel = base.join(err_p, how="left").join(mnt_p, how="left").join(fail_p, how="left")

    # Fill missing event indicators with 0
    event_cols = [
        c for c in panel.columns 
        if c.startswith((cfg.error_prefix, cfg.maint_prefix, cfg.fail_prefix))
    ]
    panel[event_cols] = panel[event_cols].fillna(0).astype("int16")

    # Add machine metadata (repeated hourly)
    machines_f = machines_f.set_index("machineID")
    if include_age:
        age = panel.index.get_level_values("machineID").map(machines_f["age"])
        panel["age"] = age.values.astype(float)

    if include_model:
        model_series = panel.index.get_level_values("machineID").map(machines_f["model"]).astype(str)
        if one_hot_model:
            dummies = pd.get_dummies(model_series, prefix=cfg.model_prefix.rstrip("_"))
            # Align index with panel
            dummies.index = panel.index
            panel = pd.concat([panel, dummies.astype(float)], axis=1)
        else:
            panel["model"] = model_series.values

    panel = panel.sort_index()

    if drop_constant_cols:
        nunique = panel.nunique(dropna=False)
        const_cols = nunique[nunique <= 1].index.tolist()
        if len(const_cols) > 0:
            panel = panel.drop(columns=const_cols)

    return panel


def infer_data_types(
    panel: pd.DataFrame, 
    *, 
    continuous_cols: Optional[Sequence[str]] = None
) -> pd.Series:
    """
    Infer column data types for causal discovery algorithms.
    
    Returns a Series indexed by column name with:
    - 0 = continuous
    - 1 = discrete/categorical
    
    By default, telemetry columns (volt, rotate, pressure, vibration, age) are 
    treated as continuous, and all other columns as discrete.
    
    Parameters
    ----------
    panel : pd.DataFrame
        The data panel
    continuous_cols : Sequence[str], optional
        If provided, REPLACES the default continuous columns with this list.
        To ADD columns, include the defaults: 
        continuous_cols=['volt', 'rotate', 'pressure', 'vibration', 'age', 'my_col']
        
    Returns
    -------
    pd.Series
        Series with column names as index and 0/1 as values
    """
    col_types = pd.Series(1, index=panel.columns, dtype=int)  # default: discrete
    
    # Default continuous columns (telemetry + age)
    default_cont = ["volt", "rotate", "pressure", "vibration", "age"]
    
    # If user provides continuous_cols, use that instead (complete replacement)
    if continuous_cols is not None:
        default_cont = list(continuous_cols)

    # Mark continuous columns
    for c in default_cont:
        if c in col_types.index:
            col_types.loc[c] = 0

    return col_types


def scale_continuous_columns(
    panel: pd.DataFrame,
    col_types: pd.Series,
    *,
    scaler: Optional[StandardScaler] = None,
):
    """
    Standardize continuous columns (type==0) using sklearn StandardScaler.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Input data
    col_types : pd.Series
        Output from infer_data_types()
    scaler : StandardScaler, optional
        Pre-fitted scaler to use. If None, a new scaler is fitted.
        
    Returns
    -------
    Tuple[pd.DataFrame, StandardScaler]
        Scaled panel and the scaler used
    """
    cont_cols = col_types[col_types == 0].index.tolist()
    if len(cont_cols) == 0:
        return panel.copy(), scaler

    X = panel[cont_cols].values.astype(float)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X)
    Xs = scaler.transform(X)
    out = panel.copy()
    out.loc[:, cont_cols] = Xs
    return out, scaler


def panel_to_machine_datasets(
    panel: pd.DataFrame, 
    machine_ids: Optional[Sequence[int]] = None
) -> Dict[int, pd.DataFrame]:
    """
    Split MultiIndex panel into a dict of machineID -> (T x N) dataframe.
    
    Parameters
    ----------
    panel : pd.DataFrame
        MultiIndex DataFrame with (datetime, machineID) index
    machine_ids : Sequence[int], optional
        Specific machine IDs to extract. If None, extracts all.
        
    Returns
    -------
    Dict[int, pd.DataFrame]
        Dictionary mapping machine ID to its time series data
    """
    if machine_ids is None:
        machine_ids = panel.index.get_level_values("machineID").unique().tolist()
    datasets = {}
    for mid in machine_ids:
        df = panel.xs(mid, level="machineID").sort_index()
        datasets[int(mid)] = df
    return datasets
