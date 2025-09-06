from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

def profit(consumption, price, fixed_cost=0.0, variable_cost_per_unit=0.0):
    consumption = np.asarray(consumption, dtype=float)
    price = np.asarray(price, dtype=float)
    return price * consumption - (fixed_cost + variable_cost_per_unit * consumption)

def margin(consumption, price, fixed_cost=0.0, variable_cost_per_unit=0.0):
    ingresos = price * consumption
    util = profit(consumption, price, fixed_cost, variable_cost_per_unit)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(ingresos != 0, util / ingresos, np.nan)
    return m

def cagr(first_value, last_value, periods):
    first_value = float(first_value)
    last_value = float(last_value)
    periods = int(periods)
    if periods <= 0 or first_value <= 0 or last_value <= 0:
        return np.nan
    return (last_value / first_value) ** (1.0 / periods) - 1.0

def market_share(df: pd.DataFrame, value_col: str, group_cols=("year",)):
    df = df.copy()
    total = df.groupby(list(group_cols))[value_col].transform("sum")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["market_share"] = np.where(total != 0, df[value_col] / total, np.nan)
    return df

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res/ss_tot

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.abs((y_true - y_pred) / y_true) * 100.0
    out = out[~np.isinf(out)]
    return np.mean(out)

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom != 0, np.abs(y_pred - y_true) / denom, np.nan) * 100.0
    return np.nanmean(out)

@dataclass
class RegressionReport:
    mae: float
    rmse: float
    mape: float
    smape: float
    r2: float

def regression_report(y_true, y_pred) -> RegressionReport:
    return RegressionReport(
        mae=mae(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mape=mape(y_true, y_pred),
        smape=smape(y_true, y_pred),
        r2=r2(y_true, y_pred),
    )
