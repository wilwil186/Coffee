import re
from pathlib import Path
import pandas as pd
import numpy as np

YEAR_COL_PATTERN = re.compile(r"^\d{4}/\d{2}$")

def _end_year_from_span(span: str) -> int:
    a, b = span.split("/")
    y1 = int(a)
    yy = int(b)
    c = (y1 // 100) * 100
    if yy < (y1 % 100):
        return c + 100 + yy
    else:
        return c + yy

def detect_id_columns(cols):
    cols_l = [c.lower() for c in cols]
    country_map = {"country", "pais", "país"}
    type_map = {"coffee type", "tipo", "tipo_cafe", "coffee_type", "tipo de café"}
    country_col = None
    type_col = None
    for c in cols:
        if c.lower() in country_map:
            country_col = c
        if c.lower() in type_map:
            type_col = c
    if country_col is None:
        country_col = cols[0]
    if type_col is None and len(cols) > 1:
        type_col = cols[1]
    return country_col, type_col

def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    country_col, type_col = detect_id_columns(df.columns)
    year_cols = [c for c in df.columns if isinstance(c, str) and YEAR_COL_PATTERN.match(c)]
    if not year_cols:
        rename_map = {}
        for c in df.columns:
            cl = c.strip().lower()
            if cl in {"country", "pais", "país"}:
                rename_map[c] = "country"
            elif cl in {"coffee type", "tipo", "tipo_cafe", "coffee_type", "tipo de café"}:
                rename_map[c] = "type"
            elif cl in {"year", "anio", "año"}:
                rename_map[c] = "year"
            elif cl in {"consumption", "consumo"}:
                rename_map[c] = "consumption"
        out = df.rename(columns=rename_map)
        return out
    id_vars = [country_col]
    if type_col:
        id_vars.append(type_col)
    long_df = df.melt(id_vars=id_vars, value_vars=year_cols, var_name="year_span", value_name="consumption")
    long_df["year"] = long_df["year_span"].apply(_end_year_from_span).astype(int)
    long_df = long_df.drop(columns=["year_span"])
    rename_map = {country_col: "country"}
    if type_col:
        rename_map[type_col] = "type"
    long_df = long_df.rename(columns=rename_map)
    long_df["country"] = long_df["country"].astype(str).str.strip()
    if "type" in long_df:
        long_df["type"] = long_df["type"].astype(str).str.strip()
    else:
        long_df["type"] = "All"
    long_df["consumption"] = pd.to_numeric(long_df["consumption"], errors="coerce")
    long_df = long_df.dropna(subset=["consumption"])
    long_df = long_df.sort_values(["country", "type", "year"]).reset_index(drop=True)
    return long_df

def load_coffee_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)
    return wide_to_long(raw)

def _fix_price_header(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # quitar filas con tickers/headers
    bad_rows = df[df.eq("KC=F").any(axis=1)].index
    df = df.drop(index=bad_rows)
    if "Price" in df.columns and (df["Price"] == "Date").any():
        first_row = df.iloc[0]
        if str(first_row["Price"]).lower() == "date":
            new_cols = first_row.tolist()
            df = df.iloc[1:].copy()
            df.columns = new_cols
    rename = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"date", "fecha"}:
            rename[c] = "date"
        elif cl in {"price", "close", "cierre"}:
            rename[c] = "close"
    df = df.rename(columns=rename)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "close" in df.columns:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    return df

def load_price_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    df = _fix_price_header(df)
    df["year"] = df["date"].dt.year
    yearly = df.groupby("year", as_index=False)["close"].mean().rename(columns={"close": "price"})
    return yearly

def merge_coffee_price(coffee_long: pd.DataFrame, price_yearly: pd.DataFrame) -> pd.DataFrame:
    merged = coffee_long.merge(price_yearly, on="year", how="left").sort_values(["country","type","year"])
    merged["price"] = merged["price"].interpolate(limit_direction="both")
    return merged

def save_clean_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
