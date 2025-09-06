# utils/io.py
# ------------------------------------------------------------
# Limpieza y transformaciones para el proyecto Coffee
# - Convierte tablas anchas (1990/91) → largo (year entero)
# - Carga y limpia precios diarios, anualiza por promedio
# - Une consumo con precio por año
# ------------------------------------------------------------

from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Iterable, Tuple

# -----------------------
# Limpieza y transformaciones (consumo)
# -----------------------

YEAR_COL_PATTERN = re.compile(r"^\d{4}/\d{2}$")

def _end_year_from_span(span: str) -> int:
    """
    Convierte '1990/91' → 1991, manejando cambio de siglo: '1999/00' → 2000.
    """
    a, b = span.split("/")
    y1 = int(a)
    yy = int(b)
    century = (y1 // 100) * 100
    if yy < (y1 % 100):
        return century + 100 + yy
    return century + yy

def detect_id_columns(cols: Iterable[str]) -> Tuple[str, str | None]:
    """
    Detecta columnas de país y tipo con nombres frecuentes; si no, hace fallback.
    """
    cols = list(cols)
    country_map = {"country", "pais", "país"}
    type_map = {"coffee type", "tipo", "tipo_cafe", "coffee_type", "tipo de café"}

    country_col = None
    type_col = None
    for c in cols:
        cl = c.lower().strip()
        if cl in country_map:
            country_col = c
        if cl in type_map:
            type_col = c

    if country_col is None:
        country_col = cols[0]
    if type_col is None and len(cols) > 1:
        type_col = cols[1]
    return country_col, type_col

def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ancho→largo. Salida estándar: country, type, year, consumption.
    Si ya viene en largo, solo normaliza nombres.
    """
    df = df.copy()
    country_col, type_col = detect_id_columns(df.columns)
    year_cols = [c for c in df.columns if isinstance(c, str) and YEAR_COL_PATTERN.match(c)]

    if not year_cols:
        # Ya puede estar en largo: normalizamos nombres
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
        if "type" not in out.columns:
            out["type"] = "All"
        return out

    # Unpivot
    id_vars = [country_col]
    if type_col:
        id_vars.append(type_col)
    long_df = df.melt(id_vars=id_vars, value_vars=year_cols,
                      var_name="year_span", value_name="consumption")
    long_df["year"] = long_df["year_span"].apply(_end_year_from_span).astype(int)
    long_df = long_df.drop(columns=["year_span"])

    # Estandarizar nombres
    rename_map = {country_col: "country"}
    if type_col:
        rename_map[type_col] = "type"
    long_df = long_df.rename(columns=rename_map)

    # Tipos y limpieza
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
    """
    Carga el dataset de consumo (CSV/Parquet) y lo transforma a formato largo.
    """
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)
    return wide_to_long(raw)

# -----------------------
# Limpieza y transformaciones (precios)
# -----------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres a minúsculas_con_guiones."""
    df = df.copy()
    norm = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    return df.rename(columns=norm)

def _looks_like_date_series(s: pd.Series) -> bool:
    """Heurística: la serie parece fecha si >80% es parseable."""
    try:
        parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        return parsed.notna().mean() > 0.8
    except Exception:
        return False

def _drop_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas típicas de export (ticker 'KC=F', header duplicado con 'Date').
    Funciona con tu precios.csv: primeras filas 'Ticker/KC=F' y 'Date'.
    """
    df = df.copy()

    # 1) Filas con el ticker literal en alguna celda
    bad_idx = df[df.apply(lambda r: any(str(x).strip() == "KC=F" for x in r), axis=1)].index
    df = df.drop(index=bad_idx, errors="ignore")

    # 2) Fila de encabezado duplicado donde 'Price' == 'Date'
    if "Price" in df.columns:
        mask_header = df["Price"].astype(str).str.strip().eq("Date")
        if mask_header.any():
            header_row = df.index[mask_header][0]
            # Re-asignar encabezados desde esa fila y descartar filas previas
            new_cols = df.loc[header_row].tolist()
            df = df.loc[df.index > header_row].copy()
            df.columns = new_cols

    # 3) También eliminar posible fila 'Ticker' en la primera columna
    if "Price" in df.columns:
        df = df[df["Price"].astype(str).str.strip().ne("Ticker")]

    return df

def _pick_date_column(df: pd.DataFrame) -> str:
    """
    Elige columna de fecha. En tu archivo, la fecha viene en 'Price' tras limpiar.
    """
    candidates = [c for c in ["date", "fecha", "price", "precio"] if c in df.columns]
    for c in candidates:
        if _looks_like_date_series(df[c]):
            return c
    for c in df.columns:
        if _looks_like_date_series(df[c]):
            return c
    raise ValueError("No se pudo identificar la columna de fecha en precios.csv")

def _pick_close_column(df: pd.DataFrame) -> str:
    """
    Elige la columna de precio de cierre entre alias típicos.
    Para tu CSV, será 'close'.
    """
    ordered_aliases = [
        "close", "adj_close",
        "cierre", "cierre_ajustado", "precio_cierre",
        "closing_price", "precio_cierre_ajustado",
    ]
    for c in ordered_aliases:
        if c in df.columns:
            if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8:
                return c
    # fallback por coincidencia parcial
    for c in df.columns:
        if ("close" in c) or ("cierre" in c):
            if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5:
                return c
    raise KeyError("No se encontró columna de precio de cierre ('close'/'adj_close'/'cierre').")

def _fix_price_header(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DF de precios y devuelve columnas: date (datetime), close (float).
    """
    # 1) Elimina filas basura y normaliza nombres
    df = _drop_bad_rows(df)
    df = _normalize_columns(df)

    # 2) Detecta columnas de fecha y cierre
    date_col = _pick_date_column(df)
    close_col = _pick_close_column(df)

    # 3) Construye salida estándar
    out = df[[date_col, close_col]].copy()
    out.columns = ["date", "close"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce", infer_datetime_format=True)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    return out

def load_price_data(path: str | Path) -> pd.DataFrame:
    """
    Carga precios diarios desde CSV, limpia encabezados/filas raras,
    y devuelve un DataFrame anual con precio promedio por año: columns = [year, price].
    """
    path = Path(path)
    df_raw = pd.read_csv(path)
    df = _fix_price_header(df_raw)
    df["year"] = df["date"].dt.year
    yearly = (df.groupby("year", as_index=False)["close"]
                .mean()
                .rename(columns={"close": "price"}))
    return yearly

# -----------------------
# Merge, interpolación y guardado
# -----------------------

def merge_coffee_price(coffee_long: pd.DataFrame, price_yearly: pd.DataFrame) -> pd.DataFrame:
    """
    Une consumo (por país/tipo/año) con precio anual global.
    Si falta precio para algún año, interpola temporalmente.
    """
    merged = (coffee_long.merge(price_yearly, on="year", how="left")
                        .sort_values(["country", "type", "year"])
                        .reset_index(drop=True))
    # Precio anual es global → interpolación simple global por año
    merged["price"] = merged["price"].interpolate(limit_direction="both")
    return merged

def save_clean_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    """Guarda el dataset limpio (CSV)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
