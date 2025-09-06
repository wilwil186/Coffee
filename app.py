# app.py — Chatbot mínimo de negocio (sin LLM por defecto)
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="High Garden Coffee — Chatbot", layout="wide")
st.title("🤖 Chatbot — High Garden Coffee")

# ---------- Carga de datos ----------
def load_data():
    # Preferir dataset maestro
    p_master = Path("data/coffee_clean.csv")
    if p_master.exists():
        df = pd.read_csv(p_master)
    else:
        # Respaldo: coffee_db.csv (formato wide) + precios.csv (opcional)
        p_db, p_prices = Path("coffee_db.csv"), Path("precios.csv")
        if not p_db.exists():
            st.error("No se encontró data/coffee_clean.csv ni coffee_db.csv")
            st.stop()
        df = pd.read_csv(p_db)
        # Intento de normalización mínima
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if "year" not in df.columns:
            # Convertir columnas tipo 1990/91 a 'year'
            wide_years = [c for c in df.columns if re.fullmatch(r"\d{4}/\d{2}", c)]
            if wide_years:
                df_long = df.melt(id_vars=[c for c in df.columns if c not in wide_years],
                                  value_vars=wide_years, var_name="year_season", value_name="consumption")
                df_long["year"] = df_long["year_season"].str.slice(0,4).astype(int)
                df = df_long.drop(columns=["year_season"])
        if p_prices.exists():
            prices = pd.read_csv(p_prices)
            prices.columns = [c.strip().lower() for c in prices.columns]
            # anualización muy simple si hay daily_date y price
            if {"date","price"} <= set(prices.columns):
                prices["year"] = pd.to_datetime(prices["date"]).dt.year
                price_year = prices.groupby("year", as_index=False)["price"].mean().rename(columns={"price":"price"})
                df = df.merge(price_year, on="year", how="left")
    # Tipos suaves
    for col in ["country","type"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    # KPIs (si faltan)
    if {"price","consumption"} <= set(df.columns) and "revenue" not in df.columns:
        df["revenue"] = df["price"] * df["consumption"]
    if "profit" not in df.columns and "revenue" in df.columns:
        df["profit"] = df["revenue"]
    if "margin" not in df.columns and "revenue" in df.columns:
        df["margin"] = np.where(df["revenue"].replace(0, np.nan).notna(), df["profit"]/df["revenue"], np.nan)
    # Integrar predicciones si existen
    preds_dir = Path("predicciones")
    if preds_dir.exists():
        for csv in preds_dir.glob("*.csv"):
            try:
                p = pd.read_csv(csv)
                # detectar target por nombre de columna
                target_cols = [c for c in p.columns if c.startswith("pred_")]
                if target_cols:
                    # merge por columnas clave
                    keys = [col for col in ["year","country","type"] if col in p.columns and col in df.columns]
                    if keys:
                        df = df.merge(p, on=keys, how="left")
            except Exception:
                pass
    return df

df = load_data()

# ---------- Utilidades de consulta ----------
def pick_country(query: str):
    countries = sorted(df["country"].dropna().unique().tolist()) if "country" in df.columns else []
    if not countries:
        return None
    # heurística: contiene el nombre
    for c in countries:
        if c.lower() in query.lower():
            return c
    return None

def parse_years(query: str):
    years = [int(y) for y in re.findall(r"(19|20)\d{2}", query)]
    if not years:
        return None, None
    return min(years), max(years)

def answer(query: str):
    q = query.strip()
    if not q:
        return "Haz una pregunta como: 'Precio en Colombia 2020' o 'Consumo Brasil 2015-2020'."
    country = pick_country(q)
    y_min, y_max = parse_years(q)
    target = None
    for t in ["price","consumption","profit","revenue","margin"]:
        if t in q.lower():
            target = t
            break
        # alias
        if "utilidad" in q.lower():
            target = "profit"
        if "ingres" in q.lower():
            target = "revenue"
        if "margen" in q.lower():
            target = "margin"

    df_q = df.copy()
    if country and "country" in df_q.columns:
        df_q = df_q[df_q["country"].str.lower() == country.lower()]
    if y_min and "year" in df_q.columns:
        df_q = df_q[(df_q["year"] >= y_min) & (df_q["year"] <= (y_max or y_min))]

    # Elegir columnas a mostrar
    show_cols = [c for c in ["year","country","type","price","consumption","revenue","profit","margin"] if c in df_q.columns]
    # Si hay predicciones para algún target
    pred_cols = [c for c in df_q.columns if c.startswith("pred_")]
    if pred_cols:
        show_cols += pred_cols

    if target and target in df_q.columns:
        show_cols = [c for c in show_cols if c in df_q.columns]
        out = df_q.sort_values(["year","country","type"], na_position="last")[show_cols].head(50)
        return out
    else:
        # si no detectamos target, devolvemos un resumen pequeño
        out = df_q.sort_values(["year","country","type"], na_position="last")[show_cols].head(50)
        if out.empty:
            return "No encontré datos que coincidan con tu consulta."
        return out

st.markdown("Escribe preguntas en español (histórico y, si existen, predicciones). Ejemplos:")
st.code("Precio en Colombia 2020\nConsumo Brasil 2015-2020\nUtilidad por tipo 2019", language="text")

user_q = st.text_input("Tu pregunta", value="Precio en Colombia 2020")
resp = answer(user_q)

if isinstance(resp, pd.DataFrame):
    st.dataframe(resp, use_container_width=True)
else:
    st.info(resp)

st.caption("Prototipo sin LLM — opcionalmente puedes integrar OpenAI para respuestas en lenguaje natural.")
