# coffee_chatbot_gemini.py
# Chat + predicciones locales usando tus modelos .joblib y Google AI Studio (Gemini)

import os
import re
import sys
from typing import Optional, Dict, Tuple
from pathlib import Path

import pandas as pd
from joblib import load
import google.generativeai as genai

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ====================== CONFIG GEMINI ======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ====================== ARTIFACTS / DATA ======================
ART_PRICE = os.getenv("ART_PRICE", "models/lasso_price.joblib")
ART_CONS  = os.getenv("ART_CONSUMPTION", "models/lasso_consumption.joblib")
ART_PROF  = os.getenv("ART_PROFIT", "models/lasso_profit.joblib")
DATA_CLEAN = os.getenv("DATA_CLEAN", "data/coffee_clean.csv")

TARGET_TO_ART = {"precio": ART_PRICE, "consumo": ART_CONS, "utilidad": ART_PROF}

# Comando: admite valores con o sin comillas (p.ej. pais="Costa Rica")
CMD_RE = re.compile(
    r"""^/pred\s+(?P<target>precio|consumo|utilidad)\s+
        (?:pais|pa[ií]s)=(?P<pais>"[^"]+"|[^ ]+)\s+
        tipo=(?P<tipo>"[^"]+"|[^ ]+)\s+
        a[nñ]o=(?P<anio>\d{4})$""",
    re.IGNORECASE | re.VERBOSE
)

# ====================== UTILIDADES DATA/MODELOS ======================
def ensure_basic_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Country" in df.columns and "country" not in df.columns:
        df = df.rename(columns={"Country": "country"})
    if "Coffee type" in df.columns and "type" not in df.columns:
        df = df.rename(columns={"Coffee type": "type"})
    if "año" in df.columns and "year" not in df.columns:
        df = df.rename(columns={"año": "year"})
    # tipificar
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "country" in df.columns:
        df["country"] = df["country"].astype(str)
    if "type" in df.columns:
        df["type"] = df["type"].astype(str)
    return df

def load_context_df() -> pd.DataFrame:
    for p in [DATA_CLEAN, "coffee_clean.csv", "coffee_db.csv"]:
        if os.path.exists(p):
            return ensure_basic_cols(pd.read_csv(p))
    raise FileNotFoundError("No encontré data histórica. Ajusta DATA_CLEAN o ubica coffee_clean.csv/coffee_db.csv.")

def try_import_build_xy():
    try:
        # Si en tu repo existe utils.features.build_xy, lo usamos.
        from utils.features import build_xy  # type: ignore
        return build_xy
    except Exception:
        return None

def build_features_fallback(df_all: pd.DataFrame, y_col: str, feat_cols: Optional[list]) -> Tuple[pd.DataFrame, Optional[pd.Series], list]:
    """
    Fallback simple:
      - Si el artifact trae feat_cols, usamos esas columnas (y dummies para country/type si están).
      - Si no, usamos ['year','country','type'] con one-hot.
    """
    df = df_all.copy()
    if feat_cols:
        keep = [c for c in feat_cols if c in df.columns]
        if not keep:
            raise ValueError("feat_cols del artifact no existen en el dataframe.")
        X = df[keep].copy()
        cat_like = [c for c in keep if c in ("country", "type")]
        if cat_like:
            X = pd.get_dummies(X, columns=cat_like, drop_first=False)
    else:
        base = [c for c in ["year", "country", "type"] if c in df.columns]
        if not base:
            raise ValueError("Faltan columnas mínimas para features (year/country/type).")
        X = pd.get_dummies(df[base], columns=[c for c in ["country","type"] if c in base], drop_first=False)
    y = df[y_col] if y_col in df.columns else None
    return X, y, list(X.columns)

# --------- NUEVO: soportar artifacts tipo dict o Pipeline ----------
def infer_y_col_from_filename(path: str) -> str:
    name = Path(path).name.lower()
    if re.search(r"consum", name): return "consumption"
    if re.search(r"profit|util", name): return "profit"
    return "price"

def load_artifact_info(artifact_path: str):
    art = load(artifact_path)
    # Caso 1: dict con metadata
    if isinstance(art, dict):
        mdl = art.get("model")
        y_col = art.get("y_col", "price")
        feat_cols = art.get("feat_cols")
        group_cols = art.get("group_cols", ["country", "type"])
        q80 = art.get("PI80_abs"); q95 = art.get("PI95_abs")
        return mdl, y_col, feat_cols, group_cols, q80, q95, art
    # Caso 2: Pipeline/estimador directo
    mdl = art
    y_col = infer_y_col_from_filename(artifact_path)
    feat_cols = None
    try:
        fni = getattr(mdl, "feature_names_in_", None)
        if fni is not None:
            feat_cols = list(fni)
    except Exception:
        pass
    group_cols = ["country", "type"]
    q80 = q95 = None
    return mdl, y_col, feat_cols, group_cols, q80, q95, {"model": mdl}

def pipeline_expects_raw(pipeline) -> bool:
    """Heurística: si hay ColumnTransformer en el Pipeline, esperamos columnas crudas."""
    try:
        if isinstance(pipeline, Pipeline):
            for _, step in pipeline.steps:
                if isinstance(step, ColumnTransformer):
                    return True
                if isinstance(step, Pipeline):
                    for __, sub in step.steps:
                        if isinstance(sub, ColumnTransformer):
                            return True
    except Exception:
        pass
    return False

def predict_with_artifact(df_all: pd.DataFrame, artifact_path: str, year: int, country: str, ctype: str) -> Dict:
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"No encuentro el artifact: {artifact_path}")

    mdl, y_col, feat_cols, group_cols, q80, q95, _ = load_artifact_info(artifact_path)
    df_all = ensure_basic_cols(df_all)

    # normaliza valores (case-insensitive)
    def norm_val(col, val):
        if col not in df_all.columns:
            return val
        for u in pd.Series(df_all[col].astype(str).unique()):
            if u.lower() == str(val).lower():
                return u
        return val

    country = norm_val("country", country)
    ctype = norm_val("type", ctype)

    # agrega fila objetivo si falta
    base_cols = ["year"] + [c for c in group_cols if c in df_all.columns]
    mask_exists = (
        (df_all.get("country") == country) &
        (df_all.get("type") == ctype) &
        (df_all.get("year") == year)
    )
    if not mask_exists.any():
        new_row = {c: None for c in df_all.columns}
        for c in base_cols:
            new_row[c] = {"year": year, "country": country, "type": ctype}.get(c, None)
        df_all = pd.concat([df_all, pd.DataFrame([new_row])], ignore_index=True)

    # Construcción de features
    build_xy = try_import_build_xy()
    if build_xy:
        try:
            X_all, _, _ = build_xy(df_all, y_col)
        except Exception:
            # Si tu build_xy falla por diferencias, usamos fallback
            X_all, _, _ = build_features_fallback(df_all, y_col, feat_cols)
    else:
        if pipeline_expects_raw(mdl):
            base = [c for c in ["year", "country", "type"] if c in df_all.columns]
            if not base:
                raise ValueError("Faltan columnas mínimas (year/country/type).")
            X_all = df_all[base].copy()
        else:
            X_all, _, _ = build_features_fallback(df_all, y_col, feat_cols)

    # localizar fila objetivo (usando df_all para filtrar)
    mask_idx = pd.Series(True, index=X_all.index)
    if "year" in df_all.columns:
        mask_idx &= (df_all.loc[X_all.index, "year"].astype(int) == int(year))
    if "country" in df_all.columns:
        mask_idx &= (df_all.loc[X_all.index, "country"].astype(str) == str(country))
    if "type" in df_all.columns:
        mask_idx &= (df_all.loc[X_all.index, "type"].astype(str) == str(ctype))
    idx = X_all.index[mask_idx]
    if len(idx) == 0:
        raise RuntimeError("No pude ubicar la fila objetivo tras construir features.")

    # Alinear columnas esperadas
    if feat_cols:
        for m in [c for c in feat_cols if c not in X_all.columns]:
            X_all[m] = 0
        X_all = X_all[feat_cols]

    X_pred = X_all.loc[idx]
    if isinstance(X_pred, pd.Series):
        X_pred = X_pred.to_frame().T

    y_hat = float(mdl.predict(X_pred)[-1])
    out = {
        "pred": y_hat,
        "group": {"country": str(country), "type": str(ctype), "year": int(year)}
    }
    if q80 is not None:
        out.update({"lo80": y_hat - float(q80), "hi80": y_hat + float(q80)})
    if q95 is not None:
        out.update({"lo95": y_hat - float(q95), "hi95": y_hat + float(q95)})
    return out

# ====================== CHAT CON GEMINI ======================
def chat_gemini(history, user_text):
    if not GEMINI_API_KEY:
        raise RuntimeError("Falta GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)

    # Convertir historial a formato que acepta Gemini
    gem_history = []
    for m in history:
        if m["role"] == "system":
            gem_history.append({"role": "user", "parts": [m["content"]]})
        elif m["role"] == "user":
            gem_history.append({"role": "user", "parts": [m["content"]]})
        elif m["role"] == "assistant":
            gem_history.append({"role": "model", "parts": [m["content"]]})

    model = genai.GenerativeModel(GEMINI_MODEL)
    chat = model.start_chat(history=gem_history)
    resp = chat.send_message(user_text)
    return resp.text

# ====================== CLI PRINCIPAL ======================
def strip_quotes(s: str) -> str:
    return s[1:-1] if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")) else s

def main():
    system_prompt = (
        "Eres CoffeeBot. Respondes en español con claridad. "
        "Si el usuario usa '/pred <target> pais=<p> tipo=<t> año=<y>', "
        "predice con los modelos locales (targets: precio, consumo, utilidad). "
        "Pide datos faltantes con precisión."
    )
    history = [{"role": "system", "content": system_prompt}]

    print(f"☕ CoffeeBot (Gemini: {GEMINI_MODEL}) listo.")
    print("Ejemplos:")
    print('  /pred precio pais=Colombia tipo=Arabica año=2021')
    print('  /pred consumo pais="Costa Rica" tipo=Robusta año=2020')
    print("Escribe 'salir' para terminar.\n")

    while True:
        try:
            user = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChao!")
            break

        if not user:
            continue
        if user.lower() in {"salir", "exit", "quit"}:
            print("Chao! 👋")
            break

        m = CMD_RE.match(user)
        if m:
            target = m.group("target").lower()
            pais = strip_quotes(m.group("pais"))
            tipo = strip_quotes(m.group("tipo"))
            anio = int(m.group("anio"))
            art_path = TARGET_TO_ART.get(target)
            if not art_path:
                print("\nBot: No hay artifact configurado para ese target.\n")
                continue
            try:
                df_hist = load_context_df()
                res = predict_with_artifact(df_hist, art_path, anio, pais, tipo)
                pi80 = f" (PI80: {res['lo80']:.3f}–{res['hi80']:.3f})" if "lo80" in res else ""
                pi95 = f" (PI95: {res['lo95']:.3f}–{res['hi95']:.3f})" if "lo95" in res else ""
                print(
                    f"\nBot: ✅ Predicción de {target} para "
                    f"{res['group']['country']} / {res['group']['type']} / {res['group']['year']}:\n"
                    f"- pred: {res['pred']:.3f}{pi80}{pi95}\n"
                )
            except Exception as e:
                print(f"\nBot: ❌ No pude predecir: {e}\n")
            continue

        # Small talk / explicaciones con Gemini
        try:
            answer = chat_gemini(history, user)
        except Exception as e:
            answer = f"No pude generar respuesta con Gemini: {e}"
        print(f"\nBot: {answer}\n")
        history.extend([{"role": "user", "content": user}, {"role": "assistant", "content": answer}])

if __name__ == "__main__":
    main()