# High Garden Coffee — Reto Técnico de ML (EDA, Modelado, Evaluación y Chatbot)

> Proyecto de analítica y machine learning para la empresa ficticia **High Garden Coffee**. 
> El objetivo es transformar datos históricos (1990–2020) en **insights accionables**: tendencias de consumo, **rangos de precio** y **métricas de rentabilidad** (ingresos, utilidad y margen). 
> Incluye una propuesta **BONUS** de Chatbot para consulta en lenguaje natural.

---

## 🧭 Contexto del reto
Con base en el set de datos de consumo doméstico de café por **país** y **tipo**, se busca:
- Analizar y documentar el dataset (EDA).
- Construir **modelos supervisados** por objetivo (`price`, `consumption`, `profit`) respetando la dimensión temporal.
- Evaluar los modelos con métricas transparentes y reproducibles.
- Presentar resultados de manera clara para negocio.
- (BONUS) Incluir un **chatbot** que responda preguntas del negocio usando las predicciones y KPIs.

> Este repositorio cubre los **requisitos mínimos** (Análisis, Solución, Implementación/Evaluación y Presentación) y agrega el **BONUS**.

---

## 🗂️ Estructura del repositorio

```
.
├── EDA.ipynb                 # Limpieza, integración de precios y KPIs
├── Inferencia.ipynb          # Modelado por objetivo (sin gráficos)
├── Evaluacion.ipynb          # Holdout, selección de artefactos y métricas
├── utils/
│   ├── io.py
│   └── metrics.py
├── data/
│   └── coffee_clean.csv      # dataset maestro generado desde EDA
├── models/                   # artefactos .joblib
├── predicciones/             # CSVs de predicciones exportadas
├── results/                  # figuras opcionales
├── coffee_db.csv             # datos fuente
├── precios.csv               # precios diarios (se anualizan en EDA)
└── requirements.txt
```

---

## 📊 KPIs de negocio (definiciones reproducibles)

Estas columnas se calculan en **EDA** y quedan listas para modelado y reporting:

```python
# Asumiendo columnas: year, country, type, price, consumption
df = df.copy()

# 1) Ingresos
if {"price","consumption"} <= set(df.columns) and "revenue" not in df.columns:
    df["revenue"] = df["price"] * df["consumption"]

# 2) Utilidad (con costo unitario opcional)
COST_PER_UNIT = None  # Ajusta según tu negocio. Si None y existe 'profit', se respeta.
if "profit" not in df.columns:
    if COST_PER_UNIT is None:
        df["profit"] = df["revenue"]
    else:
        df["profit"] = df["revenue"] - COST_PER_UNIT * df["consumption"]

# 3) Margen
if "margin" not in df.columns and "revenue" in df.columns:
    df["margin"] = df["profit"] / df["revenue"]

# 4) Participación de mercado anual
if "market_share" not in df.columns and {"year","consumption"} <= set(df.columns):
    total_year = df.groupby("year")["consumption"].transform("sum")
    df["market_share"] = df["consumption"] / total_year
```

> **Sugerencia:** documenta el valor de `COST_PER_UNIT` usado en experimentos para interpretar correctamente `profit`.

---

## 🧪 Modelado y evaluación

- **Targets:** `price`, `consumption`, `profit`.
- **Features base:** `year` (num), `country` y `type` (one-hot/ordinal). Evitar *leakage* (solo pasado).
- **Partición temporal:** `train/val/test` (holdout final: 2020).
- **Modelos:** baselines (último valor / promedio), Ridge, Lasso, RandomForest (puedes ampliar a GBMs).
- **Métricas:** `MAE`, `RMSE`, `sMAPE` y `R²` cuando aplica.

**Ejemplo real (target=`price`, holdout=2020) de la última ejecución:**  
`RMSE ≈ 11.863`, `MAE ≈ 7.893`, `sMAPE ≈ 7.07%`, `n_val = 55`, artefacto: `lasso_price.joblib`.

> Nota: estos valores dependen de las *features* y de la partición. Mantén siempre la comparación con **baselines** para validar que el modelo agrega valor.

---

## 🚀 Cómo reproducir

### 1) Entorno
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Orden recomendado
1. **EDA.ipynb** → genera `data/coffee_clean.csv` + KPIs.
2. **Inferencia.ipynb** → entrena y guarda artefactos en `models/` (uno por target).
3. **Evaluacion.ipynb** → imprime métricas del holdout y selecciona el mejor artefacto. Puede producir CSVs en `predicciones/`.

### 3) Datos
- Coloca `coffee_db.csv` y `precios.csv` en el **raíz** del repo (o ajusta rutas).

---

## 📒 “Más bonito” en `Evaluacion.ipynb` (celdas listas para pegar)

> Estas celdas **no** agregan dependencias nuevas y se degradan con gracia si faltan variables/archivos.

### A) Resumen ejecutivo (robusto)
```python
from pathlib import Path
import json

def print_header(title):
    print("\n" + "—"*80)
    print(title.upper())
    print("—"*80)

def load_ev_if_any():
    # Intenta usar la variable `ev` del notebook; si no existe, busca un JSON en /results
    out = globals().get("ev", None)
    if out is not None:
        return out
    # fallback: results/ev_<TARGET>.json
    tgt = globals().get("TARGET", None)
    if tgt:
        p = Path("results") / f"ev_{tgt}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None

print_header("Resumen ejecutivo de evaluación")
ev_local = load_ev_if_any()
if ev_local is None:
    print("No encontré `ev` ni results/ev_<TARGET>.json — ejecuta primero la celda principal de evaluación.")
else:
    print("Objetivo:", ev_local.get("target", globals().get("TARGET", "?")).upper())
    metrics = ev_local.get("metrics", {})
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<10}: {v:,.4f}")
        else:
            print(f"  {k:<10}: {v}")
    best_art = ev_local.get("best_artifact") or ev_local.get("artifact") or "¿no registrado?"
    print("Mejor artefacto:", best_art)
```

### B) Comparación vs. baselines (si existe)
```python
print_header("Comparación vs baselines")
comp = ev_local.get("comparisons") if ev_local else None
if not comp:
    print("No hay tabla de comparación. (Opcional: guarda `ev['comparisons']`).")
else:
    import pandas as pd
    dfc = pd.DataFrame(comp).T
    display(dfc)
```

### C) Muestra de predicciones (auto-descubrimiento)
```python
print_header("Muestra de predicciones")
import pandas as pd

def try_load_predictions():
    # 1) predicciones/export
    tgt = globals().get("TARGET", None)
    if tgt:
        for name in [f"pred_{tgt}.csv", f"pred_{tgt}_holdout.csv"]:
            p = Path("predicciones") / name
            if p.exists():
                return pd.read_csv(p)
    # 2) results/preds_<TARGET>.csv
    if tgt:
        p2 = Path("results") / f"preds_{tgt}.csv"
        if p2.exists():
            return pd.read_csv(p2)
    return None

preds = try_load_predictions()
if preds is None:
    print("No encontré CSVs de predicción. Genera y exporta desde tu pipeline de evaluación.")
else:
    # columnas amigables primero, si existen
    preferred = [c for c in ["year","country","type","y_true","y_pred","error","pred_price","pred_consumption","pred_profit"] if c in preds.columns]
    cols = preferred + [c for c in preds.columns if c not in preferred]
    display(preds[cols].head(10))
```

---

## 💬 BONUS — Chatbot interno (prototipo)

Un **Streamlit** mínimo que responde preguntas de negocio (histórico y, si existen, predicciones) a partir de los CSVs del repo.

### Ejecución rápida
```bash
pip install streamlit pandas numpy
streamlit run app.py
```

### `app.py` (incluido en este repositorio como ejemplo)
- Carga `data/coffee_clean.csv` (o `coffee_db.csv` + `precios.csv` como respaldo).
- Si existen, integra CSVs de `predicciones/` (p. ej., `pred_price.csv`).
- Permite preguntas como: 
  - “**Precio** en *Colombia* para **2020**”
  - “**Consumo** de *Brasil*, **2015–2020**”
  - “**Utilidad** por *tipo* en **2019**”
- Devuelve tablas/resúmenes sin depender de un servicio externo.

> ¿Quieres LLM? Puedes añadir `pip install openai` y crear una respuesta en lenguaje natural a partir de los resultados tabulares (dejamos el ‘hook’ en el código).

---

## 🗺️ Roadmap sugerido
- Añadir **rezagos/MAs** y validación **walk-forward**.
- Probar **LightGBM/XGBoost** con *early stopping*.
- Registrar experimentos (MLflow) y **model cards** por artefacto.
- Extender el chatbot a **RAG** (índice FAISS de KPIs/predicciones por país/tipo/año).

---

## 📝 Licencia
Uso académico/demostrativo.
