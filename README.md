# High Garden Coffee — EDA, Inferencia y Evaluación

> Proyecto de analítica y machine learning para la empresa ficticia **High Garden Coffee**. Incluye **Análisis Exploratorio de Datos (EDA)**, **modelado inferencial** y **evaluación**, cumpliendo los mínimos del reto técnico y agregando una propuesta **BONUS** de IA generativa.

---

## 🎯 Contexto del reto
Datos históricos (1990–2020) de consumo doméstico de café por **país** y **tipo**. Objetivo: **tendencias**, **rangos de precios futuros** y **métricas de negocio** (ingresos/utilidad) para apoyar decisiones.

**Requisitos cubiertos:**
- ✅ **Análisis de la información** (EDA)
- ✅ **Solución analítica de negocio** (Inferencia: `price`, `consumption`, `profit`)
- ✅ **Implementación y evaluación** (Evaluación)
- ✅ **Presentación de resultados** (este README + figuras)
- ⭐ **BONUS**: propuesta de **IA generativa/LLM** (abajo)

---

## 📁 Estructura del repositorio

```
.
├── EDA.ipynb
├── Inferencia.ipynb
├── Evaluacion.ipynb
├── utils/
│   ├── io.py
│   └── metrics.py
├── data/
│   └── coffee_clean.csv      # (se genera desde EDA.ipynb)
├── models/                   # artefactos .joblib
├── predicciones/             # CSVs de predicciones
├── results/                  # figuras (opcional)
├── coffee_db.csv             # datos fuente (wide)
├── precios.csv               # precios diarios (se anualizan)
└── requirements.txt
```

---

## 🧹 EDA — limpieza, integración y KPIs

### 1) Limpieza y normalización
- Convierte formato **wide → long**: años `1990/91`, `1991/92`, … pasan a una sola columna **`year`** (entero).
- Estandariza nombres a minúsculas con guion bajo: `Country`→`country`, `Coffee type`→`type`, etc.
- Control de calidad: tipos, nulos, duplicados.

### 2) Integración de **precios**
- `precios.csv` (diario) → **anualización** (p.ej. promedio anual) y unión por **`year`**.

### 3) KPIs de negocio (cómo se calculan)
> Tu EDA deja disponibles: `price`, `consumption`, **`revenue`**, **`profit`**, **`margin`**, **`market_share`**. Este es el código base para reproducirlos (idempotente: sólo calcula si falta la columna).

```python
import pandas as pd
df = df.copy()

# 1) revenue = price * consumption
if {"price","consumption"} <= set(df.columns) and "revenue" not in df.columns:
    df["revenue"] = df["price"] * df["consumption"]

# 2) profit: si no existe, permitir costo unitario configurable
COST_PER_UNIT = None  # define tu costo; si None y ya existe 'profit', se respeta
if "profit" not in df.columns:
    if COST_PER_UNIT is None:
        # Si no se especifica costo, asume 0 para reproducibilidad (ajusta según tu negocio)
        df["profit"] = df["revenue"]
    else:
        df["profit"] = df["revenue"] - COST_PER_UNIT * df["consumption"]

# 3) margin = profit / revenue (evitar división por cero)
if "margin" not in df.columns and "revenue" in df.columns:
    df["margin"] = df["profit"] / df["revenue"]
    df.loc[~df["revenue"].replace({0: pd.NA}).notna(), "margin"] = pd.NA

# 4) market_share: participación por año
if "market_share" not in df.columns and {"year","consumption"} <= set(df.columns):
    total_year = df.groupby("year")["consumption"].transform("sum")
    df["market_share"] = df["consumption"] / total_year
```

### 4) Visualización
- Series y barras por **país** y **tipo** (rotación de 45° en el eje X y etiquetas de datos).
- Exporta figuras a `results/` (opcional).
- Guarda dataset maestro en `data/coffee_clean.csv` para su uso en modelado.

---

## 🧪 Inferencia — modelos por objetivo

**Objetivos:** `price`, `consumption`, `profit`  
**Features base:** `year` (numérica), `country` y `type` (categóricas con one-hot).  
**Partición temporal:** `train/val/test` respetando el tiempo (evita *leakage*).  
**Modelos:** baselines (último valor/promedio), **Ridge**, **Lasso**, RandomForest.  
**Métricas:** MAE, RMSE, (s)MAPE y R² cuando aplica.  
**Artefactos:** se guardan en `models/` como `.joblib`.  
**Predicciones:** se exportan a `predicciones/`.

---

## 🧾 Evaluación — resultados reales del repo

**Holdout (último año = 2020) para `TARGET = "price"` con el mejor artefacto:**  
- **Artifact elegido:** `lasso_price.joblib`  
- **Métricas (2020):**
  - **RMSE:** `11.8629`
  - **MAE:** `7.8928`
  - **sMAPE:** `7.0723%`
  - **n_val:** `55`
- **Predicciones generadas:** `shape = (1760, 4)`

> Nota: estos valores son los **obtenidos en tu `Evaluacion.ipynb`**. Si cambias features o particiones, las métricas variarán. Mantén la comparación con baselines para validar mejora.

---

## 🚀 Cómo replicar

1) **Entorno**
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\activate
pip install -r requirements.txt
```

2) **Orden recomendado**
- Ejecuta `EDA.ipynb` → genera `data/coffee_clean.csv` y figuras.
- Ejecuta `Inferencia.ipynb` → entrena y guarda artefactos/predicciones.
- Ejecuta `Evaluacion.ipynb` → imprime métricas finales, elige mejor modelo y puede predecir años futuros.

3) **Datos**
- Ubica `coffee_db.csv` y `precios.csv` en el raíz (o ajusta rutas en los cuadernos).

---

## 📈 Recomendaciones
- Añadir **rezagos** y **medias móviles** del target y/o precio internacional.
- Probar **ensembles gradientes** (XGBoost/LightGBM) con *time-series CV* (walk-forward).
- Analizar **métricas por segmento** (país/tipo) para identificar dónde el modelo aporta más.
- Documentar **supuestos de costos** usados para `profit` y verificar consistencia de unidades.

---

## 🤖 BONUS — Analista Virtual (LLM)
Prototipo de **chatbot interno** con RAG sobre:
- Predicciones por país/tipo/año
- KPIs (ingresos/utilidad/margen)
- Resúmenes ejecutivos por mercado
Respuesta en lenguaje natural para preguntas como: “**¿Pronóstico de consumo en Colombia 2025?**” o “**Top 5 países por margen en 2020**”.

---

## ✅ Checklist
- [x] EDA con KPIs y dataset limpio
- [x] Modelos por objetivo
- [x] Evaluación con holdout (2020) y artefacto óptimo
- [x] Presentación (README) y guías de ejecución
- [ ] BONUS LLM (prototipo)

---

## Licencia
Uso académico/demostrativo.
