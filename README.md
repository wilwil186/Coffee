# High Garden Coffee — **Presentación de Resultados** (Reto Técnico ML)

> Solución de analítica y machine learning para la empresa **Garden Coffee**: EDA, modelado supervisado (consumo, precio, utilidad), evaluación rigurosa y demo en app. Este documento se centra en **presentar los resultados** del modelado y cómo reproducirlos.

---

## 🎯 Objetivo del reto
Usar el histórico (1990–2020) de consumo (tazas), países y tipos de café para:
- Analizar la información y formular KPI de negocio (ingresos, costos, utilidad, margen).
- Construir y evaluar modelos que **predigan** consumo, precio y utilidad anual.
- Entregar **artefactos** de modelo y un flujo reproducible con una **demo** en app.

> Alineado a los mínimos de la prueba: análisis, solución analítica, implementación y evaluación, y **presentación de resultados** (ver PDF del reto).

---

## 🧰 Datos y alcance
- **`data/coffee_clean.csv`**: dataset anual por `country` y `type` con consumo (tazas) y precio anualizado.
- **Horizonte**: 1990–2020 (holdout = último año disponible).
- **KPI de negocio** (calculados en EDA): ingresos (= precio × consumo), costos (parametrizable), utilidad (= ingresos − costos), margen, CAGR, participación por país/tipo.

---

## 🧪 Metodología (resumen)
- **Partición temporal**: `train / val / test` con holdout del último año, evitando leakage.
- **Features**: rezagos y medias móviles (solo pasado), codificación por país y tipo.
- **Modelos**: baselines (último valor, promedio histórico) y modelos supervisados (Ridge, Lasso, **RandomForest**).
- **Métricas**: MAE, RMSE, **sMAPE** y n de observaciones por split.
- **Artefactos**: se guardan en `models/` como `.joblib` (uno por objetivo).

---

## 📈 Resultados de prueba (run actual)
Entrenamiento/evaluación realizado desde el notebook **`Inferencia_patched.ipynb`**. En validación, el mejor modelo para los tres objetivos fue **RandomForest**; abajo se reporta desempeño en **test** (n_test = 55).

### 🔹 `price` (precio anual)
- **Mejor modelo**: rf (validación MAE = 10.0550)
- **Test**: **RMSE = 0.8500**, **MAE = 0.8500**, **sMAPE = 0.383%**, **n_test = 55**
- **Artefacto**: `models/price_model.joblib`

### 🔹 `consumption` (tazas/año)
- **Mejor modelo**: rf (validación MAE = 1,545,649.7304)
- **Test**: **RMSE = 5,270,065.0822**, **MAE = 2,021,107.0803**, **sMAPE = 1.547%**, **n_test = 55**
- **Artefacto**: `models/consumption_model.joblib`

### 🔹 `profit` (utilidad anual)
- **Mejor modelo**: rf (validación MAE = 819,023,721.3023)
- **Test**: **RMSE = 1,236,075,648.5734**, **MAE = 376,286,684.7858**, **sMAPE = 4.984%**, **n_test = 55**
- **Artefacto**: `models/profit_model.joblib`

> **Lectura ejecutiva**: Los sMAPE bajos (≲5%) en los tres objetivos indican **buen ajuste relativo** a la escala de cada variable. Para negocio, esto habilita estimaciones anuales de precio, consumo y utilidad con errores esperados acotados, útiles para planeación y escenarios por país/tipo.

---

## 🧪 Validación y buenas prácticas
- Comparación contra **baselines** (último valor y promedio histórico) para dimensionar la ganancia del modelo.
- **Holdout temporal** del año más reciente y features 100% causales.
- Reporte de **n** por split y por segmento (país/tipo) cuando aplique.
- Artefactos versionados y reproducibles con `requirements.txt` y notebooks.

---

## 🔁 Reproducibilidad (paso a paso)
1) Crear entorno e instalar dependencias
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) EDA y KPIs (opcional si ya existe `data/coffee_clean.csv`)
- Ejecuta `EDA.ipynb` para limpiar, anualizar precios y exportar `data/coffee_clean.csv`.

3) Entrenar y evaluar
- Ejecuta `Inferencia_patched.ipynb` (o `Inferencia.ipynb`). Al finalizar se guardan:
  - `models/price_model.joblib`
  - `models/consumption_model.joblib`
  - `models/profit_model.joblib`
- Las predicciones de test/futuro se dejan en `predicciones/` y tablas en `results/` (si aplica).

4) Demo (app)
- Sigue **[guiaApp.md](./guiaApp.md)** para correr `app.py` (usa Google AI Studio API).

---

## 🧩 Uso de artefactos (snippet)
Para cargar un artefacto y predecir sobre nuevos datos con los mismos features:
```python
from joblib import load
import pandas as pd

art = load("models/price_model.joblib")
model = art["model"]
y_col = art.get("y_col", "price")
feat_cols = art["feat_cols"]

# df_new debe tener las mismas columnas de features
X_new = df_new[feat_cols].dropna()
df_pred = df_new.loc[X_new.index, ["year","country","type"]].copy()
yhat = model.predict(X_new)
df_pred[f"pred_{y_col}"] = yhat
```

---

## 💬 BONUS — Copiloto con LLM
Se incluye `app.py`, un **chatbot** que permite consultar métricas y predicciones de forma natural (por ejemplo: “proyección de precio para Colombia 2021–2023”). La guía de uso está en **[guiaApp.md](./guiaApp.md)** (API de Google AI Studio).

---

## ⚠️ Limitaciones y próximos pasos
- Mejorar imputación/outliers antes de features.
- Probar LightGBM/XGBoost y Prophet por segmento país×tipo.
- Intervalos de predicción basados en varianza de residuales por segmento.
- MLOps: `dvc` para versionar datos/modelos y CI para validar notebooks.

---

## 📂 Estructura relevante
```
.
├── EDA.ipynb
├── Inferencia.ipynb
├── Inferencia_patched.ipynb
├── app.py
├── guiaApp.md
├── data/coffee_clean.csv
├── models/
│   ├── price_model.joblib
│   ├── consumption_model.joblib
│   └── profit_model.joblib
├── predicciones/
└── results/
```

---

## 📝 Licencia
Este proyecto se publica bajo **GNU GPL v3**. Si vas a reutilizar código o modelos, mantén el aviso de licencia y comparte mejoras bajo los mismos términos.
