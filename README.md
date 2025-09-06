# High Garden Coffee — Analítica y ML para consumo, precio y utilidad

> Proyecto de machine learning e inteligencia de negocios para una exportadora internacional de café. Cubre EDA, modelado supervisado (consumo, precio y utilidad) y evaluación reproducible.

---

## 🎯 Objetivo
Aprovechar el set de datos histórico (1990–2020) de consumo y precios para:
- **Entender** la evolución por **país** y **tipo de café** (EDA + KPIs).
- **Predecir** variables clave a nivel anual: **consumo (tazas)**, **precio** y **utilidad**.
- **Evaluar** rigurosamente los modelos con **partición temporal** y métricas estándar.
- **Entregar** artefactos y un flujo reproducible para despliegue y toma de decisiones.

## 🧰 Estructura del repo
```text
.
├── EDA.ipynb                # Limpieza, unificación, KPIs de negocio y gráficos
├── Inferencia.ipynb         # Features (rezagos, MAs), entrenamiento y predicción (sin gráficos)
├── Evaluacion.ipynb         # Comparación de modelos y holdout del último año
├── utils/
│   ├── io.py                # Utilidades de I/O y reshape (wide→long, detección automática)
│   └── metrics.py           # Métricas de negocio y evaluación (MAE, RMSE, sMAPE, etc.)
├── data/
│   └── coffee_clean.csv     # Salida de EDA (normalizado y enriquecido)
├── models/                  # Artefactos .joblib (uno por objetivo)
├── predicciones/            # Predicciones del set de test y/o futuro
├── results/                 # Tablas/figuras de resultados
├── coffee_db.csv            # Datos de consumo por país/tipo (formato ancho)
├── precios.csv              # Precios diarios (se anualizan en EDA)
└── requirements.txt         # Dependencias del proyecto
```

## 📦 Datos y supuestos
- **`coffee_db.csv`**: columnas `Country`, `Coffee type` y años tipo `1990/91`… que se normalizan a columna `year`.
- **`precios.csv`**: precios diarios del café; se **agregan a promedio anual** y se integran por país/tipo.
- **KPIs de negocio** calculados en `EDA.ipynb` (puedes ajustar supuestos al inicio del notebook):
  - **Ingresos** = precio anual * consumo.
  - **Costos** (supuesto parametrizable): costo unitario * consumo.
  - **Utilidad** = ingresos − costos.
  - **Margen**, **CAGR**, **participación** por país/tipo, etc.

## 🔁 Flujo de trabajo reproducible
1) **Crear entorno**
```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) **EDA** (`EDA.ipynb`)
- Limpia y **normaliza años** (p.ej. `1990/91` → `year=1991`).
- Une con `precios.csv` **anualizado**.
- Genera visualizaciones con **todos los años en el eje X (rotación 45°)** y **etiquetas de datos**.
- Exporta `data/coffee_clean.csv`.

3) **Modelado e inferencia** (`Inferencia.ipynb`)
- Construcción de **features**: rezagos (solo pasado), medias móviles, dummies de `country` y `type`.
- **Partición temporal**: `train/val/test` (sin leakage; holdout = último año disponible).
- Modelos probados: **baselines** (último valor, promedio histórico), **Ridge**, **Lasso**, **RandomForest** (opcional: LightGBM/XGBoost si están instalados).
- Métricas: **MAE**, **RMSE**, **MAPE**, **sMAPE**, **R²**.
- Guarda artefactos en `models/` y predicciones en `predicciones/`.

4) **Evaluación** (`Evaluacion.ipynb`)
- Centraliza la **comparación de modelos** por objetivo y el **holdout** del año más reciente.
- Imprime un **resumen ejecutivo** por objetivo (métricas + n de validación).
- Permite cargar un artefacto y **predecir años futuros** con intervalos simples (PI80/PI95 absolutos).

## 📊 Resultados (corte actual del repo)
**Holdout 2020 — target = `price`**

| Modelo | Artefacto | RMSE | MAE | sMAPE (%) | n_val |
|---|---|---:|---:|---:|---:|
| Lasso | `models/lasso_price.joblib` | **5.4913** | **3.9316** | **3.5788** | **52** |

> _Notas_: el flujo **infirió** metadatos ausentes del artefacto (p. ej. `y_col='price'`, `group_cols=['country','type']`) a partir del nombre/convenciones, y luego ejecutó la evaluación sobre el holdout 2020.

Si ya corriste `Inferencia.ipynb` para **consumption** y **profit**, agrega aquí sus tablas con las métricas análogas (RMSE/MAE/sMAPE/R²) y los artefactos elegidos.

## 🧪 Métricas y validación
- Comparación contra **baselines** (último valor, promedio histórico) para dimensionar ganancias.
- **Partición temporal** y **features causales** (sin usar futuro) para evitar leakage.
- Reporte de **número de observaciones** en validación (`n_val`) y resumen por **país**/**tipo** si aplica.

## ⚙️ Uso de artefactos (predicción)
Ejemplo de uso programático de un artefacto `.joblib` con intervalos absolutos:
```python
from joblib import load
from utils.io import build_xy  # misma lógica que en entrenamiento

art = load("models/lasso_price.joblib")
mdl = art["model"]
y_col = art.get("y_col", "price")
feat_cols = art["feat_cols"]
q80, q95 = art.get("PI80_abs", 0.0), art.get("PI95_abs", 0.0)

X_new, _, _ = build_xy(df_new, y_col)   # respeta ingeniería de features
preds = mdl.predict(X_new.dropna())

out = df_new.loc[X_new.dropna().index, ["year", *art.get("group_cols", ["country","type"]) ]].copy()
out[f"pred_{y_col}"] = preds
out[f"pred_{y_col}_lo80"] = preds - q80
out[f"pred_{y_col}_hi80"] = preds + q80
out[f"pred_{y_col}_lo95"] = preds - q95
out[f"pred_{y_col}_hi95"] = preds + q95
```

## 🧾 Cómo presentar (alineado a la prueba)
- **Análisis de la información**: EDA con KPIs (ingresos, costos, margen, CAGR, share) y gráficos por país/tipo.
- **Solución analítica**: pipelines de features + modelos supervisados por objetivo.
- **Implementación y evaluación**: artefactos versionados (`models/`), partición temporal, métricas (RMSE/MAE/sMAPE/R²), comparación con baselines.
- **Presentación de resultados**: tabla de métricas + resumen ejecutivo y predicciones exportables (`predicciones/`).

## 💬 BONUS — Copiloto con LLM
Propuesta de valor rápido:
- **Chatbot “CoffeeCopilot”** (CLI/Gradio/Streamlit) que contesta:
  - *“¿Cuál es el top-5 de países por crecimiento en 2010–2020?”*
  - *“Muéstrame la proyección de precio para Colombia, tipo Robusta, 2021–2023.”*
  - *“¿Qué factores más pesan en el modelo de precio?”* (explicabilidad básica)
- Arquitectura sugerida: loader de artefactos + capa de consultas sobre `predicciones/` + plantilla RAG (FAQ de negocio) para respuestas citadas.
- Seguridad: control de alcance (solo lectura de resultados) y registro de auditoría.

## 🧩 Limitaciones y próximos pasos
- **Imputación**: mejorar manejo de huecos y outliers antes de features.
- **Modelos**: probar **LightGBM/XGBoost** y **prophet** para series por país/tipo.
- **Intervalos**: PI dependientes de varianza del residuo por segmento (no solo absolutos).
- **MLOps**: añadir pruebas, `dvc` para data/artefactos y CI para validar notebooks.

## 📄 Licencia
Uso educativo/demostrativo. Ajusta a tu licencia preferida si publicarás.
