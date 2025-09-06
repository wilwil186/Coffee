# High Garden Coffee — EDA, Inferencia y Evaluación

> Proyecto de analítica y machine learning para la empresa ficticia **High Garden Coffee**. Incluye **Análisis Exploratorio de Datos (EDA)**, **modelado inferencial** y **evaluación**, cumpliendo los mínimos del reto técnico y agregando una propuesta **BONUS** de IA generativa.

---

## 🎯 Contexto del reto
La empresa busca aprovechar datos históricos (1990–2020) de consumo doméstico de café por país y tipo para **identificar tendencias**, **estimar precios/útiles futuros** y **apoyar decisiones comerciales**. La solución exige:
- **Análisis de la información**
- **Solución analítica a problemas de negocio**
- **Implementación y evaluación**
- **Presentación de resultados**
- *(BONUS)* Propuesta de **IA generativa/LLM**

---

## 📁 Estructura del repositorio

```
.
├── EDA.ipynb               # Limpieza, integración de fuentes, KPIs de negocio y visualización
├── Inferencia.ipynb        # Features, partición temporal, modelos y predicciones
├── Evaluacion.ipynb        # Holdout/val/test y métricas finales por objetivo
├── utils/
│   ├── io.py               # Utilidades de carga/transformación (wide→long, etc.)
│   └── metrics.py          # Métricas de negocio y evaluación (MAE, RMSE, sMAPE, etc.)
├── data/
│   └── coffee_clean.csv    # Datos limpios generados por el EDA
├── models/                 # Artefactos .joblib guardados por Inferencia
├── predicciones/           # CSVs con predicciones por objetivo/horizonte
├── results/                # Salidas de evaluación/figuras (si aplica)
├── coffee_db.csv           # Fuente en bruto (consumo por país y tipo, años anchos)
├── precios.csv             # Precios diarios (se anualizan en el EDA)
└── requirements.txt
```

---

## 🧹 ¿Qué hace el EDA?
El cuaderno **`EDA.ipynb`** deja el dataset listo para modelar y para comunicar resultados de negocio.

**1) Normalización / preparación**
- Unpivot: convierte años en columnas (`1990/91`, `1991/92`, …) a una sola columna `year` (entero).
- Estandariza nombres: `Country`→`country`, `Coffee type`→`type`, etc.
- Control de calidad: tipos, nulos y duplicados.

**2) Integración de precios**
- Lee `precios.csv` (diario) y **anualiza** (p.ej., promedio por año calendario).
- Hace **join** por año con el consumo limpio.

**3) Variables de negocio (KPIs)**
> Los supuestos de costos se ajustan al inicio del notebook. Cambia los parámetros para replicar tus cálculos.
- **`revenue` (ingreso)**: `price * consumption` (ajusta unidades si tu consumo no está en la misma unidad de precio).
- **`cost` (costo)**: función de costo configurable (p.ej., costo unitario * consumo).
- **`profit` (utilidad)**: `revenue - cost`.
- **`margin` (margen)**: `profit / revenue`.
- **`market_share`**: participación por país/año respecto al total.
- **CAGR**: tasa compuesta de crecimiento para consumo/ingresos por país.

**4) Visualización (presentación amigable)**
- Tendencias por país/tipo; top países y participación.
- Eje X con **todos los años** y **rotación 45°**.
- **Etiquetas de datos** en barras/líneas cuando aplica.
- Exporta figuras a `results/` (opcional).
- **Guarda** `data/coffee_clean.csv` como dataset maestro.

---

## 🧪 ¿Qué hace la Inferencia?
El cuaderno **`Inferencia.ipynb`** arma un set de features simple pero extensible y entrena modelos por objetivo: `price`, `consumption`, `profit`.

**Features base**
- Numéricas: `year` (y opcionalmente rezagos/medias móviles del target).
- Categóricas: `country`, `type` (one-hot encoding).

**Partición temporal (sin leakage)**
- División `train/val/test` respetando el tiempo (p.ej., 1994–2014 / 2015–2017 / 2018–2020).

**Modelos**
- **Baselines**: último valor, promedio histórico (referencia de negocio).
- **Lineales penalizados**: Ridge / Lasso.
- **No lineales**: RandomForest (y opcional **LightGBM/XGBoost** si están disponibles).

**Métricas**
- **MAE**, **RMSE**, **MAPE/sMAPE**, y **R²** cuando aplica.
- Salida estandarizada a `predicciones/` y **artefactos** a `models/` (.joblib).

---

## 🧾 ¿Qué hace la Evaluación?
El cuaderno **`Evaluacion.ipynb`** centraliza la **comparación de modelos** y el **holdout** del **último año** para cada objetivo.

- Imprime un **resumen ejecutivo** por objetivo (métricas y n de validación).
- Permite **predecir años futuros** (opcional) con el mejor pipeline.
- Ejemplo de salida para `TARGET = "price"` (holdout 2020):
  - `RMSE ≈ 37.93`, `MAE ≈ 37.92`, `sMAPE ≈ 29.27%`, `n_val = 55`, `year_val = 2020`.
  - Interpretación: ~30% de error relativo es **aceptable para exploración**, pero **mejorable** para uso operativo.
- Guarda predicciones y, si quieres, el modelo final en `models/`.

> **Tip**: compara siempre contra baselines; si un modelo no bate al baseline, no es apto para producción.

---

## 🚀 Cómo replicar en local

1) **Clonar e instalar**
```bash
git clone https://github.com/wilwil186/Coffee.git
cd Coffee
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
# .venv\Scripts\activate                          # (Windows PowerShell)
pip install -r requirements.txt
```

2) **Ubicar datos fuente**
- Coloca `coffee_db.csv` y `precios.csv` en el raíz del repo (o ajusta rutas en los notebooks).

3) **Ejecutar cuadernos (orden recomendado)**
- `EDA.ipynb` → genera `data/coffee_clean.csv` y figuras.
- `Inferencia.ipynb` → entrena modelos y guarda predicciones/artefactos.
- `Evaluacion.ipynb` → imprime métricas finales y genera predicciones por año/país/tipo.

> Si usas VSCode/Jupyter, asegúrate de seleccionar el *kernel* de la venv.

---

## 📌 Preguntas de negocio que responde
- **¿Cómo evolucionó el consumo por país y tipo (1990–2020)?**
- **¿Cuál es la participación de mercado por país y su CAGR?**
- **¿Qué variables explican mejor `price`, `consumption` y `profit`?**
- **¿Qué esperamos para 2021–2022 (o el horizonte que definas)?**

---

## 📈 Recomendaciones y siguientes pasos
- Incorporar **más señales**: macro (PIB, inflación), oferta (producción, clima), y **precios internacionales** del café (ICO).
- **Ingeniería de atributos temporal**: rezagos/ventanas, shocks exógenos, festivos.
- Probar **modelos de series** (Prophet/ARIMA) y **ensembles gradientes** (XGBoost/LightGBM).
- Validación **time-series cross‑validation** (walk‑forward) para robustez.
- Métricas **por segmento** (país/tipo) para priorizar mercados.

---

## 🤖 BONUS — Analista Virtual (IA Generativa/LLM)
Propuesta de un **chatbot interno** conectado a:
- **Predicciones** por país/tipo/año
- **KPIs de negocio** (ingresos, utilidad, margen)
- **Notas del EDA** y supuestos
Con **RAG** (Retrieval‑Augmented Generation) los equipos de ventas/estrategia consultan en lenguaje natural: *“¿Pronóstico de consumo en Colombia 2025?”*, *“Top 5 países por margen en 2020”*, o *“Resumen ejecutivo de Vietnam”*.
- Entregables: endpoint REST + UI simple (Dash/Streamlit) + cuaderno de ejemplo.

---

## ✅ Checklist de entrega
- [x] **Análisis de la información (EDA)**
- [x] **Solución analítica con modelos (Inferencia)**
- [x] **Implementación y evaluación (Evaluacion)**
- [x] **Presentación de resultados (este README + figuras)**
- [ ] *(Bonus opcional)* **Prototipo LLM/Chatbot**

---

## 📝 Notas de reproducibilidad
- Se fijan *random seeds* donde aplica.
- Las rutas de salida (`data/`, `models/`, `predicciones/`, `results/`) se crean si no existen.
- Los notebooks lanzan mensajes claros ante columnas faltantes o esquemas distintos.

---

## Licencia
Uso académico/demostrativo. Ajusta según tu necesidad.
