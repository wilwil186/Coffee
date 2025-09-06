# Refactor Coffee — EDA e Inferencia

Este paquete refactoriza el notebook original en dos cuadernos y un pequeño módulo de utilidades.

## Estructura
```
.
├── EDA.ipynb
├── Inferencia.ipynb
├── utils/
│   ├── io.py
│   └── metrics.py
├── data/
│   └── coffee_clean.csv   # (se genera al correr EDA.ipynb)
├── models/                # (se generan al correr Inferencia.ipynb)
└── predicciones/          # (se generan al correr Inferencia.ipynb)
```

## Requisitos
- Python 3.10+
- Bibliotecas: ver `requirements.txt`

## Datos de entrada
Coloca los archivos fuente en el directorio raíz (o ajusta las rutas dentro de los notebooks):
- `coffee_db.csv` con columnas `Country`, `Coffee type` y años como `1990/91`, ...
- `precios.csv` con precios diarios (se agregan a promedio anual).

## Pasos
1. **EDA.ipynb**
   - Limpia y normaliza los datos (convierte los años `1990/91` → `year`).
   - Une con `precios.csv` anualizado.
   - Calcula **KPIs** (ingresos, costos, utilidad, márgenes, CAGR, market share).
   - Genera visualizaciones (eje X con **todos los años** y **rotación 45°**, con **etiquetas**).
   - Guarda `data/coffee_clean.csv`.

2. **Inferencia.ipynb**
   - Construye *features*: rezagos, medias móviles, dummies de `country` y `type`.
   - Partición temporal (`train/val/test`) sin *leakage*.
   - Modelos: **baselines**, **Ridge**, **Lasso**, **RandomForest** (intenta **LightGBM/XGBoost** si están disponibles).
   - Métricas: **MAE, RMSE, MAPE, sMAPE, R²**.
   - Guarda modelos (`.joblib`) y predicciones (`predicciones/predicciones_test.csv`).
   - Imprime un **resumen ejecutivo** textual.

## Notas
- Los supuestos de costos se definen al inicio de `EDA.ipynb` y pueden ajustarse.
- Si tus datos ya están en formato largo, `utils.io.wide_to_long` lo detectará y no hará *unpivot*.
- El módulo `utils.metrics` incluye funciones reutilizables de negocio y evaluación.
