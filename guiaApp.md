# Cómo ejecutar `app.py` (CoffeeBot) — **Google AI Studio (Gemini)**

Usamos **Google AI Studio (Gemini)** como LLM. Solo necesitas tu **API Key**.

## Requisitos

* Python 3.9+
* API Key de **Google AI Studio (Gemini)**

## Instalación

```bash
git clone https://github.com/wilwil186/Coffee.git
cd Coffee
python -m venv .venv && source .venv/bin/activate  # (Windows: .\.venv\Scripts\Activate.ps1)
pip install -U streamlit google-generativeai pandas scikit-learn joblib
```

## API Key

Define tu llave antes de correr (o pégala luego en el sidebar):

```bash
# Linux/macOS
export GEMINI_API_KEY="TU_API_KEY"

# Windows PowerShell
$env:GEMINI_API_KEY="TU_API_KEY"
```

## Ejecutar

```bash
streamlit run app.py
```

Abre `http://localhost:8501`. En el **sidebar**:

* Pega tu `GEMINI_API_KEY` (si no lo exportaste).
* Ajusta rutas a datos/modelos si hace falta (`DATA_CLEAN` y `ART_*`).

## Uso (prompts)

Predicción con comando:

```
/pred precio pais=Colombia tipo=Arabica año=2021
/pred consumo pais="Costa Rica" tipo=Robusta año=2020
/pred utilidad pais=Brazil tipo=Arabica año=2019
```

Conversación libre (Gemini): escribe cualquier pregunta, p. ej. *“resume tendencias por país”*.

> Repo: [https://github.com/wilwil186/Coffee](https://github.com/wilwil186/Coffee) — Usa **API de Google AI Studio (Gemini)**.
