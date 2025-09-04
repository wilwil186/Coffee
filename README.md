#### **1. Análisis de la Información (Análisis Exploratorio de Datos - EDA)**

El primer paso es entender a fondo la historia que cuentan los datos.

*   **Procesamiento de Datos:**
    *   La tarea inicial es "limpiar" y reestructurar el set de datos. Los años están en formato de texto y como columnas (`1990/91`, `1991/92`, etc.). Transformaría esta tabla ancha a un formato largo, con tres columnas principales: `Pais`, `Año`, y `Consumo`. Esto facilita enormemente el análisis de series temporales.
    *   Verificaría la consistencia de la columna `Total_domestic_consumption` y, si es necesario, la recalcularía para asegurar la calidad de los datos.

*   **Visualización y Hallazgos Clave:**
    *   **Tendencia Global:** Crearía un gráfico de líneas para visualizar el consumo total de café a nivel mundial entre 1990 y 2020. Esto nos dirá si el mercado global está en crecimiento, estancado o en declive.
    *   **Principales Mercados:** Un diagrama de barras mostraría los 10 países con el mayor consumo acumulado. Esto permite identificar los mercados más grandes y consolidados. Países como Brasil, Etiopía, Indonesia y Colombia probablemente aparecerán en los primeros lugares.
    *   **Mercados Emergentes y de Alto Crecimiento:** Analizaría la tasa de crecimiento anual compuesta (CAGR) del consumo para cada país. Esto es clave para identificar mercados que, aunque no sean los más grandes hoy, tienen el mayor potencial de crecimiento a futuro. Países como Vietnam o Filipinas podrían destacar aquí.
    *   **Análisis por Tipo de Café:** Agruparía los datos por tipo de café (Arabica, Robusta, Mezclas) para analizar si hay una tendencia de consumo global o regional hacia un tipo específico de grano.

#### **2. Solución a las Problemáticas de Negocio (Modelado Analítico)**

Con los insights del análisis, procedemos a construir soluciones predictivas. La principal problemática de negocio es saber **dónde y cuánto café vender en el futuro**.

**Solución A: Modelo de Pronóstico de Demanda (Forecasting)**

Dado que el set de datos contiene información histórica anual, la técnica más adecuada es el análisis de series temporales para predecir el consumo futuro.

*   **Metodología:**
    1.  Para cada país clave, se tratará su consumo histórico como una serie temporal.
    2.  Implementaría un modelo como **ARIMA (Autoregressive Integrated Moving Average)** o **Prophet de Facebook**. Estos modelos son excelentes para capturar tendencias y estacionalidades en datos históricos.
    3.  El objetivo sería pronosticar el consumo para los próximos 5 a 10 años (ej. 2021-2030) para cada país.

*   **Abordando la Falta de Precios:** El reto pide rangos de precios futuros, pero el dataset no incluye precios. Lo abordaría de la siguiente manera:
    *   **Declarar la limitación:** Sería transparente al indicar que con los datos provistos no se pueden predecir precios directamente.
    *   **Usar el consumo como un *proxy* de la demanda:** Argumentaría que un pronóstico de alta demanda en un país sugiere un mercado saludable donde los precios probablemente se mantendrán fuertes o crecerán. Un mercado con demanda decreciente es un riesgo.
    *   **Recomendación de Siguientes Pasos:** Propondría enriquecer el análisis futuro con datos de precios históricos de fuentes públicas, como la Organización Internacional del Café (ICO).

**Solución B: Segmentación de Mercados (Clustering)**

No todos los mercados son iguales. Una estrategia de "talla única" es ineficiente. Por ello, propongo usar un modelo de clustering para agrupar a los países con características similares.

*   **Metodología:**
    1.  Utilizaría un algoritmo de clustering como **K-Means**.
    2.  Las variables para segmentar a los países serían:
        *   Volumen de consumo promedio.
        *   Tasa de crecimiento del consumo (CAGR).
        *   Preferencia por tipo de café (ej. mayormente Arabica o Robusta).
    3.  El resultado serían 3 o 4 grupos de países, por ejemplo:
        *   **Mercados Maduros/Gigantes:** Alto consumo, bajo crecimiento (Ej. Brasil). Estrategia: Mantener relaciones y enfocarse en la eficiencia.
        *   **Mercados en Expansión:** Consumo medio, alto crecimiento (Ej. Vietnam). Estrategia: Invertir agresivamente en marketing y distribución para capturar el crecimiento.
        *   **Mercados de Nicho/Pequeños:** Bajo consumo, crecimiento variable. Estrategia: Enfoque en cafés especiales o de nicho con mayor margen.

#### **3. Implementación y Evaluación de la Solución**

*   **Implementación:** Utilizaría Python con librerías como Pandas para la manipulación de datos, Matplotlib/Seaborn para visualizaciones, y `statsmodels` o `prophet` para el forecasting. Para el clustering, usaría `Scikit-learn`.
*   **Evaluación:**
    *   **Forecasting:** Para validar el modelo de pronóstico, dividiría los datos históricos en un conjunto de entrenamiento (ej. 1990-2015) y uno de prueba (2016-2020). Mediría el error del modelo en el conjunto de prueba usando métricas como el Error Absoluto Medio (MAE) para asegurar que las predicciones son fiables.
    *   **Clustering:** La evaluación sería más cualitativa, asegurando que los segmentos creados tengan sentido desde una perspectiva de negocio y sean claramente diferenciables.

#### **4. Presentación de los Resultados**

La presentación final estaría enfocada en la toma de decisiones, no en los detalles técnicos.

1.  **Dashboard Interactivo:** Crearía un dashboard (usando herramientas como Tableau, Power BI, o Dash en Python) que resuma los hallazgos. Incluiría:
    *   Un mapa mundial coloreado por clúster de mercado.
    *   Gráficos con los pronósticos de demanda para los 10 países más importantes.
    *   Filtros para que un gerente de ventas pueda explorar por región o tipo de café.
2.  **Recomendaciones Accionables:** Concluiría con 3 a 5 recomendaciones claras para "High Garden Coffee", basadas en los resultados. Por ejemplo: "Recomendamos priorizar la expansión en el Sudeste Asiático, ya que nuestros modelos predicen un crecimiento de la demanda del 25% en los próximos 5 años, liderado por Vietnam e Indonesia".

---

### **BONUS: Integración con IA Generativa y LLMs**

Para llevar la solución al siguiente nivel y demostrar habilidades en IA de vanguardia, propondría lo siguiente:

**Propuesta: Un "Analista de Mercados Virtual" para High Garden Coffee**

Crearía un chatbot interno basado en un LLM (Modelo de Lenguaje Grande) que permita a los ejecutivos y equipos de ventas interactuar con los datos y los resultados del análisis en lenguaje natural.

*   **¿Cómo funcionaría?**
    1.  **Base de Conocimiento (RAG - Retrieval-Augmented Generation):** El LLM se conectaría a una base de datos que contiene todos los hallazgos: los datos limpios, los pronósticos de consumo para cada país, la definición de cada clúster de mercado y los insights clave del análisis.
    2.  **Interfaz de Chat:** Los usuarios podrían hacer preguntas directas como:
        *   "¿Cuál es el pronóstico de consumo de café en Colombia para 2025?"
        *   "Dame un resumen de los mercados en el clúster de 'Alto Crecimiento'."
        *   "¿Qué países de África muestran una tendencia positiva en el consumo de Robusta?"
    3.  **Generación de Reportes Automáticos:** El sistema también podría generar reportes automáticos. Un gerente podría pedir: "Crea un informe de una página sobre las oportunidades de mercado en México", y el LLM redactaría un resumen coherente utilizando los datos y pronósticos disponibles.

*   **Valor Agregado:**
    *   **Democratización de Datos:** Elimina la barrera técnica. Un gerente no necesita saber de Python o SQL para obtener respuestas.
    *   **Eficiencia:** Automatiza la creación de reportes y la consulta de datos, liberando tiempo del equipo de análisis para tareas más estratégicas.
    *   **Decisiones más Rápidas:** Proporciona insights instantáneos para apoyar decisiones comerciales rápidas.

Esta propuesta de solución no solo cumple con todos los requisitos del reto, sino que también demuestra una comprensión profunda del negocio, habilidades técnicas en machine learning y una visión de futuro con la integración de IA generativa.


¡ÉXITOS
