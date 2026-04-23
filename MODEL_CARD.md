---
language:
  - es
  - en
license: apache-2.0
base_model: distilbert-base-multilingual-cased
tags:
  - text-classification
  - emotion-classification
  - music
  - lyrics
  - spanish
  - english
  - fine-tuned
  - distilbert
  - education
pipeline_tag: text-classification
library_name: transformers
metrics:
  - accuracy
  - f1
widget:
  - text: "Esta noche te escribo llorando, no puedo dejar de pensar en ti"
    example_title: "Tristeza"
  - text: "Quiero bailar contigo toda la noche, la vida es una fiesta"
    example_title: "Alegría"
  - text: "Eres mi todo, contigo quiero pasar toda mi vida"
    example_title: "Amor"
  - text: "Recuerdo esos días de infancia que nunca volverán"
    example_title: "Nostalgia"
  - text: "Estoy harto de tus mentiras, se acabó, no vuelvas más"
    example_title: "Rabia"
model-index:
  - name: musicbot-emotion-classifier
    results:
      - task:
          type: text-classification
          name: Emotion Classification
        metrics:
          - type: accuracy
            value: 0.7572
          - type: f1
            value: 0.7475
            name: F1 macro
---

# MusicBot — Clasificador de Emociones en Letras de Canciones

Clasificador multiclase que detecta la emoción predominante en fragmentos de letras de canciones en **español e inglés**. Fine-tune de `distilbert-base-multilingual-cased` entrenado sobre un corpus propio de **5 081 canciones** etiquetadas mediante un esquema combinado (keywords emocionales + modelo multilingüe de sentimiento).

Forma parte del **Proyecto 3 del curso de Minería de Textos** del Colegio Universitario de Cartago (2026). Se integra como componente de clasificación dentro del chatbot musical **MúsicBot**, que combina este modelo con un pipeline RAG (FAISS + Flan-T5) para responder consultas sobre un corpus lírico.

---

## Etiquetas

El modelo predice una de 5 emociones:

| ID | Etiqueta   | Emoji |
|----|------------|-------|
| 0  | alegria    | 😊    |
| 1  | tristeza   | 😢    |
| 2  | amor       | ❤️    |
| 3  | rabia      | 😠    |
| 4  | nostalgia  | 🌅    |

---

## Uso rápido

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_ID = "nubiaebv/musicbot-emotion-classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

texto = "Esta noche te escribo llorando, no puedo dejar de pensar en ti"
inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    logits = model(**inputs).logits
    probs  = torch.softmax(logits, dim=-1)[0]

etiquetas = ["alegria", "tristeza", "amor", "rabia", "nostalgia"]
pred_id   = int(probs.argmax())
print(f"Emoción: {etiquetas[pred_id]} ({probs[pred_id]:.1%})")
```

### Con `pipeline`

```python
from transformers import pipeline

clf = pipeline("text-classification",
               model="nubiaebv/musicbot-emotion-classifier")
print(clf("Quiero bailar contigo toda la noche"))
# [{'label': 'alegria', 'score': 0.92}]
```

---

## Datos de entrenamiento

- **Origen:** corpus de letras de canciones en español e inglés, curado durante los Proyectos 1, 2 y 3 del curso. Almacenado en MongoDB Atlas.
- **Tamaño inicial:** 6 526 canciones con letra.
- **Tamaño tras etiquetado y balanceo:** **5 081 canciones**.
- **Géneros representados:** pop, rock, reggaetón, hip-hop, baladas, k-pop, latin, electronic, entre otros.

### Pipeline de etiquetado (semiautomático)

1. **Nivel 1 — Keywords emocionales:** listas de términos por clase sobre texto normalizado → 5 002 canciones etiquetadas (cobertura 76.6%).
2. **Nivel 2 — Modelo de sentimiento multilingüe:** las 1 464 canciones sin etiqueta se clasificaron con un modelo preentrenado → cobertura final **99.1%** (6 466 de 6 526).
3. **Nivel 3 — Filtrado:** se descartaron clases con menos de 50 muestras.
4. **Balanceo:** submuestreo de la clase mayoritaria (`amor`) para que el ratio desbalance/minoritaria no supere **3.0x**.

### Distribución antes del balanceo

| Clase     | Muestras | %     |
|-----------|---------:|------:|
| amor      | 2 813    | 43.5% |
| rabia     | 1 406    | 21.7% |
| nostalgia | 1 140    | 17.6% |
| alegria   | 631      | 9.8%  |
| tristeza  | 476      | 7.4%  |
| **Total** | **6 466** | **100%** |

Ratio de desbalance inicial: **5.9×** — superior al umbral de 3× definido por el equipo.

### Distribución después del balanceo (dataset final)

| Clase     | Muestras | Acción            |
|-----------|---------:|-------------------|
| amor      | 1 428    | submuestreado desde 2 813 |
| rabia     | 1 406    | sin cambio        |
| nostalgia | 1 140    | sin cambio        |
| alegria   | 631      | sin cambio        |
| tristeza  | 476      | sin cambio        |
| **Total** | **5 081** | **ratio 3.0×**   |

### Particiones (seed fijo = 42)

| Split      | Muestras | % del total |
|------------|---------:|------------:|
| Train      | 3 557    | 70%         |
| Validation | 762      | 15%         |
| Test       | 762      | 15%         |

---

## Procedimiento de entrenamiento

- **Modelo base:** `distilbert-base-multilingual-cased` (~135M parámetros)
- **Cabeza:** `AutoModelForSequenceClassification` con 5 labels
- **Trainer:** `WeightedTrainer` personalizado (pesos inversos a frecuencia de clase) + `EarlyStoppingCallback` sobre F1 macro de validación

### Hiperparámetros

| Parámetro       | Valor  |
|-----------------|--------|
| Épocas máximas  | 3 (`MAX_EPOCHS`) con early stopping |
| Batch size      | 16     |
| Learning rate   | 2e-5   |
| Max length      | 256    |
| Optimizador     | AdamW  |
| Seed            | 42     |

### Pesos de clase usados por el `WeightedTrainer`

```
alegria   : 1.6430
tristeza  : 2.0210
amor      : 0.7215
rabia     : 0.7289
nostalgia : 0.8783
```

Los pesos compensan el desbalance residual penalizando más los errores en clases minoritarias (especialmente `tristeza`).

---

## Evaluación

Sobre el conjunto de test (762 muestras, estratificado).

### Comparación baseline vs fine-tuned

| Métrica       | Baseline (mayoría de clase) | **Fine-tuned** | Ganancia |
|---------------|----------------------------:|---------------:|---------:|
| Accuracy      | 0.3250                      | **0.7572**     | **+0.4322** |
| F1 macro      | 0.0981                      | **0.7475**     | **+0.6494** |

El baseline predice siempre la clase más frecuente (`amor`). El fine-tuning aporta **+43 puntos de accuracy** y **+65 puntos de F1 macro**, confirmando que el modelo aprende patrones reales en las letras más allá de la clase mayoritaria.

### Métricas detalladas por clase (test)

| Clase     | Precision | Recall | F1-score | Support |
|-----------|----------:|-------:|---------:|--------:|
| alegria   | 0.77      | 0.72   | 0.74     | 102     |
| tristeza  | 0.64      | 0.77   | 0.70     | 57      |
| amor      | 0.82      | 0.77   | 0.79     | 218     |
| rabia     | 0.73      | 0.72   | 0.73     | 204     |
| nostalgia | 0.76      | 0.80   | 0.78     | 181     |
| **Macro avg**    | **0.74** | **0.76** | **0.75** | 762 |
| **Weighted avg** | **0.76** | **0.76** | **0.76** | 762 |
| **Accuracy**     |          |          | **0.76** | 762 |

### Matriz de confusión

Filas = etiqueta real, columnas = predicción.

|              | alegria | tristeza | amor | rabia | nostalgia |
|--------------|--------:|---------:|-----:|------:|----------:|
| **alegria**  | **73**  | 5        | 5    | 11    | 8         |
| **tristeza** | 1       | **44**   | 4    | 4     | 4         |
| **amor**     | 7       | 5        | **168** | 22 | 16        |
| **rabia**    | 10      | 9        | 19   | **147** | 19      |
| **nostalgia**| 4       | 6        | 9    | 17    | **145**   |

**Observaciones:**
- `nostalgia` es la clase con mejor recall (80%), probablemente por su vocabulario distintivo (recuerdos, tiempo pasado, pérdida).
- La mayor confusión ocurre entre `amor` y `rabia` (22 casos): ambas clases comparten léxico relacional intenso; letras de desamor o celos pueden caer en cualquiera.
- `alegria` se confunde frecuentemente con `rabia` (11 casos) — posiblemente canciones de fiesta con letras enérgicas y agresivas (reggaetón).

---

## Limitaciones y sesgos

- **Dominio restringido:** entrenado exclusivamente con letras de canciones. No generaliza bien a otros tipos de texto (noticias, reseñas, diálogos).
- **Etiquetado semiautomático:** las etiquetas no provienen de anotación humana completa. El pipeline keywords + modelo de sentimiento arrastra sus propios sesgos — en particular, las listas de keywords pueden sobrerrepresentar términos explícitos y perder canciones con emociones expresadas metafóricamente.
- **Desbalance residual:** pese al submuestreo a ratio 3×, `tristeza` (476 muestras) y `alegria` (631) siguen siendo clases minoritarias. Los class weights compensan parcialmente, pero su precisión y recall son menos fiables que las de `amor` o `nostalgia`.
- **Bilingüe no equilibrado:** el corpus contiene canciones en español e inglés sin paridad explícita. El rendimiento por idioma no se evaluó de forma separada.
- **Longitud limitada:** la entrada está truncada a 256 tokens; canciones largas pierden contexto.
- **Uso previsto:** apoyo a sistemas de recomendación musical y análisis de corpus lírico. **No apto** para decisiones sensibles en salud mental, diagnóstico emocional o cualquier contexto clínico.

---

## Integración en MúsicBot

Este clasificador se integra en un pipeline **RAG + Fine-Tuning** dentro de la aplicación MúsicBot:

1. El usuario escribe una consulta musical.
2. El clasificador infiere la emoción predominante en la pregunta.
3. La emoción se usa como filtro sobre el índice FAISS del corpus de canciones.
4. Un generador Flan-T5 construye la respuesta en español a partir de los chunks recuperados.
5. La interfaz Dash muestra la emoción detectada y las canciones citadas.

Código fuente completo: <https://github.com/nubiaebv/chatbot-musical-inteligente>

---

## Autores

- **Nubia Elena Brenes Valerín** — [@nubiaebv](https://github.com/nubiaebv)
- **Pablo Andrés Marín Castillo** — [@pmarin2592](https://github.com/pmarin2592)

### Contexto académico

- **Curso:** Minería de Textos
- **Institución:** Colegio Universitario de Cartago (CUC)
- **Profesor:** Osvaldo González Chaves
- **Proyecto:** Proyecto 3 — Chatbot Musical con RAG + Fine-Tuning
- **Año:** 2026

### Licencia

Apache 2.0 — libre uso con atribución.

### Cómo citar

```bibtex
@misc{musicbot_emotion_classifier_2026,
  title   = {MúsicBot — Clasificador de Emociones en Letras de Canciones},
  author  = {Brenes Valerín, Nubia Elena and Marín Castillo, Pablo Andrés},
  year    = {2026},
  note    = {Proyecto 3 — Minería de Textos, Colegio Universitario de Cartago},
  url     = {https://huggingface.co/nubiaebv/musicbot-emotion-classifier}
}
```
