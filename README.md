# chatbot-musical-inteligente
Agente conversacional bilingüe que utiliza RAG y Fine-Tuning para responder consultas sobre un corpus de canciones en español e inglés.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?logo=meta&logoColor=white)
![Plotly Dash](https://img.shields.io/badge/Plotly_Dash-2.x-3F4F75?logo=plotly&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-3.x-09A3D5?logo=spacy&logoColor=white)
![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffD21E)
---

##  Descripción

**chatbot-musical-inteligente** es el producto final de una serie de tres proyectos construidos sobre el mismo corpus de letras de canciones. Integra un pipeline completo de **Retrieval-Augmented Generation (RAG)**, un **clasificador de emociones obtenido por Fine-Tuning** y una **interfaz web con Plotly Dash** para que cualquier usuario pueda interactuar con el sistema en lenguaje natural.

| Proyecto | Enfoque | Producido |
|---|---|---|
| Proyecto 1 | POS Tagging con NLTK y spaCy | Análisis morfosintáctico de 5,000–10,000 canciones |
| Proyecto 2 | BoW, TF-IDF, Word2Vec y BETO | Representaciones vectoriales y dashboard Plotly Dash |
| **Proyecto 3** | **Chatbot con RAG + Fine-Tuning + Plotly Dash** | **Agente conversacional con interfaz web funcional** |

---

##  Línea de investigación

Este chatbot sigue la **Línea B — Curador Emocional**: analiza las emociones presentes en las letras del corpus y sugiere canciones según el estado de ánimo del usuario.

**Personalidad del chatbot:** _"Soy MúsicBot, un curador emocional de música. Me especializo en encontrar canciones que conecten con cómo te sientes. Respondo basándome exclusivamente en letras de canciones de mi base de datos. Si no encuentro información relevante, te lo digo."_

**Tarea de clasificación (Fine-Tuning):** clasificador de polaridad / emoción básica (alegría, tristeza, enojo, miedo, sorpresa) entrenado sobre las letras del corpus.

---

##  Arquitectura del sistema

```
                         ┌─────────────────────────┐
Usuario ──► Pregunta ──► │    Interfaz Plotly Dash  │
                         └──────────┬──────────────┘
                                    │
                         ┌──────────▼──────────────┐
                         │    Chatbot Engine        │
                         │  (memoria N turnos)      │
                         └──────────┬──────────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              │                     │                      │
   ┌──────────▼──────────┐ ┌────────▼────────┐ ┌──────────▼──────────┐
   │  Clasificador de     │ │  Pipeline RAG   │ │   Generador LLM     │
   │  Emociones (FT)      │ │  (FAISS)        │ │  (flan-t5-base)     │
   └──────────────────────┘ └────────┬────────┘ └──────────────────────┘
                                     │
                         ┌───────────▼──────────────┐
                         │   Corpus + Embeddings     │
                         │   (CSV / embeddings_cache)│
                         └───────────────────────────┘
```

### Pipeline RAG detallado

```
Corpus de canciones (CSV / MongoDB)
        │
        ▼
  Chunking (por estrofa vs. por canción — se comparan ambas estrategias)
        │
        ▼
  Embeddings: paraphrase-multilingual-MiniLM-L12-v2
        │
        ▼
  Índice FAISS (IndexFlatIP + normalización L2)
        │
Pregunta ──► Embedding ──► Top-K chunks ──► Prompt + contexto ──► LLM ──► Respuesta
```

---

##  Estructura del repositorio

```
chatbot-musical-inteligente/
│
├── app/
│   ├── main.py              # Punto de entrada — Plotly Dash
│   └── config.py                   # Variables de entorno y rutas
│
├── data/
│   ├── corpus_canciones.csv        # Corpus base (Proyectos 1 y 2)
│   └── embeddings_cache/           # Embeddings pre-calculados (.npy / .pkl)
│
├── models/
│   └── clasificador_emocion/       # Pesos del modelo fine-tuneado
│
├── notebooks/
│   ├── 01_exploracion_corpus.ipynb         # Estadísticas del dataset
│   ├── 02_rag_pipeline.ipynb               # Chunking + embeddings + FAISS + generador
│   ├── 03_finetuning_clasificador.ipynb    # Entrenamiento del clasificador de emociones
│   └── 04_chatbot_completo.ipynb           # Chatbot integrado (pruebas en notebook)
│
├── resultados/
│   ├── metricas.json               # Accuracy, F1, matriz de confusión
│   └── conversaciones_prueba/      # 10+ conversaciones con y sin RAG
│
├── src/
│   ├── rag_utils.py                # Chunking, embeddings, FAISS, búsqueda semántica
│   ├── finetuning_utils.py         # Dataset, Trainer, evaluación del clasificador
│   └── chatbot_engine.py           # Clase del chatbot (memoria, prompt, generación)
│
├── .env.example                    # Plantilla de variables de entorno
├── .gitignore
├── requirements.txt
├── README.md
└── USO_DE_IA.md                    # Registro transparente del uso de IA
```

---

##  Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/nubiaebv/chatbot-musical-inteligente.git
cd chatbot-musical-inteligente
git checkout integracion-backend-ui
```

### 2. Crear y activar entorno virtual

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar modelos de NLP

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

```bash
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

### 5. Configurar variables de entorno

```bash
cp .env.example .env
```

Edita `.env` con tus credenciales:

```env
# MongoDB Atlas (opcional — el corpus también puede cargarse desde CSV)
MONGO_URI=mongodb+srv://<usuario>:<password>@<cluster>.mongodb.net/
MONGO_DB_NAME=chatbot_musical

# API de LLM (opcional — el sistema funciona localmente sin esto)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

> ⚠️ El chatbot **funciona sin API key**, usando `google/flan-t5-xl` como generador local. La API externa es una mejora opcional.

---

##  Uso

### Lanzar el chatbot (un solo comando)

```bash
python app/chatbot_app.py
```

Abre tu navegador en `http://127.0.0.1:8050` para interactuar con el chatbot.

### Reproducir el pipeline completo (notebooks)

```bash
jupyter notebook
```

Ejecuta los notebooks en orden numérico dentro de `notebooks/`:

| Notebook | Descripción |
|---|---|
| `01_exploracion_corpus.ipynb` | Estadísticas del dataset y distribución por género/emoción |
| `02_rag_pipeline.ipynb` | Chunking, embeddings, FAISS y generador |
| `03_finetuning_clasificador.ipynb` | Entrenamiento y evaluación del clasificador de emociones |
| `04_chatbot_completo.ipynb` | Integración completa: RAG + clasificador + memoria |

---

##  Análisis implementados

### Pipeline RAG — `02_rag_pipeline.ipynb`

- **Estrategias de chunking comparadas:** por canción completa vs. por estrofa. Se reporta cuál produce recuperaciones más relevantes.
- **Embeddings cacheados** en disco (`data/embeddings_cache/`) para evitar regenerarlos en cada ejecución.
- **Búsqueda semántica** con FAISS (`IndexFlatIP` + `faiss.normalize_L2` para similitud coseno).
- **Generador:** `google/flan-t5-base` por defecto (modo local). Conmutable a OpenAI / Anthropic / Gemini con una variable de entorno.

### Clasificador de emociones — `03_finetuning_clasificador.ipynb`

- **Modelo base:** `distilbert-base-multilingual-cased` (entrenable en CPU/Colab gratis).
- **Repositorio del Modelo:** El modelo fine-tuneado está alojado en Hugging Face: [nubiaebv/musicbot-emotion-classifier](https://huggingface.co/nubiaebv/musicbot-emotion-classifier).
- **Dataset etiquetado:** mínimo 1,500 canciones con etiqueta de emoción.
- **División:** 70% train / 15% validation / 15% test con seed fijo para reproducibilidad.
- **Entrenamiento:** máximo 3 épocas.
- **Métricas reportadas:** accuracy, F1 macro y matriz de confusión sobre el conjunto de test.
- **Comparación obligatoria** con el modelo base sin fine-tuning (zero-shot baseline) para cuantificar la ganancia.

### Chatbot conversacional — `04_chatbot_completo.ipynb`

- **Memoria de diálogo:** historial de los últimos 5 turnos enviado como contexto al generador en cada turno.
- **Flujo por turno:**
  1. Embedding de la pregunta del usuario.
  2. Clasificación de emoción/intención con el modelo fine-tuneado.
  3. Recuperación de los top-K chunks relevantes desde FAISS.
  4. Filtrado de chunks según la emoción detectada.
  5. Ensamblado del prompt (sistema + historial + contexto + pregunta).
  6. Generación de respuesta con el LLM.
- **Comparación con/sin RAG:** 10+ conversaciones documentadas en `resultados/metricas.json`.

---

##  Autores

**Nubia Elena Brenes Valerín** · [@nubiaebv](https://github.com/nubiaebv)  
**Pablo Andrés Marín Castillo** · [@pmarin2592](https://github.com/pmarin2592)

---

##  Contexto académico

- **Curso:** Minería de Textos — Colegio Universitario de Cartago (CUC)
- **Profesor:** Osvaldo González Chaves
- **Proyecto:** 3 de 3 — continuación directa de los Proyectos 1 (POS Tagging) y 2 (Análisis Semántico)

---
*Última actualización: Jueves 23 de abril del 2026*