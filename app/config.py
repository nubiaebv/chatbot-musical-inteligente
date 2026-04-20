"""
Configuración centralizada del proyecto
"""

# Importe de librerias
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Rutas del proyecto
BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / "data"
CACHE_DIR       = DATA_DIR / "embeddings_cache"
MODELS_DIR      = BASE_DIR / "models"
RESULTADOS_DIR  = BASE_DIR / "resultados"
LOGS_DIR = BASE_DIR / "logs"

# Crear directorios si no existen
for d in [CACHE_DIR, MODELS_DIR, RESULTADOS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Base de Datos
MONGO_URI       = os.getenv("MONGO_URI", "")
DB_NAME         = os.getenv("DB_NAME", "analisisMusical")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "analisisMusical")

# Modelos
EMBEDDING_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GENERATOR_MODEL    = os.getenv("GENERATOR_MODEL", "local")   # "local" | "claude" | "openai" | "gemini"
FLAN_T5_MODEL      = "google/flan-t5-base"
FINETUNE_BASE      = "distilbert-base-multilingual-cased"
FINETUNE_MODEL_DIR = str(MODELS_DIR / "clasificador_emocion")

# Archivos de caché
CACHE_EMBEDDINGS = str(CACHE_DIR / "embeddings_parrafos.npy")
CACHE_CHUNKS     = str(CACHE_DIR / "chunks_parrafos.pkl")
CACHE_FAISS      = str(CACHE_DIR / "indice_faiss.index")

# Parámetros RAG
TOP_K            = 5
CHUNK_MIN_LEN    = 30
MAX_HISTORIA     = 5     # turnos de memoria conversacional

# Parámetros Fine-Tuning
RANDOM_SEED      = int(os.getenv("RANDOM_SEED", 42))
TRAIN_SPLIT      = 0.70
VAL_SPLIT        = 0.15
TEST_SPLIT       = 0.15
MAX_EPOCHS       = 8
BATCH_SIZE       = 16
LEARNING_RATE    = 2e-5
MAX_LENGTH       = 256
MIN_SAMPLES_PER_CLASS = 100   # clases con menos muestras se descartan

# Etiquetas de emoción
EMOCIONES = ["alegria", "tristeza", "amor", "rabia", "nostalgia"]
EMOCION2ID = {e: i for i, e in enumerate(EMOCIONES)}
ID2EMOCION = {i: e for i, e in enumerate(EMOCIONES)}

# Personalidad del chatbot
SYSTEM_PROMPT = """Eres MúsicBot, un curador emocional de música. \
Tu especialidad es analizar emociones en letras de canciones y recomendar \
música según el estado de ánimo del usuario. \
Respondes basándote exclusivamente en las canciones de tu base de datos. \
Si no encuentras información relevante en el corpus, lo dices claramente \
en lugar de inventar. Eres amable, empático y apasionado por la música. \
IMPORTANTE: Responde siempre en español, sin importar el idioma de las letras \
recuperadas o el idioma en que te hagan la pregunta."""