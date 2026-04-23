# app/config.py — Configuración centralizada
# Importe de librerias
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Configuración de Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
Configuración centralizada del proyecto
"""

# Cargar variables de entorno desde .env
load_dotenv()

# Rutas del proyecto
BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / "data"
CACHE_DIR       = DATA_DIR / "embeddings_cache"
MODELS_DIR      = BASE_DIR / "models"
RESULTADOS_DIR  = BASE_DIR / "resultados"
LOGS_DIR = BASE_DIR / "logs"
ASSETS_DIR       = BASE_DIR / "app" / "assets"


# Crear directorios si no existen
for d in [CACHE_DIR, MODELS_DIR, RESULTADOS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── RUTAS DE DATOS Y CACHÉ ──────────────────────────────────────────────────
DATA_PATH        = os.getenv("DATA_PATH",   str(DATA_DIR / "corpus_canciones.csv"))
EMBED_CACHE_PATH = os.getenv("EMBED_CACHE", str(DATA_DIR / "embeddings_cache" / "faiss_index.pkl"))
MODEL_DIR        = os.getenv("MODEL_DIR",   str(MODELS_DIR / "clasificador"))

# Base de Datos
MONGO_URI       = os.getenv("MONGO_URI", "")
DB_NAME         = os.getenv("DB_NAME", "analisisMusical")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "analisisMusical")

# Modelos
EMBEDDING_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GENERATOR_MODEL    = os.getenv("GENERATOR_MODEL", "local")   # "local" | "claude" | "openai" | "gemini"
FLAN_T5_MODEL      = "google/flan-t5-large"
FINETUNE_BASE      = "distilbert-base-multilingual-cased"
FINETUNE_MODEL_DIR = str(MODELS_DIR / "clasificador_emocion")
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEN_MODEL_NAME   = "google/flan-t5-large"  # Modelo sugerido por su eficiencia local

# FORZADO LOCAL:
LLM_PROVIDER     = "local"

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
MAX_EPOCHS       = 7
BATCH_SIZE       = 16
LEARNING_RATE    = 2e-5
MAX_LENGTH       = 256
MIN_SAMPLES_PER_CLASS = 50   # clases con menos muestras se descartan

# Etiquetas de emoción
EMOCIONES = ["alegria", "tristeza", "amor", "rabia", "nostalgia"]
EMOCION2ID = {e: i for i, e in enumerate(EMOCIONES)}
ID2EMOCION = {i: e for i, e in enumerate(EMOCIONES)}
UMBRAL_CONFIANZA_EMOCION = 0.70
# Personalidad del chatbot
SYSTEM_PROMPT = """Eres MúsicBot, un curador emocional de música. \
Tu especialidad es analizar emociones en letras de canciones y recomendar \
música según el estado de ánimo del usuario. \
Respondes basándote exclusivamente en las canciones de tu base de datos. \
Si no encuentras información relevante en el corpus, lo dices claramente \
en lugar de inventar. Eres amable, empático y apasionado por la música. \
IMPORTANTE: Responde siempre en español, sin importar el idioma de las letras \
recuperadas o el idioma en que te hagan la pregunta."""

# ─── PARÁMETROS DE RAG Y CHAT ───────────────────────────────────────────────
TOP_K          = int(os.getenv("TOP_K",         "5"))  # Cantidad de chunks a recuperar
HISTORY_TURNS  = int(os.getenv("HISTORY_TURNS", "5"))  # Memoria de la conversación
CHUNKING_STRAT = "paragraph"                           # Estrategia de fragmentación

# ─── CONFIGURACIÓN DE APP (DASH) ────────────────────────────────────────────
APP_PORT  = 8050
APP_DEBUG = False  # Desactivado para mayor estabilidad en la presentación

logger.info(f"✅ Configuración cargada. Proveedor de LLM: {LLM_PROVIDER}")
