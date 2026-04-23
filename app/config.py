# app/config.py — Configuración centralizada
import os
from pathlib import Path
import logging

# Configuración de Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── RUTAS DEL SISTEMA ──────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent.parent   # Raíz del proyecto
DATA_DIR         = BASE_DIR / "data"
MODELS_DIR       = BASE_DIR / "models"
ASSETS_DIR       = BASE_DIR / "app" / "assets"

# Asegurar existencia de directorios críticos
for folder in [DATA_DIR, MODELS_DIR, ASSETS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ─── RUTAS DE DATOS Y CACHÉ ──────────────────────────────────────────────────
DATA_PATH        = os.getenv("DATA_PATH",   str(DATA_DIR / "corpus_canciones.csv"))
EMBED_CACHE_PATH = os.getenv("EMBED_CACHE", str(DATA_DIR / "embeddings_cache" / "faiss_index.pkl"))
MODEL_DIR        = os.getenv("MODEL_DIR",   str(MODELS_DIR / "clasificador"))

# ─── CONFIGURACIÓN DE IA LOCAL ──────────────────────────────────────────────
# Usamos modelos que garantizan ejecución local estable
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEN_MODEL_NAME   = "google/flan-t5-base"  # Modelo sugerido por su eficiencia local

# FORZADO LOCAL:
LLM_PROVIDER     = "local"

# ─── PARÁMETROS DE RAG Y CHAT ───────────────────────────────────────────────
TOP_K          = int(os.getenv("TOP_K",         "5"))  # Cantidad de chunks a recuperar
HISTORY_TURNS  = int(os.getenv("HISTORY_TURNS", "5"))  # Memoria de la conversación
CHUNKING_STRAT = "paragraph"                           # Estrategia de fragmentación

# ─── CONFIGURACIÓN DE APP (DASH) ────────────────────────────────────────────
APP_PORT  = 8050
APP_DEBUG = False  # Desactivado para mayor estabilidad en la presentación

logger.info(f"✅ Configuración cargada. Proveedor de LLM: {LLM_PROVIDER}")