"""
src/logic.py — Adaptador entre la interfaz visual y el ChatbotEngine.
Conecta main.py con el backend real (MongoDB + RAG + Fine-Tuning).
"""
from __future__ import annotations

import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Rutas
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS (compatibles con interface.py)
# ══════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """Fragmento recuperado por el RAG — compatible con render_chunk_card()."""
    text:   str
    song:   str
    artist: str
    genre:  str
    score:  float
    emotion: str = ""


class RAGResult:
    """Resultado del chatbot — compatible con main.py."""
    def __init__(self, answer: str, chunks: List[Chunk],
                 classifier_label: str = None,
                 classifier_conf:  float = None):
        self.answer           = answer
        self.chunks           = chunks
        self.classifier_label = classifier_label
        self.classifier_conf  = classifier_conf


# ══════════════════════════════════════════════════════════════════
# MOTOR PRINCIPAL — wrapper sobre ChatbotEngine
# ══════════════════════════════════════════════════════════════════

class ChatEngine:
    """
    Adaptador que conecta la interfaz visual (main.py / interface.py)
    con el backend real del proyecto (MongoDB + RAG + Fine-Tuning).

    main.py llama a:
      engine.initialize()    → carga modelos y RAG
      engine.chat(text)      → procesa pregunta y retorna RAGResult
      engine.clear_history() → limpia historial
      engine._initialized    → booleano de estado
    """

    def __init__(self):
        self._initialized  = False
        self._bot          = None
        self._db           = None
        self._finetuning   = None
        self._rag          = None
        logger.info("ChatEngine creado. Llama a initialize() para cargar modelos.")

    def initialize(self):
        """
        Carga todos los componentes del sistema:
        1. Conexión a MongoDB Atlas
        2. Corpus etiquetado con emociones
        3. Índice FAISS
        4. ChatbotEngine con Flan-T5
        """
        try:
            logger.info("Inicializando sistema MúsicBot...")

            from src.mongo_utils      import mongo_utils
            from src.finetuning_utils import finetuning_utils
            from src.rag_utils        import rag_utils
            from src.chatbot_engine   import chatbot_engine

            # Conexión MongoDB
            self._db = mongo_utils()
            if not self._db.verificar_conexion():
                raise RuntimeError("No se pudo conectar a MongoDB Atlas.")

            # Cargar canciones
            canciones_raw = self._db.cargar_canciones()
            logger.info(f"Canciones cargadas: {len(canciones_raw)}")

            # Corpus etiquetado (desde caché si existe)
            self._finetuning  = finetuning_utils()
            corpus_etiquetado = self._finetuning.etiquetar_corpus_con_modelo(canciones_raw)
            logger.info(f"Corpus etiquetado: {len(corpus_etiquetado)} canciones")

            # Inicializar RAG
            self._rag = rag_utils()
            self._rag.inicializar(corpus_etiquetado)
            logger.info("Índice FAISS listo.")

            # Chatbot Engine
            self._bot = chatbot_engine()
            logger.info("Flan-T5 cargado.")

            self._initialized = True
            logger.info("Sistema MúsicBot listo.")
            print(">>> SISTEMA LISTO <<<")  # ← agregar esto

        except Exception as e:
            logger.error(f"Error al inicializar: {e}")
            print(f">>> ERROR: {e}")  # ← agregar esto
            self._initialized = False
            raise

    def chat(self, user_msg: str, use_rag: bool = True) -> RAGResult:
        """
        Procesa la pregunta del usuario y retorna un RAGResult
        compatible con la interfaz visual.
        """
        if not self._initialized or self._bot is None:
            return RAGResult(
                answer="⏳ El sistema todavía está cargando. Espera un momento.",
                chunks=[],
                classifier_label=None,
                classifier_conf=None
            )

        try:
            # Llamar al ChatbotEngine
            resultado = self._bot.responder(user_msg)

            # Convertir chunks al formato de interface.py
            chunks_visual = []
            for c in resultado.get("chunks", []):
                chunks_visual.append(Chunk(
                    text    = c.get("texto",   "")[:200],
                    song    = c.get("titulo",  "Desconocido"),
                    artist  = c.get("artista", "Desconocido"),
                    genre   = c.get("genero",  "Desconocido"),
                    score   = c.get("score",   0.0),
                    emotion = c.get("emocion", ""),
                ))

            # Datos del clasificador
            emocion = resultado.get("emocion")
            label   = emocion["emocion"] if emocion else None
            conf    = emocion["score"]   if emocion else None

            return RAGResult(
                answer           = resultado["respuesta"],
                chunks           = chunks_visual,
                classifier_label = label,
                classifier_conf  = conf,
            )

        except Exception as e:
            logger.error(f"Error en chat: {e}")
            return RAGResult(
                answer="Lo siento, ocurrió un error. Intenta de nuevo.",
                chunks=[],
                classifier_label=None,
                classifier_conf=None
            )

    def clear_history(self):
        """Limpia el historial conversacional."""
        if self._bot:
            self._bot.limpiar_historial()
            logger.info("Historial limpiado.")


# Instancia global — importada por main.py
engine = ChatEngine()
