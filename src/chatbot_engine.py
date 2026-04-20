"""
Motor del Chatbot Musical MúsicBot
Integra clasificador fine-tuneado + RAG +  memoria conversacional.
"""

import os
import sys
import logging
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.config import (
    SYSTEM_PROMPT, MAX_HISTORIA, TOP_K,
    GENERATOR_MODEL, FLAN_T5_MODEL, LOGS_DIR
)
from src.rag_utils import rag_utils
from src.finetuning_utils import finetuning_utils
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Control para logs

def _configurar_logger(nombre: str) -> logging.Logger:
    logger = logging.getLogger(nombre)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    consola = logging.StreamHandler(sys.stdout)
    consola.setLevel(logging.INFO)
    consola.setFormatter(fmt)

    archivo = logging.FileHandler(Path(LOGS_DIR) / "chatbot_engine.log", encoding="utf-8")
    archivo.setLevel(logging.DEBUG)
    archivo.setFormatter(fmt)

    logger.addHandler(consola)
    logger.addHandler(archivo)
    return logger

class chatbot_engine:
    """
    Motor conversacional de MúsicBot.

    Pipeline por turno:
      1. Predecir emoción con el clasificador fine-tuneado
      2. Buscar chunks en FAISS filtrando por esa emoción
      3. Fallback sin filtro si hay pocos resultados
      4. Construir prompt con sistema + historial + contexto
      5. Generar respuesta con el backend configurado
      6. Actualizar memoria conversacional
    """

    def __init__(self):
        self._log      = _configurar_logger("chatbot_engine")
        self.historial = []
        self.generador = None
        self._flan_tokenizer = None
        self._flan_model     = None
        self._inicializar_generador()
        self._rag = rag_utils()
        self._finetunig = finetuning_utils()

    # Inicialización del generador

    def _inicializar_generador(self):
        """Carga el backend de generación según la configuración."""
        try:
            if GENERATOR_MODEL == "local":
                self._cargar_flan_t5()
            else:
                self._log.warning("Sin API key válida. Usando Flan-T5 local como fallback.")
                self._cargar_flan_t5()
        except Exception as e:
            self._log.error(f"Error al inicializar generador: {e}")
            raise RuntimeError(f"No se pudo inicializar el generador: {e}") from e

    def _cargar_flan_t5(self):
        """Carga Flan-T5 usando AutoTokenizer y AutoModelForSeq2SeqLM."""
        try:

            self._dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
            self._log.info(f"Cargando {FLAN_T5_MODEL} en {self._dispositivo.upper()}...")

            self._flan_tokenizer = AutoTokenizer.from_pretrained(FLAN_T5_MODEL)
            self._flan_model     = AutoModelForSeq2SeqLM.from_pretrained(
                FLAN_T5_MODEL).to(self._dispositivo)

            self._log.info("Flan-T5 cargado correctamente.")

        except ImportError as e:
            self._log.error(f"Librería faltante para Flan-T5: {e}")
            raise
        except Exception as e:
            self._log.error(f"Error al cargar Flan-T5: {e}")
            raise RuntimeError(f"No se pudo cargar Flan-T5: {e}") from e

    # Construcción del prompt

    def _construir_prompt(self, pregunta: str, chunks_recuperados: list) -> str:
        """Ensambla el prompt: sistema + historial + contexto RAG + pregunta."""
        try:
            partes_ctx = [
                f"[{r['titulo']} - {r['artista']} | {r['genero']} | "
                f"{r['anio']} | Emocion: {r['emocion']}]\n{r['texto']}"
                for r in chunks_recuperados
            ]
            contexto = "\n\n".join(partes_ctx)

            hist_str = ""
            for turno in self.historial[-MAX_HISTORIA:]:
                hist_str += f"Usuario: {turno['usuario']}\nMúsicBot: {turno['bot']}\n"

            return (
                f"{SYSTEM_PROMPT}\n\n"
                f"Fragmentos de canciones relevantes:\n{contexto}\n\n"
                f"{hist_str}"
                f"Usuario: {pregunta}\nMúsicBot:"
            )
        except Exception as e:
            self._log.error(f"Error al construir prompt: {e}")
            return f"{SYSTEM_PROMPT}\n\nUsuario: {pregunta}\nMúsicBot:"

    # Generación de respuesta

    def _generar(self, prompt: str) -> str:
        """Genera la respuesta con el backend configurado."""
        try:
            if self._flan_model is not None:
                return self._generar_flan_t5(prompt)

            return "No pude generar una respuesta."

        except Exception as e:
            self._log.error(f"Error al generar respuesta: {e}")
            return "Lo siento, ocurrió un error al generar la respuesta."

    def _generar_flan_t5(self, prompt: str) -> str:
        try:
            inputs = self._flan_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self._dispositivo)
            with torch.no_grad():
                outputs = self._flan_model.generate(**inputs, max_new_tokens=350)
            return self._flan_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            self._log.error(f"Error en generación Flan-T5: {e}")
            raise


    # Pipeline principal

    def responder(self, pregunta_usuario: str) -> dict:
        """
        Procesa la pregunta y retorna la respuesta del chatbot.

        Returns:
            dict con: respuesta, chunks, emocion detectada, historial.
        """
        self._log.info(f"Pregunta recibida: '{pregunta_usuario[:60]}'")

        try:
            # Paso 1: Clasificar emoción con el modelo fine-tuneado
            emocion_detectada = self._finetunig.predecir_emocion(pregunta_usuario)
            filtro_emoc = emocion_detectada["emocion"] if emocion_detectada else None
            self._log.debug(f"Emoción detectada: {emocion_detectada}")

            # Paso 2: Búsqueda semántica con filtro de emoción
            chunks = self._rag.buscar(
                pregunta=pregunta_usuario, top_k=TOP_K, filtro_emocion=filtro_emoc
            )

            # Paso 3: Fallback sin filtro si hay pocos resultados
            if len(chunks) < 2:
                self._log.debug("Pocos resultados con filtro. Buscando sin filtro...")
                chunks = self._rag.buscar(pregunta=pregunta_usuario, top_k=TOP_K)

            self._log.debug(f"Chunks recuperados: {len(chunks)}")

            # Paso 4: Construir prompt y generar respuesta
            prompt    = self._construir_prompt(pregunta_usuario, chunks)
            respuesta = self._generar(prompt)
            self._log.info(f"Respuesta generada ({len(respuesta)} chars)")

            # Paso 5: Actualizar memoria conversacional
            self.historial.append({
                "usuario": pregunta_usuario,
                "bot":     respuesta,
                "chunks":  chunks,
            })

            return {
                "respuesta": respuesta,
                "chunks":    chunks,
                "emocion":   emocion_detectada,
                "historial": self.historial,
            }

        except RuntimeError as e:
            self._log.error(f"Error de sistema al responder: {e}")
            return {
                "respuesta": "El sistema no está listo. Verifica que el RAG esté inicializado.",
                "chunks": [], "emocion": None, "historial": self.historial,
            }
        except Exception as e:
            self._log.error(f"Error inesperado al responder: {e}")
            return {
                "respuesta": "Ocurrió un error inesperado. Intenta de nuevo.",
                "chunks": [], "emocion": None, "historial": self.historial,
            }

    # Gestión del historial

    def limpiar_historial(self):
        """Reinicia la memoria conversacional."""
        self.historial = []
        self._log.info("Historial conversacional limpiado.")

    def get_historial_display(self) -> list:
        """Retorna el historial en formato para la interfaz Dash."""
        mensajes = []
        for turno in self.historial:
            mensajes.append({"role": "user",      "content": turno["usuario"]})
            mensajes.append({"role": "assistant",  "content": turno["bot"]})
        return mensajes
