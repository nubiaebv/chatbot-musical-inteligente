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
    GENERATOR_MODEL, FLAN_T5_MODEL, LOGS_DIR,UMBRAL_CONFIANZA_EMOCION
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
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            self._dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
            self._log.info(f"Cargando {FLAN_T5_MODEL} en {self._dispositivo.upper()}...")
            self._flan_tokenizer = AutoTokenizer.from_pretrained(FLAN_T5_MODEL)
            self._flan_model = AutoModelForSeq2SeqLM.from_pretrained(
                FLAN_T5_MODEL).to(self._dispositivo)
            self._log.info("Flan-T5 cargado correctamente.")
        except Exception as e:
            self._log.error(f"Error al cargar Flan-T5: {e}")
            raise RuntimeError(f"No se pudo cargar Flan-T5: {e}") from e

    # ── Detección de intención ───────────────────────────────────

    def _detectar_intencion(self, pregunta: str) -> str:
        """
        Detecta el tipo de pregunta para usar el prompt correcto.

        Returns:
            'recomendacion' → buscar canción y recomendar
            'informativa'   → pregunta sobre artistas, géneros, diferencias
            'seguimiento'   → referencia a turno anterior
            'fuera_dominio' → no relacionado con música
        """
        p = pregunta.lower().strip()

        fuera_dominio = [
            "vuelo", "avion", "madrid", "precio", "cuesta", "dinero",
            "mundial", "futbol", "deporte", "politica", "presidente",
            "clima", "tiempo", "receta", "cocina", "medicina", "doctor",
            "gano", "ganó", "campeon", "campeón", "copa", "liga",
            "pelicula", "serie", "netflix", "noticias", "historia",
            "matematica", "ciencia", "geografia"
        ]
        if any(w in p for w in fuera_dominio):
            return "fuera_dominio"

        seguimiento = [
            "esa", "ese", "otra", "similar", "parecida", "mismo",
            "artista", "quien es", "quién es", "primera", "segunda",
            "la que", "el que", "mencionaste", "dijiste", "anterior"
        ]
        if any(w in p for w in seguimiento) and self.historial:
            return "seguimiento"

        informativas = [
            "que artistas", "qué artistas", "cuales artistas",
            "diferencia entre", "que diferencia", "cual es la diferencia",
            "como es", "que generos", "qué géneros",
            "cuantas canciones", "que hay en", "cuéntame sobre"
        ]
        if any(w in p for w in informativas):
            return "informativa"

        return "recomendacion"

    # ── Generación con Flan-T5 ───────────────────────────────────

    def _generar_flan_t5(self, pregunta: str, chunks: list,
                         intencion: str = "recomendacion") -> str:
        try:
            import torch

            mejor = chunks[0] if chunks else None
            segundo = chunks[1] if len(chunks) >= 2 else None

            # Prompts específicos por intención — sin SYSTEM_PROMPT para evitar
            # que Flan-T5 lo copie en el output
            if intencion == "fuera_dominio":
                prompt = (
                    f"Question: {pregunta}\n"
                    f"This is not about music.\n"
                    f"Answer: 'Lo siento, solo puedo ayudarte con música."
                    f" ¿Te recomiendo una canción?'\n"
                    f"Repeat the Answer:"
                )

            elif intencion == "informativa" and chunks:
                artistas = list(dict.fromkeys([c["artista"] for c in chunks[:5]]))[:4]
                generos = list(dict.fromkeys([c["genero"] for c in chunks[:5]]))[:3]
                prompt = (
                    f"Artists: {', '.join(artistas)}. "
                    f"Genres: {', '.join(generos)}.\n"
                    f"Question about music database: {pregunta}\n"
                    f"Answer in Spanish mentioning these artists and genres:\n"
                    f"Answer:"
                )

            elif mejor:
                canciones = f'"{mejor["titulo"]}" by {mejor["artista"]} ({mejor["genero"]})'
                if segundo:
                    canciones += f', "{segundo["titulo"]}" by {segundo["artista"]}'
                prompt = (
                    f"MusicBot recommends songs in Spanish.\n"
                    f"Songs: {canciones}\n"
                    f"User: {pregunta}\n"
                    f"Answer starting with 'Te recomiendo':\n"
                    f"Answer:"
                )

            else:
                prompt = (
                    f"MusicBot. No songs found.\n"
                    f"User: {pregunta}\n"
                    f"Answer in Spanish that no songs were found:\n"
                    f"Answer:"
                )

            inputs = self._flan_tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=384
            ).to(self._dispositivo)

            with torch.no_grad():
                outputs = self._flan_model.generate(
                    **inputs,
                    max_new_tokens=120,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    repetition_penalty=2.5,
                )

            respuesta = self._flan_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()

            self._log.debug(f"Flan-T5 [{intencion}]: {respuesta[:100]}")

            # Detectar si el modelo copió el prompt o generó algo inútil
            palabras_invalidas = [
                "musicbot", "eres", "curador", "answer:", "repeat",
                "question:", "artists:", "songs:", "user:"
            ]
            respuesta_invalida = (
                    not respuesta or
                    len(respuesta.split()) < 4 or
                    pregunta.lower().strip() in respuesta.lower() or
                    any(p in respuesta.lower() for p in palabras_invalidas)
            )

            if respuesta_invalida:
                self._log.debug(f"Respuesta inválida detectada, usando fallback.")
                if intencion == "fuera_dominio":
                    respuesta = (
                        "Lo siento, esa información está fuera de mi dominio. "
                        "Solo puedo ayudarte con recomendaciones musicales. "
                        "¿Te recomiendo una canción?"
                    )
                elif intencion == "informativa" and chunks:
                    artistas = list(dict.fromkeys([c["artista"] for c in chunks[:4]]))
                    generos = list(dict.fromkeys([c["genero"] for c in chunks[:3]]))
                    if "artista" in pregunta.lower():
                        respuesta = (
                            f"En mi base de datos tengo canciones de artistas como: "
                            f"{', '.join(artistas)}, entre otros. "
                            f"¿Te gustaría que te recomiende una canción?"
                        )
                    elif "diferencia" in pregunta.lower():
                        respuesta = (
                            f"En mi corpus tengo canciones de géneros como "
                            f"{', '.join(generos)}. Las canciones tristes hablan de "
                            f"pérdida y desamor, mientras que las alegres hablan de "
                            f"celebración y amor positivo. ¿Quieres que te recomiende alguna?"
                        )
                    else:
                        respuesta = (
                            f"En mi base de datos tengo canciones de "
                            f"{', '.join(artistas)}, entre otros. ¿Te recomiendo alguna?"
                        )
                elif mejor:
                    respuesta = (
                        f'Te recomiendo "{mejor["titulo"]}" de {mejor["artista"]}, '
                        f'una canción de {mejor["genero"]} '
                        f'con emoción de {mejor["emocion"]}.'
                    )
                    if segundo:
                        respuesta += (
                            f' También puedes escuchar '
                            f'"{segundo["titulo"]}" de {segundo["artista"]}.'
                        )
                else:
                    respuesta = "No encontré canciones relevantes para tu pregunta."

            return respuesta

        except Exception as e:
            self._log.error(f"Error en Flan-T5: {e}")
            raise

    # Generación de respuesta

    def _generar(self, pregunta: str, chunks: list, intencion: str = "recomendacion") -> str:
        """Genera la respuesta con el backend configurado."""
        try:
            if self._flan_model is not None:
                return self._generar_flan_t5(pregunta, chunks, intencion)

            return "No pude generar una respuesta."

        except Exception as e:
            self._log.error(f"Error al generar respuesta: {e}")
            return "Lo siento, ocurrió un error al generar la respuesta."

    def _detectar_intencion(self, pregunta: str) -> str:
        """
        Detecta el tipo de pregunta antes de generar respuesta.

        Returns:
            'recomendacion' → buscar canción y recomendar
            'informativa'   → pregunta sobre artistas, géneros, diferencias
            'seguimiento'   → referencia a turno anterior
            'fuera_dominio' → no relacionado con música
        """
        p = pregunta.lower().strip()

        # Palabras que indican fuera de dominio
        fuera_dominio = [
            "vuelo", "avion", "madrid", "precio", "cuesta", "dinero",
            "mundial", "futbol", "deporte", "politica", "presidente",
            "clima", "tiempo", "receta", "cocina", "medicina", "doctor",
            "gano", "ganó", "campeon", "campeón", "copa", "liga",
            "pelicula", "serie", "netflix", "noticias"
        ]
        if any(w in p for w in fuera_dominio):
            return "fuera_dominio"

        # Palabras de seguimiento
        seguimiento = [
            "esa", "ese", "otra", "similar", "parecida", "mismo",
            "artista", "quien es", "quién es", "primera", "segunda",
            "la que", "el que", "mencionaste", "dijiste"
        ]
        if any(w in p for w in seguimiento) and self.historial:
            return "seguimiento"

        # Preguntas informativas sobre música
        informativas = [
            "que artistas", "qué artistas", "cuales artistas",
            "diferencia entre", "que diferencia", "cual es la diferencia",
            "como es", "que generos", "qué géneros",
            "cuantas canciones", "que hay en"
        ]
        if any(w in p for w in informativas):
            return "informativa"

        return "recomendacion"

    # Pipeline principal

    def responder(self, pregunta_usuario: str) -> dict:
        self._log.info(f"Pregunta recibida: '{pregunta_usuario[:60]}'")

        try:
            # Paso 1: Detectar intención PRIMERO
            intencion = self._detectar_intencion(pregunta_usuario)
            self._log.debug(f"Intención: {intencion}")

            # Paso 2: Clasificar emoción
            emocion_detectada = self._finetunig.predecir_emocion(pregunta_usuario)
            filtro_emoc = None
            if emocion_detectada and emocion_detectada["score"] >= UMBRAL_CONFIANZA_EMOCION:
                filtro_emoc = emocion_detectada["emocion"]
                self._log.debug(f"Filtro activo: {filtro_emoc} ({emocion_detectada['score']:.0%})")

            # Paso 3: Búsqueda semántica
            if intencion == "seguimiento" and self.historial:
                # Enriquecer la pregunta con contexto del turno anterior
                ultimo_turno = self.historial[-1]
                ultimo_bot = ultimo_turno["bot"]

                # Extraer artista y título del último turno si los hay
                contexto_prev = ""
                if ultimo_turno.get("chunks"):
                    mejor_prev = ultimo_turno["chunks"][0]
                    contexto_prev = (
                        f'{mejor_prev["titulo"]} {mejor_prev["artista"]} '
                        f'{mejor_prev["genero"]}'
                    )

                # Combinar pregunta actual con contexto previo para mejor búsqueda
                pregunta_enriquecida = f'{pregunta_usuario} {contexto_prev}'.strip()
                self._log.debug(f"Búsqueda enriquecida: {pregunta_enriquecida[:80]}")

                chunks = buscar_chunks(pregunta=pregunta_enriquecida, top_k=TOP_K)
            else:
                chunks = buscar_chunks(
                    pregunta=pregunta_usuario, top_k=TOP_K,
                    filtro_emocion=filtro_emoc
                )
                if len(chunks) < 2:
                    chunks = buscar_chunks(pregunta=pregunta_usuario, top_k=TOP_K)

            # Filtrar idiomas no latinos
            chunks_latinos = [c for c in chunks if c.get('idioma', '') in ['en', 'es', 'pt', 'fr', 'it']]
            if len(chunks_latinos) >= 2:
                chunks = chunks_latinos

            self._log.debug(f"Chunks recuperados: {len(chunks)}")

            # Paso 4: Generar respuesta pasando la intención
            respuesta = self._generar(pregunta_usuario, chunks, intencion)
            self._log.info(f"Respuesta generada ({len(respuesta)} chars)")

            # Paso 5: Actualizar memoria
            self.historial.append({
                "usuario": pregunta_usuario,
                "bot": respuesta,
                "chunks": chunks,
            })

            return {
                "respuesta": respuesta,
                "chunks": chunks,
                "emocion": emocion_detectada,
                "historial": self.historial,
            }

        except RuntimeError as e:
            self._log.error(f"Error de sistema: {e}")
            return {
                "respuesta": "El sistema no está listo. Verifica que el RAG esté inicializado.",
                "chunks": [], "emocion": None, "historial": self.historial,
            }
        except Exception as e:
            self._log.error(f"Error inesperado: {e}")
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
