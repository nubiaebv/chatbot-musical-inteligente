"""
Pipeline RAG como clase.
El corpus que llega ya tiene campo 'emocion' del clasificador fine-tuneado.
"""

import re
import os
import sys
import logging
import pickle
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from sentence_transformers import SentenceTransformer
import faiss

from app.config import (
    EMBEDDING_MODEL, CACHE_EMBEDDINGS, CACHE_CHUNKS,
    CACHE_FAISS, TOP_K, CHUNK_MIN_LEN, LOGS_DIR
)


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

    archivo = logging.FileHandler(Path(LOGS_DIR) / "rag_utils.log", encoding="utf-8")
    archivo.setLevel(logging.DEBUG)
    archivo.setFormatter(fmt)

    logger.addHandler(consola)
    logger.addHandler(archivo)
    return logger

class rag_utils:
    """
    Pipeline RAG completo como Singleton.
    Gestiona chunking, embeddings, índice FAISS y búsqueda semántica.
    El corpus debe venir pre-etiquetado con emociones por el clasificador.
    """

    _instancia = None

    def __new__(cls):
        if cls._instancia is None:
            cls._instancia = super().__new__(cls)
            cls._instancia._inicializado = False
        return cls._instancia

    def __init__(self):
        if self._inicializado:
            return
        self._log          = _configurar_logger("rag_utils")
        self._modelo_emb   = None
        self._indice_faiss = None
        self._chunks       = None
        self._inicializado = True
        self._log.debug("rag_utils instanciado.")

    # Chunking

    def chunking_por_parrafos(self, cancion, min_longitud=CHUNK_MIN_LEN):
        """Divide la letra por estrofas. Cada chunk hereda la emoción del clasificador."""
        try:
            letra = str(cancion.get("letra", "")).strip()
            if not letra:
                return []

            parrafos = re.split(r'\n\s*\n', letra)
            parrafos = [p.strip() for p in parrafos if p.strip()]

            if len(parrafos) <= 1:
                parrafos = [p.strip() for p in letra.split('\n') if p.strip()]

            chunks_texto, buffer = [], ""
            for p in parrafos:
                if len(buffer) + len(p) < min_longitud * 3:
                    buffer += (" " if buffer else "") + p
                else:
                    if buffer.strip():
                        chunks_texto.append(buffer.strip())
                    buffer = p
            if buffer.strip():
                chunks_texto.append(buffer.strip())

            if not chunks_texto:
                chunks_texto = [letra[:1000]]

            emocion       = cancion.get("emocion", "amor")
            emocion_score = cancion.get("emocion_score", 0.0)

            return [{
                "texto":         texto,
                "titulo":        cancion.get("titulo",  "Desconocido"),
                "artista":       cancion.get("artista", "Desconocido"),
                "genero":        cancion.get("genero",  "Desconocido"),
                "anio":          cancion.get("anio",    "?"),
                "idioma":        cancion.get("idioma",  "?"),
                "emocion":       emocion,
                "emocion_score": emocion_score,
                "chunk_id":      i,
                "cancion_id":    str(cancion.get("_id", "")),
            } for i, texto in enumerate(chunks_texto)]

        except Exception as e:
            self._log.warning(f"Error al chunkear canción '{cancion.get('titulo', '?')}': {e}")
            return []

    def chunking_cancion_completa(self, cancion):
        """Estrategia alternativa: letra completa como un único chunk."""
        try:
            letra = str(cancion.get("letra", "")).strip()
            if not letra:
                return []
            return [{
                "texto":         letra[:2000],
                "titulo":        cancion.get("titulo",  "Desconocido"),
                "artista":       cancion.get("artista", "Desconocido"),
                "genero":        cancion.get("genero",  "Desconocido"),
                "anio":          cancion.get("anio",    "?"),
                "idioma":        cancion.get("idioma",  "?"),
                "emocion":       cancion.get("emocion", "amor"),
                "emocion_score": cancion.get("emocion_score", 0.0),
                "chunk_id":      0,
                "cancion_id":    str(cancion.get("_id", "")),
            }]
        except Exception as e:
            self._log.warning(f"Error en chunking completo '{cancion.get('titulo', '?')}': {e}")
            return []

    def construir_chunks(self, canciones: list) -> list:
        """Aplica chunking por párrafos a todo el corpus etiquetado."""
        self._log.info(f"Construyendo chunks para {len(canciones)} canciones...")
        try:
            todos = []
            for c in canciones:
                todos.extend(self.chunking_por_parrafos(c))
            self._log.info(f"Chunks generados: {len(todos)}")
            return todos
        except Exception as e:
            self._log.error(f"Error al construir chunks: {e}")
            raise RuntimeError(f"Error en construcción de chunks: {e}") from e

    # Embeddings

    def _get_modelo_embeddings(self) -> SentenceTransformer:
        """Carga el modelo de embeddings una sola vez."""
        if self._modelo_emb is None:
            try:
                self._log.info(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}")
                self._modelo_emb = SentenceTransformer(EMBEDDING_MODEL)
                dim = self._modelo_emb.get_sentence_embedding_dimension()
                self._log.info(f"Modelo cargado. Dimensión: {dim}")
            except Exception as e:
                self._log.error(f"Error al cargar modelo de embeddings: {e}")
                raise RuntimeError(f"No se pudo cargar el modelo de embeddings: {e}") from e
        return self._modelo_emb

    def generar_embeddings(self, chunks: list, forzar=False):
        """Genera o carga desde caché los embeddings de los chunks."""
        if not forzar and Path(CACHE_EMBEDDINGS).exists() and Path(CACHE_CHUNKS).exists():
            try:
                self._log.info("Cargando embeddings desde caché...")
                embeddings = np.load(CACHE_EMBEDDINGS)
                with open(CACHE_CHUNKS, "rb") as f:
                    chunks_cache = pickle.load(f)
                self._log.info(f"Embeddings cargados: {embeddings.shape} | Chunks: {len(chunks_cache)}")
                return embeddings, chunks_cache
            except Exception as e:
                self._log.warning(f"Error al cargar caché de embeddings, regenerando: {e}")

        try:
            self._log.info(f"Generando embeddings para {len(chunks)} chunks...")
            modelo = self._get_modelo_embeddings()
            textos = [c["texto"] for c in chunks]
            embeddings = modelo.encode(
                textos, batch_size=64, show_progress_bar=True, convert_to_numpy=True
            ).astype(np.float32)

            np.save(CACHE_EMBEDDINGS, embeddings)
            with open(CACHE_CHUNKS, "wb") as f:
                pickle.dump(chunks, f)

            self._log.info(f"Embeddings guardados: {embeddings.shape}")
            return embeddings, chunks

        except Exception as e:
            self._log.error(f"Error al generar embeddings: {e}")
            raise RuntimeError(f"Error en generación de embeddings: {e}") from e

    # Índice FAISS

    def construir_indice(self, embeddings, forzar=False):
        """Construye o carga el índice FAISS con búsqueda por coseno."""
        if not forzar and Path(CACHE_FAISS).exists():
            try:
                self._log.info("Cargando índice FAISS desde caché...")
                indice = faiss.read_index(CACHE_FAISS)
                self._log.info(f"Índice cargado: {indice.ntotal} vectores, dim {indice.d}")
                return indice
            except Exception as e:
                self._log.warning(f"Error al cargar índice FAISS, reconstruyendo: {e}")

        try:
            self._log.info("Construyendo índice FAISS...")
            dimension = embeddings.shape[1]
            indice    = faiss.IndexFlatL2(dimension)
            emb_norm  = embeddings.copy()
            faiss.normalize_L2(emb_norm)
            indice.add(emb_norm)
            faiss.write_index(indice, CACHE_FAISS)
            self._log.info(f"Índice guardado: {indice.ntotal} vectores")
            return indice
        except Exception as e:
            self._log.error(f"Error al construir índice FAISS: {e}")
            raise RuntimeError(f"Error en construcción de índice FAISS: {e}") from e

    # Inicialización completa

    def inicializar(self, canciones_etiquetadas=None, forzar=False):
        """
        Inicializa el pipeline RAG completo.
        Si existe caché, carga sin necesitar las canciones.
        Si no, requiere el corpus etiquetado por etiquetar_corpus_con_modelo().
        """
        cache_ok = (Path(CACHE_EMBEDDINGS).exists() and
                    Path(CACHE_CHUNKS).exists() and
                    Path(CACHE_FAISS).exists())

        try:
            if cache_ok and not forzar:
                self._log.info("Caché completo encontrado. Cargando RAG desde disco...")
                embeddings, self._chunks = self.generar_embeddings([], forzar=False)
                self._indice_faiss = self.construir_indice(embeddings, forzar=False)
            else:
                if canciones_etiquetadas is None:
                    raise ValueError(
                        "No hay caché. Pasa el corpus etiquetado por etiquetar_corpus_con_modelo().")
                self._log.info("Construyendo RAG desde corpus etiquetado...")
                self._chunks = self.construir_chunks(canciones_etiquetadas)
                embeddings, self._chunks = self.generar_embeddings(self._chunks, forzar=True)
                self._indice_faiss = self.construir_indice(embeddings, forzar=True)

            self._log.info(f"RAG listo: {len(self._chunks)} chunks indexados.")
            return self._indice_faiss, self._chunks

        except ValueError as e:
            self._log.error(f"Error de configuración al inicializar RAG: {e}")
            raise
        except Exception as e:
            self._log.error(f"Error inesperado al inicializar RAG: {e}")
            raise RuntimeError(f"Error al inicializar RAG: {e}") from e

    #  Búsqueda semántica

    def buscar(self, pregunta: str, top_k=TOP_K, filtro_genero=None,
               filtro_idioma=None, filtro_emocion=None) -> list:
        """
        Busca chunks relevantes en FAISS filtrando por emoción del clasificador.

        Args:
            pregunta:       Texto de la pregunta del usuario.
            top_k:          Número de resultados a devolver.
            filtro_genero:  Filtrar por género (ej: 'pop').
            filtro_idioma:  Filtrar por idioma (ej: 'es').
            filtro_emocion: Filtrar por emoción del clasificador (ej: 'tristeza').

        Returns:
            Lista de chunks con metadatos y score de similitud.
        """
        if self._indice_faiss is None or self._chunks is None:
            msg = "RAG no inicializado. Llama a inicializar() primero."
            self._log.error(msg)
            raise RuntimeError(msg)

        try:
            self._log.debug(
                f"Buscando | pregunta='{pregunta[:50]}' | "
                f"top_k={top_k} | emocion={filtro_emocion}"
            )
            modelo = self._get_modelo_embeddings()
            emb    = modelo.encode([pregunta], convert_to_numpy=True).astype(np.float32)
            faiss.normalize_L2(emb)

            k_busq = min(top_k * 8, len(self._chunks))
            distancias, indices = self._indice_faiss.search(emb, k_busq)

            resultados = []
            for dist, idx in zip(distancias[0], indices[0]):
                if idx < 0 or idx >= len(self._chunks):
                    continue
                chunk = self._chunks[idx]
                if filtro_genero  and str(chunk.get("genero","")).lower() != filtro_genero.lower():
                    continue
                if filtro_idioma  and str(chunk.get("idioma","")).lower() != filtro_idioma.lower():
                    continue
                if filtro_emocion and str(chunk.get("emocion","")).lower() != filtro_emocion.lower():
                    continue
                resultados.append({
                    "texto":         chunk["texto"],
                    "titulo":        chunk["titulo"],
                    "artista":       chunk["artista"],
                    "genero":        chunk["genero"],
                    "anio":          chunk["anio"],
                    "idioma":        chunk["idioma"],
                    "emocion":       chunk["emocion"],
                    "emocion_score": chunk["emocion_score"],
                    "score":         float(1 - dist),
                })
                if len(resultados) >= top_k:
                    break

            self._log.debug(f"Resultados encontrados: {len(resultados)}")
            return resultados

        except RuntimeError:
            raise
        except Exception as e:
            self._log.error(f"Error inesperado en búsqueda: {e}")
            raise RuntimeError(f"Error en búsqueda semántica: {e}") from e