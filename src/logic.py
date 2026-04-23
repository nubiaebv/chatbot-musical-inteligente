# src/logic.py — Motor de IA y RAG Local
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Importación de configuración
from app.config import (
    DATA_PATH, EMBED_MODEL_NAME, GEN_MODEL_NAME,
    TOP_K, HISTORY_TURNS
)

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    song: str
    artist: str
    genre: str
    score: float


class RAGResult:
    def __init__(self, answer: str, chunks: List[Chunk], classifier_label: str = None):
        self.answer = answer
        self.chunks = chunks
        self.classifier_label = classifier_label


class ChatEngine:
    def __init__(self):
        logger.info("Cargando modelos locales (esto puede tardar unos segundos)...")
        # Modelo de Embeddings local
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        # Modelo de Generación local (Flan-T5)
        self.tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)

        self.corpus_df = None
        self.corpus_embeddings = None
        self._history = []

    def initialize(self):
        """Carga el CSV y genera/carga los embeddings locales"""
        self.corpus_df = pd.read_csv(DATA_PATH)
        # Simulación de búsqueda manual (Pipeline a mano)
        texts = self.corpus_df['lyrics'].tolist()
        self.corpus_embeddings = self.embed_model.encode(texts, convert_to_tensor=True)
        logger.info("Sistema RAG Local Inicializado.")

    def _get_context(self, query: str) -> List[Chunk]:
        """Búsqueda semántica manual usando similitud de coseno"""
        query_emb = self.embed_model.encode(query, convert_to_tensor=True)
        # Cálculo de similitud manual
        scores = torch.cos_sim(query_emb, self.corpus_embeddings)[0]
        top_results = torch.topk(scores, k=TOP_K)

        chunks = []
        for score, idx in zip(top_results.values, top_results.indices):
            row = self.corpus_df.iloc[idx.item()]
            chunks.append(Chunk(
                text=row['lyrics'][:200] + "...",  # Fragmento
                song=row['title'],
                artist=row['artist'],
                genre=row['genre'],
                score=float(score)
            ))
        return chunks

    def chat(self, user_msg: str) -> RAGResult:
        if self.corpus_df is None: self.initialize()

        # 1. Recuperación (RAG Manual)
        chunks = self._get_context(user_msg)

        # 2. Construcción de Prompt con Citas
        context_text = "\n".join([f"[{c.artist} - {c.song}]: {c.text}" for c in chunks])
        prompt = f"""Responde como MúsicBot. Usa solo este contexto:
        {context_text}

        Pregunta: {user_msg}
        Respuesta (Cita siempre artista y canción):"""

        # 3. Generación Local
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.gen_model.generate(**inputs, max_new_tokens=150)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return RAGResult(answer=answer, chunks=chunks)


engine = ChatEngine()