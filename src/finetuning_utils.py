"""
finetuning_utils.py — Clasificador de emoción como clase con logs y excepciones.

Flujo:
  1. Etiquetado combinado: modelo de sentimiento multilingüe + keywords
  2. Submuestreo para balancear clases (ratio máximo 3x)
  3. Fine-Tuning de DistilBERT con WeightedTrainer + EarlyStopping
  4. El modelo fine-tuneado etiqueta TODO el corpus para el RAG
"""

import os
import sys
import json
import logging
import random
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.config import (
    FINETUNE_BASE, FINETUNE_MODEL_DIR, RANDOM_SEED,
    VAL_SPLIT, TEST_SPLIT, MAX_EPOCHS, BATCH_SIZE,
    LEARNING_RATE, MAX_LENGTH, EMOCION2ID, ID2EMOCION,
    MIN_SAMPLES_PER_CLASS, RESULTADOS_DIR, CACHE_DIR, LOGS_DIR
)

CORPUS_ETIQUETADO_PATH = str(Path(CACHE_DIR) / "corpus_con_emociones.pkl")


# ══════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════

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
    archivo = logging.FileHandler(Path(LOGS_DIR) / "finetuning.log", encoding="utf-8")
    archivo.setLevel(logging.DEBUG)
    archivo.setFormatter(fmt)
    logger.addHandler(consola)
    logger.addHandler(archivo)
    return logger


# ══════════════════════════════════════════════════════════════════
# CLASE EmotionClassifier
# ══════════════════════════════════════════════════════════════════

class finetuning_utils:
    """
    Clasificador de emoción musical con Singleton.
    Gestiona etiquetado combinado (modelo+keywords), balanceo,
    entrenamiento con WeightedTrainer+EarlyStopping, evaluación e inferencia.
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
        self._log           = _configurar_logger("EmotionClassifier")
        self._pipeline_clf  = None
        self._etiquetador   = None   # modelo de sentimiento multilingüe
        self._inicializado  = True
        self._log.debug("EmotionClassifier instanciado.")

    # ── Keywords base ────────────────────────────────────────────

    KEYWORDS = {
        "alegria":   ["feliz","alegr","fiesta","celebr","reir","gozo","divert",
                      "happy","joy","fun","party","laugh","enjoy","smile","dance","good"],
        "tristeza":  ["trist","llor","dolor","sufr","pena","melanc","deprim",
                      "sad","cry","pain","hurt","suffer","broken","tears","alone","empty"],
        "amor":      ["amor","querer","amar","beso","corazon","enamorad","pasion",
                      "love","heart","kiss","baby","darling","together","forever","mine"],
        "rabia":     ["rabia","odio","enoj","furi","traicion","venganz",
                      "hate","anger","mad","furious","damn","never","fight","wrong","liar"],
        "nostalgia": ["recuerd","pasado","ayer","añor","extrañ","volver",
                      "remember","past","yesterday","memories","used to","back","time","gone"],
    }

    def _contar_keywords(self, texto_lower: str) -> dict:
        """Cuenta ocurrencias de keywords por emoción."""
        return {e: sum(texto_lower.count(p) for p in ps)
                for e, ps in self.KEYWORDS.items()}

    # ── Etiquetado por keywords (umbral 2) ───────────────────────

    def _etiquetar_keywords_estricto(self, texto: str):
        """Keywords con umbral 2 — alta precisión, baja cobertura."""
        texto_lower = texto.lower()
        conteos = self._contar_keywords(texto_lower)
        total   = sum(conteos.values())
        if total == 0:
            return None
        mejor = max(conteos, key=conteos.get)
        return mejor if conteos[mejor] >= 2 else None

    def _etiquetar_keywords_suave(self, texto: str):
        """Keywords con umbral 1 — mayor cobertura para canciones cortas."""
        texto_lower = texto.lower()
        conteos = self._contar_keywords(texto_lower)
        if max(conteos.values()) == 0:
            return None
        return max(conteos, key=conteos.get)

    # ── Etiquetado con modelo de sentimiento ─────────────────────

    def _cargar_etiquetador(self):
        """Carga el modelo de sentimiento multilingüe (singleton)."""
        if self._etiquetador is None:
            try:
                import torch
                from transformers import pipeline as hf_pipeline
                device = 0 if torch.cuda.is_available() else -1
                self._log.info("Cargando modelo de sentimiento multilingüe...")
                self._etiquetador = hf_pipeline(
                    "text-classification",
                    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                    top_k=None, truncation=True, max_length=128, device=device
                )
                self._log.info("Modelo de sentimiento cargado.")
            except Exception as e:
                self._log.warning(f"No se pudo cargar modelo de sentimiento: {e}")
        return self._etiquetador

    def _etiquetar_con_modelo(self, texto: str) -> str:
        """
        Combina modelo de sentimiento + keywords para etiquetar.
        Siempre retorna una emoción (nunca None).
        """
        try:
            etiquetador = self._cargar_etiquetador()
            texto_lower = texto.lower()

            tiene = {e: any(k in texto_lower for k in ks)
                     for e, ks in self.KEYWORDS.items()}

            if etiquetador is not None:
                raw = etiquetador(texto[:256])
                while isinstance(raw, list) and isinstance(raw[0], list):
                    raw = raw[0]
                scores = {r['label'].lower(): r['score'] for r in raw}
                pos = scores.get('positive', 0)
                neg = scores.get('negative', 0)

                # Reglas combinadas modelo + keywords
                if neg > 0.6 and tiene["rabia"]:     return "rabia"
                if neg > 0.5 and tiene["tristeza"]:   return "tristeza"
                if neg > 0.4 and tiene["nostalgia"]:  return "nostalgia"
                if pos > 0.6 and tiene["alegria"]:    return "alegria"
                if pos > 0.5:
                    return "alegria" if tiene["alegria"] else "amor"
                if neg > 0.5:
                    return "tristeza" if tiene["tristeza"] else "rabia"

            # Fallback: keyword con mayor conteo
            conteos = self._contar_keywords(texto_lower)
            if max(conteos.values()) > 0:
                return max(conteos, key=conteos.get)
            return "amor"

        except Exception as e:
            self._log.warning(f"Error en etiquetado con modelo: {e}")
            return "amor"

    # ── Etiquetado del corpus completo ───────────────────────────

    def etiquetar_corpus_keywords(self, canciones: list) -> list:
        """
        Etiqueta el corpus usando SOLO keywords con umbral 2.
        Alta precisión, cobertura ~48%. Las canciones sin señal
        clara quedan sin etiqueta (se descartan).

        Para ampliar la cobertura usar etiquetar_con_modelo_sentimiento()
        sobre las canciones que quedaron sin etiqueta.
        """
        from collections import Counter
        self._log.info(f"Etiquetando por keywords ({len(canciones)} canciones)...")
        dataset, sin_etiqueta = [], 0

        try:
            for cancion in canciones:
                letra = str(cancion.get("letra", "")).strip()
                if len(letra) < 50:
                    sin_etiqueta += 1
                    continue
                texto = letra[:800]
                emocion = self._etiquetar_keywords_estricto(texto)
                if emocion is None:
                    sin_etiqueta += 1
                    continue
                dataset.append({
                    "texto": texto,
                    "emocion": emocion,
                    "titulo": cancion.get("titulo", ""),
                    "artista": cancion.get("artista", ""),
                })

            dist = Counter(d["emocion"] for d in dataset)
            self._log.info(f"Etiquetadas: {len(dataset)} | Sin etiqueta: {sin_etiqueta}")
            self._log.info(f"Cobertura: {len(dataset) / len(canciones) * 100:.1f}%")
            self._log.info(f"Distribución: {dict(sorted(dist.items(), key=lambda x: -x[1]))}")
            return dataset

        except Exception as e:
            self._log.error(f"Error en etiquetado por keywords: {e}")
            raise RuntimeError(f"Error en etiquetado: {e}") from e

    def etiquetar_con_modelo_sentimiento(self, canciones_sin_etiqueta: list,
                                         titulos_ya_etiquetados: set) -> list:
        """
        Etiqueta canciones que quedaron sin etiqueta usando el modelo
        de sentimiento multilingüe + keywords como desempate.
        Llamar DESPUÉS de etiquetar_corpus_keywords() para ampliar cobertura.

        Args:
            canciones_sin_etiqueta:  Lista de docs de MongoDB sin etiqueta.
            titulos_ya_etiquetados:  Set de títulos ya etiquetados (para no duplicar).

        Returns:
            Lista de nuevos dicts con 'texto' y 'emocion'.
        """
        from collections import Counter
        self._log.info(
            f"Etiquetando {len(canciones_sin_etiqueta)} canciones "
            f"con modelo de sentimiento..."
        )
        nuevos = []

        try:
            for cancion in canciones_sin_etiqueta:
                titulo = cancion.get("titulo", "")
                if titulo in titulos_ya_etiquetados:
                    continue
                letra = str(cancion.get("letra", "")).strip()
                if len(letra) < 20:
                    continue
                emocion = self._etiquetar_con_modelo(letra[:400])
                if emocion:
                    nuevos.append({
                        "texto": letra[:800],
                        "emocion": emocion,
                        "titulo": titulo,
                        "artista": cancion.get("artista", ""),
                    })

            dist = Counter(d["emocion"] for d in nuevos)
            self._log.info(f"Nuevas etiquetas: {len(nuevos)}")
            self._log.info(f"Distribución: {dict(sorted(dist.items(), key=lambda x: -x[1]))}")
            return nuevos

        except Exception as e:
            self._log.error(f"Error en etiquetado con modelo: {e}")
            raise RuntimeError(f"Error en etiquetado con modelo: {e}") from e
    # ── Balanceo de clases ───────────────────────────────────────

    def balancear_dataset(self, dataset: list, ratio_max: float = 3.0) -> list:
        """
        Submuestrea las clases mayoritarias para reducir el desbalance.
        Mantiene todas las muestras de las clases minoritarias.

        Args:
            dataset:   Lista de dicts con campo 'emocion'.
            ratio_max: Máximo ratio permitido entre clase mayor y menor.

        Returns:
            Dataset balanceado.
        """
        from collections import Counter

        random.seed(RANDOM_SEED)
        dist = Counter(d["emocion"] for d in dataset)

        clase_min  = min(dist.values())
        limite_max = int(clase_min * ratio_max)

        self._log.info(f"Balanceando dataset | clase_min={clase_min} | limite_max={limite_max}")

        dataset_balanceado = []
        for emocion in dist:
            subset = [d for d in dataset if d["emocion"] == emocion]
            if len(subset) > limite_max:
                subset = random.sample(subset, limite_max)
                self._log.info(f"  {emocion:12s}: {dist[emocion]} → {limite_max} (submuestreado)")
            else:
                self._log.info(f"  {emocion:12s}: {len(subset)} (sin cambio)")
            dataset_balanceado.extend(subset)

        random.shuffle(dataset_balanceado)

        dist_nueva  = Counter(d["emocion"] for d in dataset_balanceado)
        nuevo_ratio = max(dist_nueva.values()) / min(dist_nueva.values())
        self._log.info(f"Total final: {len(dataset_balanceado)} | Ratio: {nuevo_ratio:.1f}x")
        return dataset_balanceado

    # ── Dataset HuggingFace ──────────────────────────────────────

    def preparar_dataset_hf(self, dataset_etiquetado: list):
        """Prepara DatasetDict con splits 70/15/15 y seed fijo."""
        from datasets import Dataset, DatasetDict
        from collections import Counter

        self._log.info("Preparando DatasetDict para HuggingFace...")
        try:
            conteo = Counter(d["emocion"] for d in dataset_etiquetado)
            clases_validas = {e for e, c in conteo.items() if c >= MIN_SAMPLES_PER_CLASS}
            datos = [d for d in dataset_etiquetado if d["emocion"] in clases_validas]

            self._log.info(f"Clases válidas (>={MIN_SAMPLES_PER_CLASS}): {sorted(clases_validas)}")
            self._log.info(f"Total muestras: {len(datos)}")

            for d in datos:
                d["label"] = EMOCION2ID.get(d["emocion"], 0)

            ds = Dataset.from_list(datos).shuffle(seed=RANDOM_SEED)
            n       = len(ds)
            n_test  = int(n * TEST_SPLIT)
            n_val   = int(n * VAL_SPLIT)

            dd = DatasetDict({
                "train":      ds.select(range(n_test + n_val, n)),
                "validation": ds.select(range(n_test, n_test + n_val)),
                "test":       ds.select(range(n_test)),
            })
            self._log.info(
                f"Splits: train={len(dd['train'])} | "
                f"val={len(dd['validation'])} | test={len(dd['test'])}"
            )
            return dd

        except Exception as e:
            self._log.error(f"Error al preparar dataset: {e}")
            raise RuntimeError(f"Error en preparación del dataset: {e}") from e

    # ── Tokenización ─────────────────────────────────────────────

    def _tokenizar_dataset(self, dataset_dict, tokenizer):
        """Tokeniza el DatasetDict."""
        def tokenizar(batch):
            return tokenizer(batch["texto"], truncation=True,
                             padding="max_length", max_length=MAX_LENGTH)
        tokenizado = dataset_dict.map(tokenizar, batched=True)
        tokenizado = tokenizado.remove_columns(
            [c for c in tokenizado["train"].column_names
             if c not in ["input_ids", "attention_mask", "label"]]
        )
        tokenizado.set_format("torch")
        return tokenizado

    # ── Entrenamiento ────────────────────────────────────────────

    def entrenar(self, dataset_dict, num_labels=None):
        """
        Fine-Tuning de DistilBERT con:
          - WeightedTrainer: corrige desbalance con CrossEntropyLoss ponderado
          - EarlyStoppingCallback: detiene si no mejora en 2 épocas
          - warmup_ratio: calentamiento gradual del learning rate
        """
        import torch
        import torch.nn as nn
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer, DataCollatorWithPadding,
            EarlyStoppingCallback
        )
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.utils.class_weight import compute_class_weight

        self._log.info(f"Iniciando fine-tuning de {FINETUNE_BASE}")

        try:
            if num_labels is None:
                num_labels = len(set(dataset_dict["train"]["label"]))

            # ── Class weights ─────────────────────────────────────
            labels_train  = np.array(dataset_dict["train"]["label"])
            clases_unicas = np.unique(labels_train)
            pesos         = compute_class_weight("balanced",
                                                  classes=clases_unicas,
                                                  y=labels_train)
            class_weights = torch.tensor(pesos, dtype=torch.float32)

            self._log.info("Class weights:")
            for idx, peso in zip(clases_unicas, pesos):
                self._log.info(f"  {ID2EMOCION.get(int(idx), idx):12s}: {peso:.4f}")

            # ── WeightedTrainer ───────────────────────────────────
            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels  = inputs.pop("labels")
                    outputs = model(**inputs)
                    loss    = nn.CrossEntropyLoss(
                        weight=class_weights.to(outputs.logits.device)
                    )(outputs.logits, labels)
                    return (loss, outputs) if return_outputs else loss

            # ── Modelo y tokenizer ────────────────────────────────
            tokenizer = AutoTokenizer.from_pretrained(FINETUNE_BASE)
            model     = AutoModelForSequenceClassification.from_pretrained(
                FINETUNE_BASE, num_labels=num_labels,
                id2label=ID2EMOCION, label2id=EMOCION2ID
            )

            dataset_tok = self._tokenizar_dataset(dataset_dict, tokenizer)

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                preds = np.argmax(logits, axis=-1)
                return {
                    "accuracy": accuracy_score(labels, preds),
                    "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
                }

            dispositivo = "GPU" if torch.cuda.is_available() else "CPU"
            self._log.info(
                f"Dispositivo: {dispositivo} | Épocas: {MAX_EPOCHS} | "
                f"Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}"
            )

            # ── TrainingArguments ─────────────────────────────────
            training_args = TrainingArguments(
                output_dir=FINETUNE_MODEL_DIR,
                num_train_epochs=MAX_EPOCHS,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                weight_decay=0.01,
                warmup_ratio=0.1,              # calentamiento gradual
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1_macro",
                greater_is_better=True,
                seed=RANDOM_SEED,
                logging_steps=50,
                report_to="none",
                fp16=torch.cuda.is_available(),
            )

            # ── Trainer con EarlyStopping ─────────────────────────
            trainer = WeightedTrainer(
                model=model, args=training_args,
                train_dataset=dataset_tok["train"],
                eval_dataset=dataset_tok["validation"],
                processing_class=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            )

            self._log.info("Entrenando...")
            trainer.train()

            trainer.save_model(FINETUNE_MODEL_DIR)
            tokenizer.save_pretrained(FINETUNE_MODEL_DIR)
            self._log.info(f"Modelo guardado en {FINETUNE_MODEL_DIR}")
            return trainer, tokenizer, model

        except Exception as e:
            self._log.error(f"Error durante el entrenamiento: {e}")
            raise RuntimeError(f"Error en fine-tuning: {e}") from e

    # ── Evaluación ───────────────────────────────────────────────

    def evaluar(self, trainer, dataset_dict, tokenizer) -> dict:
        """Evalúa en test set y guarda métricas en resultados/metricas.json."""
        from sklearn.metrics import (
            accuracy_score, f1_score, classification_report, confusion_matrix
        )

        self._log.info("Evaluando modelo en test set...")
        try:
            dataset_tok  = self._tokenizar_dataset(dataset_dict, tokenizer)
            predicciones = trainer.predict(dataset_tok["test"])
            preds  = np.argmax(predicciones.predictions, axis=-1)
            labels = predicciones.label_ids

            acc     = accuracy_score(labels, preds)
            f1      = f1_score(labels, preds, average="macro", zero_division=0)
            cm      = confusion_matrix(labels, preds).tolist()
            reporte = classification_report(
                labels, preds,
                target_names=[ID2EMOCION.get(i, str(i)) for i in sorted(set(labels))],
                zero_division=0
            )

            metricas = {
                "accuracy": round(acc, 4), "f1_macro": round(f1, 4),
                "confusion_matrix": cm, "classification_report": reporte,
                "num_test_samples": len(labels),
            }

            self._log.info(f"Accuracy: {acc:.4f} | F1 Macro: {f1:.4f}")

            Path(RESULTADOS_DIR).mkdir(exist_ok=True)
            with open(Path(RESULTADOS_DIR) / "metricas.json", "w", encoding="utf-8") as f:
                json.dump(metricas, f, ensure_ascii=False, indent=2)
            self._log.info("Métricas guardadas en resultados/metricas.json")
            return metricas

        except Exception as e:
            self._log.error(f"Error en evaluación: {e}")
            raise RuntimeError(f"Error al evaluar el modelo: {e}") from e

    # ── Etiquetar corpus completo con el modelo ──────────────────

    def etiquetar_corpus_con_modelo(self, canciones: list,
                                     batch_size=64, forzar=False) -> list:
        """
        Usa el clasificador fine-tuneado para etiquetar TODO el corpus.
        Resultado cacheado en disco para no repetir la inferencia.
        """
        import pickle
        import torch

        cache_path = Path(CORPUS_ETIQUETADO_PATH)

        if cache_path.exists() and not forzar:
            self._log.info("Cargando corpus etiquetado desde caché...")
            try:
                with open(cache_path, "rb") as f:
                    corpus = pickle.load(f)
                self._log.info(f"{len(corpus)} canciones cargadas con emoción.")
                return corpus
            except Exception as e:
                self._log.warning(f"Error al cargar caché, regenerando: {e}")

        if not Path(FINETUNE_MODEL_DIR).exists():
            msg = "No hay modelo fine-tuneado. Ejecuta entrenar() primero."
            self._log.error(msg)
            raise RuntimeError(msg)

        try:
            from transformers import pipeline as hf_pipeline
            device = 0 if torch.cuda.is_available() else -1
            self._log.info(f"Cargando clasificador desde {FINETUNE_MODEL_DIR}...")
            clf = hf_pipeline(
                "text-classification", model=FINETUNE_MODEL_DIR,
                tokenizer=FINETUNE_MODEL_DIR, truncation=True,
                max_length=MAX_LENGTH, device=device
            )

            textos  = [str(c.get("letra", ""))[:512] for c in canciones]
            validos = [i for i, t in enumerate(textos) if len(t.strip()) >= 30]
            self._log.info(
                f"Clasificando {len(validos)} canciones en batches de {batch_size}..."
            )

            resultados_clf = {}
            for i in range(0, len(validos), batch_size):
                batch_idx   = validos[i:i+batch_size]
                batch_texts = [textos[j] for j in batch_idx]
                outputs     = clf(batch_texts)
                for j, raw in zip(batch_idx, outputs):
                    item = raw
                    while isinstance(item, list):
                        item = item[0]
                    resultados_clf[j] = {
                        "emocion":       item["label"],
                        "emocion_score": round(item["score"], 4)
                    }
                if (i // batch_size) % 10 == 0:
                    self._log.info(
                        f"  Procesadas: {min(i+batch_size, len(validos))}/{len(validos)}"
                    )

            corpus_etiquetado = []
            for i, cancion in enumerate(canciones):
                c = dict(cancion)
                if i in resultados_clf:
                    c["emocion"]       = resultados_clf[i]["emocion"]
                    c["emocion_score"] = resultados_clf[i]["emocion_score"]
                else:
                    c["emocion"]       = "amor"
                    c["emocion_score"] = 0.0
                corpus_etiquetado.append(c)

            Path(cache_path.parent).mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(corpus_etiquetado, f)

            from collections import Counter
            dist = Counter(c["emocion"] for c in corpus_etiquetado)
            self._log.info(f"Corpus etiquetado: {len(corpus_etiquetado)} canciones")
            for emocion, count in sorted(dist.items(), key=lambda x: -x[1]):
                self._log.info(
                    f"  {emocion:12s}: {count:5d} ({count/len(corpus_etiquetado)*100:.1f}%)"
                )
            self._log.info(f"Guardado en {cache_path}")
            return corpus_etiquetado

        except RuntimeError:
            raise
        except Exception as e:
            self._log.error(f"Error al etiquetar corpus con modelo: {e}")
            raise RuntimeError(f"Error en etiquetado con modelo: {e}") from e

    # ── Inferencia en tiempo real ────────────────────────────────

    def cargar_clasificador(self):
        """Carga el clasificador fine-tuneado para inferencia (singleton)."""
        if self._pipeline_clf is None:
            if not Path(FINETUNE_MODEL_DIR).exists():
                self._log.warning("No hay modelo fine-tuneado disponible.")
                return None
            try:
                from transformers import pipeline as hf_pipeline
                self._pipeline_clf = hf_pipeline(
                    "text-classification", model=FINETUNE_MODEL_DIR,
                    tokenizer=FINETUNE_MODEL_DIR,
                    truncation=True, max_length=MAX_LENGTH
                )
                self._log.info("Clasificador cargado para inferencia.")
            except Exception as e:
                self._log.error(f"Error al cargar clasificador: {e}")
                return None
        return self._pipeline_clf

    def predecir_emocion(self, texto: str):
        """
        Predice la emoción de un texto.
        Robusto a cualquier estructura de output del pipeline.

        Returns:
            dict con 'emocion' y 'score', o None si no hay modelo.
        """
        try:
            clf = self.cargar_clasificador()
            if clf is None:
                return None
            raw = clf(texto[:MAX_LENGTH])
            while isinstance(raw, list):
                raw = raw[0]
            resultado = {"emocion": raw["label"], "score": round(raw["score"], 4)}
            self._log.debug(f"Emoción predicha: {resultado}")
            return resultado
        except Exception as e:
            self._log.error(f"Error al predecir emoción: {e}")
            return None



