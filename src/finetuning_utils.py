"""
Clasificador de emoción

Flujo:
  1. Etiquetado inicial por keywords (para entrenar el clasificador)
  2. Fine-Tuning de DistilBERT con WeightedTrainer
  3. El modelo fine-tuneado etiqueta TODO el corpus
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import pickle
from transformers import pipeline as hf_pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.config import (
    FINETUNE_BASE, FINETUNE_MODEL_DIR, RANDOM_SEED,
    VAL_SPLIT, TEST_SPLIT, MAX_EPOCHS, BATCH_SIZE,
    LEARNING_RATE, MAX_LENGTH, EMOCION2ID, ID2EMOCION,
    MIN_SAMPLES_PER_CLASS, RESULTADOS_DIR, CACHE_DIR, LOGS_DIR
)

CORPUS_ETIQUETADO_PATH = str(Path(CACHE_DIR) / "corpus_con_emociones.pkl")


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

    archivo = logging.FileHandler(Path(LOGS_DIR) / "finetuning.log", encoding="utf-8")
    archivo.setLevel(logging.DEBUG)
    archivo.setFormatter(fmt)

    logger.addHandler(consola)
    logger.addHandler(archivo)
    return logger


class finetuning_utils:
    """
    Clasificador de emoción musical con Singleton.
    Gestiona etiquetado, entrenamiento, evaluación e inferencia.
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
        self._log          = _configurar_logger("finetuning_utils")
        self._pipeline_clf = None
        self._inicializado = True
        self._log.debug("finetuning_utils instanciado.")

    # Etiquetado por keywords

    def _etiquetar_keywords(self, texto: str):
        """Etiqueta emoción por keywords. Solo para generar datos de entrenamiento."""
        texto_lower = texto.lower()
        reglas = {
            "alegria":   ["feliz","alegr","fiesta","celebr","reir","gozo","divert","happy","joy","fun","party","celebrat","laugh","enjoy"],
            "tristeza":  ["trist","llor","dolor","sufr","pena","melanc","deprim","sad","cry","pain","suffer","sorrow","heartbreak","lonely"],
            "amor":      ["amor","querer","amar","beso","corazon","enamorad","pasion","love","heart","kiss","adore","darling","romance","together"],
            "rabia":     ["rabia","odio","enoj","furi","maldic","traicion","venganz","hate","anger","rage","mad","furious","betrayal","revenge"],
            "nostalgia": ["recuerd","pasado","ayer","tiempo","añor","extrañ","volver","remember","past","yesterday","memories","miss","gone","return"],
        }
        conteos = {e: sum(texto_lower.count(p) for p in ps) for e, ps in reglas.items()}
        total = sum(conteos.values())
        if total == 0:
            return None
        mejor = max(conteos, key=conteos.get)
        return mejor if conteos[mejor] >= 2 else None

    def etiquetar_corpus_keywords(self, canciones: list) -> list:
        """Etiqueta el corpus por keywords para generar el dataset de entrenamiento."""

        self._log.info(f"Etiquetando corpus por keywords ({len(canciones)} canciones)...")
        dataset, sin_etiqueta = [], 0
        try:
            for cancion in canciones:
                letra = str(cancion.get("letra", "")).strip()
                if len(letra) < 50:
                    continue
                texto   = letra[:800]
                emocion = self._etiquetar_keywords(texto)
                if emocion is None:
                    sin_etiqueta += 1
                    continue
                dataset.append({
                    "texto": texto, "emocion": emocion,
                    "titulo": cancion.get("titulo",""), "artista": cancion.get("artista","")
                })

            dist = Counter(d["emocion"] for d in dataset)
            self._log.info(f"Etiquetadas: {len(dataset)} | Sin etiqueta: {sin_etiqueta}")
            self._log.info(f"Distribución: {dict(sorted(dist.items(), key=lambda x:-x[1]))}")
            return dataset

        except Exception as e:
            self._log.error(f"Error en etiquetado por keywords: {e}")
            raise RuntimeError(f"Error en etiquetado: {e}") from e

    #Dataset HuggingFace

    def preparar_dataset_hf(self, dataset_etiquetado: list):
        """Prepara DatasetDict con splits 70/15/15 y seed fijo."""


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
            n  = len(ds)
            n_test = int(n * TEST_SPLIT)
            n_val  = int(n * VAL_SPLIT)

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

    # Tokenización

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

    # Entrenamiento

    def entrenar(self, dataset_dict, num_labels=None):
        """Fine-Tuning de DistilBERT con WeightedTrainer para corregir desbalance."""


        self._log.info(f"Iniciando fine-tuning de {FINETUNE_BASE}")

        try:
            if num_labels is None:
                num_labels = len(set(dataset_dict["train"]["label"]))

            # Class weights para corregir desbalance
            labels_train  = np.array(dataset_dict["train"]["label"])
            clases_unicas = np.unique(labels_train)
            pesos         = compute_class_weight("balanced", classes=clases_unicas, y=labels_train)
            class_weights = torch.tensor(pesos, dtype=torch.float32)

            self._log.info("Class weights calculados:")
            for idx, peso in zip(clases_unicas, pesos):
                self._log.info(f"  {ID2EMOCION.get(int(idx), idx):12s}: {peso:.4f}")

            # WeightedTrainer
            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels  = inputs.pop("labels")
                    outputs = model(**inputs)
                    loss    = nn.CrossEntropyLoss(
                        weight=class_weights.to(outputs.logits.device)
                    )(outputs.logits, labels)
                    return (loss, outputs) if return_outputs else loss

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
            self._log.info(f"Dispositivo: {dispositivo} | Épocas: {MAX_EPOCHS} | Batch: {BATCH_SIZE}")

            training_args = TrainingArguments(
                output_dir=FINETUNE_MODEL_DIR,
                num_train_epochs=MAX_EPOCHS,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1_macro",
                seed=RANDOM_SEED,
                logging_steps=50,
                report_to="none",
                fp16=torch.cuda.is_available(),
            )

            trainer = WeightedTrainer(
                model=model, args=training_args,
                train_dataset=dataset_tok["train"],
                eval_dataset=dataset_tok["validation"],
                processing_class=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=compute_metrics,
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

    # Evaluación

    def evaluar(self, trainer, dataset_dict, tokenizer) -> dict:
        """Evalúa en test set y guarda métricas."""


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

    # Etiquetar corpus completo con el modelo

    def etiquetar_corpus_con_modelo(self, canciones: list, batch_size=64, forzar=False) -> list:
        """
        Usa el clasificador fine-tuneado para etiquetar TODO el corpus.
        Resultado cacheado en disco para no repetir la inferencia.
        """

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

            device = 0 if torch.cuda.is_available() else -1
            self._log.info(f"Cargando clasificador desde {FINETUNE_MODEL_DIR}...")
            clf = hf_pipeline("text-classification", model=FINETUNE_MODEL_DIR,
                               tokenizer=FINETUNE_MODEL_DIR, truncation=True,
                               max_length=MAX_LENGTH, device=device)

            textos  = [str(c.get("letra", ""))[:512] for c in canciones]
            validos = [i for i, t in enumerate(textos) if len(t.strip()) >= 30]
            self._log.info(f"Clasificando {len(validos)} canciones en batches de {batch_size}...")

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
                    self._log.info(f"  Procesadas: {min(i+batch_size, len(validos))}/{len(validos)}")

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


            dist = Counter(c["emocion"] for c in corpus_etiquetado)
            self._log.info(f"Corpus etiquetado: {len(corpus_etiquetado)} canciones")
            for emocion, count in sorted(dist.items(), key=lambda x: -x[1]):
                self._log.info(f"  {emocion:12s}: {count:5d} ({count/len(corpus_etiquetado)*100:.1f}%)")
            self._log.info(f"Guardado en {cache_path}")
            return corpus_etiquetado

        except RuntimeError:
            raise
        except Exception as e:
            self._log.error(f"Error al etiquetar corpus con modelo: {e}")
            raise RuntimeError(f"Error en etiquetado con modelo: {e}") from e

    #Inferencia en tiempo real

    def cargar_clasificador(self):
        """Carga el clasificador fine-tuneado para inferencia (singleton)."""
        if self._pipeline_clf is None:
            if not Path(FINETUNE_MODEL_DIR).exists():
                self._log.warning("No hay modelo fine-tuneado disponible.")
                return None
            try:

                self._pipeline_clf = hf_pipeline(
                    "text-classification", model=FINETUNE_MODEL_DIR,
                    tokenizer=FINETUNE_MODEL_DIR, truncation=True, max_length=MAX_LENGTH
                )
                self._log.info("Clasificador cargado para inferencia.")
            except Exception as e:
                self._log.error(f"Error al cargar clasificador: {e}")
                return None
        return self._pipeline_clf

    def predecir_emocion(self, texto: str):
        """
        Predice la emoción de un texto. Robusto a cualquier estructura del pipeline.

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



