"""
Conexión persistente a MongoDB Atlas
"""

# Importe de librerias
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from pymongo import MongoClient
from pymongo.errors import (
    ConnectionFailure, ServerSelectionTimeoutError,
    OperationFailure, ConfigurationError
)
from app.config import MONGO_URI, DB_NAME, COLLECTION_NAME, LOGS_DIR

# Control para logs

def _configurar_logger(nombre: str) -> logging.Logger:
    """
    Configura un logger que escribe tanto en consola como en archivo.
    El archivo de log se guarda en logs/mongo_utils.log
    """
    logger = logging.getLogger(nombre)

    if logger.handlers:
        return logger  # ya configurado, no duplicar handlers

    logger.setLevel(logging.DEBUG)

    formato = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler de consola (INFO en adelante)
    consola = logging.StreamHandler(sys.stdout)
    consola.setLevel(logging.INFO)
    consola.setFormatter(formato)

    # Handler de archivo (DEBUG en adelante — registro completo)
    ruta_log = Path(LOGS_DIR) / "BDConexion.log"
    archivo = logging.FileHandler(ruta_log, encoding="utf-8")
    archivo.setLevel(logging.DEBUG)
    archivo.setFormatter(formato)

    logger.addHandler(consola)
    logger.addHandler(archivo)

    return logger


class mongo_utils:

    _instancia = None

    def __new__(cls):
        if cls._instancia is None:
            cls._instancia = super().__new__(cls)
            cls._instancia._inicializado = False
        return cls._instancia

    def __init__(self):
        if self._inicializado:
            return
        self._cliente  = None
        self._log      = _configurar_logger("mongo_utils")
        self._inicializado = True
        self._log.debug("mongo_utils instanciado.")

    def _conectar(self) -> MongoClient:
        if self._cliente is not None:
            return self._cliente
        if not MONGO_URI or "<usuario>" in MONGO_URI:
            msg = "MONGO_URI no configurada. Edita el archivo .env."
            self._log.error(msg)
            raise ConfigurationError(msg)
        try:
            self._log.info(f"Conectando a MongoDB Atlas | DB: {DB_NAME} | Col: {COLLECTION_NAME}")
            self._cliente = MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=8000,
                connectTimeoutMS=8000,
                socketTimeoutMS=30000,
            )
            self._log.debug("Cliente MongoClient creado.")
            return self._cliente
        except ConfigurationError as e:
            self._log.error(f"Error de configuración en MONGO_URI: {e}")
            raise
        except Exception as e:
            self._log.error(f"Error inesperado al crear el cliente: {e}")
            raise

    def _get_coleccion(self):
        try:
            return self._conectar()[DB_NAME][COLLECTION_NAME]
        except Exception as e:
            self._log.error(f"Error al obtener la colección: {e}")
            raise

    def cerrar(self):
        if self._cliente is not None:
            try:
                self._cliente.close()
                self._cliente = None
                self._log.info("Conexión cerrada correctamente.")
            except Exception as e:
                self._log.warning(f"Error al cerrar la conexión: {e}")
        else:
            self._log.debug("No había conexión activa.")

    def verificar_conexion(self) -> bool:
        try:
            self._conectar().admin.command("ping")
            self._log.info("Conexión verificada correctamente.")
            return True
        except ServerSelectionTimeoutError as e:
            self._log.error(f"Timeout (revisa tu IP en Network Access): {e}")
            return False
        except ConnectionFailure as e:
            self._log.error(f"Fallo de conexión: {e}")
            return False
        except ConfigurationError as e:
            self._log.error(f"Error de configuración: {e}")
            return False
        except Exception as e:
            self._log.error(f"Error inesperado: {e}")
            return False

    def cargar_canciones(self, limite=None, solo_con_letra=True) -> list:
        self._log.info(f"Cargando canciones | limite={limite} | solo_con_letra={solo_con_letra}")
        try:
            col = self._get_coleccion()
            filtro = {"letra": {"$exists": True, "$nin": [None, "", "null"]}} if solo_con_letra else {}
            proyeccion = {"_id":1,"titulo":1,"artista":1,"genero":1,"anio":1,"letra":1,"idioma":1}
            cursor = col.find(filtro, proyeccion)
            if limite:
                cursor = cursor.limit(limite)
            canciones = list(cursor)
            self._log.info(f"Canciones cargadas: {len(canciones)}")
            return canciones
        except OperationFailure as e:
            self._log.error(f"Error de operación (permisos o query): {e}")
            raise RuntimeError(f"Error al consultar MongoDB: {e}") from e
        except ServerSelectionTimeoutError as e:
            self._log.error(f"Timeout al cargar canciones: {e}")
            raise RuntimeError("No se pudo conectar a MongoDB Atlas.") from e
        except Exception as e:
            self._log.error(f"Error inesperado al cargar canciones: {e}")
            raise

    def estadisticas_corpus(self) -> dict:
        self._log.info("Calculando estadísticas del corpus...")
        try:
            col   = self._get_coleccion()
            total = col.count_documents({})
            con_letra = col.count_documents({"letra": {"$exists": True, "$nin": [None, ""]}})
            generos = list(col.aggregate([{"$group":{"_id":"$genero","total":{"$sum":1}}},{"$sort":{"total":-1}}]))
            idiomas = list(col.aggregate([{"$group":{"_id":"$idioma","total":{"$sum":1}}},{"$sort":{"total":-1}}]))
            anios   = list(col.aggregate([{"$match":{"anio":{"$exists":True,"$ne":None}}},
                                          {"$group":{"_id":None,"min_anio":{"$min":"$anio"},"max_anio":{"$max":"$anio"}}}]))
            stats = {
                "total_documentos": total,
                "con_letra":        con_letra,
                "generos":  {g["_id"]: g["total"] for g in generos if g["_id"]},
                "idiomas":  {i["_id"]: i["total"] for i in idiomas if i["_id"]},
                "anio_min": anios[0]["min_anio"] if anios else "?",
                "anio_max": anios[0]["max_anio"] if anios else "?",
            }
            self._log.info(f"Estadísticas listas | Géneros: {len(stats['generos'])} | Idiomas: {len(stats['idiomas'])}")
            return stats
        except OperationFailure as e:
            self._log.error(f"Error en aggregation pipeline: {e}")
            raise RuntimeError(f"Error al calcular estadísticas: {e}") from e
        except Exception as e:
            self._log.error(f"Error inesperado en estadísticas: {e}")
            raise