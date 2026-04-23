"""
subir_modelo_hf.py — Sube el clasificador fine-tuneado a Hugging Face Hub
"""

import sys
from pathlib import Path
from getpass import getpass

from huggingface_hub import HfApi, login, whoami
from transformers import AutoTokenizer, AutoModelForSequenceClassification



# Configuracion—


USUARIO       = "nubiaebv"                   # usuario de Hugging Face
NOMBRE_REPO   = "musicbot-emotion-classifier"  # nombre del modelo en el Hub
RUTA_MODELO   = "models/clasificador_emocion"  # ruta local al modelo fine-tuneado
RUTA_CARD     = "MODEL_CARD.md"              # archivo de la Model Card
PRIVADO       = False                        # True si quieres que sea privado


HF_TOKEN      = ""                           # deja "" para que lo pida interactivo por consola.



# Lógica

def main():
    repo_id = f"{USUARIO}/{NOMBRE_REPO}"

    print("═" * 60)
    print(f"  Subiendo modelo a Hugging Face Hub")
    print(f"  Repo destino: {repo_id}")
    print(f"  Ruta local:   {RUTA_MODELO}")
    print("═" * 60)

    # 1. Validar ruta del modelo
    ruta_modelo = Path(RUTA_MODELO)
    if not ruta_modelo.exists():
        print(f"❌ No existe la ruta: {ruta_modelo.resolve()}")
        sys.exit(1)

    archivos_esperados = ["config.json"]
    faltantes = [a for a in archivos_esperados if not (ruta_modelo / a).exists()]
    if faltantes:
        print(f"❌ Faltan archivos en el modelo: {faltantes}")
        sys.exit(1)

    # Verificar pesos (formato nuevo o viejo)
    tiene_pesos = (ruta_modelo / "model.safetensors").exists() or \
                  (ruta_modelo / "pytorch_model.bin").exists()
    if not tiene_pesos:
        print("❌ No se encontró 'model.safetensors' ni 'pytorch_model.bin'")
        sys.exit(1)

    print(f"✅ Modelo encontrado en: {ruta_modelo.resolve()}")

    # 2. Login en HF
    token = HF_TOKEN or getpass("🔑 Pega tu token de Hugging Face (hf_...): ").strip()
    if not token:
        print("❌ Token vacío.")
        sys.exit(1)

    login(token=token, add_to_git_credential=False)
    info = whoami()
    print(f"✅ Autenticado como: {info['name']}")

    # 3. Cargar modelo y tokenizer localmente
    print("🔄 Cargando modelo y tokenizer para validar...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(ruta_modelo))
        model     = AutoModelForSequenceClassification.from_pretrained(str(ruta_modelo))
    except Exception as e:
        print(f"❌ Error cargando el modelo local: {e}")
        sys.exit(1)
    print(f"✅ Modelo cargado: {model.config.num_labels} etiquetas | "
          f"{sum(p.numel() for p in model.parameters()):,} parámetros")

    # 4. Crear repositorio y subir modelo + tokenizer
    api = HfApi()
    print(f"📦 Creando/verificando repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model",
                    private=PRIVADO, exist_ok=True)

    print("⬆️  Subiendo pesos del modelo...")
    model.push_to_hub(repo_id, token=token)

    print("⬆️  Subiendo tokenizer...")
    tokenizer.push_to_hub(repo_id, token=token)

    # 5. Subir Model Card
    if Path(RUTA_CARD).exists():
        print(f"⬆️  Subiendo Model Card desde {RUTA_CARD}...")
        api.upload_file(
            path_or_fileobj=RUTA_CARD,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
            token=token,
        )
    else:
        print(f"⚠️  No se encontró {RUTA_CARD}. Puedes subirla después manualmente.")

    # 6. Final
    url = f"https://huggingface.co/{repo_id}"
    print()
    print("═" * 60)
    print(f"✅ LISTO — modelo disponible en:")
    print(f"   {url}")
    print("═" * 60)
    print()
    print("Para usarlo desde código:")
    print(f'    from transformers import pipeline')
    print(f'    clf = pipeline("text-classification", model="{repo_id}")')
    print(f'    clf("Esta noche te escribo llorando")')


if __name__ == "__main__":
    main()
