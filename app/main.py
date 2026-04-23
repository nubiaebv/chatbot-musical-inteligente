"""
app/main.py — Orquestador Final de MúsicBot
"""

import sys
import threading
import webbrowser
import time
import logging
from dataclasses import asdict
from pathlib import Path

# Configurar rutas
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, ctx, ALL, no_update
from dash.exceptions import PreventUpdate

# Componentes locales
from src.interface import (
    build_layout, render_message, render_chunks_panel,
    _empty_chat_state, _empty_sidebar_state, SUGGESTIONS
)
from src.logic import engine
from app.config import APP_PORT, APP_DEBUG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar Dash
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="MúsicBot — N&P",
    assets_folder="assets",
    suppress_callback_exceptions=True
)

app.layout = build_layout()

# Inicializar motor en segundo plano
threading.Thread(target=lambda: engine.initialize(), daemon=True).start()


# ─── CALLBACK PRINCIPAL: ENVIAR MENSAJE ──────────────────────────────────────

@app.callback(
    [Output("conversation-store", "data"),
     Output("chunks-store", "data"),
     Output("classifier-store", "data"),
     Output("user-input", "value"),
     Output("suggestions-bar", "style")],
    [Input("send-btn", "n_clicks"),
     Input({"type": "suggestion-pill", "index": ALL}, "n_clicks")],
    [State("user-input", "value"),
     State("conversation-store", "data")],
    prevent_initial_call=True
)
def handle_send(n_btn, n_suggestions, user_text, conversation):
    trigger = ctx.triggered_id

    # Determinar si el mensaje viene del input o de una sugerencia
    if isinstance(trigger, dict) and trigger.get("type") == "suggestion-pill":
        text = SUGGESTIONS[trigger["index"]]
    else:
        text = (user_text or "").strip()

    if not text:
        raise PreventUpdate

    # Respuesta mientras carga el motor
    if not engine._initialized:
        new_conv = (conversation or []) + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": "⏳ Cargando modelos locales... espera un momento."}
        ]
        return new_conv, [], {}, "", {"display": "none"}

    # Llamada al RAG
    result = engine.chat(text, use_rag=True)

    # Actualizar historial de datos
    new_conv = (conversation or []) + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": result.answer}
    ]

    chunks_data = [asdict(c) for c in result.chunks]
    classifier_data = {"label": result.classifier_label, "conf": result.classifier_conf}

    return new_conv, chunks_data, classifier_data, "", {"display": "none"}


# ─── CALLBACK DE RENDERIZADO: ACTUALIZAR VISTA DEL CHAT ──────────────────────

@app.callback(
    Output("chat-history", "children"),
    Input("conversation-store", "data")
)
def update_chat_view(conversation):
    if not conversation:
        return [_empty_chat_state()]
    return [render_message(m["role"], m["content"]) for m in conversation]


# ─── CALLBACK DE RENDERIZADO: ACTUALIZAR PANEL LATERAL ───────────────────────

@app.callback(
    [Output("chunks-panel", "children"),
     Output("chunk-count-badge", "children")],
    [Input("chunks-store", "data"),
     Input("classifier-store", "data")]
)
def update_sidebar_view(chunks, classifier):
    if not chunks:
        return [_empty_sidebar_state()], "0"

    label = (classifier or {}).get("label")
    conf = (classifier or {}).get("conf")

    return render_chunks_panel(chunks, label, conf), str(len(chunks))


# ─── CALLBACK: LIMPIAR CHAT ──────────────────────────────────────────────────

@app.callback(
    [Output("conversation-store", "data", allow_duplicate=True),
     Output("chunks-store", "data", allow_duplicate=True),
     Output("classifier-store", "data", allow_duplicate=True),
     Output("suggestions-bar", "style", allow_duplicate=True)],
    Input("clear-btn", "n_clicks"),
    prevent_initial_call=True
)
def clear_chat(n):
    if n:
        engine.clear_history()
        return [], [], {}, {"display": "flex"}
    return no_update


# ─── ARRANQUE ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Abrir navegador automáticamente
    def open_b():
        time.sleep(2)
        webbrowser.open(f"http://127.0.0.1:{APP_PORT}")


    threading.Thread(target=open_b, daemon=True).start()
    app.run(port=APP_PORT, debug=APP_DEBUG)