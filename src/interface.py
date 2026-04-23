"""
src/interface.py — Capa de presentación (Sincronizada con style.css)
"""
from __future__ import annotations
import dash_bootstrap_components as dbc
from dash import dcc, html

SUGGESTIONS = [
    "¿Qué canciones hablan de desamor?",
    "Dame una canción de rock sobre libertad",
    "¿Cómo suena el reggaetón vs el pop?",
    "Recomiéndame algo alegre",
]


def _header() -> html.Header:
    return html.Header(
        className="app-header",
        children=[
            html.Div("🎵", className="logo-icon"),
            html.Span(["Músic", html.Span("Bot", style={"color": "var(--accent-amber)"})], className="logo-text"),
            html.Span("RAG + Fine-Tuning", className="header-badge"),
        ],
    )


def _empty_chat_state():
    return html.Div(className="chat-empty", children=[
        html.Div("🎶", className="empty-icon"),
        html.H3("¿Qué escuchamos hoy?"),
        html.P("Haz una pregunta sobre las letras de tu corpus musical.")
    ])


def _empty_sidebar_state():
    return html.Div(className="sidebar-empty", children=[
        html.Div("📂", className="empty-icon"),
        html.P("Aquí aparecerán los fragmentos de las canciones recuperadas.")
    ])


def render_message(role: str, content: str) -> html.Div:
    is_bot = role == "assistant"
    return html.Div(
        className=f"message-row {'user' if not is_bot else 'bot'}",
        children=[
            html.Div("MB" if is_bot else "TÚ", className=f"avatar {'bot' if is_bot else 'user'}"),
            html.Div(content, className=f"bubble {'bot' if is_bot else 'user'}")
        ]
    )


def render_chunks_panel(chunks: list[dict], label: str | None, conf: float | None) -> list:
    """Esta es la función que orquesta todo el panel lateral"""
    items = []
    if label and label != "general":
        items.append(html.Div(className="classifier-result", children=[
            html.Span("🏷 Género Detectado:", className="label"),
            html.Span(label.upper(), className="value"),
            html.Span(f"{conf:.0%}", className="confidence")
        ]))

    for i, c in enumerate(chunks):
        # Aquí llamamos a la función que crea cada tarjeta individual
        items.append(render_chunk_card(c, i + 1))
    return items


def render_chunk_card(chunk: dict, rank: int) -> html.Div:
    """Esta es la función que main.py o el panel necesitan para crear cada tarjeta"""
    return html.Div(className="chunk-card", children=[
        html.Div(className="chunk-card-header", children=[
            html.Span(f"#{rank}", className="chunk-rank"),
            html.Span(chunk.get("song", "S/T"), className="chunk-song-title"),
            html.Span(f"{chunk.get('score', 0):.1%}", className="chunk-score"),
        ]),
        html.Div(className="chunk-meta", children=[
            html.Span(chunk.get("artist", "Artista"), className="chunk-meta-tag artist"),
            html.Span(chunk.get("genre", "Género"), className="chunk-meta-tag genre"),
        ]),
        html.Div(chunk.get("text", ""), className="chunk-text"),
    ])


def build_layout() -> html.Div:
    return html.Div(className="app-wrapper", children=[
        _header(),
        html.Div(className="main-content", children=[
            # Izquierda: Chat
            html.Div(className="chat-column", children=[
                html.Div(id="chat-history", className="chat-history"),
                # Sugerencias
                html.Div(id="suggestions-bar", className="suggestions", children=[
                    html.Div(s, id={"type": "suggestion-pill", "index": i}, className="suggestion-pill")
                    for i, s in enumerate(SUGGESTIONS)
                ]),
                # Input
                html.Div(className="input-area", children=[
                    html.Button("🗑", id="clear-btn", className="clear-btn"),
                    dcc.Textarea(id="user-input", className="chat-textarea", placeholder="Escribe tu duda musical..."),
                    html.Button("➤", id="send-btn", className="send-btn")
                ])
            ]),
            # Derecha: Sidebar
            html.Div(className="sidebar", children=[
                html.Div(className="sidebar-header", children=[
                    html.H4("FUENTES (RAG)"),
                    html.Span("0", id="chunk-count-badge", className="count-badge")
                ]),
                html.Div(id="chunks-panel", className="sidebar-scroll")
            ])
        ]),
        # Almacenamiento
        dcc.Store(id="conversation-store", data=[]),
        dcc.Store(id="chunks-store", data=[]),
        dcc.Store(id="classifier-store", data={})
    ])