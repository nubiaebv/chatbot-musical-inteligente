"""
src/interface.py — # Gestiona la interfaz gráfica del bot,
incorporando un monitor dinámico para visualizar el estado de los procesos del sistema.

"""
from __future__ import annotations
import dash_bootstrap_components as dbc
from dash import dcc, html

SUGGESTIONS = [
    "¿Qué canciones hablan de desamor?",
    "Estoy muy feliz, recomiéndame algo alegre",
    "Necesito algo para llorar un rato",
    "Dame una canción romántica de amor",
]

EMOTION_EMOJI = {
    "alegria":   "😊",
    "tristeza":  "😢",
    "amor":      "❤️",
    "rabia":     "😠",
    "nostalgia": "🌅",
}


#  Header

def _header() -> html.Header:
    return html.Header(
        className="app-header",
        children=[
            html.Div("🎵", className="logo-icon"),
            html.Span(
                ["Músic", html.Span("Bot", style={"color": "var(--accent-amber)"})],
                className="logo-text"
            ),
            html.Span("RAG + Fine-Tuning", className="header-badge"),
            # Indicador de estado del sistema
            html.Span(
                id="system-status",
                children="⏳ Cargando...",
                style={
                    "marginLeft":  "auto",
                    "fontSize":    "12px",
                    "color":       "#f5a623",
                    "fontFamily":  "monospace",
                    "paddingRight":"16px",
                }
            ),
            # Intervalo para verificar estado cada 3 segundos
            dcc.Interval(id="status-interval", interval=3000, n_intervals=0),
        ],
    )


# Estados vacíos

def _empty_chat_state():
    return html.Div(className="chat-empty", children=[
        html.Div("🎶", className="empty-icon"),
        html.H3("¿Qué escuchamos hoy?"),
        html.P("Haz una pregunta y encuentra la canción ideal para cada momento.")
    ])


def _empty_sidebar_state():
    return html.Div(className="sidebar-empty", children=[
        html.Div("📂", className="empty-icon"),
        html.P("Fuentes de consulta: artista, canción y fragmentos de estrofas recuperadas.")
    ])


# Mensajes del chat

def render_message(role: str, content: str) -> html.Div:
    is_bot = role == "assistant"
    return html.Div(
        className=f"message-row {'user' if not is_bot else 'bot'}",
        children=[
            html.Div(
                "MB" if is_bot else "TÚ",
                className=f"avatar {'bot' if is_bot else 'user'}"
            ),
            html.Div(content, className=f"bubble {'bot' if is_bot else 'user'}")
        ]
    )


# Panel lateral de chunks

def render_chunks_panel(chunks: list[dict],
                        label: str | None,
                        conf:  float | None) -> list:
    # Orquesta el panel lateral con clasificador y tarjetas de chunks.
    items = []

    # Emoción detectada en la pregunta
    if label:
        emoji = EMOTION_EMOJI.get(label, "🎵")
        conf_str = f"{conf:.0%}" if conf else ""
        items.append(html.Div(
            className="classifier-result",
            children=[
                html.Span("🏷 Emoción detectada:", className="label"),
                html.Span(f"{emoji} {label.upper()}", className="value"),
                html.Span(conf_str, className="confidence"),
            ]
        ))

    for i, c in enumerate(chunks):
        items.append(render_chunk_card(c, i + 1))

    return items


def render_chunk_card(chunk: dict, rank: int) -> html.Div:
    # Tarjeta individual de chunk con emoción del clasificador fine-tuneado.
    emocion = chunk.get("emotion", "")
    emoji   = EMOTION_EMOJI.get(emocion, "🎵")

    return html.Div(className="chunk-card", children=[
        html.Div(className="chunk-card-header", children=[
            html.Span(f"#{rank}",                       className="chunk-rank"),
            html.Span(chunk.get("song",  "S/T"),        className="chunk-song-title"),
            html.Span(f"{chunk.get('score', 0):.1%}",   className="chunk-score"),
        ]),
        html.Div(className="chunk-meta", children=[
            html.Span(chunk.get("artist", "Artista"),   className="chunk-meta-tag artist"),
            html.Span(chunk.get("genre",  "Género"),    className="chunk-meta-tag genre"),
            html.Span(f"{emoji} {emocion}",             className="chunk-meta-tag")
            if emocion else html.Span(""),
        ]),
        html.Div(chunk.get("text", ""), className="chunk-text"),
    ])


# Layout principal

def build_layout() -> html.Div:
    return html.Div(className="app-wrapper", children=[
        _header(),
        html.Div(className="main-content", children=[

            # Columna del chat
            html.Div(className="chat-column", children=[
                html.Div(id="chat-history", className="chat-history"),

                # Sugerencias
                html.Div(
                    id="suggestions-bar",
                    className="suggestions",
                    children=[
                        html.Div(
                            s,
                            id={"type": "suggestion-pill", "index": i},
                            className="suggestion-pill"
                        )
                        for i, s in enumerate(SUGGESTIONS)
                    ]
                ),

                # Input + botones
                html.Div(className="input-area", children=[
                    html.Button("🗑", id="clear-btn", className="clear-btn"),
                    dcc.Textarea(
                        id="user-input",
                        className="chat-textarea",
                        placeholder="Escribe tu pregunta musical... (Enter para enviar)"
                    ),
                    html.Button("➤", id="send-btn", className="send-btn"),
                ]),
            ]),

            # Sidebar de contexto
            html.Div(className="sidebar", children=[
                html.Div(className="sidebar-header", children=[
                    html.H4("CONTEXTO MUSICAL"),
                    html.Span("0", id="chunk-count-badge", className="count-badge"),
                ]),
                html.Div(id="chunks-panel", className="sidebar-scroll"),
            ]),
        ]),

        # Almacenamiento en cliente
        dcc.Store(id="conversation-store", data=[]),
        dcc.Store(id="chunks-store",       data=[]),
        dcc.Store(id="classifier-store",   data={}),
    ])
