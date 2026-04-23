# USO DE INTELIGENCIA ARTIFICIAL

> ** Documentación del uso de IA**  
> Este archivo documenta el uso de herramientas
> de Inteligencia Artificial durante el desarrollo de este proyecto.

---

## 1. Herramientas de IA Utilizadas

| Herramienta    | Tareas en las que se usó                                                          |
|----------------|-----------------------------------------------------------------------------------|
| Claude         | _Estructuración de Dash, control de excepciones y análisis de métricas._          |
| Gemini         | _Consulta de redacción de mensajes de commit y generación de preguntas de prueba_ |
| GitHub Copilot | _Sugerencias de estructuras de documentación_                                     |

---
## 2. Ejemplos de Prompts Utilizados

### Prompt 1
**Herramienta:** _Claude_  
**Contexto / tarea:** _ Diseñar una arquitectura modular en Python con Plotly Dash para el chatbot musical._
```
Rol: Ingeniero de Software senior especializado en arquitecturas modulares con Python y experto en Plotly Dash.
Tarea: Generar una estructura limpia (Clean Architecture) separando interfaz, estilos y lógica en archivos independientes:
- assets/style.css → estilos visuales
- interface.py → layout con Dash Chat Component y Bootstrap
- logic.py → recuperación de datos (RAG)
- app.py → orquestador que une lógica e interfaz
Requisitos: interfaz con citas explícitas, panel lateral dinámico para chunks, uso de dash-bootstrap-components, servidor con debug=False.

```

**Resultado obtenido:** _Se creó una estructura modular base clara, con separación de responsabilidades y fácil de mantener._  

---
### Prompt 2
**Herramienta:** _Claude_  
**Contexto / tarea:** _Configurar PyTorch con soporte CUDA para aprovechar la GPU NVIDIA GTX en el proyecto._
```
El motivo por el cual Python te sigue diciendo cpu es meramente un tema de software: tienes instalado el "paquete básico" de PyTorch que no sabe cómo hablar con esa tarjeta.
Cómo activarla ahora mismo
Sigue estos pasos en el orden exacto dentro de tu terminal de VS Code (o donde estés corriendo el proyecto):
Borra la versión actual de Torch (que solo usa CPU):

pip uninstall torch torchvision torchaudio -y

Instala la versión con soporte CUDA:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```
**Resultado obtenido:** _Se logró activar la GPU NVIDIA GTX para el proyecto, permitiendo entrenar el modelo fine-tuneado con mayor velocidad y procesar lotes de datos más grandes en la laptop ASUS TUF._

---

### Prompt 3
**Herramienta:** _Claude_  
**Contexto / tarea:** _Analizar los resultados del fine-tuning del clasificador musical y recomendar el mejor punto de entrenamiento._
```
El mejor punto fue época 6:
Época 5: Val Loss 0.848 | F1 0.724
Época 6: Val Loss 0.858 | F1 0.738 ← mejor F1
Época 7: Val Loss 0.888 | F1 0.723 ← F1 baja
El EarlyStopping se activó correctamente en época 7 — el modelo guardado es el de época 6 con F1 0.738.
Comparación completa de todos los entrenamientos:
- Sin class weights (3,343 muestras): F1 = 0.558
- Con class weights (3,343 muestras): F1 = 0.770
- Dataset balanceado + modelo (2,945 muestras): F1 = 0.802 ← mejor
- 3 niveles combinados (2,063 muestras): F1 = 0.764
- Actual (keywords + modelo separado, 3,557 muestras): F1 = 0.738

```

**Resultado obtenido:** _La IA recomendó aceptar el F1 = 0.738 como resultado sólido, señalando que el dataset balanceado previo (F1 = 0.802) no podía reproducirse exactamente. Se documentó la comparación entre entrenamientos y se justificó la decisión de continuar con el modelo de época 6._

---

## 3. Reflexión sobre el Aprendizaje

_El uso de asistentes de IA (Claude, Gemini y Copilot) optimizó el desarrollo al estructurar la arquitectura modular, refinar el pipeline RAG y agilizar la documentación del código. Estas herramientas permitieron resolver bloqueos técnicos complejos y mejorar la claridad de la interfaz, actuando como un motor de productividad para el equipo._

_Sin embargo, el criterio de los integrantes del equipo de trabajo prevaleció en las decisiones críticas, como la estrategia de chunking y el análisis de métricas. Cada salida fue validada, asegurando que la IA funcionara como un apoyo técnico bajo la supervisión y responsabilidad final del equipo._

---

## 4. Modificaciones al Código / Análisis Generado por IA

- **Modificación 1:** _Se adaptaron los prompts sugeridos por IA para alinearlos con la personalidad definida del chatbot (experto en géneros musicales)._
- **Modificación 2:** _Se añadieron validaciones en el clasificador para evitar fugas de datos y garantizar reproducibilidad._
- **Modificación 3:** _Se ajustó el pipeline RAG para comparar chunking por canción completa vs. por estrofa._

---
*Última actualización: Jueves 23 de abril del 2026*