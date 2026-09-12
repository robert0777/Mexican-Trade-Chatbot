# 🛒 Asistente Retail AI - Chatbot Inteligente para Retail

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://asistente-inteligente-ventas-y-soporte-retail.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange.svg)
![OpenRouter](https://img.shields.io/badge/OpenRouter-API-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Aplicación web en vivo:** [🌐 https://asistente-inteligente-ventas-y-soporte-retail.streamlit.app/](https://asistente-inteligente-ventas-y-soporte-retail.streamlit.app/)

## 📋 Descripción

**Asistente Retail AI** es una solución conversacional basada en **IA Generativa y RAG (Retrieval-Augmented Generation)** diseñada para el sector retail mexicano. El sistema permite realizar consultas en lenguaje natural sobre catálogos de productos, políticas comerciales, garantías, manuales de gestión de pedidos y términos de venta.

Esta versión utiliza la arquitectura de modelos LLM disponible en **OpenRouter** incorporando un mecanismo de **fallback multi-modelo streaming** (con soporte para modelos gratuitos como `minimax/minimax-m3:free`, `google/gemma-4-31b:free`, `cohere/north-mini-code:free` y `openrouter/free-models-router`), detección inteligente de saludos contextuales (`GreetingHandler`) según la hora local, y scoring heurístico de relevancia de chunks con truncado de contexto optimizado mediante `tiktoken`.

---

## ✨ Características Principales

- 🤖 **Múltiples Modelos LLM vía OpenRouter con Fallback:** Intenta llamadas en streaming a través de una lista de modelos gratuitos de alta capacidad (`minimax/minimax-m3:free`, `google/gemma-4-31b:free`, `cohere/north-mini-code:free`, `openrouter/free-models-router`). Si un modelo falla, el sistema conmuta automáticamente al siguiente.
- 💬 **Manejador Inteligente de Saludos (`GreetingHandler`):** Filtra y procesa saludos en español ("hola", "buenos días", "buenas noches"), respondiendo dinámicamente según la hora local (mañana, tarde, noche) sin realizar llamadas innecesarias a la API ni procesar la base vectorial si no existe una pregunta técnica asociada.
- 📚 **Carga y Normalización Automática de Documentos PDF:** Extrae y normaliza texto comercial en español desde el directorio `./pdf_files_retail/`, reemplazando abreviaturas comunes (ej. *D.* -> *Doctor*, *Dra.* -> *Doctora*) y estandarizando espacios y caracteres.
- 🎯 **Algoritmo de Scoring & Relevancia de Chunks:** Utiliza `tiktoken` (modelo `gpt-3.5-turbo`) para calcular la superposición léxica y ajustar la ventana de contexto sin exceder el límite seguro de tokens (`max_total_tokens=6000`).
- 🔄 **Respuestas en Streaming & Fuentes Citadas:** Presenta las respuestas generadas en tiempo real mediante respuestas en streaming (`st.write_stream`), desglosando y deduplicando las fuentes consultadas en un menú desplegable (`expander`) para garantizar transparencia e inspección detallada.
- 🎨 **Interfaz Personalizada en Streamlit:** Diseño limpio con barra lateral interactiva, branding del autor, enlaces a artículos en Medium, publicaciones académicas y credenciales profesionales.

---

## 🏗️ Estructura del Proyecto

```text
.
├── app_retail 1.0.py              # Código fuente principal de la aplicación Streamlit
├── requirements.txt               # Lista de dependencias del proyecto
├── README.md                      # Documentación completa del proyecto
├── retail-icon.svg                # Icono vectorial de la aplicación
├── Data Flow Diagram.jpg          # Diagrama de arquitectura RAG y flujo de interacción
└── pdf_files_retail/              # Directorio con los documentos comerciales en PDF
    ├── Catálogo de Productos 2022_Comercializadora SECTH.pdf
    ├── Catálogo de Productos y Servicios_CLOUD Comercializadora.pdf
    ├── Generación de Pedidos Seguimiento Manual y Automático_Aspel_Amazon.pdf
    ├── Gestión de Pedidos y Distribución_Manual de Consulta.pdf
    ├── Política de Devolución y Garantía 2025_Syscom.pdf
    ├── Política de Venta y Devoluciones_Grupo Biomaster.pdf
    └── Términos y Condiciones Cliente Final_Transbel.pdf
```

---

## 📊 Arquitectura del Sistema y Flujo de Datos

El flujo de procesamiento del **Asistente Retail AI** combina un preprocesamiento contextual con un pipeline RAG resiliente:

```text
[Usuario: Ingresa consulta en Streamlit]
                   │
                   ▼
       [GreetingHandler.process_input]
                   │
         ┌─────────┴─────────┐
         │ ¿Se detectó       │
         │   saludo?         │
         └────┬─────────┬────┘
           Sí │         │ No
              ▼         │
   [Imprime saludo según]│
        [hora local]    │
              │         │
              └────┬────┘
                   ▼
         ┌───────────────────┐
         │ ¿Existe pregunta  │
         │   técnica?        │
         └────┬─────────┬────┘
           No │         │ Sí
              ▼         ▼
  [Despliega st.info  ┌───────────────────┐
   con menú de temas] │ ¿Documentos en    │
                      │ session_state?    │
                      └────┬─────────┬────┘
                        No │         │ Sí
                           ▼         ▼
               [Muestra warning:  [select_relevant_chunks
                Cargar primero]    & scoring léxico]
                                             │
                                             ▼
                                  [Construye contexto
                                  unificado por fuente]
                                             │
                                             ▼
                                  [Truncate context:
                                  Límite tokens tiktoken]
                                             │
                                             ▼
                                  [generate_completion_with_fallback:
                                  OpenRouter API]
                                             │
                                    ┌────────┴────────┐
                                    │ ¿Respuesta     │
                                    │  exitosa?       │
                                    └────┬───────┬────┘
                                      Sí │       │ Error
                                         ▼       ▼
                            [Muestra respuesta   [Prueba siguiente
                             streaming y         modelo en
                             expanders fuentes]  FREE_MODELS]
```

### Explicación del Flujo:
1. **Entrada de Usuario:** El usuario ingresa una consulta en la interfaz de Streamlit.
2. **Procesamiento de Saludos (`GreetingHandler`):** Se verifica si la entrada contiene saludos en español. Si está presente, el sistema genera una bienvenida dinámica acorde a la hora del día (mañana, tarde o noche).
3. **Validación de Pregunta Técnica:** 
   - Si **no** hay una pregunta técnica tras el saludo, el sistema muestra un mensaje informativo (`st.info`) con sugerencias de temas.
   - Si **sí** hay una pregunta técnica, el sistema procede al flujo RAG.
4. **Verificación de Documentos:** Se comprueba si los documentos están cargados en `st.session_state`. Si no han sido cargados, se solicita al usuario presionar el botón de carga.
5. **Retrieval & Scoring:** Se filtran los fragmentos de documentos (`chunks`) utilizando un cálculo de densidad léxica y superposición de términos (`calculate_chunk_relevance`).
6. **Construcción y Truncado de Contexto:** Se agrupan los extractos por archivo origen y se ajusta el contexto al límite máximo de tokens (`max_total_tokens=6000`) evaluado con `tiktoken`.
7. **Inferencia con Fallback en OpenRouter:** Se envía el prompt a la API de OpenRouter probando secuencialmente la lista de modelos gratuitos (`FREE_MODELS`). Si un modelo falla, conmuta automáticamente al siguiente.
8. **Renderizado de Resultados:** Se despliega la respuesta generada mediante streaming (`st.write_stream`) junto con el modelo activo utilizado y los acordeones desplegables (`st.expander`) con las fuentes consultadas.

---

## ⚙️ Deployment & Streamlit Setup

### 1. Repository Setup
Push the application repository to GitHub:
```bash
git clone https://github.com/robert0777/asistente-retail-ai.git
cd asistente-retail-ai
```

### 2. Streamlit Cloud Secrets Setup
In your deployed app dashboard on Streamlit Cloud, navigate to **Settings** -> **Secrets**, and store your OpenRouter API key:
```toml
OPENROUTER_API_KEY = "your_openrouter_api_key_here"
```

### 3. Add Target PDF Documents
Place official retail documentation PDF files into:
```text
./pdf_files_retail/
```

### 4. Application Execution Entry
Point Streamlit Cloud deployment to:
```text
app_retail 1.0.py
```

---

## 📦 Dependencias (`requirements.txt`)

```txt
streamlit>=1.30.0
langchain-core>=0.1.0
langchain-community>=0.0.20
langchain-nvidia-ai-endpoints>=0.1.0
langgraph>=0.0.20
openai>=1.0.0
tiktoken>=0.5.0
pypdf>=3.0.0
reportlab>=4.0.0
faiss-cpu>=1.7.4
```

---

## 👤 Autor

**Dr. Robert Hernández Martínez**  
*Consultor en Ciencia Actuarial, Finanzas, Modelación de Riesgos y IA Aplicada*

- 📧 **Correo Electrónico:** [robert@actuariayfinanzas.net](mailto:robert@actuariayfinanzas.net)
- 📝 **Medium:** [@chomchom216](https://chomchom216.medium.com/)
- 🎓 **Publicaciones Académicas:** [UNAM Academia](https://unam1.academia.edu/Robert_Hernandez_Martinez)
- 🏆 **Certificaciones:** [Credly Profile](https://www.credly.com/users/robert-hernandez.89bffe7b)
- 🐙 **GitHub:** [@robert0777](https://github.com/robert0777)

---

© 2026 Asistente Retail AI.
