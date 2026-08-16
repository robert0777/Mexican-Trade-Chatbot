import streamlit as st
import os
import re
import datetime
from pathlib import Path
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

FAISS_INDEX_PATH = "./faiss_index_comercio"

# --- CACHED INITIALIZATIONS (NVIDIA & FAISS) ---

@st.cache_resource
def get_nvidia_llm():
    """Cache the NVIDIA Chat model instance across user sessions."""
    return ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.3,
        max_tokens=2500
    )

@st.cache_resource
def get_nvidia_embeddings():
    """Cache the NVIDIA Embeddings instance using a valid text embedding model."""
    return NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        model_type="query"
    )

@st.cache_resource
def get_vectorstore():
    """Loads FAISS index into memory once and caches it across re-runs."""
    embeddings = get_nvidia_embeddings()
    if Path(FAISS_INDEX_PATH).exists():
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def build_and_save_vectorstore():
    """Only called explicitly when re-indexing files."""
    pdf_dir = Path("./pdf_files_comercio_exterior")
    if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
        raise FileNotFoundError("No se encontraron regulaciones en PDF en './pdf_files_comercio_exterior'")
    
    loader = PyPDFDirectoryLoader(path="./pdf_files_comercio_exterior", silent_errors=True, recursive=False)
    raw_docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", "Artículo", "ARTÍCULO", ".", ";", " ", ""]
    )
    split_docs = text_splitter.split_documents(raw_docs)
    
    embeddings = get_nvidia_embeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    get_vectorstore.clear()  # Clear cache to force reload on next call
    return vectorstore

# --- GREETING HANDLER ---
class TradeGreetingHandler:
    def __init__(self):
        self.greetings = {
            'hola', 'buenos días', 'buenas tardes', 'buenas noches', 'saludos',
            'qué tal', 'cómo estás', 'mucho gusto', 'hey', 'hi', 'hello', 'buen día'
        }
        self.greeting_pattern = re.compile(
            '|'.join(r'\b{}\b'.format(re.escape(g)) for g in self.greetings),
            re.IGNORECASE
        )

    def normalize_text(self, text: str) -> str:
        text = text.lower()
        replacements = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u', 'ñ': 'n'}
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def extract_question(self, text: str) -> str:
        matches = list(self.greeting_pattern.finditer(text))
        if not matches:
            return text
        last_match_end = matches[-1].end()
        question = text[last_match_end:].strip(' ,.!?¿¡')
        return question if question else ""

    def process_input(self, user_input: str) -> tuple[bool, str | None, str | None]:
        if not user_input:
            return False, None, None
            
        normalized_input = self.normalize_text(user_input)
        is_greeting = bool(self.greeting_pattern.search(normalized_input))
        actual_question = self.extract_question(user_input)
        
        current_hour = datetime.datetime.now().hour
        if is_greeting:
            if current_hour < 12:
                greeting_response = "¡Buenos días! ¿En qué puedo asesorarle hoy en materia de comercio exterior, clasificación arancelaria o cumplimiento T-MEC?"
            elif current_hour < 18:
                greeting_response = "¡Buenas tardes! ¿En qué puedo asesorarle hoy en materia de comercio exterior, clasificación arancelaria o cumplimiento T-MEC?"
            else:
                greeting_response = "¡Buenas noches! ¿En qué puedo asesorarle hoy en materia de comercio exterior, clasificación arancelaria o cumplimiento T-MEC?"
        else:
            greeting_response = None
            
        return is_greeting, greeting_response, actual_question

# --- SYSTEM PROMPT TEMPLATE ---
TRADE_SME_PROMPT_TEMPLATE = """
Eres un Asistente Especialista Senior (SME) en Comercio Exterior y Legislación Aduanera Mexicana, operando en los corredores comerciales EE.UU.-México (T-MEC / USMCA).

Tu objetivo es guiar a importadores, exportadores y agentes aduanales respondiendo sus consultas operativas y normativas con base en el contexto legal proporcionado.

Instrucciones de Respuesta:
1. **Análisis Normativo Riguroso**: Utiliza la legislación oficial presentada en el contexto (Ley Aduanera, LIGIE, RGCE, CFF, Tratados Comerciales).
2. **Formato Financiero e Impuestos**: Si la consulta menciona montos o tasas de contribuciones, presenta los datos organizados formalmente con la notación y moneda correspondiente (ej. Valor en USD $, Contribuciones en MXN $, Tasas % e IGIE/DTA/IVA).
3. **Referencias Legales**: Cita explícitamente los Artículos, Leyes, Anexos o Reglas que respaldan tu respuesta.
4. **Opinión Experta & Disclaimers**: Proporciona tu opinión técnica e includes las advertencias/deslindes de responsabilidad normativa aplicables.
5. **Matriz RAID OBLIGATORIA**: Finaliza SIEMPRE tu dictamen con una tabla Markdown de Matriz RAID (Riesgos, Acciones, Incidentes/Issues, Decisiones).

Contexto Legal Disponible:
{context}

Consulta del Usuario: {question}

Reporte Técnico Formal:
"""

# --- UI SETUP ---
st.set_page_config(layout="wide", page_title="Mexican Customs & Trade Assistant", page_icon="🛃")

st.header("🛃 Asistente Especialista en Comercio Exterior y Operaciones Aduaneras (México - USMCA)")
st.markdown("""
Plataforma para la consulta de legislación aduanera, verificación de Reglas de Origen T-MEC, análisis normativo 
y generación de reportes de cumplimiento e impacto operativo.
""")

with st.sidebar:
    st.markdown("### 🛃 Trade SME Chatbot")
    st.markdown("---")
    st.markdown("### 👤 Autor")
    st.markdown("**Dr. Robert Hernández Martínez**")
    st.markdown("""
        <a href="https://chomchom216.medium.com/" target="_blank">📝 Artículos en Medium</a><br>
        <a href="https://unam1.academia.edu/Robert_Hernandez_Martinez" target="_blank">🎓 Publicaciones Académicas</a><br>
        <a href="https://www.credly.com/users/robert-hernandez.89bffe7b" target="_blank">🏆 Credenciales</a><br>
        <a href="https://github.com/robert0777" target="_blank">🐙 GitHub</a><br>
        <a href="mailto:robert@actuariayfinanzas.net">📧 Contacto</a>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("© 2026 Mexican Customs AI Chatbot")

if 'greeting_handler' not in st.session_state:
    st.session_state.greeting_handler = TradeGreetingHandler()

# Load vectorstore from cache
vectorstore = get_vectorstore()

if st.button("Cargar e Indizar Regulaciones PDF (FAISS VectorStore)"):
    with st.spinner("Indizando/Cargando documentos normativos..."):
        try:
            start_time = datetime.datetime.now()
            vectorstore = build_and_save_vectorstore()
            duration = (datetime.datetime.now() - start_time).total_seconds()
            st.success(f"Base documental cargada exitosamente en {duration:.2f} segundos.")
        except Exception as e:
            st.error(f"Error al procesar la carpeta de regulaciones: {str(e)}")

user_query = st.text_input("Ingrese su consulta operativa o normativa de comercio exterior:")

if user_query:
    is_greeting, greeting_resp, actual_q = st.session_state.greeting_handler.process_input(user_query)
    
    if is_greeting and greeting_resp:
        st.info(greeting_resp)
        
    if actual_q:
        if vectorstore is None:
            st.warning("⚠️ Primero cargue los documentos ejecutando el botón 'Cargar e Indizar Regulaciones PDF'")
        else:
            try:
                with st.status("Procesando dictamen técnico...", expanded=True) as status:
                    st.write("🔍 Consultando índice vectorial FAISS...")
                    relevant_docs = vectorstore.similarity_search(actual_q, k=3)
                    
                    docs_context = [
                        f"--- Documento Legal #{i} [{Path(doc.metadata.get('source', 'Desconocido')).name}] ---\n{doc.page_content}"
                        for i, doc in enumerate(relevant_docs, 1)
                    ]
                    formatted_context = "\n\n".join(docs_context)
                    
                    st.write("🤖 Generando respuesta con NVIDIA Nemotron...")
                    llm = get_nvidia_llm()
                    prompt = TRADE_SME_PROMPT_TEMPLATE.format(context=formatted_context, question=actual_q)
                    
                    response_container = st.empty()
                    chunks = []
                    for chunk in llm.stream([HumanMessage(content=prompt)]):
                        chunks.append(chunk.content)
                        response_container.markdown("".join(chunks))
                    
                    final_output = "".join(chunks)
                    status.update(label="Dictamen completado", state="complete", expanded=False)
                
                # Footnotes / Referenced Extracts
                with st.expander("📂 Ver extractos normativos consultados"):
                    for doc in relevant_docs:
                        source_name = Path(doc.metadata.get('source', 'Desconocido')).name
                        st.write(f"**{source_name}**")
                        st.caption(doc.page_content)
                    
            except Exception as e:
                st.error(f"Error durante la consulta del modelo: {str(e)}")