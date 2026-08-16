import streamlit as st
import os
import re
import datetime
from pathlib import Path
from dotenv import load_dotenv
import tiktoken
import io

# ReportLab Imports for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# LangChain Imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

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


# --- DOCUMENT PROCESSING & VECTORSTORE ---

def count_tokens(text):
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(tokenizer.encode(text))

def load_and_index_documents():
    pdf_dir = Path("./pdf_files_comercio_exterior")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No se encontraron regulaciones en PDF en el directorio './pdf_files_comercio_exterior'")
    
    loader = PyPDFDirectoryLoader(path="./pdf_files_comercio_exterior", silent_errors=True, recursive=False)
    raw_docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        length_function=count_tokens,
        separators=["\n\n", "\n", "Artículo", "ARTÍCULO", ".", ";", " ", ""]
    )
    split_docs = text_splitter.split_documents(raw_docs)
    
    embeddings = NVIDIAEmbeddings(model="NV-Embed-QA")
    st.session_state.vectorstore = FAISS.from_documents(split_docs, embeddings)


# --- PDF REPORT GENERATOR ---

def generate_pdf_report(report_text: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    story = []
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, leading=20, textColor='#1e3d59')
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=10, leading=14)
    
    story.append(Paragraph("Dictamen Técnico de Comercio Exterior y Cumplimiento Aduanero", title_style))
    story.append(Spacer(1, 12))
    
    for line in report_text.split('\n'):
        clean_line = line.replace('*', '').strip()
        if clean_line:
            story.append(Paragraph(clean_line, body_style))
            story.append(Spacer(1, 4))
            
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# --- SYSTEM PROMPT TEMPLATE ---

TRADE_SME_PROMPT_TEMPLATE = """
Eres un Asistente Especialista Senior (SME) en Comercio Exterior y Legislación Aduanera Mexicana, operando en los corredores comerciales EE.UU.-México (T-MEC / USMCA) y Unión Europea-México (TLCUEM).

Tu objetivo es guiar a importadores, exportadores y agentes aduanales respondiendo sus consultas operativas y normativas con base en el contexto legal proporcionado.

Instrucciones de Respuesta:
1. **Análisis Normativo Riguroso**: Utiliza la legislación oficial presentada en el contexto (Ley Aduanera, LIGIE, RGCE, CFF, Tratados Comerciales).
2. **Formato Financiero e Impuestos**: Si la consulta menciona montos o tasas de contribuciones, presenta los datos organizados formalmente con la notación y moneda correspondiente (ej. Valor en USD $, Contribuciones en MXN $, Tasas % e IGIE/DTA/IVA).
3. **Referencias Legales**: Cita explícitamente los Artículos, Leyes, Anexos o Reglas que respaldan tu respuesta.
4. **Opinión Experta & Disclaimers**: Proporciona tu opinión técnica e incluye las advertencias/deslindes de responsabilidad normativa aplicables.
5. **Matriz RAID OBLIGATORIA**: Finaliza SIEMPRE tu dictamen con una tabla Markdown de Matriz RAID (Riesgos, Acciones, Incidentes/Issues, Decisiones).

Contexto Legal Disponible:
{context}

Consulta del Usuario: {question}

Reporte Técnico Formal:
"""

# --- UI SETUP ---
st.set_page_config(layout="wide", page_title="Mexican Customs & Trade Assistant", page_icon="🛃")

st.header("🛃 Asistente Especialista en Comercio Exterior y Operaciones Aduaneras (México / USMCA / UE)")
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

if st.button("Cargar e Indizar Regulaciones PDF (FAISS VectorStore)"):
    with st.spinner("Indizando documentos normativos mediante NVIDIA Embeddings..."):
        try:
            start_time = datetime.datetime.now()
            load_and_index_documents()
            duration = (datetime.datetime.now() - start_time).total_seconds()
            st.success(f" Base documental cargada e indizada exitosamente en {duration:.2f} segundos.")
        except Exception as e:
            st.error(f"Error al procesar la carpeta de regulaciones: {str(e)}")

user_query = st.text_input("Ingrese su consulta operativa o normativa de comercio exterior:")

if user_query:
    is_greeting, greeting_resp, actual_q = st.session_state.greeting_handler.process_input(user_query)
    
    if is_greeting and greeting_resp:
        st.info(greeting_resp)
        
    if actual_q:
        if "vectorstore" not in st.session_state:
            st.warning("⚠️ Primero cargue los documentos ejecutando el botón 'Cargar e Indizar Regulaciones PDF'")
        else:
            try:
                with st.spinner("Consultando legislación y redactando respuesta técnica..."):
                    
                    # 1. Similarity Search over FAISS Store
                    relevant_docs = st.session_state.vectorstore.similarity_search(actual_q, k=4)
                    
                    docs_context = []
                    for i, doc in enumerate(relevant_docs, 1):
                        source_name = Path(doc.metadata.get('source', 'Desconocido')).name
                        docs_context.append(f"--- Documento Legal #{i} [{source_name}] ---\n{doc.page_content}")
                    
                    formatted_context = "\n\n".join(docs_context)
                    
                    # 2. Call NVIDIA Chat Model
                    llm = ChatNVIDIA(
                        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
                        temperature=0.3,
                        max_tokens=4000
                    )
                    
                    prompt = TRADE_SME_PROMPT_TEMPLATE.format(
                        context=formatted_context,
                        question=actual_q
                    )
                    
                    response = llm.invoke([HumanMessage(content=prompt)])
                    final_output = response.content
                    
                    # 3. Render Output
                    st.markdown("### 📜 Dictamen de Cumplimiento y Análisis Técnico")
                    st.markdown(final_output)
                    
                    # 4. Source Footnotes
                    with st.expander("📂 Ver extractos normativos consultados"):
                        for doc in relevant_docs:
                            source_name = Path(doc.metadata.get('source', 'Desconocido')).name
                            st.write(f"**{source_name}**")
                            st.caption(doc.page_content)
                    
                    # 5. Download PDF
                    pdf_bytes = generate_pdf_report(final_output)
                    st.download_button(
                        label="📄 Descargar Dictamen Técnico en PDF",
                        data=pdf_bytes,
                        file_name="Dictamen_Comercio_Exterior_SME.pdf",
                        mime="application/pdf"
                    )
                    
            except Exception as e:
                st.error(f"Error durante la consulta del modelo: {str(e)}")