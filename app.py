import streamlit as st
import os
import re
import datetime
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
import tiktoken

from openai import OpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# --- GREETING HANDLER FOR CUSTOMS ADVISORY ---
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

# --- UTILITIES & DOCUMENT PROCESSING ---
@lru_cache(maxsize=None)
def get_tokenizer():
    return tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text):
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))

def get_pdf_files(directory="./pdf_files_comercio_exterior"):
    pdf_dir = Path(directory)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No se encontraron regulaciones o documentos PDF en el directorio './pdf_files_comercio_exterior'")
    return pdf_files

def normalize_spanish_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('Art.', 'Artículo').replace('Fracc.', 'Fracción')
    return text.strip()

def load_trade_documents():
    if "documents" not in st.session_state:
        pdf_files = get_pdf_files()
        
        chunk_size = min(1000, max(400, 600 * (50 / len(pdf_files))))
        chunk_overlap = max(80, chunk_size // 8)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=count_tokens,
            separators=["\n\n", "\n", "Artículo", "ARTÍCULO", ".", ";", " ", ""]
        )
        
        loader = PyPDFDirectoryLoader(
            path="./pdf_files_comercio_exterior", 
            silent_errors=True,
            recursive=False
        )
        docs = loader.load()
        
        processed_docs = []
        for doc in docs:
            doc.page_content = normalize_spanish_text(doc.page_content)
            processed_docs.append(doc)
        
        st.session_state.documents = text_splitter.split_documents(processed_docs)

def calculate_chunk_relevance(chunk, question):
    question_words = set(question.lower().split())
    chunk_words = set(chunk.page_content.lower().split())
    word_overlap = len(question_words.intersection(chunk_words))
    length_factor = 1 / (len(chunk.page_content.split()) + 1)
    return word_overlap * (1 - length_factor)

def select_relevant_chunks(question, chunks, max_total_tokens=6000):
    prompt_tokens = count_tokens(question) + 800
    available_tokens = max_total_tokens - prompt_tokens
    
    scored_chunks = [(chunk, calculate_chunk_relevance(chunk, question)) for chunk in chunks]
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    selected_chunks = []
    used_documents = set()
    current_tokens = 0
    
    for chunk, score in scored_chunks:
        doc_name = Path(chunk.metadata['source']).name
        chunk_tokens = count_tokens(chunk.page_content)
        
        if doc_name not in used_documents and current_tokens + chunk_tokens <= available_tokens:
            selected_chunks.append(chunk)
            used_documents.add(doc_name)
            current_tokens += chunk_tokens
            
        if current_tokens >= available_tokens * 0.9:
            break
            
    return selected_chunks

# --- UI INITIALIZATION ---
st.set_page_config(layout="wide", page_title="Agentic AI Mexican Customs & Trade SME", page_icon="🛃")

st.header("🛃 Asistente Especialista en Comercio Exterior y Operaciones Aduaneras (México / USMCA / UE)")
st.markdown("""
Plataforma experta para la consulta de legislación aduanera, cálculo de contribuciones (IGIE, DTA, IVA), 
verificación de Reglas de Origen T-MEC y generación de reportes de cumplimiento e impacto normativo.
""")

# Sidebar Setup
with st.sidebar:
    st.markdown("### 🛃 Agentic Trade SME")
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
    st.caption("© 2026 Mexican Customs AI Agent")

# LLM Client Setup
try:
    nvidia_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
except Exception as e:
    st.error(f"Error al inicializar cliente NVIDIA API: {str(e)}")
    st.stop()

# SYSTEM PROMPT FOR CUSTOMS SME
TRADE_SME_PROMPT_TEMPLATE = """
Eres un Agente Especialista Senior (SME) en Comercio Exterior y Legislación Aduanera Mexicana, operando en los corredores comerciales EE.UU.-México (T-MEC / USMCA) y Unión Europea-México (TLCUEM).

Tu objetivo es guiar a importadores, exportadores y agentes aduanales en la toma de decisiones normativas y operativas.

Instrucciones de Respuesta:
1. **Análisis Normativo Riguroso**: Utiliza la legislación oficial presentada en las fuentes (Ley Aduanera, LIGIE, RGCE, CFF, Tratados Comerciales).
2. **Cálculos Financieros e Impuestos**: Si la consulta requiere costos o impuestos, presenta las cifras desglosadas formalmente utilizando la notación y moneda correspondiente (ej. Valor en Aduana en USD $, Contribuciones en MXN $, Tasas % e IGIE/DTA/IVA).
3. **Referencias Legales**: Cita explícitamente los Artículos, Leyes, Anexos o Reglas que respaldan la respuesta.
4. **Opinión Experta & Disclaimers**: Proporciona tu opinión técnica clara e incluye las advertencias/deslindes de responsabilidad normativa aplicables.
5. **Matriz RAID OBLIGATORIA**: Finaliza SIEMPRE tu informe con una tabla Markdown de Matriz RAID (Riesgos, Acciones, Incidentes/Issues, Decisiones).

Contexto Legal Disponible:
{context}

Consulta del Usuario: {question}

Reporte Técnico Formal:
"""

if 'greeting_handler' not in st.session_state:
    st.session_state.greeting_handler = TradeGreetingHandler()

# Interface Input
user_query = st.text_input("Ingrese su consulta operativa o normativa de comercio exterior:")

if st.button("Cargar y Procesar Regulaciones PDF"):
    with st.spinner("Procesando archivos y compilando índice normativo en ./pdf_files_comercio_exterior..."):
        try:
            start_time = datetime.datetime.now()
            load_trade_documents()
            duration = (datetime.datetime.now() - start_time).total_seconds()
            st.success(f" Base documental cargada exitosamente en {duration:.2f} segundos.")
        except Exception as e:
            st.error(f"Error al procesar la carpeta de regulaciones: {str(e)}")

if user_query:
    is_greeting, greeting_resp, actual_q = st.session_state.greeting_handler.process_input(user_query)
    
    if is_greeting and greeting_resp:
        st.info(greeting_resp)
        
    if actual_q:
        if "documents" in st.session_state:
            try:
                with st.spinner("Ejecutando análisis normativo y financiero..."):
                    selected_chunks = select_relevant_chunks(actual_q, st.session_state.documents)
                    
                    docs_used = {}
                    for chunk in selected_chunks:
                        doc_name = Path(chunk.metadata['source']).name
                        if doc_name not in docs_used:
                            docs_used[doc_name] = []
                        docs_used[doc_name].append(chunk.page_content)
                    
                    context_parts = [f"[Documento Legal: {doc}]\n" + "\n".join(texts) for doc, texts in docs_used.items()]
                    context = "\n\n".join(context_parts)
                    
                    completion = nvidia_client.chat.completions.create(
                        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
                        messages=[
                            {"role": "system", "content": "/think"},
                            {"role": "user", "content": TRADE_SME_PROMPT_TEMPLATE.format(context=context, question=actual_q)}
                        ],
                        temperature=0.4,
                        max_tokens=4000
                    )
                    
                    st.markdown("### 📜 Dictamen de Cumplimiento y Análisis Técnico")
                    st.write(completion.choices[0].message.content)
                    
                    st.markdown("---")
                    with st.expander("📂 Ver extractos normativos consultados"):
                        for doc_name, chunks in docs_used.items():
                            st.write(f"**{doc_name}**")
                            for c in chunks:
                                st.caption(c)
                                
            except Exception as e:
                st.error(f"Error durante el procesamiento: {str(e)}")
        else:
            st.warning("⚠️ Primero cargue los documentos ejecutando el botón 'Cargar y Procesar Regulaciones PDF'")