import streamlit as st
import os
import time
import datetime
import re
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
import tiktoken

from openai import OpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Target Directory for Trade PDFs
DATA_DIR = "./pdf_files_comercio_exterior"

class GreetingHandler:
    def __init__(self):
        self.greetings = {
            'hola', 'buenos días', 'buenas tardes', 'buenas noches', 'saludos',
            'qué tal', 'cómo estás', 'como estas', 'qué onda', 'que onda',
            'cómo le va', 'cómo está usted', 'mucho gusto', 'bonito día', 'buen día',
            'hey', 'hi', 'hello', 'cómo te llamas', 'me llamo', 'mi nombre es'
        }
        
        self.time_based_responses = {
            'morning': "¡Buenos días! ¿En qué puedo ayudarte respecto a comercio exterior, legislación aduanera y T-MEC?",
            'afternoon': "¡Buenas tardes! ¿En qué puedo ayudarte respecto a comercio exterior, legislación aduanera y T-MEC?",
            'evening': "¡Buenas noches! ¿En qué puedo ayudarte respecto a comercio exterior, legislación aduanera y T-MEC?"
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
                greeting_response = self.time_based_responses['morning']
            elif current_hour < 18:
                greeting_response = self.time_based_responses['afternoon']
            else:
                greeting_response = self.time_based_responses['evening']
        else:
            greeting_response = None
            
        return is_greeting, greeting_response, actual_question

@lru_cache(maxsize=None)
def get_tokenizer():
    return tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text):
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))

def get_pdf_files(directory=DATA_DIR):
    pdf_dir = Path(directory)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No se encontraron archivos PDF en el directorio: {directory}")
    return pdf_files

def normalize_trade_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_documents():
    if "documents" not in st.session_state:
        pdf_files = get_pdf_files()
        
        chunk_size = min(800, max(300, int(500 * (50 / len(pdf_files)))))
        chunk_overlap = max(50, chunk_size // 10)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=count_tokens,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        )
        
        loader = PyPDFDirectoryLoader(
            path=DATA_DIR, 
            silent_errors=True,
            recursive=False
        )
        docs = loader.load()
        
        processed_docs = []
        for doc in docs:
            normalized_text = normalize_trade_text(doc.page_content)
            doc.page_content = normalized_text
            processed_docs.append(doc)
        
        st.session_state.documents = text_splitter.split_documents(processed_docs)

def calculate_chunk_relevance(chunk, question):
    question_words = set(question.lower().split())
    chunk_words = set(chunk.page_content.lower().split())
    word_overlap = len(question_words.intersection(chunk_words))
    length_factor = 1 / (len(chunk.page_content.split()) + 1)
    return word_overlap * (1 - length_factor)

def select_relevant_chunks(question, chunks, max_total_tokens=6000):
    prompt_tokens = count_tokens(question) + 500
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

def truncate_context(context, max_tokens=6000):
    tokens = count_tokens(context)
    if tokens > max_tokens:
        lines = context.split('\n')
        truncated_context = []
        current_tokens = 0
        for line in lines:
            line_tokens = count_tokens(line)
            if current_tokens + line_tokens <= max_tokens:
                truncated_context.append(line)
                current_tokens += line_tokens
            else:
                break
        return '\n'.join(truncated_context)
    return context

# UI Setup
st.set_page_config(layout="wide", page_title="Asesor AI de Comercio Exterior y Aduanas", page_icon="🌐")
st.header("Asistente AI Especializado en Legislación Aduanera y T-MEC")
st.markdown("Sistema de consulta para importadores, exportadores y agentes aduanales sobre la normativa del corredor US-MX.")





# Sidebar content
with st.sidebar:
    # App Logo and Title
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("ai-advisor-icon.svg", width=50)
    with col2:
        st.markdown('<p class="sidebar-app-name">AI Chatbot Asesor de Normativa Aduanera Mexicana</p>', unsafe_allow_html=True)
    
    
    # About Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📖 About this App")
    st.write("Analizador automatizado de normativa aduanera mexicana, Reglas Generales de Comercio Exterior y T-MEC.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Author Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 👤 Author")
    st.markdown("**Dr. Robert Hernández Martínez**")


    
    # Contact Links
   
    st.markdown(
    """
    <div class="sidebar-link-container">
        <div><a href="https://chomchom216.medium.com/" class="sidebar-link">📝 Articles on Medium</a></div>
        <div><a href="https://unam1.academia.edu/Robert_Hernandez_Martinez" class="sidebar-link">🎓 Academic Publications</a></div>
        <div><a href="https://www.credly.com/users/robert-hernandez.89bffe7b" class="sidebar-link">🏆 Credentials</a></div>
        <div><a href="https://github.com/robert0777" class="sidebar-link">🐙 GitHub</a></div>
        <div><a href="mailto:robert@actuariayfinanzas.net" class="sidebar-link">📧 Contact</a></div>
    </div>
""",
    unsafe_allow_html=True,
)
    
    
    
    # Footer
    st.markdown("""
        <div style="position: fixed; bottom: 0; padding: 1rem; text-align: center; font-size: 0.8rem; color: #6B7280;">
            © 2026 Asistente AI Especializado en Legislación Aduanera y T-MEC
        </div>
    """, unsafe_allow_html=True)










# Client Initialization
try:
    nvidia_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
except Exception as e:
    st.error(f"Error al inicializar el cliente NVIDIA NIMS: {str(e)}")
    st.stop()

# System Prompt specialized in Trade
TRADE_PROMPT_TEMPLATE = """
Eres un Asistente Experto (SME) en Comercio Exterior y Legislación Aduanera Mexicana/EE.UU.
Tu objetivo es ayudar a importadores, exportadores y agentes aduanales a realizar análisis de cumplimiento.

Instrucciones de Respuesta:
1. Analiza exhaustivamente la consulta usando únicamente los fragmentos proporcionados.
2. Presenta todas las cifras, tasas tributarias (IGE, IVA, DTA), valoraciones y métricas en sus formatos correspondientes (USD, MXN, %, etc.).
3. Incluye referencias normativas específicas para cada afirmación (artículos de Ley Aduanera, Anexo 22, CFF, Capítulos T-MEC, etc.).
4. Ofrece una opinión técnica sustentada y agrega un aviso de exención de responsabilidad (Disclaimer) al inicio o final.
5. Finaliza OBLIGATORIAMENTE con una tabla RAID (Risks, Actions, Issues, Decisions) formateada en Markdown.

Extractos Normativos Disponibles:
{context}

Consulta del Usuario: {question}

Informe de Cumplimiento:
"""

if 'greeting_handler' not in st.session_state:
    st.session_state.greeting_handler = GreetingHandler()

prompt1 = st.text_input(
    "Introduzca su consulta sobre comercio exterior:",
    placeholder="Ej: ¿Cuáles son las reglas de origen aplicables bajo el T-MEC para el sector automotriz?"
)

if st.button("Cargar y Procesar Documentos Aduaneros"):
    with st.spinner('Cargando expediente de normativa en pdf_files_comercio_exterior...'):
        try:
            start_time = time.process_time()
            load_documents()
            processing_time = time.process_time() - start_time
            st.success(f"📚 Expediente cargado exitosamente en {processing_time:.2f} segundos.")
        except Exception as e:
            st.error(f"Error al cargar los documentos: {str(e)}")

if prompt1:
    is_greeting, greeting_response, actual_question = st.session_state.greeting_handler.process_input(prompt1)
    
    if is_greeting:
        st.write(greeting_response)
        
    if actual_question:
        if "documents" in st.session_state:
            try:
                with st.spinner('Generando reporte de cumplimiento aduanero...'):
                    start = time.process_time()
                    selected_chunks = select_relevant_chunks(actual_question, st.session_state.documents)
                    
                    docs_used = {}
                    for chunk in selected_chunks:
                        doc_name = Path(chunk.metadata['source']).name
                        if doc_name not in docs_used:
                            docs_used[doc_name] = []
                        docs_used[doc_name].append(chunk.page_content)
                    
                    context_parts = []
                    for doc_name, contents in docs_used.items():
                        context_parts.append(f"[Documento Legal: {doc_name}]\n" + "\n".join(contents))
                    
                    context = truncate_context("\n\n".join(context_parts))
                    
                    completion = nvidia_client.chat.completions.create(
                        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
                        messages=[
                            {"role": "system", "content": "/think"},
                            {
                                "role": "user",
                                "content": TRADE_PROMPT_TEMPLATE.format(
                                    context=context,
                                    question=actual_question
                                )
                            }
                        ],
                        temperature=0.3,
                        top_p=0.95,
                        max_tokens=4000,
                        stream=False
                    )
                    
                    st.write("📋 Reporte Técnico de Cumplimiento:")
                    st.write(completion.choices[0].message.content)
                    st.info(f"⏱️ Tiempo de respuesta: {time.process_time() - start:.2f} segundos")
                    
                    with st.expander("Ver Fuentes Consultadas"):
                        for doc_name, doc_chunks in docs_used.items():
                            st.write(f"**{doc_name}**")
                            for chunk in doc_chunks:
                                st.caption(chunk)
            except Exception as e:
                st.error(f"Error en la consulta: {str(e)}")
        else:
            st.warning("⚠️ Por favor cargue los documentos antes de realizar la consulta.")