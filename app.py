from itertools import chain
import pdfplumber
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from docx import Document
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="QueryMind",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Clean UI (remove avatars)
# -----------------------------
st.markdown("""
<style>
[data-testid="stChatMessageAvatar"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.title("QueryMind")
st.caption("Ask questions from your documents")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Upload Documents")

    files = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "docx", "csv"],
        accept_multiple_files=True
    )

    url = st.text_input("Or enter website URL")

    st.markdown("---")
    temperature = st.slider("Creativity", 0.0, 1.0, 0.3)

# -----------------------------
# Helper Functions (IMPORTANT: BEFORE USE)
# -----------------------------
def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text

def read_txt(file):
    return file.read().decode("utf-8")

def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def read_url(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        return soup.get_text(separator="\n")
    except:
        return ""

# -----------------------------
# Chat History
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Read All Inputs
# -----------------------------
all_text = ""

if files:
    for file in files:
        if file.name.endswith(".pdf"):
            all_text += read_pdf(file)
        elif file.name.endswith(".txt"):
            all_text += read_txt(file)
        elif file.name.endswith(".docx"):
            all_text += read_docx(file)
        elif file.name.endswith(".csv"):
            all_text += read_csv(file)

if url:
    all_text += read_url(url)

# Stop if nothing uploaded
if not all_text.strip():
    st.info("Please upload files or enter a URL to start chatting.")
    st.stop()

# -----------------------------
# Text Splitting
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)

chunks = text_splitter.split_text(all_text)

# -----------------------------
# Embeddings + Vector Store
# -----------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

vector_store = FAISS.from_texts(chunks, embeddings)

# -----------------------------
# Retriever
# -----------------------------
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

# -----------------------------
# LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=temperature,
    max_tokens=1000,
    openai_api_key=OPENAI_API_KEY
)

# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant answering questions based ONLY on provided context.\n\n"
     "Context:\n{context}"),
    ("human", "{question}")
])

# -----------------------------
# Chain
# -----------------------------
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -----------------------------
# Chat UI
# -----------------------------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

user_question = st.chat_input("Ask something about your data...")

if user_question:
    st.session_state.chat_history.append(("user", user_question))
    with st.chat_message("user"):
        st.markdown(user_question)

    response = chain.invoke(user_question)

    st.session_state.chat_history.append(("assistant", response))
    with st.chat_message("assistant"):
        st.markdown(response)
