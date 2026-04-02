from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import fitz  # PyMuPDF
import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

APP_TITLE = "Course Slides RAG"
DATA_DIR = Path("rag_data")
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
CHUNKS_FILE = INDEX_DIR / "chunks.pkl"
META_FILE = INDEX_DIR / "meta.json"
FAISS_FILE = INDEX_DIR / "slides.index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

for folder in [DATA_DIR, UPLOAD_DIR, INDEX_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


@dataclass
class Chunk:
    text: str
    source_file: str
    page_num: int
    chunk_id: str


def file_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_pdf(pdf_path: Path) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            text = normalize_text(page.get_text("text"))
            if text:
                pages.append((i + 1, text))
    finally:
        doc.close()
    return pages


def chunk_slide_text(
    source_file: str,
    page_num: int,
    text: str,
    max_words: int = 120,
    overlap: int = 30,
) -> List[Chunk]:
    words = text.split()
    if not words:
        return []

    chunks: List[Chunk] = []
    start = 0
    idx = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_text = " ".join(words[start:end]).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source_file=source_file,
                    page_num=page_num,
                    chunk_id=f"{source_file}-p{page_num}-c{idx}",
                )
            )
        if end == len(words):
            break
        start += max_words - overlap
        idx += 1

    return chunks


def build_chunks_from_pdfs(pdf_paths: List[Path]) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for pdf_path in pdf_paths:
        for page_num, text in extract_text_from_pdf(pdf_path):
            all_chunks.extend(chunk_slide_text(pdf_path.name, page_num, text))
    return all_chunks


@st.cache_resource(show_spinner=False)
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_texts(texts: List[str]) -> np.ndarray:
    model = load_embedder()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, chunks: List[Chunk], metadata: Dict) -> None:
    faiss.write_index(index, str(FAISS_FILE))
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_index() -> Tuple[faiss.IndexFlatIP | None, List[Chunk], Dict]:
    if not (FAISS_FILE.exists() and CHUNKS_FILE.exists() and META_FILE.exists()):
        return None, [], {}

    index = faiss.read_index(str(FAISS_FILE))
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, chunks, metadata


def retrieve(query: str, index: faiss.IndexFlatIP, chunks: List[Chunk], top_k: int = 5) -> List[Tuple[Chunk, float]]:
    query_emb = embed_texts([query])
    scores, indices = index.search(query_emb, top_k)
    results: List[Tuple[Chunk, float]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append((chunks[idx], float(score)))
    return results


def build_prompt(question: str, retrieved_chunks: List[Tuple[Chunk, float]]) -> str:
    context = "\n\n".join(
        f"[Source {i}] File: {chunk.source_file} | Slide/Page: {chunk.page_num}\n{chunk.text}"
        for i, (chunk, _score) in enumerate(retrieved_chunks, start=1)
    )
    return f"""
You are a study assistant for university course slides.
Answer only from the retrieved slide context.
If the answer is not clearly in the slides, say exactly:
I couldn't find that clearly in the uploaded slides.
Keep the answer accurate, concise, and helpful for studying.
When useful, mention the source slide/page.

Question:
{question}

Retrieved slide context:
{context}
""".strip()


def ollama_available(model_name: str) -> bool:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model_name, "prompt": "ping", "stream": False},
            timeout=10,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def ask_ollama(prompt: str, model_name: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def save_uploaded_pdf(uploaded_file) -> Path:
    file_bytes = uploaded_file.read()
    digest = file_sha256(file_bytes)[:12]
    saved_path = UPLOAD_DIR / f"{digest}_{safe_filename(uploaded_file.name)}"
    with open(saved_path, "wb") as f:
        f.write(file_bytes)
    return saved_path


def index_uploaded_pdfs(uploaded_files) -> Tuple[bool, str]:
    if not uploaded_files:
        return False, "Please upload at least one PDF slide file."

    saved_paths: List[Path] = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.lower().endswith(".pdf"):
            saved_paths.append(save_uploaded_pdf(uploaded_file))

    if not saved_paths:
        return False, "No valid PDF files were uploaded."

    with st.spinner("Extracting text and building the vector index..."):
        chunks = build_chunks_from_pdfs(saved_paths)
        if not chunks:
            return False, "No extractable text was found. Your slides may be image-only PDFs."

        embeddings = embed_texts([chunk.text for chunk in chunks])
        index = build_faiss_index(embeddings)
        metadata = {
            "files": [path.name for path in saved_paths],
            "chunk_count": len(chunks),
            "embedding_model": EMBED_MODEL_NAME,
        }
        save_index(index, chunks, metadata)

    return True, f"Indexed {len(saved_paths)} file(s) into {len(chunks)} chunks."


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Free local RAG for course slides using Streamlit + FAISS + sentence-transformers + Ollama")

with st.sidebar:
    st.header("Settings")
    ollama_model = st.text_input("Ollama model", value=DEFAULT_OLLAMA_MODEL)
    top_k = st.slider("Top retrieved chunks", 3, 10, 5)

    st.markdown("---")
    st.subheader("Upload slides")
    uploaded_files = st.file_uploader(
        "Upload PDF course slides",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button("Build / Rebuild Index", use_container_width=True):
        ok, msg = index_uploaded_pdfs(uploaded_files)
        st.success(msg) if ok else st.error(msg)

index, chunks, metadata = load_index()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Ask a question")
    question = st.text_area(
        "Example: Explain deadlock in simple terms from the slides",
        height=120,
        placeholder="Ask anything about the uploaded course slides...",
    )

    if metadata:
        st.info(
            f"Indexed files: {', '.join(metadata.get('files', []))}\n\n"
            f"Chunks: {metadata.get('chunk_count', 0)} | Embeddings: {metadata.get('embedding_model', '')}"
        )
    else:
        st.warning("No index found yet. Upload PDFs and click Build / Rebuild Index.")

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.error("Please enter a question.")
        elif index is None or not chunks:
            st.error("No slide index is available yet.")
        else:
            retrieved = retrieve(question.strip(), index, chunks, top_k)
            st.markdown("### Answer")
            if ollama_available(ollama_model):
                try:
                    st.write(ask_ollama(build_prompt(question.strip(), retrieved), ollama_model))
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")
                    st.write("Showing retrieved evidence instead.")
            else:
                st.warning("Ollama is not running, so only retrieval results are shown.")

with right:
    st.subheader("Retrieved Evidence")
    if index is not None and chunks and question.strip():
        for rank, (chunk, score) in enumerate(retrieve(question.strip(), index, chunks, top_k), start=1):
            with st.expander(
                f"#{rank} | {chunk.source_file} | Slide/Page {chunk.page_num} | score={score:.3f}",
                expanded=(rank == 1),
            ):
                st.write(chunk.text)
    else:
        st.write("Relevant chunks will appear here after you ask a question.")

st.markdown("---")
st.markdown(
    "**Flow:** upload PDFs → extract text → chunk text → embed with a free model → search with FAISS → answer with a free local Ollama model."
)
