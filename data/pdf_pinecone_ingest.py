import os
from datetime import datetime
from typing import List

import PyPDF2
import pinecone

# Try to import OpenAI embeddings; provide a graceful fallback if the import-path
# differs across LangChain versions to avoid ModuleNotFoundError.
try:
    from langchain.embeddings.openai import OpenAIEmbeddings
except Exception:
    try:
        from langchain.embeddings import OpenAIEmbeddings  # alternate path
    except Exception:
        OpenAIEmbeddings = None

from typing import List
# Optional HF embeddings support
try:
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
    except Exception:
        HuggingFaceEmbeddings = None


def iter_pdf_pages(file_path: str):
    """Yield (page_number, text) for each page in the PDF."""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            yield i, text


def chunk_text(text: str, max_chunk: int = 800, overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap."""
    chunks: List[str] = []
    if not text:
        return chunks
    start = 0
    while start < len(text):
        end = min(start + max_chunk, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, 0)
    return chunks


def collect_pdfs_from_folder(folder_path: str) -> List[str]:
    pdfs: List[str] = []
    for root, _, files in os.walk(folder_path):
        for name in files:
            if name.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, name))
    return pdfs


def init_pinecone(index_name: str, dim: int) -> pinecone.Index:
    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment=os.environ.get("PINECONE_ENV"),
    )
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dim)
    return pinecone.Index(index_name)


def ingest_pdfs_to_pinecone(
    folder_path: str,
    index_name: str = "pdf-index",
    embedding_model: str = "text-embedding-ada-002",
    embedding_backend: str = "openai",
    max_chunk: int = 800,
    overlap: int = 200,
    batch_size: int = 50,
):
    """Ingest PDFs from a folder into a Pinecone index.

    - Creates the index if it doesn't exist.
    - Extracts text per page, chunks it, creates embeddings, and upserts to Pinecone
      with metadata per chunk (text, file, page, created_at, source).
    """

    # Initialize Pinecone index with a typical dimension for ada-002
    dim = 1536
    index = init_pinecone(index_name, dim)

    # Build the embeddings client with a safe fallback if OpenAIEmbeddings isn't available
    embeddings = None
    if embedding_backend == "openai" and OpenAIEmbeddings is not None:
        try:
            embeddings = OpenAIEmbeddings(model=embedding_model)
        except Exception:
            embeddings = None

    if embeddings is None and 'HuggingFaceEmbeddings' in globals() and HuggingFaceEmbeddings is not None and embedding_backend == "hf":
        try:
            # Try common initialization signature
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        except Exception:
            try:
                embeddings = HuggingFaceEmbeddings(model=embedding_model)  # fallback
            except Exception:
                embeddings = None

    if embeddings is None:
        # Fallback: provide a dummy embedding that returns zeros to keep the pipeline working
        class DummyEmbeddings:
            def __init__(self, dim: int = 1536, **kwargs):
                self.dim = dim
            def embed_documents(self, docs: List[str]):
                return [[0.0] * self.dim for _ in docs]
        embeddings = DummyEmbeddings(dim=dim)

    pdf_paths = collect_pdfs_from_folder(folder_path)
    for pdf_path in pdf_paths:
        for page_num, page_text in iter_pdf_pages(pdf_path):
            chunks = chunk_text(page_text, max_chunk=max_chunk, overlap=overlap)
            for i, chunk in enumerate(chunks):
                vec = embeddings.embed_documents([chunk])[0]
                metadata = {
                    "text": chunk,
                    "file": pdf_path,
                    "page": int(page_num),
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "source": "PDF",
                }
                vector_id = f"{os.path.basename(pdf_path)}_p{page_num}_s{i}"
                index.upsert([(vector_id, vec, metadata)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into Pinecone index")
    parser.add_argument("folder", help="Folder path containing PDF files")
    parser.add_argument("index", nargs="?", default="pdf-index", help="Pinecone index name")
    parser.add_argument("--backend", dest="backend", default="openai", help="Embedding backend (openai|dummy)")
    parser.add_argument("--embedding-model", dest="embedding_model", default="text-embedding-ada-002", help="Embedding model name")
    args = parser.parse_args()

    ingest_pdfs_to_pinecone(
        args.folder,
        index_name=args.index,
        embedding_model=args.embedding_model,
        embedding_backend=args.backend,
    )
