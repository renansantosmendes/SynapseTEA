import os
from datetime import datetime
from typing import List

import PyPDF2
import pinecone

from langchain.embeddings.openai import OpenAIEmbeddings


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

    embeddings = OpenAIEmbeddings(model=embedding_model)

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
    # Simple CLI usage example (adjust path as needed)
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDFs into Pinecone index")
    parser.add_argument("folder", help="Folder path containing PDF files")
    parser.add_argument("index", nargs="?", default="pdf-index", help="Pinecone index name")
    args = parser.parse_args()

    ingest_pdfs_to_pinecone(args.folder, index_name=args.index)
