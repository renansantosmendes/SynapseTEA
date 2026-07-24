import os
import logging
import unicodedata
from datetime import datetime
from typing import List, Tuple, Optional
from pathlib import Path

import PyPDF2
from pinecone import Pinecone, ServerlessSpec

# Import OpenAI embeddings from langchain_openai package with graceful fallback
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None

# Optional HF embeddings support
try:
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
    except Exception:
        HuggingFaceEmbeddings = None
        
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Model dimension mappings
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 512,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _sanitize_filename(filename: str) -> str:
    """Remove non-ASCII characters from filename for Pinecone vector IDs.

    Args:
        filename: Original filename with potential non-ASCII characters.

    Returns:
        Sanitized filename with only ASCII characters.
    """
    # Normalize Unicode characters (decompose accents)
    normalized = unicodedata.normalize("NFKD", filename)
    # Keep only ASCII characters
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    # Replace spaces and special chars with underscores
    sanitized = "".join(c if c.isalnum() else "_" for c in ascii_only)
    return sanitized


def _is_zero_vector(vec: List[float]) -> bool:
    """Check if a vector contains only zeros.

    Args:
        vec: Embedding vector to check.

    Returns:
        True if vector contains only zeros, False otherwise.
    """
    return all(v == 0.0 for v in vec)


def _get_model_dimension(model_name: str) -> int:
    """Get the dimension for a specific embedding model."""
    return MODEL_DIMENSIONS.get(model_name, 1536)


def _normalize_vector(vec: List[float], dim: int) -> List[float]:
    """Normalize embedding vector to required dimension.

    Args:
        vec: Embedding vector.
        dim: Target dimension.

    Returns:
        Normalized vector with exact dimension size.
    """
    if len(vec) == dim:
        return vec
    if len(vec) > dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))


def iter_pdf_pages(file_path: str) -> Tuple[int, str]:
    """Yield (page_number, text) for each page in the PDF.

    Args:
        file_path: Path to the PDF file.

    Yields:
        Tuple of (page_number, page_text).
    """
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                yield i, text
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        raise


def chunk_text(
    text: str,
    max_chunk: int = 800,
    overlap: int = 200,
) -> List[str]:
    """Split text into chunks with overlap.

    Args:
        text: Text to split.
        max_chunk: Maximum chunk size in characters.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    chunks: List[str] = []
    if not text or not text.strip():
        return chunks
    
    start = 0
    while start < len(text):
        end = min(start + max_chunk, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, 0)
    
    return chunks


def collect_pdfs_from_folder(folder_path: str) -> List[str]:
    """Collect all PDF files from a folder (recursively).

    Args:
        folder_path: Path to the folder containing PDFs.

    Returns:
        List of full paths to PDF files.
    """
    pdfs: List[str] = []
    try:
        folder = Path(folder_path)
        if not folder.exists():
            logger.warning(f"Folder does not exist: {folder_path}")
            return pdfs
        
        for pdf_file in folder.glob("**/*.pdf"):
            pdfs.append(str(pdf_file))
        
        logger.info(f"Found {len(pdfs)} PDF files in {folder_path}")
    except Exception as e:
        logger.error(f"Error collecting PDFs from {folder_path}: {e}")
    
    return pdfs


def init_pinecone(
    index_name: str,
    dim: int,
    api_key: Optional[str] = None,
    environment: Optional[str] = None,
):
    """Initialize Pinecone client and create index if needed.

    Args:
        index_name: Name of the Pinecone index.
        dim: Dimension of the embedding vectors.
        api_key: Pinecone API key (defaults to PINECONE_API_KEY env var).
        environment: Pinecone environment (defaults to PINECONE_ENV env var).

    Returns:
        Pinecone Index object.
    """
    api_key = api_key or os.environ.get("PINECONE_API_KEY")
    environment = environment or os.environ.get("PINECONE_ENV")
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set")
    
    try:
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists, if not create it
        indexes = pc.list_indexes()
        if index_name not in [idx.name for idx in indexes.indexes]:
            logger.info(f"Creating Pinecone index: {index_name} with dimension {dim}")
            pc.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        
        index = pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")
        return index
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        raise


def ingest_pdfs_to_pinecone(
    folder_path: str,
    index_name: str = "pdf-index",
    embedding_model: str = "text-embedding-3-small",
    embedding_backend: str = "openai",
    max_chunk: int = 800,
    overlap: int = 200,
    batch_size: int = 50,
) -> dict:
    """Ingest PDFs from a folder into a Pinecone index.

    Args:
        folder_path: Path to folder containing PDFs.
        index_name: Name of the Pinecone index.
        embedding_model: Embedding model to use.
        embedding_backend: Embedding backend ('openai', 'hf', or 'dummy').
        max_chunk: Maximum chunk size in characters.
        overlap: Overlap between consecutive chunks.
        batch_size: Batch size for Pinecone upserts.

    Returns:
        Dictionary with ingestion statistics:
        - total_pdfs: Number of PDFs processed.
        - total_chunks: Total number of chunks created.
        - failed_pdfs: List of PDFs that failed to process.
    """
    stats = {
        "total_pdfs": 0,
        "total_chunks": 0,
        "failed_pdfs": [],
    }
    
    # Get correct dimension for model
    dim = _get_model_dimension(embedding_model)
    logger.info(f"Using model {embedding_model} with dimension {dim}")
    
    # Initialize Pinecone index
    try:
        index = init_pinecone(index_name, dim)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        return stats
    
    # Build the embeddings client with a safe fallback
    embeddings = None
    if embedding_backend == "openai" and OpenAIEmbeddings is not None:
        try:
            embeddings = OpenAIEmbeddings(model=embedding_model)
            logger.info(f"Initialized OpenAI embeddings with model: {embedding_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI embeddings: {e}")
            embeddings = None
    
    if embeddings is None and embedding_backend == "hf" and HuggingFaceEmbeddings is not None:
        try:
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            logger.info(f"Initialized HuggingFace embeddings with model: {embedding_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize HF embeddings: {e}")
            embeddings = None
    
    if embeddings is None:
        # Fallback: dummy embedding that returns zeros
        logger.error("CRITICAL: No embedding backend available! OpenAI embeddings failed to initialize.")
        logger.error("Please check:")
        logger.error("  1. OPENAI_API_KEY environment variable is set and valid")
        logger.error("  2. langchain-openai package is installed (pip install langchain-openai)")
        logger.error("  3. You have API credits available in your OpenAI account")
        raise ValueError(
            "Cannot ingest PDFs without a working embedding backend. "
            "OpenAI embeddings failed to initialize. Check logs above for details."
        )
    
    # Collect PDFs
    pdf_paths = collect_pdfs_from_folder(folder_path)
    stats["total_pdfs"] = len(pdf_paths)
    
    if not pdf_paths:
        logger.warning(f"No PDFs found in {folder_path}")
        return stats
    
    # Process PDFs
    vectors_to_upsert = []
    processed_count = 0
    
    for pdf_path in pdf_paths:
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            for page_num, page_text in iter_pdf_pages(pdf_path):
                chunks = chunk_text(page_text, max_chunk=max_chunk, overlap=overlap)
                for chunk_idx, chunk in enumerate(chunks):
                    try:
                        vec = embeddings.embed_documents([chunk])[0]
                        vec = _normalize_vector(vec, dim)
                        
                        # Validate vector is not all zeros
                        if _is_zero_vector(vec):
                            logger.error(
                                f"Vector contains only zeros for chunk from {pdf_path} "
                                f"(page {page_num}, chunk {chunk_idx}). "
                                "This indicates embedding backend is not working correctly. "
                                "Check OPENAI_API_KEY and API connectivity."
                            )
                            continue
                        
                        filename = os.path.basename(pdf_path)
                        sanitized_filename = _sanitize_filename(filename)
                        
                        metadata = {
                            "text": chunk[:1000],  # Store only first 1000 chars in metadata
                            "file": filename,
                            "page": int(page_num),
                            "chunk_idx": chunk_idx,
                            "created_at": datetime.utcnow().isoformat() + "Z",
                            "source": "PDF",
                        }
                        vector_id = f"{sanitized_filename}_p{page_num}_c{chunk_idx}"
                        vectors_to_upsert.append((vector_id, vec, metadata))
                        stats["total_chunks"] += 1
                        
                        # Upsert in batches
                        if len(vectors_to_upsert) >= batch_size:
                            try:
                                index.upsert(vectors=vectors_to_upsert)
                                logger.info(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone")
                                vectors_to_upsert = []
                            except Exception as e:
                                logger.error(f"Error upserting vectors to Pinecone: {e}")
                    
                    except Exception as e:
                        logger.error(f"Error embedding chunk from {pdf_path} (page {page_num}, chunk {chunk_idx}): {e}")
            
            processed_count += 1
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            stats["failed_pdfs"].append(pdf_path)
    
    # Upsert remaining vectors
    if vectors_to_upsert:
        try:
            index.upsert(vectors=vectors_to_upsert)
            logger.info(f"Upserted final {len(vectors_to_upsert)} vectors to Pinecone")
        except Exception as e:
            logger.error(f"Error upserting final vectors to Pinecone: {e}")
    
    # Log summary
    logger.info(
        f"Ingestion complete: {processed_count}/{stats['total_pdfs']} PDFs processed, "
        f"{stats['total_chunks']} chunks created"
    )
    if stats["failed_pdfs"]:
        logger.warning(f"Failed PDFs: {stats['failed_pdfs']}")
    
    return stats


if __name__ == "__main__":
    # Configuration variables
    folder_path = "C:\\Users\\renan\\Downloads\\data_synapse_tea"
    index_name = "synapse-tea-index"
    embedding_model = "text-embedding-3-large"
    embedding_backend = "openai"
    batch_size = 50
    
    stats = ingest_pdfs_to_pinecone(
        folder_path=folder_path,
        index_name=index_name,
        embedding_model=embedding_model,
        embedding_backend=embedding_backend,
        batch_size=batch_size,
    )
    
    print("\n--- Ingestion Summary ---")
    print(f"Total PDFs: {stats['total_pdfs']}")
    print(f"Total Chunks: {stats['total_chunks']}")
    if stats['failed_pdfs']:
        print(f"Failed PDFs: {len(stats['failed_pdfs'])}")
        for pdf in stats['failed_pdfs']:
            print(f"  - {pdf}")
