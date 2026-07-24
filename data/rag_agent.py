"""RAG Agent with tools for querying Pinecone vector database.

This module provides a DeepAgent-based agent with RAG tools for semantic search
and retrieval from the synapse-tea-index Pinecone database.
"""

import os
import json
import logging
from typing import Any, Literal

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Try importing deepagents, with helpful error message if not available
try:
    from deepagents import SubAgent, create_deep_agent
except ImportError as e:
    raise ImportError(
        "deepagents not installed. Install with: pip install deepagents"
    ) from e

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PineconeRAGTools:
    """Container for RAG tools that query Pinecone."""

    def __init__(
        self,
        index_name: str = "synapse-tea-index",
        embedding_model: str = "text-embedding-3-large",
    ):
        """Initialize RAG tools.

        Args:
            index_name: Name of the Pinecone index.
            embedding_model: Embedding model to use for queries.
        """
        self.index_name = index_name
        self.embedding_model = embedding_model
        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize Pinecone and embedding clients."""
        try:
            api_key = os.environ.get("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY not set")

            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index(self.index_name)
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
            logger.info(f"Initialized RAG tools with index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to initialize RAG tools: {e}")
            raise

    def search_documents(self, query: str, top_k: int = 5) -> str:
        """Search for relevant documents in the knowledge base.

        Args:
            query: The search query in natural language.
            top_k: Number of top results to return (default: 5).

        Returns:
            JSON string with search results containing document chunks and metadata.
        """
        try:
            logger.info(f"Searching for: {query} (top {top_k})")

            # Embed the query
            query_vector = self.embeddings.embed_query(query)

            # Query Pinecone
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
            )

            # Format results
            formatted_results = []
            for match in results.get("matches", []):
                formatted_results.append(
                    {
                        "id": match.get("id"),
                        "score": match.get("score"),
                        "text": match.get("metadata", {}).get("text", ""),
                        "file": match.get("metadata", {}).get("file", ""),
                        "page": match.get("metadata", {}).get("page", 0),
                        "chunk_idx": match.get("metadata", {}).get("chunk_idx", 0),
                    }
                )

            response = {
                "query": query,
                "total_results": len(formatted_results),
                "results": formatted_results,
            }
            logger.info(f"Found {len(formatted_results)} relevant documents")
            return json.dumps(response, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"Error searching documents: {e}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

    def get_document_info(self, file_name: str) -> str:
        """Get information about a specific document.

        Args:
            file_name: Name of the document to get info about.

        Returns:
            JSON string with document information and statistics.
        """
        try:
            logger.info(f"Getting info for document: {file_name}")

            # Query all chunks from this file
            # For simplicity, we'll do a filter by searching for chunks from this file
            query_vector = self.embeddings.embed_query(file_name)
            results = self.index.query(
                vector=query_vector,
                top_k=1000,
                include_metadata=True,
                filter={"file": {"$eq": file_name}},
            )

            chunks = [m for m in results.get("matches", []) if m["metadata"]["file"] == file_name]
            pages = set(m["metadata"].get("page") for m in chunks)

            info = {
                "file_name": file_name,
                "total_chunks": len(chunks),
                "pages": sorted(list(pages)),
                "chunks": [
                    {
                        "page": m["metadata"].get("page"),
                        "chunk_idx": m["metadata"].get("chunk_idx"),
                        "text_preview": m["metadata"].get("text", "")[:100] + "...",
                    }
                    for m in chunks[:10]
                ],
            }
            return json.dumps(info, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"Error getting document info: {e}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context for a specific query.

        Use this to gather background information before answering user questions.

        Args:
            query: The context query.
            top_k: Number of context chunks to retrieve.

        Returns:
            Formatted context string ready for RAG.
        """
        try:
            logger.info(f"Retrieving context for: {query}")

            # Embed the query
            query_vector = self.embeddings.embed_query(query)

            # Query Pinecone with lower threshold for more results
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
            )

            # Build context string
            context_parts = []
            for i, match in enumerate(results.get("matches", []), 1):
                file = match.get("metadata", {}).get("file", "Unknown")
                page = match.get("metadata", {}).get("page", "?")
                text = match.get("metadata", {}).get("text", "")
                score = match.get("score", 0)

                context_parts.append(
                    f"[{i}] (from {file}, p.{page}, score: {score:.3f})\n{text}"
                )

            context = "\n\n".join(context_parts)
            return context if context else "No relevant context found."

        except Exception as e:
            error_msg = f"Error retrieving context: {e}"
            logger.error(error_msg)
            return f"Error: {error_msg}"


def create_rag_agent(
    index_name: str = "synapse-tea-index",
    embedding_model: str = "text-embedding-3-large",
    model: str = "gpt-4o-mini",
):
    """Create an agent with RAG tools.

    Args:
        index_name: Pinecone index name.
        embedding_model: Embedding model to use.
        model: LLM model to use for the agent.

    Returns:
        Configured agent instance.
    """
    # Initialize RAG tools
    rag_tools = PineconeRAGTools(index_name, embedding_model)

    # Create agent using create_deep_agent with direct callable tools
    agent = create_deep_agent(
        name="SynapseTEA RAG Agent",
        model=model,
        tools=[
            rag_tools.search_documents,
            rag_tools.retrieve_context,
            rag_tools.get_document_info,
        ],
        system_prompt=(
            "Você é um assistente de IA útil e conversacional especializado em documentos médicos e terapêuticos. "
            "Responda de forma natural e em português quando o usuário escrever em português; adapte o tom ao usuário (claro, conciso e empático). "
            "Responda diretamente sempre que possível com seu conhecimento interno. Só consulte ferramentas externas (search_documents, retrieve_context, get_document_info) quando for realmente necessário para obter evidências ou dados factuais do banco de documentos. "
            "Antes de chamar uma ferramenta, avalie se a busca é necessária para evitar buscas desnecessárias e alucinações. Se chamar uma ferramenta, use a saída apenas internamente para formar a resposta final — NÃO retorne a saída bruta das ferramentas ao usuário. "
            "Sempre entregue uma resposta final concisa em texto simples. Se usar fontes, cite-as de forma breve no final da resposta (ex.: [Avaliação_ABC.pdf, p.3]). Se não souber, admita honestamente e sugira próximos passos práticos."
        ),
    )

    logger.info(f"Created RAG Agent with model: {model}")
    return agent


if __name__ == "__main__":
    # Test the RAG tools directly
    rag_tools = PineconeRAGTools()

    # Test search
    print("=" * 60)
    print("Testing search_documents tool:")
    print("=" * 60)
    results = rag_tools.search_documents("avaliação de fonoaudiologia", top_k=3)
    try:
        print(results)
    except UnicodeEncodeError:
        print("Results retrieved successfully (contains special characters)")

    print("\n" + "=" * 60)
    print("Testing retrieve_context tool:")
    print("=" * 60)
    context = rag_tools.retrieve_context("terapia ocupacional", top_k=2)
    try:
        print(context)
    except UnicodeEncodeError:
        print("Context retrieved successfully (contains special characters)")
