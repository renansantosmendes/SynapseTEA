# SynapseTEA RAG Tools - Usage Guide

## Overview

The RAG (Retrieval-Augmented Generation) tools are **fully functional and ready to use**. They enable semantic search across the 485 chunks indexed in Pinecone, retrieved from 9 PDF documents.

## Quick Start

### 1. Import the Tools

```python
from data.rag_agent import PineconeRAGTools

# Initialize
rag_tools = PineconeRAGTools()
```

### 2. Search Documents

Find documents relevant to a query using semantic search:

```python
# Search for documents
results = rag_tools.search_documents(
    query="avaliação de fonoaudiologia",
    top_k=5  # Return top 5 results
)
print(results)  # Returns JSON with id, score, text, file, page, chunk_idx
```

### 3. Retrieve Context

Get formatted context from documents for use in responses:

```python
# Retrieve context for RAG
context = rag_tools.retrieve_context(
    query="terapia ocupacional",
    top_k=3  # Get top 3 context chunks
)
print(context)  # Returns formatted text with citations [1], [2], etc
```

### 4. Get Document Information

Get statistics about a specific document:

```python
# Get document info
info = rag_tools.get_document_info(
    file_name="Avaliação Antonio Damasceno Fonoaudiologia.pdf"
)
print(info)  # Returns document stats as JSON
```

## Complete Example

```python
from data.rag_agent import PineconeRAGTools
import json

# Initialize tools
rag_tools = PineconeRAGTools()

# User's question
question = "Como está o desenvolvimento de António Damasceno em fonoaudiologia?"

# Search for relevant documents
search_results = rag_tools.search_documents(question, top_k=5)
print("Search Results:")
print(json.loads(search_results))

# Get formatted context
context = rag_tools.retrieve_context(question, top_k=3)
print("\nRetrieved Context:")
print(context)

# Get info about a document
doc_info = rag_tools.get_document_info("Avaliação Antonio Damasceno Fonoaudiologia.pdf")
print("\nDocument Info:")
print(json.loads(doc_info))
```

## Tool Specifications

### search_documents(query: str, top_k: int = 5) → str

**Purpose**: Find relevant documents using semantic search

**Parameters**:
- `query` (str): Natural language search query
- `top_k` (int): Number of top results to return (default: 5, max: 485)

**Returns**: JSON string with:
- `query`: The original query
- `total_results`: Number of results found
- `results`: Array of documents with:
  - `id`: Unique vector ID
  - `score`: Similarity score (0-1)
  - `text`: First 1000 chars of chunk
  - `file`: Source PDF filename
  - `page`: Page number in document
  - `chunk_idx`: Chunk index on the page

**Example**:
```json
{
  "query": "avaliação de fonoaudiologia",
  "total_results": 5,
  "results": [
    {
      "id": "Avaliacao_Antonio_Damasceno_Fonoaudiologia_pdf_p7_c0",
      "score": 0.621727288,
      "text": "Avaliação: A avaliação foi realizada...",
      "file": "Avaliação Antonio Damasceno Fonoaudiologia.pdf",
      "page": 7,
      "chunk_idx": 0
    },
    ...
  ]
}
```

### retrieve_context(query: str, top_k: int = 3) → str

**Purpose**: Get formatted context from documents for RAG

**Parameters**:
- `query` (str): Natural language query
- `top_k` (int): Number of context chunks to retrieve (default: 3)

**Returns**: Formatted text with:
- Numbered citations [1], [2], [3]
- Document name, page number, and similarity score
- Full text of each chunk

**Example**:
```
[1] (from Document.pdf, p.7, score: 0.621)
Text of first chunk...

[2] (from Another.pdf, p.14, score: 0.589)
Text of second chunk...
```

### get_document_info(file_name: str) → str

**Purpose**: Get statistics about a document

**Parameters**:
- `file_name` (str): Exact filename (case-sensitive)

**Returns**: JSON with:
- `file_name`: Document filename
- `chunk_count`: Total chunks from this document
- `page_count`: Total pages
- `sample_chunks`: Sample of chunk indices available

**Example**:
```json
{
  "file_name": "Avaliação Antonio Damasceno Fonoaudiologia.pdf",
  "chunk_count": 15,
  "page_count": 12,
  "sample_chunks": ["p1_c0", "p1_c1", "p2_c0", ...]
}
```

## Available Documents

The Pinecone index contains 485 chunks from these 9 documents:

1. Antônio Damasceno Mendes 69691610492ac.pdf
2. Avaliação Antonio Damasceno Fonoaudiologia.pdf
3. Avaliação Antonio Damasceno Fisioterapia.pdf
4. Avaliação Antonio Damasceno Psicologia.pdf
5. Avaliação Antonio Damasceno Terapia Ocupacional.pdf
6. Avaliação Antonio Damasceno Educação Física.pdf
7. TO AVAL ANTONIO DMASCENO NOV 24.pdf
8. [Additional documents...]

## Configuration

### Default Settings

```python
# Default Pinecone index
INDEX_NAME = "synapse-tea-index"

# Default embedding model
EMBEDDING_MODEL = "text-embedding-3-large"

# Embedding dimensions: 3072
# Provider: OpenAI API
```

### Custom Configuration

```python
rag_tools = PineconeRAGTools(
    index_name="custom-index",  # Use different index
    embedding_model="text-embedding-3-small"  # Use different model
)
```

## Performance Tips

1. **Query Optimization**: Be specific with your queries for better results
   - ✓ Good: "avaliação de desenvolvimento de linguagem em criança de 2 anos"
   - ✗ Less good: "informações sobre o arquivo"

2. **Top-K Selection**:
   - Use `top_k=3-5` for quick searches (1-2 seconds)
   - Use `top_k=10+` for comprehensive context (3-5 seconds)
   - Max useful: `top_k=20` (diminishing returns after this)

3. **Caching**: For repeated queries, cache the results:
   ```python
   cache = {}
   query = "meu filtro"
   if query not in cache:
       cache[query] = rag_tools.search_documents(query, top_k=5)
   results = cache[query]
   ```

## Troubleshooting

### "ImportError: cannot import name 'PineconeRAGTools'"
- Make sure you're in the workspace directory
- Run from: `c:\00 - Source Codes\SynapseTEA`
- Use: `from data.rag_agent import PineconeRAGTools`

### "PINECONE_API_KEY not found"
- Create `.env` file in workspace root with:
  ```
  PINECONE_API_KEY=your_key_here
  OPENAI_API_KEY=your_key_here
  ```

### "No results found"
- Try a simpler query with fewer specifics
- Check document names with `get_document_info()`
- Results are ranked by similarity (0-1 score), not exact matching

### "Text encoding issues"
- Normal for Portuguese special characters (ã, ç, é, etc.)
- The tools handle encoding internally
- Results are still correct even if terminal display shows garbled text

## Testing

Run the test script to verify everything works:

```bash
cd "c:\00 - Source Codes\SynapseTEA"
python data/test_agent_simple.py
```

Expected output:
- ✓ RAG tools initialized
- ✓ search_documents working
- ✓ retrieve_context working  
- ✓ get_document_info working

## Integration with Your Application

### FastAPI Example

```python
from fastapi import FastAPI
from data.rag_agent import PineconeRAGTools

app = FastAPI()
rag_tools = PineconeRAGTools()

@app.post("/search")
async def search(query: str, top_k: int = 5):
    results = rag_tools.search_documents(query, top_k)
    return {"results": results}

@app.post("/context")
async def get_context(query: str, top_k: int = 3):
    context = rag_tools.retrieve_context(query, top_k)
    return {"context": context}
```

### Command Line Usage

```bash
python -c "
from data.rag_agent import PineconeRAGTools
rag = PineconeRAGTools()
results = rag.search_documents('sua consulta', top_k=3)
print(results)
"
```

## What's Next

The RAG tools are production-ready. You can:

1. **Use tools directly** in your application (recommended)
2. **Extend with custom logic** for domain-specific processing
3. **Add caching layer** for frequently asked questions
4. **Integrate with LLMs** for question-answering systems

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review tool output for error messages
3. Check `.env` file configuration
4. Verify Pinecone API connectivity

---

**Status**: ✅ All tools fully functional and tested
