# SynapseTEA RAG Agent Guide

## Overview

The RAG (Retrieval-Augmented Generation) Agent is an AI-powered system that can answer questions about your documents by:

1. **Searching** the Pinecone vector database for relevant documents
2. **Retrieving** contextual information from those documents
3. **Generating** intelligent responses using GPT-4 with the retrieved context

## Architecture

```
User Question
    ↓
RAG Agent (DeepAgent)
    ↓
Tools:
  ├─ search_documents: Find relevant chunks in Pinecone
  ├─ retrieve_context: Get background information
  └─ get_document_info: Show document statistics
    ↓
Pinecone Vector Database
    ↓
LLM (GPT-4o-mini)
    ↓
Response with Citations
```

## Setup

### 1. Install Dependencies

```bash
cd "c:\00 - Source Codes\SynapseTEA"

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install/update packages
uv pip install -r requirements.txt
```

### 2. Verify Environment Variables

Make sure your `.env` file has:
```
OPENAI_API_KEY=sk-proj-...
PINECONE_API_KEY=...
```

### 3. Ensure Pinecone Index Exists

Run the ingestion script if you haven't already:
```bash
python data/pdf_pinecone_ingest.py
```

## Usage

### Interactive Mode (Recommended)

Start the interactive agent loop where you can ask natural language questions:

```bash
python data/test_agent_loop.py
```

**Example Session:**
```
You: Quais são as avaliações disponíveis para o Antonio?
Agent: Based on the documents in the database, I found several evaluations for Antonio Damasceno...

You: search Fonoaudiologia
Searching for: Fonoaudiologia
[Results displayed in JSON format with sources]

You: quit
```

### Direct Tool Access

You can also use the RAG tools directly in your own scripts:

```python
from rag_agent import PineconeRAGTools

rag_tools = PineconeRAGTools()

# Search documents
results = rag_tools.search_documents("fonoaudiologia", top_k=5)
print(results)

# Get context for RAG
context = rag_tools.retrieve_context("terapia ocupacional", top_k=3)
print(context)

# Get document info
info = rag_tools.get_document_info("Avaliação Antonio Damasceno Fonoaudiologia.pdf")
print(info)
```

### Using the Agent Programmatically

```python
from rag_agent import create_rag_agent

# Create agent
agent = create_rag_agent(
    index_name="synapse-tea-index",
    embedding_model="text-embedding-3-large",
    model="gpt-4o-mini"
)

# Ask a question
response = agent.run("O que diz a avaliação do Antonio em fevereiro?")
print(response)
```

## Available Commands in Interactive Mode

| Command | Description | Example |
|---------|-------------|---------|
| `<question>` | Ask agent a question | `Quais são as avaliações de Antonio?` |
| `search <query>` | Direct document search | `search fonoaudiologia` |
| `context <query>` | Retrieve context | `context terapia ocupacional` |
| `info <filename>` | Get document info | `info "Avaliação Antonio.pdf"` |
| `help` | Show help | `help` |
| `quit` | Exit | `quit` |

## Demo Mode

Run pre-defined demo queries:

```bash
python data/test_agent_loop.py --demo
```

This will automatically ask several example questions and show the agent's responses.

## Tool Specifications

### search_documents
- **Purpose**: Find relevant document chunks in the vector database
- **Parameters**:
  - `query` (str): Natural language search query
  - `top_k` (int): Number of results (default: 5)
- **Returns**: JSON with results including score, text, file, page, chunk_idx
- **Use Case**: Direct semantic search, finding all relevant documents

### retrieve_context
- **Purpose**: Get formatted context for RAG (Retrieval-Augmented Generation)
- **Parameters**:
  - `query` (str): Context query
  - `top_k` (int): Number of chunks (default: 3)
- **Returns**: Formatted text string with citations and scores
- **Use Case**: Preparing background info for LLM to generate responses

### get_document_info
- **Purpose**: Get metadata and statistics about a specific document
- **Parameters**:
  - `file_name` (str): Name of the document
- **Returns**: JSON with chunk count, pages, sample chunks
- **Use Case**: Understanding document structure, finding pages to review

## Customization

### Change the LLM Model

```python
agent = create_rag_agent(model="gpt-4-turbo")  # Use GPT-4 instead
```

Available models:
- `gpt-4o-mini` (default, fast & cheap)
- `gpt-4o` (better quality, more expensive)
- `gpt-4-turbo` (high quality)
- `gpt-3.5-turbo` (fast but lower quality)

### Change Embedding Model

```python
agent = create_rag_agent(embedding_model="text-embedding-3-small")
```

Available models:
- `text-embedding-3-small` (512 dims, faster)
- `text-embedding-3-large` (3072 dims, better quality, default)
- `text-embedding-ada-002` (1536 dims, older)

### Modify System Prompt

Edit the `system_prompt` parameter in `create_rag_agent()` in `rag_agent.py`:

```python
system_prompt=(
    "You are a specialized medical document analyst. "
    "Your expertise is in pediatric therapy and development. "
    # ... customize as needed
)
```

## Troubleshooting

### Error: "deepagents not installed"
```bash
uv pip install deepagents
```

### Error: "OPENAI_API_KEY not set"
Check that `.env` file exists and has valid API key:
```bash
# In .env file:
OPENAI_API_KEY=sk-proj-...
```

### Error: "Pinecone index not found"
The index might not exist. Run the ingestion script:
```bash
python data/pdf_pinecone_ingest.py
```

### Agent giving poor responses
- Try a better LLM model: `model="gpt-4o"`
- Increase context size: `top_k=7` in retrieve_context
- Review the retrieved documents: Use `search <query>` first to verify results

### Slow responses
- Use faster embedding: `embedding_model="text-embedding-3-small"`
- Use cheaper LLM: `model="gpt-3.5-turbo"`
- Reduce `top_k` when retrieving context

## Performance Tips

1. **Start with demo mode** to verify everything works
2. **Test with simple questions** before complex ones
3. **Use search/context commands** to verify data availability
4. **Monitor API costs** - each query uses OpenAI and Pinecone APIs
5. **Batch similar questions** to reuse context retrieval

## Next Steps

- [ ] Integrate agent into web API
- [ ] Add memory/conversation history
- [ ] Create specialized agents for different document types
- [ ] Add document refinement/extraction tools
- [ ] Implement user feedback loop for agent improvement
- [ ] Add support for real-time document indexing

## References

- [deepagents Documentation](https://github.com/derobertis/deepagents)
- [LangChain OpenAI Integration](https://python.langchain.com/docs/integrations/providers/openai)
- [Pinecone API](https://docs.pinecone.io/)
- [RAG Best Practices](https://docs.anthropic.com/en/docs/build-a-system#retrieval-augmented-generation)
