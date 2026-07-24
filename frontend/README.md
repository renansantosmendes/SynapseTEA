# SynapseTEA - Frontend (Chainlit)

This folder contains a minimal Chainlit frontend to interact with the local RAG backend implemented in the repository under data/rag_agent.py.

What this provides
- A frontend UI served locally via Chainlit that lets you run simple commands against the Rag backend:
  - search: <query>  -> semantic search across indexed PDF chunks
  - context: <query> -> retrieve contextual information with citations
  - info: <filename> -> get document statistics

How to run locally
1) Install dependencies (in a dedicated Python env):
   - cd frontend
   - python -m venv .venv
   - .\\.venv\\Scripts\\pip install -r requirements.txt
   - Ensure the root repo remains importable so the backend can be loaded (the frontend app.py adds the repo root to PYTHONPATH).

2) Start Chainlit frontend:
   - cd frontend
   - chainlit run app.py
   - The UI will open at a local URL (usually http://localhost:8000).

3) Interact using the chat prompt; try:
   - search: avaliação de fonoaudiologia
   - context: terapia ocupacional desenvolvimento motor
   - info: Avaliação Antonio Damasceno Fonoaudiologia.pdf

Notes
- The frontend relies on the existing Rag agent backend (Pinecone + OpenAI). Ensure API keys for Pinecone and OpenAI are set in the environment (OPENAI_API_KEY, PINECONE_API_KEY).
- The frontend app.py loads the repository path so you can import from data.rag_agent.

If you want to customize exposure (e.g., allow direct LLM fallbacks or more commands), update frontend/app.py accordingly.