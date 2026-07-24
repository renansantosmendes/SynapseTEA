#!/usr/bin/env python3
"""Chainlit frontend for SynapseTEA RAG.

This frontend talks to the local Rag agent implemented in the repository
and exposes a simple chat interface via Chainlit.
"""

import os
import sys
import json
import pathlib

# Make sure the repository root is on the Python path so we can import the Rag agent
ROOT = pathlib.Path(__file__).resolve().parents[1]  # two levels up: frontend/ -> SynapseTEA
ROOT = str(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from data.rag_agent import PineconeRAGTools
except Exception as e:
    # If import fails, provide a friendly message
    PineconeRAGTools = None
    print(f"Warning: could not import Rag agent backend: {e}")

import chainlit as cl
import asyncio
import openai
import logging

LOG = logging.getLogger("frontend.app")

# Default LLM model for replies
DEFAULT_LLM = os.environ.get("SYNAPSE_LLM", "gpt-4o-mini")

rag_tools = None


@cl.on_chat_start
async def on_start():
    await cl.Message(content=(
        "Welcome to SynapseTEA RAG Frontend. Use commands like 'search: ...', 'context: ...', 'info: ...'."
    )).send()


@cl.on_message
async def on_message(message):
    global rag_tools
    if PineconeRAGTools is None:
        cl.Message(content="Backend not available. Check environment and imports.").send()
        return
    if rag_tools is None:
        rag_tools = PineconeRAGTools()

    # Accept both plain strings and Chainlit Message objects
    if isinstance(message, str):
        text = message.strip()
    else:
        text = getattr(message, "text", None) or getattr(message, "content", None)
        if isinstance(text, str):
            text = text.strip()
        else:
            text = str(text)
    if not text:
        return

    if text.lower().startswith("search:"):
        query = text[7:].strip()
        if query:
            results = rag_tools.search_documents(query, top_k=5)
            await cl.Message(content=results).send()
        else:
            await cl.Message(content="Please provide a search query after 'search:'.").send()
        return

    if text.lower().startswith("context:"):
        query = text[8:].strip()
        if query:
            context = rag_tools.retrieve_context(query, top_k=3)
            await cl.Message(content=context).send()
        else:
            await cl.Message(content="Please provide a context query after 'context:'.").send()
        return

    if text.lower().startswith("info:"):
        filename = text[5:].strip()
        if filename:
            info = rag_tools.get_document_info(filename)
            await cl.Message(content=info).send()
        else:
            await cl.Message(content="Please provide a filename after 'info:'.").send()
        return

    # Non-command: treat as a natural question to the agent -> RAG + LLM response
    try:
        await cl.Message(content="Thinking...").send()

        # Retrieve context and search results
        context = rag_tools.retrieve_context(text, top_k=3)
        search_results = rag_tools.search_documents(text, top_k=5)

        # Compose prompt for LLM
        system_prompt = (
            "You are a helpful assistant that answers questions using the provided context. "
            "Cite sources when possible in the form [file, p.X]. If unsure, say you don't know."
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {text}\n\nAnswer based only on the context above."

        async def call_openai(prompt: str) -> str:
            try:
                # Use blocking call in thread to avoid blocking the event loop
                def _call():
                    openai.api_key = os.environ.get("OPENAI_API_KEY")
                    resp = openai.ChatCompletion.create(
                        model=DEFAULT_LLM,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=512,
                        temperature=0.2,
                    )
                    # Extract text
                    return resp["choices"][0]["message"]["content"].strip()

                return await asyncio.to_thread(_call)
            except Exception as e:
                LOG.exception("OpenAI call failed")
                return f"Error generating response: {e}"

        answer = await call_openai(user_prompt)

        # Send combined response with context summary and LLM answer
        out = f"Context:\n{context}\n\nAnswer:\n{answer}\n\nRaw search results:\n{search_results}"
        await cl.Message(content=out).send()

    except Exception as e:
        LOG.exception("Failed to generate RAG answer")
        await cl.Message(content=f"Error: {e}").send()
