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
import logging
from data.rag_agent import create_rag_agent
import requests
from datetime import datetime
import pathlib

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

LOG = logging.getLogger("frontend.app")

# Agent instance (created on first use)
agent = None
# Last tool outputs (populated after agent runs)
last_tool_calls = None
last_tool_outputs = None
# Streaming flag (default off)
STREAMING = os.environ.get("SYNAPSE_STREAM", "0") in ("1", "true", "yes")

rag_tools = None

# Langfuse settings (optional)
LANGFUSE_API_KEY = os.environ.get("LANGFUSE_API_KEY")
LANGFUSE_PROJECT = os.environ.get("LANGFUSE_PROJECT")
LANGFUSE_API_URL = os.environ.get("LANGFUSE_API_URL", "https://api.langfuse.com/v1/events")

# Ensure logs directory
LOGS_DIR = pathlib.Path(ROOT) / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LF_LOGFILE = LOGS_DIR / "langfuse_integration.log"


def send_langfuse_event(event_type: str, payload: dict) -> None:
    """Send an event to Langfuse (best-effort)."""
    try:
        if not LANGFUSE_API_KEY or not LANGFUSE_PROJECT:
            return
        headers = {
            "Authorization": f"Bearer {LANGFUSE_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "project": LANGFUSE_PROJECT,
            "type": event_type,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        resp = requests.post(LANGFUSE_API_URL, headers=headers, json=body, timeout=10)
        with open(LF_LOGFILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()} {event_type} => {resp.status_code} {resp.text}\n")
    except Exception as e:
        LOG.warning("Langfuse event error: %s", e)
    try:
        with open(LF_LOGFILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()} ERROR {e}\n")
    except Exception:
        pass


@cl.on_chat_start
async def on_start():
    await cl.Message(content=(
        "Bem-vindo ao SynapseTEA RAG — interface de consulta dos documentos clínicos.\n\n"
        "Comandos disponíveis:\n"
        "- search: <consulta> — busca semântica (retorna títulos e trechos relevantes).\n"
        "- context: <consulta> — recupera contexto formatado para RAG.\n"
        "- info: <nome_do_arquivo.pdf> — exibe estatísticas e amostras do documento.\n"
        "- show tools — mostra detalhes das ferramentas usadas na última resposta (se houver).\n"
        "- stream:on / stream:off — ativa/desativa streaming de resposta.\n\n"
        "Digite sua pergunta ou um comando. O agente decidirá automaticamente se precisa consultar a base de documentos.")
    ).send()


@cl.on_message
async def on_message(message):
    global rag_tools
    if PineconeRAGTools is None:
        await cl.Message(content="Backend not available. Check environment and imports.").send()
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

    # Show stored tool outputs on demand
    if text.lower() in ("show tools", "show tool outputs", "show-tool-calls"):
        if last_tool_outputs:
            await cl.Message(content=json.dumps(last_tool_outputs, ensure_ascii=False, indent=2)).send()
        else:
            await cl.Message(content="No tool outputs available.").send()
        return

    # Toggle streaming
    if text.lower() in ("stream:on", "stream on"):
        global STREAMING
        STREAMING = True
        await cl.Message(content="Streaming enabled.").send()
        return
    if text.lower() in ("stream:off", "stream off"):
        STREAMING = False
        await cl.Message(content="Streaming disabled.").send()
        return

    # Non-command: treat as a natural question to the agent -> RAG + LLM response
    try:
        await cl.Message(content="Thinking...").send()

        # Ask the deepagents agent to answer the question (agent uses internal tools)
        async def call_agent(question: str) -> str:
            global agent
            if agent is None:
                agent = create_rag_agent()

            def _call():
                # Try several invocation patterns until one works (prefer invoke with input)
                errs = []
                # 1) invoke with {'input': question}
                try:
                    if hasattr(agent, "invoke"):
                        return agent.invoke({"input": question})
                except Exception as e:
                    errs.append(("invoke_input", e))

                # 2) run(str)
                try:
                    if hasattr(agent, "run"):
                        return agent.run(question)
                except Exception as e:
                    errs.append(("run", e))

                # 3) invoke with messages
                try:
                    if hasattr(agent, "invoke"):
                        payload = {"messages": [{"role": "user", "content": question}]}
                        return agent.invoke(payload)
                except Exception as e:
                    errs.append(("invoke_messages", e))

                # 4) callable with dict
                try:
                    if callable(agent):
                        return agent({"input": question})
                except Exception as e:
                    errs.append(("callable_dict", e))

                msg = "Agent invocation failed. Attempts: " + ", ".join(f"{k}:{type(v).__name__}" for k, v in errs)
                raise RuntimeError(msg)

            return await asyncio.to_thread(_call)

        async def call_agent_with_messages(messages: list) -> str:
            """Call agent with a messages payload (LangChain-style)."""
            global agent
            if agent is None:
                agent = create_rag_agent()

            def _call():
                # Prefer invoke with messages
                if hasattr(agent, "invoke"):
                    return agent.invoke({"messages": messages})
                # fallback: try calling with dict
                if callable(agent):
                    return agent({"messages": messages})
                raise RuntimeError("Agent does not support messages invocation")

            return await asyncio.to_thread(_call)

        # Pre-check: run a light semantic search to decide if documents are likely relevant
        try:
            pre_search_raw = rag_tools.search_documents(text, top_k=1)
            pre_search = json.loads(pre_search_raw)
            top_score = 0.0
            if pre_search.get("results"):
                top_score = float(pre_search["results"][0].get("score", 0.0))
            # send langfuse pre-check event
            send_langfuse_event("pre_search", {"query": text, "top_score": top_score})
        except Exception:
            pre_search = None
            top_score = 0.0

        # If top_score is high, retrieve context and include in agent messages
        if top_score >= 0.60:
            try:
                context = rag_tools.retrieve_context(text, top_k=3)
            except Exception:
                context = None

            # Build messages array to pass to agent.invoke
            messages = []
            # system instruction in pt-br: short
            messages.append({"role": "system", "content": "Use o contexto fornecido a seguir apenas se for necessário para responder; não retorne o contexto bruto, apenas use-o internamente e cite fontes se usadas."})
            messages.append({"role": "user", "content": text})
            if context:
                messages.append({"role": "tool_context", "content": context})

            # Call the agent preferring messages payload
            try:
                result = await call_agent_with_messages(messages)
            except Exception:
                result = await call_agent(text)
        else:
            result = await call_agent(text)


        # Capture tool call metadata if present
        try:
            if isinstance(result, dict):
                last_tool_calls = result.get("tool_calls") or result.get("tools_called") or result.get("toolOutputs")
                # store full outputs if present
                last_tool_outputs = result.get("tool_outputs") or result.get("tools_output") or result.get("toolOutputs")
        except Exception:
            pass

        # Send logs to Langfuse (best-effort)
        try:
            payload = {
                "user_query": text,
                "tools_called": last_tool_calls,
                "tools_outputs": last_tool_outputs,
                "agent_result_type": type(result).__name__,
            }
            send_langfuse_event("agent_response", payload)
            LOG.info("Langfuse event sent for query")
        except Exception as e:
            LOG.warning("Failed sending langfuse event: %s", e)

        # Parse agent result to extract final reply text (prefer last AI message)
        reply = None
        try:
            if isinstance(result, str):
                reply = result
            elif isinstance(result, dict):
                # If result contains a 'messages' sequence (LangChain style), extract last AI message
                if "messages" in result and isinstance(result["messages"], (list, tuple)):
                    msgs = result["messages"]
                    # Iterate in reverse to find the last non-empty assistant/AI message
                    for m in reversed(msgs):
                        # support dict-like message
                        if isinstance(m, dict):
                            content = m.get("content") or m.get("text")
                            role = m.get("role")
                            if content and (role in ("assistant", "ai", None) or "ai" in str(type(m)).lower()):
                                reply = content
                                break
                        else:
                            # object with attributes
                            content = getattr(m, "content", None) or getattr(m, "text", None)
                            role = getattr(m, "role", None)
                            # AIMessage often has no role but is not a HumanMessage
                            if content and (role in ("assistant", "ai") or "ai" in type(m).__name__.lower() or "ai" in str(getattr(m, "__class__", "")).lower()):
                                reply = content
                                break
                    # if still no reply, fallback to stringifying the messages
                    if reply is None:
                        reply = "\n".join(
                            (getattr(m, "content", None) or m.get("content") if isinstance(m, dict) else str(m)) for m in msgs
                        )
                else:
                    # common keys
                    for key in ("output", "result", "response", "text", "answer"):
                        if key in result:
                            reply = result[key]
                            break
                    # fallback: stringify
                    if reply is None:
                        reply = json.dumps(result, ensure_ascii=False, indent=2)
            else:
                reply = str(result)
        except Exception:
            reply = str(result)

        # Ensure reply is a plain string
        if not isinstance(reply, str):
            reply = str(reply)

        # If tools were used, inform user briefly but keep reply final
        if last_tool_calls:
            tools_list = []
            try:
                # normalize tools list
                if isinstance(last_tool_calls, (list, tuple)):
                    tools_list = [t if isinstance(t, str) else str(t) for t in last_tool_calls]
                else:
                    tools_list = [str(last_tool_calls)]
            except Exception:
                tools_list = [str(last_tool_calls)]
            await cl.Message(content=f"(Tools used: {', '.join(tools_list)}). Reply 'show tools' to view details.").send()

        # Streaming or normal send
        if STREAMING:
            # pseudo-stream: send chunks sequentially
            chunk_size = 120
            for i in range(0, len(reply), chunk_size):
                part = reply[i : i + chunk_size]
                await cl.Message(content=part).send()
                await asyncio.sleep(0.06)
        else:
            await cl.Message(content=reply).send()

    except Exception as e:
        LOG.exception("Failed to generate RAG answer")
        await cl.Message(content=f"Error: {e}").send()
