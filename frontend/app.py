"""Chainlit chat app for the SynapseTEA RAG Agent.

Com:
- Streaming de tokens (resposta aparece em tempo real)
- Steps de "pensando/buscando" quando o agente usa ferramentas
- Memória persistente entre mensagens (checkpointer SQLite por thread_id)

Run with:
    chainlit run frontend/app.py -w
"""

import sys
import uuid
from pathlib import Path

# Garante que a raiz do projeto (SynapseTEA/) esteja no sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chainlit as cl
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from data.rag_agent import create_rag_agent  # ajuste o import conforme seu projeto

from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()

DB_PATH = str(PROJECT_ROOT / "data" / "chat_memory.sqlite")

# Guarda o context manager do checkpointer aberto durante toda a vida do app.
_checkpointer_cm = None
_checkpointer = None


@cl.on_chat_start
async def on_chat_start():
    """Executa uma vez, quando uma nova sessão de chat é iniciada."""
    global _checkpointer_cm, _checkpointer

    # Abre a conexão SQLite uma única vez (compartilhada entre sessões/threads).
    if _checkpointer is None:
        _checkpointer_cm = AsyncSqliteSaver.from_conn_string(DB_PATH)
        _checkpointer = await _checkpointer_cm.__aenter__()

    agent = create_rag_agent(checkpointer=_checkpointer)
    thread_id = str(uuid.uuid4())

    cl.user_session.set("agent", agent)
    cl.user_session.set("thread_id", thread_id)

    await cl.Message(
        content="Olá! Sou o assistente da SynapseTEA. Pode me perguntar sobre os documentos."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Executa a cada mensagem enviada pelo usuário, com streaming e steps de ferramentas."""
    agent = cl.user_session.get("agent")
    thread_id = cl.user_session.get("thread_id")

    # Como o checkpointer já guarda o histórico da conversa por thread_id,
    # só precisamos mandar a mensagem NOVA do usuário — não a lista inteira.
    messages = [{"role": "user", "content": message.content}]

    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
        # "metadata": {
        #     "langfuse_session_id": thread_id,
        #     "langfuse_tags": ["synapse-tea", "rag-agent", "pinecone"],
        # },
    }

    final_msg = cl.Message(content="")
    await final_msg.send()

    open_steps: dict[str, cl.Step] = {}

    async for event in agent.astream_events(
        {"messages": messages}, config=config, version="v2"
    ):
        kind = event["event"]

        if kind == "on_tool_start":
            tool_name = event.get("name", "tool")
            run_id = event.get("run_id")
            step = cl.Step(name=f"🔎 {tool_name}", type="tool")
            step.input = event["data"].get("input")
            await step.__aenter__()
            open_steps[run_id] = step

        elif kind == "on_tool_end":
            run_id = event.get("run_id")
            step = open_steps.pop(run_id, None)
            if step is not None:
                step.output = event["data"].get("output")
                await step.__aexit__(None, None, None)

        elif kind == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            token = getattr(chunk, "content", None)
            if token:
                await final_msg.stream_token(token)

    await final_msg.update()