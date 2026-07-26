"""
Agente construído com o pacote `deepagents` (LangGraph por baixo), usando
gpt-5-nano como modelo. O harness do deepagents já traz planejamento (TODOs),
gestão de contexto e (opcionalmente) subagentes - mais robusto para perguntas
que exigem vários passos de raciocínio encadeados.

Antes de rodar:
    pip install deepagents langchain-openai psycopg2-binary openai --break-system-packages

Variáveis de ambiente necessárias:
    OPENAI_API_KEY   (usado tanto pelo modelo gpt-5-nano quanto pelos embeddings)
    PATIENT_DB_URL   (connection string do usuário agent_readonly)
    PATIENT_ID       (id do paciente no banco, ex: "1")
"""

import os
from deepagents import create_deep_agent

from langchain_tools import ALL_TOOLS
from pubmed_tool import search_pubmed
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
 
AGENT_TOOLS = ALL_TOOLS + [search_pubmed]


INSTRUCTIONS = """Você é um assistente clínico que ajuda a acompanhar a evolução de um paciente em
terapia multidisciplinar (TEA), cruzando dados de fonoaudiologia, terapia ocupacional, psicologia,
fisioterapia e AT (acompanhante terapêutico).

Use as ferramentas disponíveis para responder com base em dados reais do banco - nunca invente
informação clínica. Sempre que possível, cite a data e a disciplina/terapeuta de origem da informação.

Diretrizes de qual ferramenta usar:
- Pergunta geral de contexto sobre o paciente -> get_patient_summary
- Pergunta sobre o que aconteceu num período específico -> get_sessions_by_period
- Pergunta sobre evolução/tendência de um domínio específico ao longo do tempo -> get_domain_trend
- Pergunta que pede para cruzar/comparar visões de disciplinas diferentes -> compare_therapists_observations
- Pergunta de nuance/contexto qualitativo (não é só data ou domínio) -> search_session_notes
- Se nenhuma ferramenta específica cobrir a pergunta -> run_readonly_sql como último recurso

Para perguntas complexas que envolvem múltiplos aspectos (ex: cruzar tendência de um domínio com
o que terapeutas relataram qualitativamente sobre ele), planeje os passos necessários, chame as
ferramentas várias vezes se preciso, e só então sintetize uma resposta final direta e clínica,
evitando floreios."""


def build_agent():
    return create_deep_agent(
        model="openai:gpt-5-nano",
        tools=AGENT_TOOLS,
        system_prompt=INSTRUCTIONS,
    )


if __name__ == "__main__":
    agent = build_agent()

    test_questions = [
        "Existe evidência científica sobre o uso do hiperfoco como estratégia terapêutica em crianças com TEA?",
    ]

    for q in test_questions:
        print(f"\n{'='*70}\nPERGUNTA: {q}\n{'='*70}")
        result = agent.invoke({"messages": [{"role": "user", "content": q}]})
        final_message = result["messages"][-1]
        print(f"\nRESPOSTA:\n{final_message.content}\n")
