"""
Mesmas 5 consultas de queries.py + o fallback SQL, agora expostas como
LangChain tools (@tool), prontas para serem passadas a um agente
(deepagents, LangGraph, ou qualquer agent runtime compatível com LangChain).

Cada tool abre sua própria conexão de curta duração com o Postgres
(simples e seguro para o volume de chamadas de um agente conversacional).
Se o volume crescer, trocar por um pool (psycopg2.pool ou SQLAlchemy) é o
próximo passo natural.
"""

import os
import json
from datetime import date
from typing import Optional

import psycopg2
from openai import OpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv

from sql_fallback import run_readonly_sql as _run_readonly_sql

load_dotenv()  # Load environment variables from .env file

CONNECTION_STRING = f"postgresql://neondb_owner:{os.environ.get('NEON_PASSWORD')}@ep-noisy-boat-axo08bx1.c-4.us-east-2.aws.neon.tech/neondb?sslmode=require"
PATIENT_ID = int(os.environ.get("PATIENT_ID", "2"))
EMBEDDING_MODEL = "text-embedding-3-large"

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _connect():
    return psycopg2.connect(CONNECTION_STRING)


def _rows_to_dicts(cur):
    cols = [desc[0] for desc in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _json(obj):
    def default(o):
        if isinstance(o, date):
            return o.isoformat()
        return str(o)
    return json.dumps(obj, ensure_ascii=False, default=default)


@tool
def get_patient_summary() -> str:
    """Retorna dados básicos do paciente, diagnóstico e contagem de sessões por especialidade.
    Use para perguntas gerais de contexto sobre o paciente."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT p.full_name, p.date_of_birth, p.sex, d.name AS diagnosis
            FROM patients p
            LEFT JOIN diagnoses d ON d.patient_id = p.id
            WHERE p.id = %s
            """,
            (PATIENT_ID,),
        )
        patient = _rows_to_dicts(cur)

        cur.execute(
            """
            SELECT specialty, COUNT(*) AS total_sessions,
                   MIN(session_date) AS first_session, MAX(session_date) AS last_session
            FROM sessions
            WHERE patient_id = %s
            GROUP BY specialty
            ORDER BY total_sessions DESC
            """,
            (PATIENT_ID,),
        )
        sessions_by_specialty = _rows_to_dicts(cur)
        cur.close()
        return _json({"patient": patient, "sessions_by_specialty": sessions_by_specialty})
    finally:
        conn.close()


@tool
def get_sessions_by_period(start_date: str, end_date: str, specialty: Optional[str] = None) -> str:
    """Lista sessões e notas de progresso num intervalo de datas (formato YYYY-MM-DD),
    opcionalmente filtrando por especialidade (ex: 'Fonoaudiologia', 'Terapia Ocupacional',
    'Psicologia', 'Fisioterapia'). Use para perguntas sobre o que aconteceu num período específico."""
    conn = _connect()
    try:
        cur = conn.cursor()
        query = """
            SELECT s.session_date, s.specialty, pe.canonical_name AS therapist,
                   s.progress_notes, s.homework_assigned
            FROM sessions s
            LEFT JOIN people pe ON pe.id = s.therapist_id
            WHERE s.patient_id = %s AND s.session_date BETWEEN %s AND %s
        """
        params = [PATIENT_ID, date.fromisoformat(start_date), date.fromisoformat(end_date)]
        if specialty:
            query += " AND s.specialty = %s"
            params.append(specialty)
        query += " ORDER BY s.session_date"

        cur.execute(query, params)
        result = _rows_to_dicts(cur)
        cur.close()
        return _json(result)
    finally:
        conn.close()


@tool
def get_domain_trend(domain_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Cruza skills, metas de PEI e desafios comportamentais ligados a um domínio específico
    (ex: 'Regulação Emocional', 'Motor Fino', 'Comunicação Expressiva'), mostrando status ao
    longo do tempo. Use para perguntas de evolução/tendência num domínio específico."""
    conn = _connect()
    try:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT 'skill' AS source, sk.name, sk.status, sk.evidence, sk.recorded_date AS event_date
            FROM skills sk
            JOIN domains d ON d.id = sk.domain_id
            WHERE sk.patient_id = %s AND d.name = %s
            """,
            (PATIENT_ID, domain_name),
        )
        skills = _rows_to_dicts(cur)

        cur.execute(
            """
            SELECT 'iep_goal' AS source, ig.objective AS name, ig.acquisition_status AS status,
                   ig.description AS evidence, ig.last_updated AS event_date
            FROM iep_goals ig
            JOIN domains d ON d.id = ig.domain_id
            WHERE d.name = %s
            """,
            (domain_name,),
        )
        iep_goals = _rows_to_dicts(cur)

        cur.execute(
            """
            SELECT 'behavioural_challenge' AS source, bc.name, bc.current_status AS status,
                   bc.description AS evidence, NULL::date AS event_date
            FROM behavioural_challenges bc
            JOIN domains d ON d.id = bc.domain_id
            WHERE bc.patient_id = %s AND d.name = %s
            """,
            (PATIENT_ID, domain_name),
        )
        challenges = _rows_to_dicts(cur)
        cur.close()

        all_events = skills + iep_goals + challenges
        if start_date and end_date:
            sd, ed = date.fromisoformat(start_date), date.fromisoformat(end_date)
            all_events = [e for e in all_events if e["event_date"] is None or (sd <= e["event_date"] <= ed)]

        all_events = sorted(all_events, key=lambda e: (e["event_date"] is None, e["event_date"]))
        return _json(all_events)
    finally:
        conn.close()


@tool
def compare_therapists_observations(start_date: str, end_date: str) -> str:
    """Agrupa observações de todas as disciplinas num mesmo período (formato YYYY-MM-DD),
    lado a lado, para comparar perspectivas diferentes sobre o mesmo momento. Use quando a
    pergunta pede para cruzar visões de terapeutas/disciplinas diferentes."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT s.specialty, s.session_date, pe.canonical_name AS therapist, s.progress_notes
            FROM sessions s
            LEFT JOIN people pe ON pe.id = s.therapist_id
            WHERE s.patient_id = %s AND s.session_date BETWEEN %s AND %s
            ORDER BY s.session_date, s.specialty
            """,
            (PATIENT_ID, date.fromisoformat(start_date), date.fromisoformat(end_date)),
        )
        result = _rows_to_dicts(cur)
        cur.close()
        return _json(result)
    finally:
        conn.close()


@tool
def search_session_notes(query_text: str, note_type: Optional[str] = None, top_k: int = 8) -> str:
    """Busca semântica (RAG) nas notas de sessão por significado, não por palavra exata.
    Use para perguntas de nuance/contexto que não são só data ou domínio, ex: 'o que os
    terapeutas notaram sobre frustração'. note_type pode ser 'activity', 'observation',
    'directive' ou 'principle'."""
    embedding_response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[query_text])
    query_embedding = embedding_response.data[0].embedding

    conn = _connect()
    try:
        cur = conn.cursor()
        query = """
            SELECT sn.content, sn.note_type, s.session_date, s.specialty,
                   sn.embedding <-> %s::vector AS distance
            FROM session_notes sn
            JOIN sessions s ON s.id = sn.session_id AND sn.source_table = 'sessions'
            WHERE s.patient_id = %s AND sn.embedding IS NOT NULL
        """
        params = [query_embedding, PATIENT_ID]
        if note_type:
            query += " AND sn.note_type = %s"
            params.append(note_type)
        query += " ORDER BY distance LIMIT %s"
        params.append(top_k)

        cur.execute(query, params)
        result = _rows_to_dicts(cur)
        cur.close()
        return _json(result)
    finally:
        conn.close()


@tool
def run_readonly_sql(query: str) -> str:
    """Fallback: roda uma consulta SQL SELECT customizada quando nenhuma outra ferramenta
    cobre a pergunta. A query DEVE conter o placeholder literal {patient_id} no WHERE
    (será substituído pelo id real). Apenas SELECT é permitido; sem INSERT/UPDATE/DELETE/DROP.
    Tabelas disponíveis: patients, diagnoses, clinics, people, domains, sessions, session_notes,
    at_supervisions, evaluations, ieps, iep_goals, skills, behavioural_challenges, strategies, profiles."""
    conn = _connect()
    try:
        result = _run_readonly_sql(conn, query, PATIENT_ID)
        return _json(result)
    except Exception as e:
        return f"Erro ao executar a consulta: {e}"
    finally:
        conn.close()


ALL_TOOLS = [
    get_patient_summary,
    get_sessions_by_period,
    get_domain_trend,
    compare_therapists_observations,
    search_session_notes,
    run_readonly_sql,
]
