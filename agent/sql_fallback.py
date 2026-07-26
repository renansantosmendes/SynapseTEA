"""
Fallback SQL: permite ao agente rodar consultas SELECT arbitrárias para perguntas
que as funções fixas não cobrem. Protegido em duas camadas:
  1. Validação no código (bloqueia qualquer coisa que não seja SELECT, exige filtro de patient_id)
  2. Permissão real no Postgres (usuário agent_readonly só tem GRANT SELECT)
A camada 2 é a que realmente importa - a camada 1 é só uma primeira barreira,
nunca confie só nela.
"""

import re

MAX_ROWS = 200
BLOCKED_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "truncate",
    "grant", "revoke", "create", "copy", "execute", "call",
]


def run_readonly_sql(conn, query: str, patient_id: int):
    """
    Executa uma consulta SELECT de forma protegida.
    Espera que a query contenha um placeholder literal '{patient_id}' que será
    substituído pelo id real - isso evita que o agente esqueça o filtro
    e vaze dados de outro paciente (se algum dia houver mais de um no banco).
    """
    normalized = query.strip().lower()

    if not normalized.startswith("select"):
        raise ValueError("Apenas consultas SELECT são permitidas.")

    if ";" in query.strip().rstrip(";"):
        raise ValueError("Múltiplos statements não são permitidos (detectado ';' no meio da query).")

    for keyword in BLOCKED_KEYWORDS:
        if re.search(rf"\b{keyword}\b", normalized):
            raise ValueError(f"Palavra-chave não permitida detectada: '{keyword}'.")

    if "patient_id" not in normalized and "{patient_id}" not in query:
        raise ValueError(
            "A query deve filtrar por patient_id explicitamente (segurança: evita "
            "vazamento de dados entre pacientes caso o banco cresça)."
        )

    # Substitui o placeholder por um parâmetro real (nunca por concatenação de string)
    query_with_placeholder = query.replace("{patient_id}", "%s")

    if "limit" not in normalized:
        query_with_placeholder = query_with_placeholder.rstrip().rstrip(";") + f" LIMIT {MAX_ROWS}"

    cur = conn.cursor()
    try:
        if "%s" in query_with_placeholder:
            cur.execute(query_with_placeholder, (patient_id,))
        else:
            cur.execute(query_with_placeholder)
        cols = [desc[0] for desc in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        cur.close()

    return rows
