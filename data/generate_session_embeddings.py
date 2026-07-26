"""
Gera embeddings para todos os registros de session_notes que ainda não têm embedding,
e salva de volta na coluna `embedding` (pgvector).

Antes de rodar:
    pip install psycopg2-binary openai --break-system-packages

Você precisa de uma OPENAI_API_KEY (https://platform.openai.com/api-keys).
Se preferir outro provider de embeddings (ex: Voyage, Cohere), me avisa que adapto o script.
"""

import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Carrega variáveis de ambiente do arquivo .env

# Use a connection string do usuário com permissão de ESCRITA aqui
# (agent_readonly não serve pra isso, pois só tem SELECT)
CONNECTION_STRING = f"postgresql://neondb_owner:{os.environ.get('NEON_PASSWORD')}@ep-noisy-boat-axo08bx1.c-4.us-east-2.aws.neon.tech/neondb?sslmode=require"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # ou cole direto aqui (não recomendado)
EMBEDDING_MODEL = "text-embedding-3-large"  # 3072 dimensões por padrão
BATCH_SIZE = 50  # quantos textos enviar por chamada de API


def get_rows_without_embedding(cur):
    cur.execute(
        """
        SELECT id, content FROM session_notes
        WHERE embedding IS NULL AND content IS NOT NULL AND content != ''
        ORDER BY id
        """
    )
    return cur.fetchall()


def generate_embeddings(client, texts):
    """Envia uma lista de textos e retorna lista de vetores na mesma ordem."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def main():
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Defina a variável de ambiente OPENAI_API_KEY antes de rodar. "
            "Ex (PowerShell): $env:OPENAI_API_KEY='sk-...'"
        )

    client = OpenAI(api_key=OPENAI_API_KEY)
    conn = psycopg2.connect(CONNECTION_STRING)
    cur = conn.cursor()

    rows = get_rows_without_embedding(cur)
    print(f"Registros sem embedding encontrados: {len(rows)}")

    if not rows:
        print("Nada a fazer - todos os registros já têm embedding.")
        return

    total_processed = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]

        try:
            embeddings = generate_embeddings(client, texts)
        except Exception as e:
            print(f"[ERRO] Falha ao gerar embeddings para o lote {i}-{i+len(batch)}: {e}")
            conn.rollback()
            raise

        for note_id, embedding in zip(ids, embeddings):
            cur.execute(
                "UPDATE session_notes SET embedding = %s WHERE id = %s",
                (embedding, note_id),
            )

        conn.commit()
        total_processed += len(batch)
        print(f"  Processados {total_processed}/{len(rows)}...")

    print(f"\nConcluído! {total_processed} registros com embedding gerado e salvo.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()