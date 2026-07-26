"""
Script para criar as tabelas no seu banco Neon.
Rode isso NO SEU COMPUTADOR (não neste ambiente), pois aqui não tenho acesso de rede ao Postgres.

Antes de rodar:
    pip install psycopg2-binary --break-system-packages
    (ou apenas `pip install psycopg2-binary` no seu ambiente local)
"""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Cole sua connection string aqui (ou melhor: use variável de ambiente)
CONNECTION_STRING = f"postgresql://neondb_owner:{os.environ['NEON_PASSWORD']}@ep-noisy-boat-axo08bx1.c-4.us-east-2.aws.neon.tech/neondb?sslmode=require"

def main():
    conn = psycopg2.connect(CONNECTION_STRING)
    cur = conn.cursor()

    with open("C:\\00 - Source Codes\\SynapseTEA\\data\\neon\\schema.sql", "r", encoding="utf-8") as f:
        schema_sql = f.read()

    cur.execute(schema_sql)
    conn.commit()

    print("Tabelas criadas com sucesso!")

    # Verifica quais tabelas foram criadas
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    tables = cur.fetchall()
    print("\nTabelas no banco:")
    for t in tables:
        print(f"  - {t[0]}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
