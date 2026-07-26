"""
Lê patient_record.json e insere todos os dados no banco Neon.
Rode NO SEU COMPUTADOR (este ambiente não tem acesso de rede ao Postgres).

Antes de rodar:
    pip install psycopg2-binary --break-system-packages

Passos:
    1. Rode primeiro setup_db.py (cria as tabelas)
    2. Ajuste CONNECTION_STRING abaixo
    3. Coloque patient_record.json na mesma pasta
    4. Rode: python insert_data.py
"""
import os
import json
import psycopg2
from dotenv import load_dotenv

load_dotenv()

CONNECTION_STRING = f"postgresql://neondb_owner:{os.environ['NEON_PASSWORD']}@ep-noisy-boat-axo08bx1.c-4.us-east-2.aws.neon.tech/neondb?sslmode=require"
JSON_PATH = "C:\\00 - Source Codes\\SynapseTEA\\patient_record.json"

# ---------------------------------------------------------------------------
# MAPA DE DEDUPLICAÇÃO MANUAL
# IMPORTANTE: dedup automático de nomes é arriscado (dado sensível de família).
# Revise esse mapa você mesmo antes de rodar - eu apenas agrupei pelo que
# pareceu óbvio a partir dos nomes/apelidos que apareceram no JSON.
# Chave = como aparece no JSON, Valor = nome canônico que você decide usar.
# ---------------------------------------------------------------------------
FAMILY_NAME_MAP = {
    "Gabriela": "Gabriela Rodrigues Damasceno Mendes",
    "Gabriela Rodrigues Damasceno": "Gabriela Rodrigues Damasceno Mendes",
    "Gabriela Rodrigues Damasceno Mendes": "Gabriela Rodrigues Damasceno Mendes",
    "Mãe": "Gabriela Rodrigues Damasceno Mendes",
    "": None,  # nome vazio - será ignorado
    "Renan Santos": "Renan Santos Mendes",
    "Renan": "Renan Santos Mendes",
    "Renan Santos Mendes": "Renan Santos Mendes",
    "Maria Clara": "Maria Clara Damasceno Mendes",
    "Clarinha": "Maria Clara Damasceno Mendes",
    "Clara": "Maria Clara Damasceno Mendes",
    "Manu": "Maria Clara Damasceno Mendes",  # ATENÇÃO: revise - "Manu" pode ser apelido da irmã ou outra pessoa (ex: AT). Confirme antes de rodar.
}

THERAPIST_NAME_MAP = {
    "Helen Rose": "Helen Rose Oliveira Alves",
    "Helen Rose Oliveira Alves": "Helen Rose Oliveira Alves",
    "Ana Clara Bacelar": "Ana Clara Bacelar Silva",
    "Ana Clara Bacelar Silva": "Ana Clara Bacelar Silva",
}



def load_json():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
 
 
def clean(value):
    """Trata string vazia como None."""
    if value in ("", None):
        return None
    return value
 
 
def insert_patient(cur, patient):
    cur.execute(
        """
        INSERT INTO patients (full_name, date_of_birth, sex, speaks_in_third_person,
                               echolalia_present, toe_walking)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            clean(patient.get("full_name")),
            clean(patient.get("date_of_birth")),
            clean(patient.get("sex")),
            patient.get("speaks_in_third_person"),
            patient.get("echolalia_present"),
            patient.get("toe_walking"),
        ),
    )
    return cur.fetchone()[0]
 
 
def insert_diagnosis(cur, patient_id, diagnosis):
    cur.execute(
        """
        INSERT INTO diagnoses (patient_id, name, icd_code, date_received, diagnosing_physician)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            patient_id,
            clean(diagnosis.get("name")),
            clean(diagnosis.get("icd_code")),
            clean(diagnosis.get("date_received")),
            clean(diagnosis.get("diagnosing_physician")),
        ),
    )
 
 
def insert_clinics(cur, clinics):
    """Retorna dict nome -> id"""
    clinic_ids = {}
    for c in clinics:
        name = clean(c.get("name"))
        if not name:
            continue
        cur.execute(
            "INSERT INTO clinics (name, address, phone) VALUES (%s, %s, %s) RETURNING id",
            (name, clean(c.get("address")), clean(c.get("phone"))),
        )
        clinic_ids[name] = cur.fetchone()[0]
    return clinic_ids
 
 
def insert_people(cur, family_members, therapists):
    """
    Deduplica usando os mapas manuais acima e insere em `people`.
    Retorna dict nome_original -> id (para podermos ligar sessions/evaluations depois)
    """
    canonical_to_id = {}
    name_to_id = {}
 
    # Família
    for fm in family_members:
        raw_name = clean(fm.get("name"))
        if raw_name is None:
            continue
        canonical = FAMILY_NAME_MAP.get(raw_name, raw_name)
        if canonical is None:
            continue
        if canonical not in canonical_to_id:
            cur.execute(
                """
                INSERT INTO people (canonical_name, aliases, person_type, relationship)
                VALUES (%s, %s, 'family', %s)
                RETURNING id
                """,
                (canonical, [raw_name], clean(fm.get("relationship"))),
            )
            pid = cur.fetchone()[0]
            canonical_to_id[canonical] = pid
        else:
            pid = canonical_to_id[canonical]
            cur.execute(
                "UPDATE people SET aliases = array_append(aliases, %s) WHERE id = %s AND NOT (%s = ANY(aliases))",
                (raw_name, pid, raw_name),
            )
        name_to_id[raw_name] = pid
 
    # Terapeutas
    for t in therapists:
        raw_name = clean(t.get("name"))
        if raw_name is None:
            continue
        canonical = THERAPIST_NAME_MAP.get(raw_name, raw_name)
        if canonical not in canonical_to_id:
            cur.execute(
                """
                INSERT INTO people (canonical_name, aliases, person_type, specialty, registration)
                VALUES (%s, %s, 'therapist', %s, %s)
                RETURNING id
                """,
                (canonical, [raw_name], clean(t.get("specialty")), clean(t.get("registration"))),
            )
            pid = cur.fetchone()[0]
            canonical_to_id[canonical] = pid
        else:
            pid = canonical_to_id[canonical]
            cur.execute(
                "UPDATE people SET aliases = array_append(aliases, %s) WHERE id = %s AND NOT (%s = ANY(aliases))",
                (raw_name, pid, raw_name),
            )
        name_to_id[raw_name] = pid
 
    return name_to_id
 
 
def get_or_create_domain(cur, domain_cache, name):
    name = clean(name)
    if name is None:
        return None
    if name in domain_cache:
        return domain_cache[name]
    cur.execute(
        """
        INSERT INTO domains (name) VALUES (%s)
        ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
        RETURNING id
        """,
        (name,),
    )
    domain_id = cur.fetchone()[0]
    domain_cache[name] = domain_id
    return domain_id
 
 
def insert_sessions(cur, patient_id, sessions, name_to_id):
    for s in sessions:
        therapist_name = clean(s.get("therapist_name"))
        therapist_id = name_to_id.get(therapist_name)
 
        cur.execute(
            """
            INSERT INTO sessions (patient_id, therapist_id, specialty, session_date,
                                   progress_notes, homework_assigned)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                patient_id,
                therapist_id,
                clean(s.get("specialty")),
                clean(s.get("session_date")),
                clean(s.get("progress_notes")),
                clean(s.get("homework_assigned")),
            ),
        )
        session_id = cur.fetchone()[0]
 
        for activity in s.get("main_activities", []):
            if clean(activity):
                cur.execute(
                    """
                    INSERT INTO session_notes (session_id, source_table, note_type, content)
                    VALUES (%s, 'sessions', 'activity', %s)
                    """,
                    (session_id, activity),
                )
 
 
def insert_at_supervisions(cur, patient_id, at_supervisions):
    for a in at_supervisions:
        cur.execute(
            "INSERT INTO at_supervisions (patient_id, session_date) VALUES (%s, %s) RETURNING id",
            (patient_id, clean(a.get("session_date"))),
        )
        at_id = cur.fetchone()[0]
 
        for obs in a.get("key_observations", []):
            if clean(obs):
                cur.execute(
                    """
                    INSERT INTO session_notes (session_id, source_table, note_type, content)
                    VALUES (%s, 'at_supervisions', 'observation', %s)
                    """,
                    (at_id, obs),
                )
        for d in a.get("directives", []):
            if clean(d):
                cur.execute(
                    """
                    INSERT INTO session_notes (session_id, source_table, note_type, content)
                    VALUES (%s, 'at_supervisions', 'directive', %s)
                    """,
                    (at_id, d),
                )
        for p in a.get("session_structure_principles", []):
            if clean(p):
                cur.execute(
                    """
                    INSERT INTO session_notes (session_id, source_table, note_type, content)
                    VALUES (%s, 'at_supervisions', 'principle', %s)
                    """,
                    (at_id, p),
                )
 
 
def insert_evaluations(cur, patient_id, evaluations, name_to_id, clinic_ids):
    for e in evaluations:
        evaluator_name = clean(e.get("evaluator"))
        evaluator_id = name_to_id.get(evaluator_name)
        clinic_id = clinic_ids.get(clean(e.get("clinic_name")))
 
        cur.execute(
            """
            INSERT INTO evaluations (patient_id, evaluation_type, evaluator_id, evaluation_date,
                                      clinic_id, instruments_used, main_findings, recommendations)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                patient_id,
                clean(e.get("evaluation_type")),
                evaluator_id,
                clean(e.get("evaluation_date")),
                clinic_id,
                e.get("instruments_used", []),
                e.get("main_findings", []),
                e.get("recommendations", []),
            ),
        )
 
 
def insert_ieps(cur, patient_id, ieps, domain_cache):
    for i in ieps:
        cur.execute(
            """
            INSERT INTO ieps (patient_id, start_date, institution, review_date)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (
                patient_id,
                clean(i.get("start_date")),
                clean(i.get("institution")),
                clean(i.get("review_date")),
            ),
        )
        iep_id = cur.fetchone()[0]
 
        for goal in i.get("goals", []):
            domain_id = get_or_create_domain(cur, domain_cache, goal.get("domain"))
            cur.execute(
                """
                INSERT INTO iep_goals (iep_id, domain_id, objective, description,
                                        acquisition_status, last_updated, strategies)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    iep_id,
                    domain_id,
                    clean(goal.get("objective")),
                    clean(goal.get("description")),
                    clean(goal.get("acquisition_status")),
                    clean(goal.get("last_updated")),
                    goal.get("strategies", []),
                ),
            )
 
 
def insert_skills(cur, patient_id, skills, domain_cache):
    for s in skills:
        domain_id = get_or_create_domain(cur, domain_cache, s.get("domain"))
        cur.execute(
            """
            INSERT INTO skills (patient_id, name, domain_id, status, evidence)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (patient_id, clean(s.get("name")), domain_id, clean(s.get("status")), clean(s.get("evidence"))),
        )
 
 
def insert_behavioural_challenges(cur, patient_id, challenges, domain_cache):
    for c in challenges:
        domain_id = get_or_create_domain(cur, domain_cache, c.get("domain"))
        cur.execute(
            """
            INSERT INTO behavioural_challenges (patient_id, name, description, domain_id, current_status)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (patient_id, clean(c.get("name")), clean(c.get("description")), domain_id, clean(c.get("current_status"))),
        )
 
 
def insert_strategies(cur, patient_id, strategies, domain_cache):
    for s in strategies:
        domains = s.get("target_domains") or []
        domain_id = get_or_create_domain(cur, domain_cache, domains[0]) if domains else None
        cur.execute(
            """
            INSERT INTO strategies (patient_id, name, description, domain_id)
            VALUES (%s, %s, %s, %s)
            """,
            (patient_id, clean(s.get("name")), clean(s.get("description")), domain_id),
        )
 
 
def insert_profiles(cur, patient_id, sensory, motor, communication):
    for profile_type, data in [
        ("sensory", sensory),
        ("motor", motor),
        ("communication", communication),
    ]:
        data_copy = {k: v for k, v in data.items() if k != "patient_name"}
        cur.execute(
            """
            INSERT INTO profiles (patient_id, profile_type, data)
            VALUES (%s, %s, %s)
            """,
            (patient_id, profile_type, json.dumps(data_copy, ensure_ascii=False)),
        )
 
 
def main():
    data = load_json()
    conn = psycopg2.connect(CONNECTION_STRING)
    cur = conn.cursor()
 
    try:
        print("Inserindo paciente...")
        patient_id = insert_patient(cur, data["patient"])
        print(f"  OK - paciente inserido, id={patient_id}")
 
        print("Inserindo diagnóstico...")
        insert_diagnosis(cur, patient_id, data["diagnosis"])
        print("  OK")
 
        print(f"Inserindo clínicas ({len(data['clinics'])})...")
        clinic_ids = insert_clinics(cur, data["clinics"])
        print(f"  OK - {len(clinic_ids)} clínicas inseridas")
 
        print(f"Inserindo pessoas (família: {len(data['family_members'])}, terapeutas: {len(data['therapists'])})...")
        name_to_id = insert_people(cur, data["family_members"], data["therapists"])
        print(f"  OK - {len(set(name_to_id.values()))} pessoas únicas após dedup (de {len(name_to_id)} nomes brutos)")
 
        domain_cache = {}
 
        print(f"Inserindo sessões ({len(data['sessions'])})...")
        insert_sessions(cur, patient_id, data["sessions"], name_to_id)
        print("  OK")
 
        print(f"Inserindo supervisões de AT ({len(data['at_supervisions'])})...")
        insert_at_supervisions(cur, patient_id, data["at_supervisions"])
        print("  OK")
 
        print(f"Inserindo avaliações ({len(data['evaluations'])})...")
        insert_evaluations(cur, patient_id, data["evaluations"], name_to_id, clinic_ids)
        print("  OK")
 
        print(f"Inserindo PEIs ({len(data['ieps'])})...")
        insert_ieps(cur, patient_id, data["ieps"], domain_cache)
        print("  OK")
 
        print(f"Inserindo skills ({len(data['skills'])})...")
        insert_skills(cur, patient_id, data["skills"], domain_cache)
        print("  OK")
 
        print(f"Inserindo desafios comportamentais ({len(data['behavioural_challenges'])})...")
        insert_behavioural_challenges(cur, patient_id, data["behavioural_challenges"], domain_cache)
        print("  OK")
 
        print(f"Inserindo estratégias ({len(data['strategies'])})...")
        insert_strategies(cur, patient_id, data["strategies"], domain_cache)
        print("  OK")
 
        print("Inserindo perfis (sensorial, motor, comunicação)...")
        insert_profiles(cur, patient_id, data["sensory_profile"], data["motor_profile"], data["communication_profile"])
        print("  OK")
 
        conn.commit()
        print("\nTodos os dados inseridos com sucesso! (commit realizado)\n")
 
        # Resumo rápido
        print("--- Resumo final ---")
        cur.execute("SELECT COUNT(*) FROM sessions")
        print(f"  sessions: {cur.fetchone()[0]}")
        cur.execute("SELECT COUNT(*) FROM at_supervisions")
        print(f"  at_supervisions: {cur.fetchone()[0]}")
        cur.execute("SELECT COUNT(*) FROM evaluations")
        print(f"  evaluations: {cur.fetchone()[0]}")
        cur.execute("SELECT COUNT(*) FROM ieps")
        print(f"  ieps: {cur.fetchone()[0]}")
        cur.execute("SELECT COUNT(*) FROM iep_goals")
        print(f"  iep_goals: {cur.fetchone()[0]}")
        cur.execute("SELECT COUNT(*) FROM session_notes")
        print(f"  session_notes: {cur.fetchone()[0]}")
        cur.execute("SELECT COUNT(*) FROM people")
        print(f"  people (deduplicado): {cur.fetchone()[0]}")
        cur.execute("SELECT COUNT(*) FROM domains")
        print(f"  domains: {cur.fetchone()[0]}")
 
    except Exception as e:
        conn.rollback()
        print(f"\n[ERRO] Falhou durante a inserção, rollback aplicado.")
        print(f"[ERRO] Tipo: {type(e).__name__}")
        print(f"[ERRO] Mensagem: {e}")
        raise
    finally:
        cur.close()
        conn.close()
 
 
if __name__ == "__main__":
    main()
