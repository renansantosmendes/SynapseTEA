-- Habilita extensão de vetores (para embeddings do RAG, usaremos depois)
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Pacientes
CREATE TABLE patients (
    id SERIAL PRIMARY KEY,
    full_name TEXT,
    date_of_birth DATE,
    sex TEXT,
    speaks_in_third_person BOOLEAN,
    echolalia_present BOOLEAN,
    toe_walking BOOLEAN
);

-- 2. Diagnósticos
CREATE TABLE diagnoses (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(id),
    name TEXT,
    icd_code TEXT,
    date_received DATE,
    diagnosing_physician TEXT
);

-- 3. Clínicas
CREATE TABLE clinics (
    id SERIAL PRIMARY KEY,
    name TEXT,
    address TEXT,
    phone TEXT
);

-- 4. Pessoas (família + terapeutas, já normalizados/deduplicados)
CREATE TABLE people (
    id SERIAL PRIMARY KEY,
    canonical_name TEXT,
    aliases TEXT[],
    person_type TEXT,      -- 'family' ou 'therapist'
    relationship TEXT,     -- se family: 'pai', 'mãe', 'irmã'...
    specialty TEXT,        -- se therapist: 'Fonoaudiologia'...
    registration TEXT,
    clinic_id INT REFERENCES clinics(id)
);

-- 5. Domínios (taxonomia compartilhada entre disciplinas)
CREATE TABLE domains (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE
);

-- 6. Sessões regulares (fono, TO, fisio, psico) - linha do tempo principal
CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(id),
    therapist_id INT REFERENCES people(id),
    specialty TEXT,
    session_date DATE,
    progress_notes TEXT,
    homework_assigned TEXT
);

-- 7. Notas de sessão (main_activities, key_observations, directives, principles - uma linha por item)
-- Cobre sessions, at_supervisions (que tem estrutura parecida mas é sessão de AT)
CREATE TABLE session_notes (
    id SERIAL PRIMARY KEY,
    session_id INT,                -- referencia sessions.id OU at_supervisions.id
    source_table TEXT,              -- 'sessions' | 'at_supervisions'
    note_type TEXT,                 -- 'activity' | 'observation' | 'directive' | 'principle'
    content TEXT,
    embedding VECTOR(1536)          -- preenchido depois, quando gerarmos embeddings
);

-- 6b. Supervisões de AT (Acompanhante Terapêutico) - estrutura própria, mesma ideia temporal
CREATE TABLE at_supervisions (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(id),
    session_date DATE
);

-- 6c. Avaliações formais (fisioterapêutica, etc.)
CREATE TABLE evaluations (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(id),
    evaluation_type TEXT,
    evaluator_id INT REFERENCES people(id),
    evaluation_date DATE,
    clinic_id INT REFERENCES clinics(id),
    instruments_used TEXT[],
    main_findings TEXT[],
    recommendations TEXT[]
);

-- 6d. PEI (Plano Educacional Individualizado)
CREATE TABLE ieps (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(id),
    start_date DATE,
    institution TEXT,
    goals TEXT[],
    review_date DATE
);

-- 8. Skills (habilidades trabalhadas)
CREATE TABLE skills (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(id),
    name TEXT,
    domain_id INT REFERENCES domains(id),
    status TEXT,
    evidence TEXT,
    recorded_date DATE
);

-- 9. Desafios comportamentais
CREATE TABLE behavioural_challenges (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(id),
    name TEXT,
    description TEXT,
    domain_id INT REFERENCES domains(id),
    current_status TEXT
);

-- 10. Estratégias
CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(id),
    name TEXT,
    description TEXT,
    domain_id INT REFERENCES domains(id)
);

-- 11. Perfis (sensorial, motor, comunicação) - snapshot por data de reavaliação
CREATE TABLE profiles (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(id),
    profile_type TEXT,     -- 'sensory' | 'motor' | 'communication'
    data JSONB,             -- guarda o conteúdo flexível de cada perfil
    recorded_date DATE
);

-- Índices úteis para consulta por data (o seu caso de uso principal)
CREATE INDEX idx_sessions_date ON sessions(session_date);
CREATE INDEX idx_sessions_patient ON sessions(patient_id);
CREATE INDEX idx_session_notes_session ON session_notes(session_id);