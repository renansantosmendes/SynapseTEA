from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Optional, Type, TypeVar

import pdfplumber
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from models import (
    ATSupervisionSession,
    BehaviouralChallenge,
    Clinic,
    CommunicationProfile,
    Diagnosis,
    Evaluation,
    FamilyMember,
    IEP,
    MotorProfile,
    Patient,
    PatientRecord,
    SensoryProfile,
    Therapist,
    TherapeuticGoal,
    TherapeuticStrategy,
    TherapySession,
    DevelopmentalSkill,
)

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gpt-5-nano"
CHUNK_SIZE = 6000
CHUNK_OVERLAP = 400

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Schema "parcial": mesma forma do PatientRecord, mas tudo opcional, usado
# para a extração por pedaço de documento (um chunk pode não conter todos os
# tipos de informação).
# ---------------------------------------------------------------------------

class ChunkExtraction(BaseModel):
    """Informações clínicas estruturadas extraídas de UM trecho de documento. Todos os campos são opcionais: preencha apenas o que estiver explicitamente presente no texto, sem inferir ou inventar dados."""
    patient: Optional[Patient] = Field(None, description="Dados do paciente, se mencionados neste trecho")
    diagnosis: Optional[Diagnosis] = Field(None, description="Diagnóstico do paciente, se mencionado neste trecho")
    family_members: list[FamilyMember] = Field(default_factory=list, description="Familiares mencionados neste trecho")
    therapists: list[Therapist] = Field(default_factory=list, description="Profissionais mencionados neste trecho")
    clinics: list[Clinic] = Field(default_factory=list, description="Clínicas/instituições mencionadas neste trecho")
    sessions: list[TherapySession] = Field(default_factory=list, description="Evoluções de sessão descritas neste trecho")
    evaluations: list[Evaluation] = Field(default_factory=list, description="Avaliações formais descritas neste trecho")
    ieps: list[IEP] = Field(default_factory=list, description="Planos de Ensino Individualizado descritos neste trecho")
    at_supervisions: list[ATSupervisionSession] = Field(
        default_factory=list, description="Supervisões de Acompanhamento Terapêutico descritas neste trecho"
    )
    skills: list[DevelopmentalSkill] = Field(default_factory=list, description="Habilidades específicas mencionadas neste trecho")
    sensory_profile: Optional[SensoryProfile] = Field(None, description="Informações de perfil sensorial presentes neste trecho")
    motor_profile: Optional[MotorProfile] = Field(None, description="Informações de perfil motor presentes neste trecho")
    communication_profile: Optional[CommunicationProfile] = Field(
        None, description="Informações de perfil de comunicação presentes neste trecho"
    )
    behavioural_challenges: list[BehaviouralChallenge] = Field(
        default_factory=list, description="Desafios comportamentais descritos neste trecho"
    )
    strategies: list[TherapeuticStrategy] = Field(
        default_factory=list, description="Estratégias terapêuticas descritas neste trecho"
    )


SYSTEM_PROMPT = """Você é um assistente especializado em estruturar prontuários clínicos \
multidisciplinares (fonoaudiologia, terapia ocupacional, fisioterapia, psicologia, \
acompanhamento terapêutico) de uma criança com TEA.

Vai receber um trecho de um documento (pode ser: evolução de sessão, avaliação, \
reavaliação, PEI/plano de ensino, ou ata de supervisão de AT).

Regras:
- Extraia SOMENTE informação explicitamente presente no texto. Nunca infira, \
complete ou invente dados que não estejam escritos.
- Se o trecho não contiver um determinado tipo de informação, deixe o campo \
correspondente vazio/nulo — não tente forçar preenchimento.
- Datas devem ser extraídas no formato ISO (AAAA-MM-DD). Se só houver dia/mês, \
use o ano mais provável pelo contexto do documento; se não for possível \
determinar o ano com confiança, não invente — omita o registro.
- Nomes de profissionais, clínicas e registros profissionais devem ser \
copiados exatamente como aparecem no texto.
- Um mesmo trecho pode conter múltiplas evoluções de sessão — extraia cada \
uma como um item separado da lista `sessions`.
"""


# ---------------------------------------------------------------------------
# Leitura de PDFs
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: str) -> str:
    """Extrai todo o texto de um PDF (assume que o PDF tem camada de texto)."""
    text_parts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def chunk_text(text: str, source_name: str) -> list[str]:
    """Divide o texto em pedaços menores, respeitando quebras de parágrafo quando possível."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [f"[Documento fonte: {source_name}]\n\n{chunk}" for chunk in chunks]


# ---------------------------------------------------------------------------
# Extração via LangChain + gpt-5-nano
# ---------------------------------------------------------------------------

def build_extraction_chain():
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    structured_llm = llm.with_structured_output(ChunkExtraction)
    return structured_llm


def extract_chunk(structured_llm, chunk: str) -> ChunkExtraction:
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", chunk),
    ]
    result = structured_llm.invoke(messages)
    return result


# ---------------------------------------------------------------------------
# Merge: consolida vários ChunkExtraction em um único PatientRecord
# ---------------------------------------------------------------------------

def _merge_singular(current: Optional[T], new: Optional[T]) -> Optional[T]:
    """Combina dois objetos singulares (ex.: Patient, Diagnosis), preferindo
    preencher campos vazios do `current` com valores não-nulos do `new`."""
    if current is None:
        return new
    if new is None:
        return current
    merged_data = current.model_dump()
    new_data = new.model_dump()
    for key, value in new_data.items():
        existing = merged_data.get(key)
        is_empty = existing in (None, "", [])
        if is_empty and value not in (None, "", []):
            merged_data[key] = value
    return type(current)(**merged_data)


def _dedupe(items: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    seen = set()
    result = []
    for item in items:
        key = tuple(item.get(f) for f in key_fields)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def merge_extractions(chunks: list[ChunkExtraction]) -> PatientRecord:
    patient: Optional[Patient] = None
    diagnosis: Optional[Diagnosis] = None
    sensory_profile: Optional[SensoryProfile] = None
    motor_profile: Optional[MotorProfile] = None
    communication_profile: Optional[CommunicationProfile] = None

    family_members: list[dict] = []
    therapists: list[dict] = []
    clinics: list[dict] = []
    sessions: list[dict] = []
    evaluations: list[dict] = []
    ieps: list[dict] = []
    at_supervisions: list[dict] = []
    skills: list[dict] = []
    behavioural_challenges: list[dict] = []
    strategies: list[dict] = []

    for chunk in chunks:
        patient = _merge_singular(patient, chunk.patient)
        diagnosis = _merge_singular(diagnosis, chunk.diagnosis)
        sensory_profile = _merge_singular(sensory_profile, chunk.sensory_profile)
        motor_profile = _merge_singular(motor_profile, chunk.motor_profile)
        communication_profile = _merge_singular(communication_profile, chunk.communication_profile)

        family_members.extend(m.model_dump() for m in chunk.family_members)
        therapists.extend(m.model_dump() for m in chunk.therapists)
        clinics.extend(m.model_dump() for m in chunk.clinics)
        sessions.extend(m.model_dump() for m in chunk.sessions)
        evaluations.extend(m.model_dump() for m in chunk.evaluations)
        ieps.extend(m.model_dump() for m in chunk.ieps)
        at_supervisions.extend(m.model_dump() for m in chunk.at_supervisions)
        skills.extend(m.model_dump() for m in chunk.skills)
        behavioural_challenges.extend(m.model_dump() for m in chunk.behavioural_challenges)
        strategies.extend(m.model_dump() for m in chunk.strategies)

    if patient is None:
        raise ValueError(
            "Nenhum dado de paciente foi extraído de nenhum documento — "
            "verifique se os PDFs contêm o cabeçalho do prontuário."
        )

    return PatientRecord(
        patient=patient,
        diagnosis=diagnosis,
        family_members=[FamilyMember(**d) for d in _dedupe(family_members, ("name",))],
        therapists=[Therapist(**d) for d in _dedupe(therapists, ("name", "specialty"))],
        clinics=[Clinic(**d) for d in _dedupe(clinics, ("name",))],
        sessions=[TherapySession(**d) for d in _dedupe(sessions, ("session_date", "therapist_name", "specialty"))],
        evaluations=[Evaluation(**d) for d in _dedupe(evaluations, ("evaluation_date", "evaluator", "evaluation_type"))],
        ieps=[IEP(**d) for d in _dedupe(ieps, ("start_date", "institution"))],
        at_supervisions=[ATSupervisionSession(**d) for d in _dedupe(at_supervisions, ("session_date",))],
        skills=[DevelopmentalSkill(**d) for d in _dedupe(skills, ("name", "domain"))],
        sensory_profile=sensory_profile,
        motor_profile=motor_profile,
        communication_profile=communication_profile,
        behavioural_challenges=[BehaviouralChallenge(**d) for d in _dedupe(behavioural_challenges, ("name",))],
        strategies=[TherapeuticStrategy(**d) for d in _dedupe(strategies, ("name",))],
    )


# ---------------------------------------------------------------------------
# Orquestração principal
# ---------------------------------------------------------------------------

def run_pipeline(pdf_dir: str, out_path: str) -> PatientRecord:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Defina a variável de ambiente OPENAI_API_KEY antes de rodar o pipeline.")

    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise RuntimeError(f"Nenhum PDF encontrado em {pdf_dir}")

    structured_llm = build_extraction_chain()
    all_chunk_results: list[ChunkExtraction] = []

    for pdf_path in pdf_paths:
        source_name = os.path.basename(pdf_path)
        print(f"[extração] lendo {source_name}")
        raw_text = extract_pdf_text(pdf_path)
        if not raw_text.strip():
            print(f"[aviso] {source_name} não retornou texto (pode ser um PDF escaneado) — pulando")
            continue

        chunks = chunk_text(raw_text, source_name)
        print(f"[extração] {source_name}: {len(chunks)} trecho(s)")

        for i, chunk in enumerate(chunks, start=1):
            print(f"  -> processando trecho {i}/{len(chunks)}")
            try:
                result = extract_chunk(structured_llm, chunk)
                all_chunk_results.append(result)
            except Exception as exc:  # noqa: BLE001
                print(f"  [erro] falha ao extrair trecho {i} de {source_name}: {exc}")

    patient_record = merge_extractions(all_chunk_results)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(patient_record.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    print(f"\n[ok] PatientRecord consolidado salvo em: {out_path}")
    print(f"  - sessões: {len(patient_record.sessions)}")
    print(f"  - avaliações: {len(patient_record.evaluations)}")
    print(f"  - PEIs: {len(patient_record.ieps)}")
    print(f"  - supervisões de AT: {len(patient_record.at_supervisions)}")
    print(f"  - terapeutas: {len(patient_record.therapists)}")

    return patient_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrai e consolida dados clínicos de PDFs em um PatientRecord.")
    parser.add_argument("--pdf-dir", required=True, help="Pasta contendo os PDFs a processar")
    parser.add_argument("--out", default="patient_record.json", help="Caminho do JSON de saída")
    args = parser.parse_args()

    run_pipeline(args.pdf_dir, args.out)