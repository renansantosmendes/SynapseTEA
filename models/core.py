from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from .enums import AcquisitionStatus, TherapyType, DomainArea, SensoryModality


class Patient(BaseModel):
    """
    Central node of the knowledge graph.
    Represents the child receiving multidisciplinary ASD treatment.

    Source: prontuário header, all clinical reports.
    """
    full_name: str = "Antônio Damasceno Mendes"
    date_of_birth: date = date(2022, 5, 20)
    diagnosis: str = "Transtorno do Espectro Autista (TEA) regressivo"
    diagnosis_date: date = date(2024, 10, 1)
    sex: str = "Masculino"
    city: str = "Belo Horizonte / MG"
    hyperfocus_topics: list[str] = Field(
        default_factory=lambda: ["eletrônicos", "escavadeiras", "mangueiras", "cabos"],
        description="Restricted-interest topics used as therapeutic engagement hooks.",
    )
    speaks_in_third_person: bool = Field(
        True,
        description="Transitioning to first-person pronoun use (documented 2025).",
    )
    echolalia_present: bool = Field(
        False,
        description="Resolved by the time of the fono re-evaluation (Aug/25).",
    )
    toe_walking: bool = Field(
        True,
        description="Padrão equino — still present under stress/frustration (Aug/25).",
    )


class FamilyMember(BaseModel):
    """
    A family member who participates in or provides context for the treatment.
    Family involvement is a recurring theme in AT supervision notes.
    """
    name: str
    relationship: str = Field(description='e.g. "mãe", "pai", "irmã"')
    role_in_treatment: str


class Therapist(BaseModel):
    """
    A licensed professional providing direct therapeutic service to the patient.
    Multiple therapists are listed across the prontuário evolutions.
    """
    name: str
    specialty: TherapyType
    registration: str = Field(
        description="Professional council registration number (CRFA, CREFITO, etc.)."
    )
    clinic: str


class Clinic(BaseModel):
    """
    A treatment institution or practice where therapy sessions occur.
    """
    name: str
    address: str
    phone: str
    services: list[TherapyType]


class Diagnosis(BaseModel):
    """
    The formal clinical diagnosis assigned to the patient, including severity
    modifiers noted in the documents.
    """
    name: str = "Transtorno do Espectro Autista"
    subtype: str = Field(
        "TEA Regressivo",
        description="Regressive onset documented in all referral reports.",
    )
    comorbidities: list[str] = Field(
        default_factory=list,
        description="No comorbidities documented.",
    )
    icd_code: str = "F84.0"
    date_received: date = date(2024, 10, 1)
    diagnosing_physician: str = "Dra. Mariana Valadão (neuropediatra)"


class TherapySession(BaseModel):
    """
    A single recorded clinical evolution (evolução) from the prontuário.
    101 evolutions are present in the full file.
    """
    session_date: date
    therapist_name: str
    specialty: TherapyType
    main_activities: list[str]
    progress_notes: str
    homework_assigned: Optional[str] = None


class TherapeuticGoal(BaseModel):
    """
    A specific developmental objective listed in the Plano de Ensino
    Multidisciplinar (IEP), tied to a domain area.
    """
    domain: DomainArea
    objective: str
    description: str
    acquisition_status: AcquisitionStatus
    last_updated: date
    strategies: list[str] = Field(default_factory=list)


class DevelopmentalSkill(BaseModel):
    """
    A discrete skill or competency tracked over time.
    Skills may be acquired, in progress, or not yet started.
    """
    name: str
    domain: DomainArea
    status: AcquisitionStatus
    evidence: str = Field(
        description="Observable behaviour noted in clinical documents."
    )


class SensoryProfile(BaseModel):
    """
    Sensory processing characteristics observed during OT and physiotherapy
    evaluations. Used to guide sensory integration interventions.
    """
    patient_name: str
    hypersensitivities: list[str]
    hypersensitivities_detail: str
    hyposensitivities: list[str]
    notable_behaviours: list[str]
    sensory_seeking: list[SensoryModality]


class MotorProfile(BaseModel):
    """
    Gross and fine motor characteristics derived from physiotherapy and OT
    evaluations (Nov/24 initial eval, Mar/25 physio, Aug/25 re-eval).
    """
    patient_name: str
    gross_motor_strengths: list[str]
    gross_motor_challenges: list[str]
    fine_motor_strengths: list[str]
    fine_motor_challenges: list[str]
    postural_findings: list[str]


class CommunicationProfile(BaseModel):
    """
    Language and communication characteristics from the speech-language
    (fonoaudiologia) evaluation and ongoing session evolutions.
    """
    patient_name: str
    vocabulary_level: str
    sentence_complexity: str
    pragmatic_skills: list[str]
    challenges: list[str]
    tools_used: list[str] = Field(
        description='e.g. "apoio visual", "livros pictográficos"'
    )


class BehaviouralChallenge(BaseModel):
    """
    A recurring behavioural pattern that requires therapeutic management.
    Documented in AT supervision notes and OT evaluations.
    """
    name: str
    description: str
    domain: DomainArea
    management_strategies: list[str]
    current_status: str = Field(
        description='e.g. "reduzido", "persistente", "em manejo"'
    )


class TherapeuticStrategy(BaseModel):
    """
    A specific clinical or behavioural technique applied across one or
    more disciplines. Extracted from session notes and supervision records.
    """
    name: str
    description: str
    applicable_disciplines: list[TherapyType]
    target_domains: list[DomainArea]
    evidence_basis: str = Field(
        description='e.g. "ABA", "Premack principle", "sensory integration"'
    )


class Evaluation(BaseModel):
    """
    A formal assessment conducted by a licensed professional,
    resulting in a written report.
    """
    evaluation_type: str = Field(
        description='e.g. "Avaliação Terapêutica Ocupacional"'
    )
    evaluator: str
    date: date
    clinic: str
    instruments_used: list[str]
    main_findings: list[str]
    recommendations: list[str]


class IEP(BaseModel):
    """
    Plano de Ensino Individualizado / Multidisciplinar (PEI).
    The structured learning and therapy plan reviewed periodically by the team.
    """
    start_date: date
    institution: str
    goals: list[TherapeuticGoal]
    review_date: Optional[date] = None


class ATSupervisionSession(BaseModel):
    """
    A supervision meeting for the Therapeutic Companion (Acompanhante Terapêutico)
    where the AT's work is reviewed, strategies adjusted, and guidance given.
    """
    session_date: date
    key_observations: list[str]
    directives: list[str]
    session_structure_principles: list[str]
