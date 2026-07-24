from __future__ import annotations

from enum import Enum


class AcquisitionStatus(str, Enum):
    """Tracks whether a therapeutic goal has been achieved."""
    ACQUIRED = "ADQUIRIDO"
    IN_PROGRESS = "Em progresso"
    NOT_STARTED = "Não iniciado"


class TherapyType(str, Enum):
    """Types of therapeutic disciplines involved in the treatment plan."""
    SPEECH_LANGUAGE = "Fonoaudiologia"
    OCCUPATIONAL = "Terapia Ocupacional"
    PHYSIOTHERAPY = "Fisioterapia"
    THERAPEUTIC_COMPANION = "Acompanhante Terapêutico (AT)"
    PSYCHOLOGY = "Psicologia"
    NEUROPEDIATRICS = "Neuropediatria"
    PEDIATRICS = "Pediatria"


class DomainArea(str, Enum):
    """Developmental domain areas addressed in the IEP (Plano de Ensino)."""
    RECEPTIVE_COMMUNICATION = "Comunicação Receptiva"
    EXPRESSIVE_COMMUNICATION = "Comunicação Expressiva"
    SOCIAL_COMPETENCIES = "Competências Sociais"
    COGNITION = "Cognição"
    GROSS_MOTOR = "Motor Grosso"
    FINE_MOTOR = "Motor Fino"
    SENSORY_PROCESSING = "Processamento Sensorial"
    DAILY_LIVING_ACTIVITIES = "Atividades de Vida Diária"
    EMOTIONAL_REGULATION = "Regulação Emocional"


class SensoryModality(str, Enum):
    """Sensory channels relevant to the patient's sensory profile."""
    TACTILE = "Tátil"
    PROPRIOCEPTIVE = "Proprioceptivo"
    VESTIBULAR = "Vestibular"
    AUDITORY = "Auditivo"
    VISUAL = "Visual"
    ORAL = "Oral"
