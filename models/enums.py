from __future__ import annotations

from enum import Enum


class AcquisitionStatus(str, Enum):
    """Status de aquisição de um objetivo terapêutico ou habilidade acompanhada ao longo do tempo."""
    ACQUIRED = "Adquirido"
    IN_PROGRESS = "Em progresso"
    NOT_STARTED = "Não iniciado"
    REGRESSED = "Regrediu"


class TherapyType(str, Enum):
    """Disciplina/especialidade envolvida no plano de tratamento multidisciplinar."""
    SPEECH_LANGUAGE = "Fonoaudiologia"
    OCCUPATIONAL = "Terapia Ocupacional"
    PHYSIOTHERAPY = "Fisioterapia"
    THERAPEUTIC_COMPANION = "Acompanhante Terapêutico (AT)"
    PSYCHOLOGY = "Psicologia"
    NEUROPEDIATRICS = "Neuropediatria"
    PEDIATRICS = "Pediatria"


class DomainArea(str, Enum):
    """Área de domínio de desenvolvimento trabalhada no Plano de Ensino Individualizado (PEI)."""
    RECEPTIVE_COMMUNICATION = "Comunicação Receptiva"
    EXPRESSIVE_COMMUNICATION = "Comunicação Expressiva"
    SOCIAL_COMPETENCIES = "Competências Sociais"
    COGNITION = "Cognição"
    GROSS_MOTOR = "Motor Grosso"
    FINE_MOTOR = "Motor Fino"
    SENSORY_PROCESSING = "Processamento Sensorial"
    DAILY_LIVING_ACTIVITIES = "Atividades de Vida Diária"
    EMOTIONAL_REGULATION = "Regulação Emocional"
    PLAY_SKILLS = "Habilidades de Brincar"


class SensoryModality(str, Enum):
    """Canal sensorial relevante para o perfil sensorial do paciente."""
    TACTILE = "Tátil"
    PROPRIOCEPTIVE = "Proprioceptivo"
    VESTIBULAR = "Vestibular"
    AUDITORY = "Auditivo"
    VISUAL = "Visual"
    ORAL = "Oral"