"""Compatibility wrapper importing from the modularized models package."""

from .enums import AcquisitionStatus, TherapyType, DomainArea, SensoryModality  # noqa: F401
from .core import (
    Patient,
    FamilyMember,
    Therapist,
    Clinic,
    Diagnosis,
    TherapySession,
    TherapeuticGoal,
    DevelopmentalSkill,
    SensoryProfile,
    MotorProfile,
    CommunicationProfile,
    BehaviouralChallenge,
    TherapeuticStrategy,
    Evaluation,
    IEP,
    ATSupervisionSession,
)

# Backwards-compatibility alias
ATSupervisionSession = ATSupervisionSession

__all__ = [
    "AcquisitionStatus",
    "TherapyType",
    "DomainArea",
    "SensoryModality",
    "Patient",
    "FamilyMember",
    "Therapist",
    "Clinic",
    "Diagnosis",
    "TherapySession",
    "TherapeuticGoal",
    "DevelopmentalSkill",
    "SensoryProfile",
    "MotorProfile",
    "CommunicationProfile",
    "BehaviouralChallenge",
    "TherapeuticStrategy",
    "Evaluation",
    "IEP",
    "ATSupervisionSession",
]