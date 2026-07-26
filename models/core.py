from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from models import AcquisitionStatus, TherapyType, DomainArea, SensoryModality


# ---------------------------------------------------------------------------
# NÓS CENTRAIS: PACIENTE, FAMÍLIA, PROFISSIONAIS, INSTITUIÇÕES
# ---------------------------------------------------------------------------

class Patient(BaseModel):
    """Nó central do grafo de conhecimento. Representa a criança em acompanhamento multidisciplinar. Fonte: cabeçalho do prontuário e relatórios clínicos."""
    full_name: str = Field(..., description="Nome completo do paciente")
    date_of_birth: date = Field(..., description="Data de nascimento do paciente")
    sex: str = Field(..., description="Sexo do paciente")
    city: Optional[str] = Field(None, description="Cidade/UF de residência do paciente")
    hyperfocus_topics: list[str] = Field(
        default_factory=list,
        description="Temas de interesse restrito (hiperfoco) usados como gancho de engajamento terapêutico",
    )
    speaks_in_third_person: Optional[bool] = Field(
        None, description="Indica se o paciente fala em terceira pessoa, com transição documentada para uso de pronome pessoal"
    )
    echolalia_present: Optional[bool] = Field(
        None, description="Indica se a ecolalia ainda está presente na fala do paciente"
    )
    toe_walking: Optional[bool] = Field(
        None, description="Indica se o paciente apresenta marcha na ponta dos pés (padrão equino)"
    )


class FamilyMember(BaseModel):
    """Membro da família que participa do tratamento ou fornece contexto para ele. O envolvimento familiar é tema recorrente nas notas de supervisão de AT."""
    name: str = Field(..., description="Nome do familiar")
    relationship: str = Field(..., description='Grau de parentesco com o paciente (ex.: "mãe", "pai", "irmã")')
    role_in_treatment: Optional[str] = Field(
        None, description="Papel desse familiar no tratamento (ex.: reforça estratégias em casa, participa das sessões)"
    )


class Therapist(BaseModel):
    """Profissional licenciado que presta atendimento terapêutico direto ao paciente. Vários terapeutas aparecem ao longo das evoluções do prontuário."""
    name: str = Field(..., description="Nome do profissional")
    specialty: TherapyType = Field(..., description="Especialidade/disciplina do profissional")
    registration: Optional[str] = Field(
        None, description="Número de registro no conselho profissional (CRFA, CREFITO etc.)"
    )
    clinic_name: Optional[str] = Field(None, description="Nome da clínica onde o profissional atende")


class Clinic(BaseModel):
    """Instituição ou consultório onde ocorrem os atendimentos terapêuticos."""
    name: str = Field(..., description="Nome da clínica/instituição")
    address: Optional[str] = Field(None, description="Endereço da clínica")
    phone: Optional[str] = Field(None, description="Telefone de contato da clínica")
    services: list[TherapyType] = Field(
        default_factory=list, description="Serviços/especialidades oferecidos por essa clínica"
    )


class Diagnosis(BaseModel):
    """Diagnóstico clínico formal atribuído ao paciente, incluindo especificadores de gravidade documentados."""
    name: str = Field(..., description="Nome do diagnóstico principal")
    subtype: Optional[str] = Field(
        None, description="Subtipo ou especificador do diagnóstico (ex.: 'TEA regressivo')"
    )
    comorbidities: list[str] = Field(
        default_factory=list, description="Comorbidades documentadas, se houver"
    )
    icd_code: Optional[str] = Field(None, description="Código CID correspondente ao diagnóstico")
    date_received: Optional[date] = Field(None, description="Data em que o diagnóstico foi recebido/firmado")
    diagnosing_physician: Optional[str] = Field(
        None, description="Nome e especialidade do médico que firmou o diagnóstico"
    )


# ---------------------------------------------------------------------------
# SESSÕES, OBJETIVOS E HABILIDADES
# ---------------------------------------------------------------------------

class TherapySession(BaseModel):
    """Uma evolução clínica individual registrada no prontuário (uma sessão de atendimento)."""
    session_date: date = Field(..., description="Data em que a sessão ocorreu")
    therapist_name: str = Field(..., description="Nome do terapeuta que realizou o atendimento")
    specialty: TherapyType = Field(..., description="Disciplina responsável pelo atendimento")
    main_activities: list[str] = Field(
        default_factory=list, description="Principais atividades realizadas durante a sessão"
    )
    progress_notes: str = Field(..., description="Texto da evolução: observações, comportamentos e resposta do paciente")
    homework_assigned: Optional[str] = Field(
        None, description="Atividade ou orientação repassada para realização em casa, se houver"
    )


class TherapeuticGoal(BaseModel):
    """Objetivo de desenvolvimento específico listado no Plano de Ensino Multidisciplinar (PEI), vinculado a uma área de domínio."""
    domain: DomainArea = Field(..., description="Área de domínio à qual o objetivo pertence")
    objective: str = Field(..., description="Nome/título curto do objetivo")
    description: str = Field(..., description="Descrição detalhada do objetivo a ser trabalhado")
    acquisition_status: AcquisitionStatus = Field(..., description="Status atual de aquisição deste objetivo")
    last_updated: date = Field(..., description="Data da última atualização de status deste objetivo")
    strategies: list[str] = Field(
        default_factory=list, description="Estratégias utilizadas para trabalhar este objetivo"
    )


class DevelopmentalSkill(BaseModel):
    """Habilidade ou competência discreta acompanhada ao longo do tempo, podendo estar adquirida, em progresso ou não iniciada."""
    name: str = Field(..., description="Nome da habilidade")
    domain: DomainArea = Field(..., description="Domínio de desenvolvimento ao qual a habilidade pertence")
    status: AcquisitionStatus = Field(..., description="Status atual de aquisição da habilidade")
    evidence: str = Field(..., description="Comportamento observável registrado nos documentos clínicos que evidencia esse status")


# ---------------------------------------------------------------------------
# PERFIS POR DOMÍNIO
# ---------------------------------------------------------------------------

class SensoryProfile(BaseModel):
    """Características de processamento sensorial observadas em avaliações de TO e fisioterapia, usadas para guiar intervenções de integração sensorial."""
    patient_name: str = Field(..., description="Nome do paciente ao qual este perfil se refere")
    hypersensitivities: list[str] = Field(
        default_factory=list, description="Estímulos aos quais o paciente apresenta hipersensibilidade"
    )
    hypersensitivities_detail: Optional[str] = Field(
        None, description="Detalhamento textual das hipersensibilidades observadas"
    )
    hyposensitivities: list[str] = Field(
        default_factory=list, description="Estímulos aos quais o paciente apresenta hipossensibilidade"
    )
    notable_behaviours: list[str] = Field(
        default_factory=list, description="Comportamentos sensoriais notáveis (ex.: busca sensorial aumentada)"
    )
    sensory_seeking: list[SensoryModality] = Field(
        default_factory=list, description="Modalidades sensoriais nas quais o paciente apresenta busca ativa"
    )


class MotorProfile(BaseModel):
    """Características motoras grossas e finas derivadas de avaliações de fisioterapia e TO."""
    patient_name: str = Field(..., description="Nome do paciente ao qual este perfil se refere")
    gross_motor_strengths: list[str] = Field(default_factory=list, description="Pontos fortes em motricidade grossa")
    gross_motor_challenges: list[str] = Field(default_factory=list, description="Dificuldades em motricidade grossa")
    fine_motor_strengths: list[str] = Field(default_factory=list, description="Pontos fortes em motricidade fina")
    fine_motor_challenges: list[str] = Field(default_factory=list, description="Dificuldades em motricidade fina")
    postural_findings: list[str] = Field(
        default_factory=list, description="Achados posturais relevantes (ex.: alinhamento de tronco, marcha)"
    )


class CommunicationProfile(BaseModel):
    """Características de linguagem e comunicação a partir da avaliação fonoaudiológica e das evoluções de sessão."""
    patient_name: str = Field(..., description="Nome do paciente ao qual este perfil se refere")
    vocabulary_level: Optional[str] = Field(None, description="Nível de vocabulário atual do paciente")
    sentence_complexity: Optional[str] = Field(None, description="Nível de complexidade das frases produzidas")
    pragmatic_skills: list[str] = Field(
        default_factory=list, description="Habilidades pragmáticas observadas (ex.: intenção comunicativa, reconto)"
    )
    challenges: list[str] = Field(default_factory=list, description="Dificuldades de linguagem/comunicação observadas")
    tools_used: list[str] = Field(
        default_factory=list, description='Recursos usados para apoiar a comunicação (ex.: "apoio visual", "livros pictográficos")'
    )


class BehaviouralChallenge(BaseModel):
    """Padrão comportamental recorrente que exige manejo terapêutico, documentado em notas de supervisão de AT e avaliações de TO."""
    name: str = Field(..., description="Nome curto do comportamento/desafio")
    description: str = Field(..., description="Descrição do comportamento e contexto em que ocorre")
    domain: DomainArea = Field(..., description="Domínio de desenvolvimento relacionado a este comportamento")
    management_strategies: list[str] = Field(
        default_factory=list, description="Estratégias de manejo utilizadas ou recomendadas"
    )
    current_status: str = Field(
        ..., description='Situação atual do comportamento (ex.: "reduzido", "persistente", "em manejo")'
    )


class TherapeuticStrategy(BaseModel):
    """Técnica clínica ou comportamental específica aplicada em uma ou mais disciplinas, extraída de notas de sessão e registros de supervisão."""
    name: str = Field(..., description="Nome da estratégia/técnica")
    description: str = Field(..., description="Descrição de como a estratégia é aplicada")
    applicable_disciplines: list[TherapyType] = Field(
        default_factory=list, description="Disciplinas em que essa estratégia é/pode ser aplicada"
    )
    target_domains: list[DomainArea] = Field(
        default_factory=list, description="Domínios de desenvolvimento visados por essa estratégia"
    )
    evidence_basis: Optional[str] = Field(
        None, description='Base teórica/evidência da estratégia (ex.: "ABA", "princípio de Premack", "integração sensorial")'
    )


# ---------------------------------------------------------------------------
# AVALIAÇÕES, PEI E SUPERVISÃO DE AT
# ---------------------------------------------------------------------------

class Evaluation(BaseModel):
    """Avaliação formal conduzida por um profissional licenciado, resultando em um relatório escrito."""
    evaluation_type: str = Field(..., description='Tipo de avaliação (ex.: "Avaliação Terapêutica Ocupacional")')
    evaluator: str = Field(..., description="Nome do profissional avaliador")
    evaluation_date: date = Field(..., description="Data da avaliação")
    clinic_name: Optional[str] = Field(None, description="Clínica onde a avaliação foi realizada")
    instruments_used: list[str] = Field(
        default_factory=list, description="Instrumentos, escalas ou protocolos utilizados na avaliação"
    )
    main_findings: list[str] = Field(default_factory=list, description="Principais achados da avaliação")
    recommendations: list[str] = Field(default_factory=list, description="Recomendações resultantes da avaliação")


class IEP(BaseModel):
    """Plano de Ensino Individualizado / Multidisciplinar (PEI). Plano estruturado de ensino e terapia, revisado periodicamente pela equipe."""
    start_date: date = Field(..., description="Data de início/emissão deste plano")
    institution: str = Field(..., description="Instituição responsável pela elaboração do plano")
    goals: list[TherapeuticGoal] = Field(default_factory=list, description="Objetivos terapêuticos definidos neste plano")
    review_date: Optional[date] = Field(None, description="Data prevista ou realizada de revisão do plano")


class ATSupervisionSession(BaseModel):
    """Reunião de supervisão do Acompanhante Terapêutico (AT), na qual o trabalho do AT é revisado, estratégias são ajustadas e orientações são dadas."""
    session_date: date = Field(..., description="Data em que a supervisão ocorreu")
    key_observations: list[str] = Field(
        default_factory=list, description="Observações-chave levantadas durante a supervisão"
    )
    directives: list[str] = Field(
        default_factory=list, description="Direcionamentos/combinados definidos para os próximos atendimentos"
    )
    session_structure_principles: list[str] = Field(
        default_factory=list, description="Princípios de estruturação de sessão discutidos (ex.: princípio de Premack)"
    )


# ---------------------------------------------------------------------------
# REGISTRO CONSOLIDADO (usado pelo agente para relacionar tudo)
# ---------------------------------------------------------------------------

class PatientRecord(BaseModel):
    """Registro consolidado de todos os nós do grafo referentes a um paciente, usado como base para o agente relacionar eventos, profissionais e domínios ao longo do tempo."""
    patient: Patient = Field(..., description="Dados do paciente")
    diagnosis: Optional[Diagnosis] = Field(None, description="Diagnóstico formal do paciente")
    family_members: list[FamilyMember] = Field(default_factory=list, description="Familiares envolvidos no tratamento")
    therapists: list[Therapist] = Field(default_factory=list, description="Profissionais envolvidos no tratamento")
    clinics: list[Clinic] = Field(default_factory=list, description="Clínicas/instituições envolvidas no tratamento")
    sessions: list[TherapySession] = Field(default_factory=list, description="Histórico de evoluções de sessão")
    evaluations: list[Evaluation] = Field(default_factory=list, description="Histórico de avaliações formais")
    ieps: list[IEP] = Field(default_factory=list, description="Histórico de Planos de Ensino Individualizado")
    at_supervisions: list[ATSupervisionSession] = Field(
        default_factory=list, description="Histórico de supervisões de Acompanhamento Terapêutico"
    )
    skills: list[DevelopmentalSkill] = Field(default_factory=list, description="Habilidades acompanhadas ao longo do tempo")
    sensory_profile: Optional[SensoryProfile] = Field(None, description="Perfil sensorial consolidado do paciente")
    motor_profile: Optional[MotorProfile] = Field(None, description="Perfil motor consolidado do paciente")
    communication_profile: Optional[CommunicationProfile] = Field(
        None, description="Perfil de comunicação consolidado do paciente"
    )
    behavioural_challenges: list[BehaviouralChallenge] = Field(
        default_factory=list, description="Desafios comportamentais acompanhados ao longo do tempo"
    )
    strategies: list[TherapeuticStrategy] = Field(
        default_factory=list, description="Estratégias terapêuticas catalogadas, reutilizáveis entre disciplinas"
    )