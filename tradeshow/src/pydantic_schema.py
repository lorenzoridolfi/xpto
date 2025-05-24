from typing import Optional, List, Literal
from pydantic import BaseModel, Field, confloat


class Segmento(BaseModel):
    valor: Literal[
        "Planejadores",
        "Poupadores",
        "Materialistas",
        "Batalhadores",
        "Céticos",
        "Endividados",
    ]


class Filosofia(BaseModel):
    valor: Literal["Multiplicar", "Guardar", "Gastar", "Ganhar", "Evitar", "Pagar"]


class RendaMensal(BaseModel):
    valor: float = Field(..., ge=0)


class Escolaridade(BaseModel):
    valor: Literal["Ensino Fundamental", "Ensino Médio", "Superior Completo"]


class Ocupacao(BaseModel):
    valor: str


class UsaBancoTradicional(BaseModel):
    valor: bool


class UsaBancoDigital(BaseModel):
    valor: bool


class UsaCorretora(BaseModel):
    valor: bool


class FrequenciaPoupancaMensal(BaseModel):
    valor: float = Field(..., ge=0)


class ComportamentoGastos(BaseModel):
    valor: Literal["cauteloso", "consumo_imediato", "necessidades_basicas"]


class ComportamentoInvestimentos(BaseModel):
    valor: Literal["diversificado", "basico", "nenhum"]


class Avaliacao(BaseModel):
    """
    Avaliação do perfil pelo agente crítico (critica) e comentários do revisor (revisao).
    Ambos os campos são obrigatórios na versão revisada.
    """

    critica: str
    revisao: str


class SyntheticUserDraft(BaseModel):
    """
    Pydantic model for a synthetic user profile (DRAFT, before review), matching synthetic_user_schema.json.
    This model is derived from the same JSON Schema as SyntheticUserReviewed, but does NOT include the 'avaliacao' field.
    Use this for user creation (before review).
    """

    id_usuario: str = Field(..., alias="id_usuario")
    segmento: Segmento
    filosofia: Filosofia
    renda_mensal: RendaMensal
    escolaridade: Escolaridade
    ocupacao: Ocupacao
    usa_banco_tradicional: UsaBancoTradicional
    usa_banco_digital: UsaBancoDigital
    usa_corretora: UsaCorretora
    frequencia_poupanca_mensal: FrequenciaPoupancaMensal
    comportamento_gastos: ComportamentoGastos
    comportamento_investimentos: ComportamentoInvestimentos
    score_risco_financeiro: Optional[confloat(ge=0.0, le=1.0)] = Field(
        None, alias="score_risco_financeiro"
    )
    engajamento_digital: Optional[Literal["baixo", "medio", "alto"]] = Field(
        None, alias="engajamento_digital"
    )
    comportamento_poupanca: Optional[
        Literal["disciplinado", "ocasional", "ausente"]
    ] = Field(None, alias="comportamento_poupanca")

    class Config:
        allow_population_by_field_name = True
        extra = "forbid"


class SyntheticUserReviewed(BaseModel):
    """
    Pydantic model for a synthetic user profile (REVIEWED, after review), matching synthetic_user_schema.json.
    This model is derived from the same JSON Schema as SyntheticUserDraft, but REQUIRES the 'avaliacao' field.
    Use this for post-review validation.
    """

    id_usuario: str = Field(..., alias="id_usuario")
    segmento: Segmento
    filosofia: Filosofia
    renda_mensal: RendaMensal
    escolaridade: Escolaridade
    ocupacao: Ocupacao
    usa_banco_tradicional: UsaBancoTradicional
    usa_banco_digital: UsaBancoDigital
    usa_corretora: UsaCorretora
    frequencia_poupanca_mensal: FrequenciaPoupancaMensal
    comportamento_gastos: ComportamentoGastos
    comportamento_investimentos: ComportamentoInvestimentos
    score_risco_financeiro: Optional[confloat(ge=0.0, le=1.0)] = Field(
        None, alias="score_risco_financeiro"
    )
    engajamento_digital: Optional[Literal["baixo", "medio", "alto"]] = Field(
        None, alias="engajamento_digital"
    )
    comportamento_poupanca: Optional[
        Literal["disciplinado", "ocasional", "ausente"]
    ] = Field(None, alias="comportamento_poupanca")
    avaliacao: Avaliacao = Field(..., alias="avaliacao")

    class Config:
        allow_population_by_field_name = True
        extra = "forbid"


class CriticOutput(BaseModel):
    """
    Pydantic model for the output of the ValidatorAgent (critic), matching critic_schema.json.
    """

    score: confloat(ge=0, le=1)
    issues: List[str]
    recommendation: Literal["accept", "flag for review"]
