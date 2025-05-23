from typing import Optional, List, Literal
from pydantic import BaseModel, Field, constr, conlist, conint, confloat


class SegmentLabel(BaseModel):
    value: Literal[
        "Planejadores",
        "Poupadores",
        "Materialistas",
        "Batalhadores",
        "Céticos",
        "Endividados",
    ]
    source: Optional[str] = None


class Philosophy(BaseModel):
    value: Literal["Multiplicar", "Guardar", "Gastar", "Ganhar", "Evitar", "Pagar"]
    source: Optional[str] = None


class MonthlyIncome(BaseModel):
    value: float = Field(..., ge=0)
    source: Optional[str] = None


class EducationLevel(BaseModel):
    value: Literal["Ensino Fundamental", "Ensino Médio", "Superior Completo"]
    source: Optional[str] = None


class Occupation(BaseModel):
    value: str
    source: Optional[str] = None


class UsesTraditionalBank(BaseModel):
    value: bool
    source: Optional[str] = None


class UsesDigitalBank(BaseModel):
    value: bool
    source: Optional[str] = None


class UsesBroker(BaseModel):
    value: bool
    source: Optional[str] = None


class SavingsFrequencyPerMonth(BaseModel):
    value: float = Field(..., ge=0)
    source: Optional[str] = None


class SpendingBehavior(BaseModel):
    value: Literal["cautious", "immediate_consumption", "basic_needs"]
    source: Optional[str] = None


class InvestmentBehavior(BaseModel):
    value: Literal["diversified", "basic", "none"]
    source: Optional[str] = None


class SyntheticUser(BaseModel):
    """
    Pydantic model for a synthetic user profile, matching synthetic_user_schema.json.
    """

    user_id: str
    segment_label: SegmentLabel
    philosophy: Philosophy
    monthly_income: MonthlyIncome
    education_level: EducationLevel
    occupation: Occupation
    uses_traditional_bank: UsesTraditionalBank
    uses_digital_bank: UsesDigitalBank
    uses_broker: UsesBroker
    savings_frequency_per_month: SavingsFrequencyPerMonth
    spending_behavior: SpendingBehavior
    investment_behavior: InvestmentBehavior
    predicted_financial_risk_score: Optional[confloat(ge=0.0, le=1.0)] = None
    inferred_digital_engagement: Optional[Literal["low", "medium", "high"]] = None
    inferred_savings_behavior: Optional[
        Literal["disciplined", "occasional", "absent"]
    ] = None


class CriticOutput(BaseModel):
    """
    Pydantic model for the output of the ValidatorAgent (critic), matching critic_schema.json.
    """

    score: confloat(ge=0, le=1)
    issues: List[str]
    recommendation: Literal["accept", "flag for review"]
