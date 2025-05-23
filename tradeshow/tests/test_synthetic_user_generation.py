import os
import json
import pytest
from tradeshow.src.synthetic_user_generator import (
    UserGeneratorAgent,
    ValidatorAgent,
    ReviewerAgent,
    TracedGroupChat,
)
from tradeshow.src.pydantic_schema import SyntheticUser, CriticOutput


def test_user_generator_agent():
    """
    Test that the UserGeneratorAgent correctly generates a synthetic user
    as a SyntheticUser Pydantic model with the expected fields and values.
    """
    segment = {
        "nome": "Planejadores",
        "descricao": "Segmento focado em planejamento financeiro e investimentos.",
        "atributos": [
            {
                "categoria": "Demografia",
                "atributo": "Idade Média",
                "valor": "40",
                "fonte": "Test",
            },
            {
                "categoria": "Comportamento",
                "atributo": "Poupança",
                "valor": "Alta",
                "fonte": "Test",
            },
        ],
    }
    agent_config = {"temperature": 0.7, "description": "desc", "system_message": "msg"}
    agent_state = {"user_id": 1}
    user_id_field = "user_id"
    agent = UserGeneratorAgent(segment, agent_config, agent_state, user_id_field)
    user = agent.generate_user()
    assert isinstance(user, SyntheticUser)
    assert user.user_id == "1"
    assert user.segment_label.value in [
        "Planejadores",
        "Poupadores",
        "Materialistas",
        "Batalhadores",
        "Céticos",
        "Endividados",
    ]
    assert user.monthly_income.value >= 0
    assert user.education_level.value in [
        "Ensino Fundamental",
        "Ensino Médio",
        "Superior Completo",
    ]
    assert isinstance(user.occupation.value, str)
    assert isinstance(user.uses_traditional_bank.value, bool)
    assert isinstance(user.uses_digital_bank.value, bool)
    assert isinstance(user.uses_broker.value, bool)
    assert user.savings_frequency_per_month.value >= 0
    assert user.spending_behavior.value in [
        "cautious",
        "immediate_consumption",
        "basic_needs",
    ]
    assert user.investment_behavior.value in [
        "diversified",
        "basic",
        "none",
    ]


def test_user_id_sequential():
    """
    Test that the user_id is assigned sequentially for each generated user.
    """
    segment = {
        "nome": "Planejadores",
        "descricao": "Segmento focado em planejamento financeiro e investimentos.",
        "atributos": [
            {
                "categoria": "Demografia",
                "atributo": "Idade Média",
                "valor": "40",
                "fonte": "Test",
            },
            {
                "categoria": "Comportamento",
                "atributo": "Poupança",
                "valor": "Alta",
                "fonte": "Test",
            },
        ],
    }
    agent_config = {"temperature": 0.7, "description": "desc", "system_message": "msg"}
    agent_state = {"user_id": 5}
    user_id_field = "user_id"
    agent = UserGeneratorAgent(segment, agent_config, agent_state, user_id_field)
    users = [agent.generate_user() for _ in range(3)]
    assert [u.user_id for u in users] == ["5", "6", "7"]
    assert agent_state["user_id"] == 8


def test_validator_agent_valid():
    """
    Test that the ValidatorAgent returns a valid CriticOutput for a valid SyntheticUser.
    """
    schema = {}  # Not used in mock
    agent_config = {"temperature": 0.3, "description": "desc", "system_message": "msg"}
    agent = ValidatorAgent(schema, agent_config)
    # Use a minimal valid SyntheticUser
    user = SyntheticUser(
        user_id="1",
        segment_label={"value": "Planejadores"},
        philosophy={"value": "Multiplicar"},
        monthly_income={"value": 1000.0},
        education_level={"value": "Ensino Médio"},
        occupation={"value": "Analista"},
        uses_traditional_bank={"value": True},
        uses_digital_bank={"value": False},
        uses_broker={"value": False},
        savings_frequency_per_month={"value": 2.0},
        spending_behavior={"value": "cautious"},
        investment_behavior={"value": "basic"},
    )
    output = agent.validate_user(user)
    assert isinstance(output, CriticOutput)
    assert output.score == 1.0
    assert output.issues == []
    assert output.recommendation == "accept"


def test_validator_agent_invalid():
    """
    Test that the ValidatorAgent returns invalid for a user missing required fields.
    """
    schema = {}  # Not used in mock
    agent_config = {"temperature": 0.3, "description": "desc", "system_message": "msg"}
    agent = ValidatorAgent(schema, agent_config)
    user = {"perfil": {}}
    valid, error = agent.validate_user(user)
    # Should be invalid and error message should mention missing fields
    assert not valid
    assert "Missing required fields" in error


def test_reviewer_agent():
    """
    Test that the ReviewerAgent returns a dict with update_synthetic_user as a SyntheticUser.
    """
    agent_config = {"temperature": 0.3, "description": "desc", "system_message": "msg"}
    agent = ReviewerAgent(agent_config)
    user = SyntheticUser(
        user_id="1",
        segment_label={"value": "Planejadores"},
        philosophy={"value": "Multiplicar"},
        monthly_income={"value": 1000.0},
        education_level={"value": "Ensino Médio"},
        occupation={"value": "Analista"},
        uses_traditional_bank={"value": True},
        uses_digital_bank={"value": False},
        uses_broker={"value": False},
        savings_frequency_per_month={"value": 2.0},
        spending_behavior={"value": "cautious"},
        investment_behavior={"value": "basic"},
    )
    critic_output = CriticOutput(score=1.0, issues=[], recommendation="accept")
    reviewed = agent.review_user(user, critic_output)
    assert "update_synthetic_user" in reviewed
    assert isinstance(reviewed["update_synthetic_user"], SyntheticUser)


def test_traced_group_chat(tmp_path):
    """
    Test that the TracedGroupChat logs messages and data and saves them to a file.
    """
    log_path = tmp_path / "trace.json"
    tracer = TracedGroupChat(str(log_path))
    tracer.log("Test message", {"foo": "bar"})
    tracer.save()
    with open(log_path) as f:
        trace = json.load(f)
    assert trace[0]["message"] == "Test message"
    assert trace[0]["data"]["foo"] == "bar"
