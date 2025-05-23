import os
import json
import pytest
from tradeshow.src.synthetic_user_generator import (
    UserGeneratorAgent,
    ValidatorAgent,
    ReviewerAgent,
    TracedGroupChat,
)


def test_user_generator_agent():
    """
    Test that the UserGeneratorAgent correctly generates a synthetic user
    with the expected fields and values based on the segment input.
    """
    # Define a mock segment for testing
    segment = {
        "nome": "TestSegment",
        "descricao": "Test description",
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
    # Mock agent config and state
    agent_config = {"temperature": 0.7, "description": "desc", "system_message": "msg"}
    agent_state = {"user_id": 1}
    user_id_field = "user_id"
    agent = UserGeneratorAgent(segment, agent_config, agent_state, user_id_field)
    user = agent.generate_user()
    # Check that the user fields are correct
    assert user["user_id"] == 1
    assert user["segmento"] == "TestSegment"
    assert user["perfil"]["idade"] == "40"
    assert user["perfil"]["poupanca"] == "Alta"
    assert user["perfil"]["descricao"] == "Test description"


def test_validator_agent_valid():
    """
    Test that the ValidatorAgent returns valid for a user with required fields.
    """
    schema = {}  # Not used in mock
    agent_config = {"temperature": 0.3, "description": "desc", "system_message": "msg"}
    agent = ValidatorAgent(schema, agent_config)
    user = {"segmento": "Test", "perfil": {}}
    valid, error = agent.validate_user(user)
    # Should be valid and no error message
    assert valid
    assert error == ""


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
    Test that the ReviewerAgent adds a review note to the user when reviewing.
    """
    agent_config = {"temperature": 0.3, "description": "desc", "system_message": "msg"}
    agent = ReviewerAgent(agent_config)
    user = {"segmento": "Test", "perfil": {}}
    error = "Missing required fields"
    reviewed = agent.review_user(user, error)
    # The review note should be present and contain the error message
    assert "review_note" in reviewed
    assert error in reviewed["review_note"]


def test_traced_group_chat(tmp_path):
    """
    Test that the TracedGroupChat logs messages and data and saves them to a file.
    """
    log_path = tmp_path / "trace.json"
    tracer = TracedGroupChat(str(log_path))
    # Log a test message with data
    tracer.log("Test message", {"foo": "bar"})
    tracer.save()
    # Load the trace and check its contents
    with open(log_path) as f:
        trace = json.load(f)
    assert trace[0]["message"] == "Test message"
    assert trace[0]["data"]["foo"] == "bar"
