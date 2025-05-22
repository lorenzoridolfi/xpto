import os
import json
import pytest
from tradeshow.src.synthetic_user_generator import UserGeneratorAgent, ValidatorAgent, ReviewerAgent, TracedGroupChat

def test_user_generator_agent():
    segment = {
        "nome": "TestSegment",
        "descricao": "Test description",
        "atributos": [
            {"categoria": "Demografia", "atributo": "Idade Média", "valor": "40", "fonte": "Test"},
            {"categoria": "Comportamento", "atributo": "Poupança", "valor": "Alta", "fonte": "Test"}
        ]
    }
    agent = UserGeneratorAgent(segment)
    user = agent.generate_user()
    assert user["segmento"] == "TestSegment"
    assert user["perfil"]["idade"] == "40"
    assert user["perfil"]["poupanca"] == "Alta"
    assert user["perfil"]["descricao"] == "Test description"

def test_validator_agent_valid():
    schema = {}  # Not used in mock
    agent = ValidatorAgent(schema)
    user = {"segmento": "Test", "perfil": {}}
    valid, error = agent.validate_user(user)
    assert valid
    assert error == ""

def test_validator_agent_invalid():
    schema = {}  # Not used in mock
    agent = ValidatorAgent(schema)
    user = {"perfil": {}}
    valid, error = agent.validate_user(user)
    assert not valid
    assert "Missing required fields" in error

def test_reviewer_agent():
    agent = ReviewerAgent()
    user = {"segmento": "Test", "perfil": {}}
    error = "Missing required fields"
    reviewed = agent.review_user(user, error)
    assert "review_note" in reviewed
    assert error in reviewed["review_note"]

def test_traced_group_chat(tmp_path):
    log_path = tmp_path / "trace.json"
    tracer = TracedGroupChat(str(log_path))
    tracer.log("Test message", {"foo": "bar"})
    tracer.save()
    with open(log_path) as f:
        trace = json.load(f)
    assert trace[0]["message"] == "Test message"
    assert trace[0]["data"]["foo"] == "bar" 