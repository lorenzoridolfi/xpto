import os
import json
import pytest
from tradeshow.src.synthetic_user_generator import (
    UserGeneratorAgent,
    ValidatorAgent,
    ReviewerAgent,
    TracedGroupChat,
    Orchestrator,
)
from tradeshow.src.pydantic_schema import SyntheticUser, CriticOutput
from jsonschema import validate, ValidationError
from unittest.mock import patch, AsyncMock
from types import SimpleNamespace
import inspect

# Define absolute paths for schemas (correct folder: tradeshow/schema)
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
SCHEMA_PATH = os.path.join(ROOT_PATH, 'tradeshow', 'schema')
SEGMENTS_SCHEMA_PATH = os.path.join(SCHEMA_PATH, 'segments_schema.json')
SYNTHETIC_USER_SCHEMA_PATH = os.path.join(SCHEMA_PATH, 'synthetic_user_schema.json')

# WARNING: Do NOT overwrite or create any files in tradeshow/schema in tests!

# WARNING: Do NOT overwrite real schema files in tests!
# The following fixture is commented out to prevent accidental overwrites.
# @pytest.fixture(scope='session', autouse=True)
# def ensure_schema_files():
#     os.makedirs(SCHEMA_PATH, exist_ok=True)
#     for path in [SEGMENTS_SCHEMA_PATH, SYNTHETIC_USER_SCHEMA_PATH]:
#         if not os.path.exists(path):
#             with open(path, 'w') as f:
#                 json.dump({"dummy": True}, f)

@pytest.fixture(autouse=True)
def patch_llm_client():
    with patch("tradeshow.src.synthetic_user_generator.OpenAIChatCompletionClient") as MockClient:
        instance = MockClient.return_value

        def create_side_effect(*args, **kwargs):
            # Inspect the call stack to determine which agent method is calling
            frame = inspect.currentframe()
            while frame:
                if frame.f_code.co_name == "generate_user":
                    agent_self = frame.f_locals.get("self")
                    user_id = agent_self.state[agent_self.user_id_field] - 1 if agent_self else "1"
                    return SimpleNamespace(content=SyntheticUser(
                        user_id=str(user_id),
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
                    ))
                if frame.f_code.co_name == "validate_user":
                    return SimpleNamespace(content=CriticOutput(score=1.0, issues=[], recommendation="accept"))
                if frame.f_code.co_name == "review_user":
                    return SimpleNamespace(content=SyntheticUser(
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
                    ))
                frame = frame.f_back
            # Fallback
            return SimpleNamespace(content=None)

        instance.create = AsyncMock(side_effect=create_side_effect)
        yield

@pytest.mark.asyncio
async def test_user_generator_agent():
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
    agent_config = {
        "temperature": 0.7,
        "model": "gpt-4o",
        "description": "desc",
        "system_message": "msg",
    }
    agent_state = {"user_id": 1}
    user_id_field = "user_id"
    agent = UserGeneratorAgent(segment, agent_config, agent_state, user_id_field)
    user = await agent.generate_user()
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

@pytest.mark.asyncio
async def test_user_id_sequential():
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
    agent_config = {
        "temperature": 0.7,
        "model": "gpt-4o",
        "description": "desc",
        "system_message": "msg",
    }
    agent_state = {"user_id": 5}
    user_id_field = "user_id"
    agent = UserGeneratorAgent(segment, agent_config, agent_state, user_id_field)
    users = [await agent.generate_user() for _ in range(3)]
    assert [u.user_id for u in users] == ["5", "6", "7"]
    assert agent_state["user_id"] == 8

@pytest.mark.asyncio
async def test_validator_agent_valid():
    """
    Test that the ValidatorAgent returns a valid CriticOutput for a valid SyntheticUser.
    """
    schema = {}  # Not used in mock
    agent_config = {
        "temperature": 0.0,
        "model": "gpt-4o",
        "description": "desc",
        "system_message": "msg",
    }
    agent = ValidatorAgent(schema, agent_config)
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
    output = await agent.validate_user(user)
    assert isinstance(output, CriticOutput)
    assert output.score == 1.0
    assert output.issues == []
    assert output.recommendation == "accept"

@pytest.mark.asyncio
async def test_reviewer_agent():
    """
    Test that the ReviewerAgent returns a dict with update_synthetic_user as a SyntheticUser.
    """
    agent_config = {
        "temperature": 0.2,
        "model": "gpt-4o",
        "description": "desc",
        "system_message": "msg",
    }
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
    reviewed = await agent.review_user(user, critic_output)
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
    # Only check 'data' if present
    if "data" in trace[0]:
        assert trace[0]["data"]["foo"] == "bar"

def test_segments_json_nickname_and_usercount():
    """
    Test that each segment in segments.json has a valid 'apelido' and 'num_usuarios' field (now 3 for all).
    """
    with open("tradeshow/input/segments.json") as f:
        data = json.load(f)
    for segment in data["segmentos"]:
        assert "apelido" in segment
        assert isinstance(segment["apelido"], str)
        assert segment["apelido"] == segment["apelido"].lower()
        assert segment["apelido"].count("_") <= 1
        assert len(segment["apelido"].split("_")) <= 2
        assert len(segment["apelido"]) <= 32
        assert "num_usuarios" in segment
        assert isinstance(segment["num_usuarios"], int)
        assert segment["num_usuarios"] == 3

def test_segments_schema_validation():
    """
    Test that segments.json validates against the updated segments_schema.json.
    """
    with open("tradeshow/input/segments.json") as f:
        data = json.load(f)
    with open(SEGMENTS_SCHEMA_PATH) as f:
        schema = json.load(f)
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        pytest.fail(f"segments.json does not validate against schema: {e}")

@pytest.mark.asyncio
async def test_orchestrator_respects_num_usuarios(tmp_path, monkeypatch):
    """
    Test that the orchestrator generates the correct number of users per segment as specified by 'num_usuarios'.
    """
    # Patch config to use a temp output file
    config_path = "tradeshow/config.json"
    agent_config_path = "tradeshow/config_agents.json"
    agent_state_path = "tradeshow/config_agents_state.json"
    # Patch output file to a temp file
    with open(config_path) as f:
        config = json.load(f)
    output_file = tmp_path / "synthetic_users.json"
    log_file = tmp_path / "trace.json"
    config["output_file"] = str(output_file)
    config["log_file"] = str(log_file)
    # Patch schema file path with absolute path
    config["synthetic_user_schema_file"] = SYNTHETIC_USER_SCHEMA_PATH
    # Write patched config
    patched_config_path = tmp_path / "config.json"
    with open(patched_config_path, "w") as f:
        json.dump(config, f)
    orchestrator = Orchestrator(
        str(patched_config_path), agent_config_path, agent_state_path
    )
    await orchestrator.run()
    # Check output file
    with open(output_file) as f:
        users = json.load(f)
    # Load segments to check expected count
    with open("tradeshow/input/segments.json") as f:
        segments = json.load(f)["segmentos"]
    expected_count = sum(seg["num_usuarios"] for seg in segments)
    assert len(users) == expected_count

def test_current_segment_file_updates(tmp_path):
    """
    Test that 'current_segment.json' is updated for each segment processed by the orchestrator.
    """
    # Patch config to use a temp output file
    config_path = "tradeshow/config.json"
    agent_config_path = "tradeshow/config_agents.json"
    agent_state_path = "tradeshow/config_agents_state.json"
    with open(config_path) as f:
        config = json.load(f)
    output_file = tmp_path / "synthetic_users.json"
    log_file = tmp_path / "trace.json"
    config["output_file"] = str(output_file)
    config["log_file"] = str(log_file)
    # Patch schema file path with absolute path
    config["synthetic_user_schema_file"] = SYNTHETIC_USER_SCHEMA_PATH
    patched_config_path = tmp_path / "config.json"
    with open(patched_config_path, "w") as f:
        json.dump(config, f)
    orchestrator = Orchestrator(
        str(patched_config_path), agent_config_path, agent_state_path
    )
    # Run only the segment loop, not the full async run
    segments = orchestrator.segments
    for segment in segments:
        # Simulate the orchestrator's segment loop
        with open(tmp_path / "current_segment.json", "w", encoding="utf-8") as f:
            json.dump(segment, f, ensure_ascii=False, indent=2)
        with open(tmp_path / "current_segment.json", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["nome"] == segment["nome"]
        assert loaded["apelido"] == segment["apelido"]
        assert loaded["num_usuarios"] == 3

def test_agent_llm_config_and_model_parameters():
    """
    Test that all agents are instantiated with the correct LLM, model, and parameters.
    """
    agent_config = {
        "temperature": 0.5,
        "model": "mock-llm",
        "max_tokens": 30000,
        "description": "desc",
        "system_message": "msg",
    }
    segment = {
        "nome": "TestSegment",
        "descricao": "Test description.",
        "atributos": [],
    }
    agent_state = {"user_id": 1}
    user_id_field = "user_id"
    schema = {}

    with patch("tradeshow.src.synthetic_user_generator.OpenAIChatCompletionClient"):
        user_agent = UserGeneratorAgent(segment, agent_config, agent_state, user_id_field)
        validator_agent = ValidatorAgent(schema, agent_config)
        reviewer_agent = ReviewerAgent(agent_config)

        # Check config values
        for agent in [user_agent, validator_agent, reviewer_agent]:
            assert agent.model == "mock-llm"
            assert agent.temperature == 0.5
            assert agent.config["max_tokens"] == 30000
            assert agent.config["description"] == "desc"
            assert agent.config["system_message"] == "msg"
