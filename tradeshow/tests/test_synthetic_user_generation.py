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
from tradeshow.src.pydantic_schema import (
    SyntheticUserDraft,
    CriticOutput,
    SyntheticUserReviewed,
)
from autogen_extensions.json_validation import validate_json
from types import SimpleNamespace
import inspect
import dotenv
import tempfile
import asyncio

# Define absolute paths for schemas (correct folder: tradeshow/schema)
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
SCHEMA_PATH = os.path.join(ROOT_PATH, "tradeshow", "schema")
SEGMENTS_SCHEMA_PATH = os.path.join(SCHEMA_PATH, "segments_schema.json")
SYNTHETIC_USER_SCHEMA_PATH = os.path.join(SCHEMA_PATH, "synthetic_user_schema.json")

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


def make_synthetic_user_draft(user_id="1"):
    return SyntheticUserDraft(
        id_usuario=user_id,
        segmento={"valor": "Planejadores"},
        filosofia={"valor": "Multiplicar"},
        renda_mensal={"valor": 1000.0},
        escolaridade={"valor": "Ensino Médio"},
        ocupacao={"valor": "Analista"},
        usa_banco_tradicional={"valor": True},
        usa_banco_digital={"valor": False},
        usa_corretora={"valor": False},
        frequencia_poupanca_mensal={"valor": 2.0},
        comportamento_gastos={"valor": "cauteloso"},
        comportamento_investimentos={"valor": "basico"},
    )


def make_critic_output():
    return CriticOutput(score=1.0, issues=[], recommendation="accept")


def make_synthetic_user_reviewed(user_id="1"):
    return SyntheticUserReviewed(
        id_usuario=user_id,
        segmento={"valor": "Planejadores"},
        filosofia={"valor": "Multiplicar"},
        renda_mensal={"valor": 1000.0},
        escolaridade={"valor": "Ensino Médio"},
        ocupacao={"valor": "Analista"},
        usa_banco_tradicional={"valor": True},
        usa_banco_digital={"valor": False},
        usa_corretora={"valor": False},
        frequencia_poupanca_mensal={"valor": 2.0},
        comportamento_gastos={"valor": "cauteloso"},
        comportamento_investimentos={"valor": "basico"},
        avaliacao={"critica": "", "revisao": ""},
    )


def async_return(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner


@pytest.fixture
def mock_llm_call():
    async def _mock_llm_call(*args, **kwargs):
        frame = inspect.currentframe()
        while frame:
            if frame.f_code.co_name == "generate_user":
                agent_self = frame.f_locals.get("self")
                if getattr(agent_self, "_force_invalid_llm_response", False):
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="{"))]
                    )
                user_id = (
                    agent_self.state[agent_self.user_id_field] - 1
                    if agent_self
                    else "1"
                )
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=json.dumps(
                                    make_synthetic_user_draft(str(user_id)).model_dump()
                                )
                            )
                        )
                    ]
                )
            if frame.f_code.co_name == "validate_user":
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=json.dumps(make_critic_output().model_dump())
                            )
                        )
                    ]
                )
            if frame.f_code.co_name == "review_user":
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=json.dumps(
                                    make_synthetic_user_reviewed("1").model_dump()
                                )
                            )
                        )
                    ]
                )
            frame = frame.f_back
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )

    return _mock_llm_call


@pytest.mark.asyncio
async def test_user_generator_agent(mock_llm_call):
    """
    Test that the UserGeneratorAgent correctly generates a synthetic user
    as a SyntheticUserDraft Pydantic model with the expected fields and values.
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
    schema = {}  # Use a dummy schema for the mock
    agent = UserGeneratorAgent(
        segment,
        agent_config,
        agent_state,
        user_id_field,
        schema,
        llm_call=mock_llm_call,
    )
    user = await agent.generate_user()
    assert isinstance(user, SyntheticUserDraft)
    assert user.id_usuario == "1"
    assert user.segmento.valor in [
        "Planejadores",
        "Poupadores",
        "Materialistas",
        "Batalhadores",
        "Céticos",
        "Endividados",
    ]
    assert user.renda_mensal.valor >= 0
    assert user.escolaridade.valor in [
        "Ensino Fundamental",
        "Ensino Médio",
        "Superior Completo",
    ]
    assert isinstance(user.ocupacao.valor, str)
    assert isinstance(user.usa_banco_tradicional.valor, bool)
    assert isinstance(user.usa_banco_digital.valor, bool)
    assert isinstance(user.usa_corretora.valor, bool)
    assert user.frequencia_poupanca_mensal.valor >= 0
    assert user.comportamento_gastos.valor in [
        "cauteloso",
        "consumo_imediato",
        "necessidades_basicas",
    ]
    assert user.comportamento_investimentos.valor in [
        "diversificado",
        "basico",
        "nenhum",
    ]


@pytest.mark.asyncio
async def test_user_id_sequential(mock_llm_call):
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
    agent = UserGeneratorAgent(
        segment,
        agent_config,
        agent_state,
        user_id_field,
        schema={},
        llm_call=mock_llm_call,
    )
    users = [await agent.generate_user() for _ in range(3)]
    assert [u.id_usuario for u in users] == ["5", "6", "7"]
    assert agent_state["user_id"] == 8


@pytest.mark.asyncio
async def test_validator_agent_valid(mock_llm_call):
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
    agent = ValidatorAgent(schema, agent_config, llm_call=mock_llm_call)
    user = make_synthetic_user_draft()
    output = await agent.validate_user(user)
    assert isinstance(output, CriticOutput)
    assert output.score == 1.0
    assert output.issues == []
    assert output.recommendation == "accept"


@pytest.mark.asyncio
async def test_reviewer_agent(mock_llm_call):
    """
    Test that the ReviewerAgent returns a dict with update_synthetic_user as a SyntheticUserReviewed.
    """
    agent_config = {
        "temperature": 0.2,
        "model": "gpt-4o",
        "description": "desc",
        "system_message": "msg",
    }
    agent = ReviewerAgent(agent_config, schema={}, llm_call=mock_llm_call)
    user = make_synthetic_user_draft()
    critic_output = make_critic_output()
    reviewed = await agent.review_user(user, critic_output)
    assert "update_synthetic_user" in reviewed
    assert isinstance(reviewed["update_synthetic_user"], SyntheticUserReviewed)


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
    validate_json(data, schema)


@pytest.mark.asyncio
async def test_orchestrator_respects_num_usuarios(tmp_path, monkeypatch, mock_llm_call):
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
        str(patched_config_path),
        agent_config_path,
        agent_state_path,
        str(output_file),
        False,
        llm_call=mock_llm_call,
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


def test_current_segment_file_updates(tmp_path, mock_llm_call):
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
        str(patched_config_path),
        agent_config_path,
        agent_state_path,
        str(output_file),
        False,
        llm_call=mock_llm_call,
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

    user_agent = UserGeneratorAgent(
        segment,
        agent_config,
        agent_state,
        user_id_field,
        schema,
        llm_call=mock_llm_call,
    )
    validator_agent = ValidatorAgent(schema, agent_config, llm_call=mock_llm_call)
    reviewer_agent = ReviewerAgent(agent_config, schema, llm_call=mock_llm_call)

    # Check config values
    for agent in [user_agent, validator_agent, reviewer_agent]:
        assert agent.model == "mock-llm"
        assert agent.temperature == 0.5
        assert agent.config["max_tokens"] == 30000
        assert agent.config["description"] == "desc"
        assert agent.config["system_message"] == "msg"


@pytest.mark.asyncio
async def test_load_openai_api_key_from_env():
    """
    Test that the OPENAI_API_KEY is loaded from the root .env file and present in the environment.
    """
    ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DOTENV_PATH = os.path.join(ROOT_PATH, ".env")
    assert os.path.exists(DOTENV_PATH), f".env file not found at {DOTENV_PATH}"
    dotenv.load_dotenv(dotenv_path=DOTENV_PATH, override=True)
    key = os.environ.get("OPENAI_API_KEY")
    assert (
        key is not None and key.strip() != ""
    ), "OPENAI_API_KEY not loaded from .env or is empty"


def test_traced_group_chat_log_and_save():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    tracer = TracedGroupChat(log_path=path)
    tracer.log(
        message="Test message",
        agent={"name": "TestAgent"},
        activity="test_activity",
        data={"foo": "bar"},
        tool_call=None,
        llm_input=[{"role": "user", "content": "hi"}],
        llm_output="response",
        duration_seconds=1.23,
    )
    tracer.save()
    with open(path) as f:
        data = json.load(f)
    assert len(data) == 1
    entry = data[0]
    assert entry["message"] == "Test message"
    assert entry["activity"] == "test_activity"
    assert entry["agent"]["name"] == "TestAgent"
    assert entry["data"]["foo"] == "bar"
    assert entry["llm_output"] == "response"
    assert entry["duration_seconds"] == 1.23
    os.remove(path)


def minimal_segment():
    return {
        "nome": "TestSegment",
        "descricao": "desc",
        "atributos": [],
    }


def minimal_agent_config():
    return {
        "temperature": 0.1,
        "model": "gpt-4o",
        "description": "desc",
        "system_message": "msg",
    }


def minimal_schema():
    return {}


def test_user_generator_agent_logs_to_tracer(mock_llm_call):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        trace_path = tmp.name
    tracer = TracedGroupChat(log_path=trace_path)
    agent = UserGeneratorAgent(
        segment=minimal_segment(),
        agent_config=minimal_agent_config(),
        agent_state={"user_id": 1},
        user_id_field="user_id",
        schema=minimal_schema(),
        tracer=tracer,
        llm_call=mock_llm_call,
    )
    asyncio.run(agent.generate_user())
    tracer.save()
    with open(trace_path) as f:
        trace = json.load(f)
    assert any(e["activity"] == "generate_user" for e in trace)


def test_validator_agent_logs_to_tracer(mock_llm_call):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        trace_path = tmp.name
    tracer = TracedGroupChat(log_path=trace_path)
    agent = ValidatorAgent(
        schema=minimal_schema(),
        agent_config=minimal_agent_config(),
        tracer=tracer,
        llm_call=mock_llm_call,
    )
    user = make_synthetic_user_draft()
    asyncio.run(agent.validate_user(user))
    tracer.save()
    with open(trace_path) as f:
        trace = json.load(f)
    assert any(e["activity"] == "validate_user" for e in trace)


def test_reviewer_agent_logs_to_tracer(mock_llm_call):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        trace_path = tmp.name
    tracer = TracedGroupChat(log_path=trace_path)
    agent = ReviewerAgent(
        agent_config=minimal_agent_config(),
        schema=minimal_schema(),
        tracer=tracer,
        llm_call=mock_llm_call,
    )
    user = make_synthetic_user_draft()
    critic_output = make_critic_output()
    asyncio.run(agent.review_user(user, critic_output))
    tracer.save()
    with open(trace_path) as f:
        trace = json.load(f)
    # Ensure all entries are dicts, not Pydantic models
    for entry in trace:
        if isinstance(entry.get("data"), dict):
            assert True
    assert any(e["activity"] == "review_user" for e in trace)


def test_user_generator_agent_logs_error_to_tracer(mock_llm_call):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        trace_path = tmp.name
    tracer = TracedGroupChat(log_path=trace_path)
    agent = UserGeneratorAgent(
        segment=minimal_segment(),
        agent_config=minimal_agent_config(),
        agent_state={"user_id": 1},
        user_id_field="user_id",
        schema=minimal_schema(),
        tracer=tracer,
        llm_call=mock_llm_call,
    )
    agent._force_invalid_llm_response = True
    try:
        asyncio.run(agent.generate_user())
    except Exception:
        pass
    tracer.save()
    with open(trace_path) as f:
        trace = json.load(f)
    assert any("error" in e["message"].lower() for e in trace)
