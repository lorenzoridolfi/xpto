from autogen_extensions.llm_mock import LLMMock

def test_llm_mock_returns_static_response():
    mock = LLMMock(static_response="DESCRIÇÃO: Test\nRESUMO: Test summary")
    response = mock.create(model="gpt-4", messages=[{"role": "user", "content": "foo"}])
    assert response.choices[0].message.content == "DESCRIÇÃO: Test\nRESUMO: Test summary"
    assert len(mock.calls) == 1
    assert mock.calls[0][1]["model"] == "gpt-4"

def test_llm_mock_default_response():
    mock = LLMMock()
    response = mock.create()
    assert "Mock summary" in response.choices[0].message.content 