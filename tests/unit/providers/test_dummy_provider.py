from rag_evaluation.models.outputs import LLMProviderErrorType
from rag_evaluation.providers.generation.dummy import DummyLLMProvider


def test_dummy_llm_provider_for_success():
    prompt = "you are dummy judge"

    llm_provider = DummyLLMProvider()
    result = llm_provider.call(prompt)

    assert result.success is True
    assert result.content is not None
    assert result.error is None
    assert result.error_type is None


def test_dummy_llm_provider_for_failure():
    prompt = ""

    llm_provider = DummyLLMProvider()
    result = llm_provider.call(prompt)

    assert result.success is False
    assert result.content is None
    assert result.error is not None
    assert result.error_type == LLMProviderErrorType.INVALID_REQUEST
