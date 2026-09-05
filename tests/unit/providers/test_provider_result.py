import pytest

from rag_evaluation.models.outputs import LLMProviderErrorType, LLMProviderResult


def test_LLMProviderResult_for_success_with_conent():
    LLMProviderResult(success=True, content="exists", error=None, error_type=None)


def test_LLMProviderResult_for_rate_limit():
    LLMProviderResult(
        success=False,
        content=None,
        error="Rate limit reached",
        error_type=LLMProviderErrorType.RATE_LIMIT,
    )


def test_LLMProviderResult_for_tiemout():
    LLMProviderResult(
        success=False,
        content=None,
        error="Timeout",
        error_type=LLMProviderErrorType.TIMEOUT,
    )


def test_LLMProviderResult_for_invalid_request():
    LLMProviderResult(
        success=False,
        content=None,
        error="INVALID_REQUEST",
        error_type=LLMProviderErrorType.INVALID_REQUEST,
    )


def test_LLMProviderResult_for_success_with_error():
    with pytest.raises(ValueError):
        LLMProviderResult(
            success=True,
            content=None,
            error="INVALID_REQUEST",
            error_type=LLMProviderErrorType.INVALID_REQUEST,
        )


def test_LLMProviderResult_for_fail_with_no_error():
    with pytest.raises(ValueError):
        LLMProviderResult(
            success=False,
            content=None,
            error=None,
            error_type=LLMProviderErrorType.INVALID_REQUEST,
        )
