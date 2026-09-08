import pytest
from rag_evaluation.embeddings import DummyEmbeddingProvider

embedding_provider = DummyEmbeddingProvider()


def test_dummy_embedding_provider_for_single_text():
    text = "Hello"

    embed_text = embedding_provider.embed(text)

    assert len(embed_text) == 768

    assert embedding_provider.model_name == "Dummy"
    assert embedding_provider.dimension == 768


def test_dummy_embedding_provider_for_batch_text():
    texts = ["Hello The", "World"]

    embed_texts = embedding_provider.embed_batch(texts)

    assert len(embed_texts) == 2
    assert all([len(emb) == 768 for emb in embed_texts])

    assert embed_texts[0] != embed_texts[1]


def test_dummy_embedding_provider_for_empty():
    with pytest.raises(ValueError):
        text = ""
        embedding_provider.embed(text)


def test_dummy_embedding_provider_for_empty_text_in_batch():
    with pytest.raises(ValueError):
        texts = ["hello", ""]

        embedding_provider.embed_batch(texts)
