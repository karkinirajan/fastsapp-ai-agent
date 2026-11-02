"""Tests for app/services/vectorstore.py"""

from unittest.mock import MagicMock, mock_open, patch


@patch("app.services.vectorstore.chromadb")
@patch("app.services.vectorstore.OpenAIEmbeddingFunction")
def test_get_collection_ingests_when_empty(MockEF, MockChroma, mock_settings, tmp_path):
    """When the collection is empty, documents from the CSV are ingested."""
    import app.services.vectorstore as vs

    vs._collection = None  # reset module-level cache

    mock_col = MagicMock()
    mock_col.count.side_effect = [0, 2]  # 0 before ingest, 2 after
    MockChroma.PersistentClient.return_value.get_or_create_collection.return_value = mock_col

    csv_content = "Question one answer one\nQuestion two answer two\n"
    with (
        patch("builtins.open", mock_open(read_data=csv_content)),
        patch("app.services.vectorstore.Path.exists", return_value=True),
    ):
        vs._init_collection()

    mock_col.add.assert_called_once()
    vs._collection = None  # cleanup


@patch("app.services.vectorstore.chromadb")
@patch("app.services.vectorstore.OpenAIEmbeddingFunction")
def test_get_collection_skips_ingestion_when_populated(MockEF, MockChroma, mock_settings):
    """When the collection already has documents, ingestion is skipped."""
    import app.services.vectorstore as vs

    vs._collection = None

    mock_col = MagicMock()
    mock_col.count.return_value = 10
    MockChroma.PersistentClient.return_value.get_or_create_collection.return_value = mock_col

    vs._init_collection()

    mock_col.add.assert_not_called()
    vs._collection = None


def test_query_vectorstore_returns_docs(mock_settings):
    """query_vectorstore returns the flat list of document strings."""
    import app.services.vectorstore as vs

    mock_col = MagicMock()
    mock_col.count.return_value = 5
    mock_col.query.return_value = {
        "documents": [["Doc A", "Doc B", "Doc C"]],
    }
    vs._collection = mock_col

    results = vs.query_vectorstore("query", n_results=3)

    assert results == ["Doc A", "Doc B", "Doc C"]
    vs._collection = None


def test_query_vectorstore_empty_collection(mock_settings):
    """Returns empty list when the collection has no documents."""
    import app.services.vectorstore as vs

    mock_col = MagicMock()
    mock_col.count.return_value = 0
    vs._collection = mock_col

    results = vs.query_vectorstore("query")
    assert results == []
    vs._collection = None


@patch("app.services.vectorstore.cohere")
def test_get_rag_context_returns_formatted_string(MockCohere, mock_settings):
    """get_rag_context returns a non-empty formatted reference string."""
    import app.services.vectorstore as vs

    mock_col = MagicMock()
    mock_col.count.return_value = 3
    mock_col.query.return_value = {"documents": [["Doc1", "Doc2", "Doc3"]]}
    vs._collection = mock_col

    rerank_result = MagicMock()
    r0, r1, r2 = MagicMock(), MagicMock(), MagicMock()
    r0.index = 0
    r1.index = 1
    r2.index = 2
    rerank_result.results = [r0, r1, r2]

    co_client = MagicMock()
    co_client.rerank.return_value = rerank_result
    MockCohere.ClientV2.return_value = co_client

    context = vs.get_rag_context("test query")

    assert "參考資料" in context
    assert "Doc1" in context
    vs._collection = None


@patch("app.services.vectorstore.cohere")
def test_get_rag_context_falls_back_on_cohere_error(MockCohere, mock_settings):
    """When Cohere rerank fails, falls back to raw ChromaDB order."""
    import app.services.vectorstore as vs

    mock_col = MagicMock()
    mock_col.count.return_value = 2
    mock_col.query.return_value = {"documents": [["Doc1", "Doc2"]]}
    vs._collection = mock_col

    co_client = MagicMock()
    co_client.rerank.side_effect = RuntimeError("cohere down")
    MockCohere.ClientV2.return_value = co_client

    context = vs.get_rag_context("test query")

    # Should still return something with the raw docs
    assert "Doc1" in context
    vs._collection = None
