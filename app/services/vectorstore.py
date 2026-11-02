"""
Vectorstore service — ChromaDB + Cohere reranking.

Documents from the FAQ CSV are ingested once into a persistent ChromaDB
collection at startup (skipped if the collection is already populated).
Queries embed the user's message, retrieve candidate documents, and then
rerank them with Cohere for precision before returning formatted context.
"""

import csv
import logging
from pathlib import Path

import chromadb
import cohere
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from app.config import get_settings

logger = logging.getLogger(__name__)

_collection: chromadb.Collection | None = None


# ── Collection initialisation ─────────────────────────────────────────────────


def get_collection() -> chromadb.Collection:
    """Return the shared ChromaDB collection, initialising it on first call."""
    global _collection
    if _collection is None:
        _collection = _init_collection()
    return _collection


def _init_collection() -> chromadb.Collection:
    settings = get_settings()
    ef = OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.openai_embedding_model,
    )
    client = chromadb.PersistentClient(path=settings.vectorstore_path)
    collection = client.get_or_create_collection(name="faq", embedding_function=ef)

    if collection.count() == 0:
        logger.info("Vectorstore empty — ingesting FAQ documents…")
        _ingest(collection, settings.content_csv_path)
    else:
        logger.info("Vectorstore has %d documents — skipping ingestion.", collection.count())

    return collection


def _ingest(collection: chromadb.Collection, csv_path: str) -> None:
    path = Path(csv_path)
    if not path.exists():
        logger.warning("CSV not found at %s — skipping ingestion.", csv_path)
        return

    texts: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            text = row[0].strip() if row else ""
            if text:
                texts.append(text)

    if not texts:
        logger.warning("No non-empty rows found in %s.", csv_path)
        return

    ids = [str(i) for i in range(len(texts))]
    collection.add(documents=texts, ids=ids)
    logger.info("Ingested %d FAQ documents into vectorstore.", len(texts))


# ── Query + rerank ────────────────────────────────────────────────────────────


def query_vectorstore(query: str, n_results: int = 8) -> list[str]:
    """Return top n_results raw documents for the given query."""
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, count),
    )
    docs = results.get("documents", [[]])[0]
    return [d for d in docs if d]


def get_rag_context(query: str) -> str:
    """
    Query the vectorstore, rerank with Cohere, and return a formatted
    context string ready to be injected into the system prompt.
    Returns an empty string if no documents are found.
    """
    settings = get_settings()
    docs = query_vectorstore(query, n_results=settings.vectorstore_query_n)
    if not docs:
        return ""

    try:
        co = cohere.ClientV2(api_key=settings.cohere_api_key)
        response = co.rerank(
            model=settings.cohere_rerank_model,
            query=query,
            documents=docs,
            top_n=min(settings.cohere_rerank_top_n, len(docs)),
        )
        top_docs = [docs[r.index] for r in response.results]
    except Exception as exc:
        logger.error("Cohere rerank failed (%s) — using raw ChromaDB order.", exc)
        top_docs = docs[: settings.cohere_rerank_top_n]

    formatted = "\n\n".join(f"參考資料{i}：{doc}" for i, doc in enumerate(top_docs, start=1))
    return f"相關參考資料：\n\n{formatted}"
