import json
import random
from pathlib import Path

from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.components.joiners.document_joiner import DocumentJoiner
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.writers import DocumentWriter

PASSAGES_PATH = Path("corpus/passages.jsonl")
EMB_MODEL = "BAAI/bge-base-en-v1.5"
RANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_DOCS = 50000
SEED = 42


def load_documents(path: Path, max_docs: int | None = None, seed: int = 42) -> list[Document]:
    reservoir: list[dict] = []
    rng = random.Random(seed)

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            ex = json.loads(line)
            text = (ex.get("text") or "").strip()
            if not text:
                continue

            if max_docs is None:
                reservoir.append(ex)
            elif len(reservoir) < max_docs:
                reservoir.append(ex)
            else:
                j = rng.randint(1, i)
                if j <= max_docs:
                    reservoir[j - 1] = ex

    docs: list[Document] = []
    for i, ex in enumerate(reservoir):
        docs.append(
            Document(
                content=(ex.get("text") or "").strip(),
                meta={
                    "doc_id": str(ex.get("doc_id", i)),
                    "title": ex.get("title", ""),
                },
            )
        )
    return docs


def build_store(docs: list[Document]) -> InMemoryDocumentStore:
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

    doc_embedder = SentenceTransformersDocumentEmbedder(
        model=EMB_MODEL,
        meta_fields_to_embed=["title"],
    )
    doc_embedder.warm_up()

    indexing = Pipeline()
    indexing.add_component("embedder", doc_embedder)
    indexing.add_component("writer", DocumentWriter(document_store=document_store))
    indexing.connect("embedder.documents", "writer.documents")

    indexing.run({"embedder": {"documents": docs}})
    return document_store


def run_queries(document_store: InMemoryDocumentStore) -> None:
    text_embedder = SentenceTransformersTextEmbedder(model=EMB_MODEL)
    text_embedder.warm_up()

    ranker = SentenceTransformersSimilarityRanker(
        model=RANKER_MODEL,
        top_k=5,
    )
    ranker.warm_up()

    query = Pipeline()
    query.add_component("text_embedder", text_embedder)
    query.add_component("bm25", InMemoryBM25Retriever(document_store=document_store, top_k=50))
    query.add_component(
        "embedding",
        InMemoryEmbeddingRetriever(document_store=document_store, top_k=50),
    )
    query.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion"))
    query.add_component("ranker", ranker)

    query.connect("text_embedder.embedding", "embedding.query_embedding")
    query.connect("bm25.documents", "joiner.documents")
    query.connect("embedding.documents", "joiner.documents")
    query.connect("joiner.documents", "ranker.documents")

    queries = [
        "who does jennifer lawrence play in the hunger games",
        "which NFL team did Kirk Cousins play for before the Vikings",
        "who was Ottaviano Petrucci",
        "what bones form the acetabulum",
    ]

    for q in queries:
        print("\n" + "=" * 100, flush=True)
        print("QUESTION:", q, flush=True)

        result = query.run(
            {
                "text_embedder": {"text": q},
                "bm25": {"query": q},
                "ranker": {"query": q},
            }
        )

        docs = result["ranker"]["documents"][:5]
        for rank, doc in enumerate(docs, start=1):
            title = doc.meta.get("title", "")
            doc_id = doc.meta.get("doc_id", "")
            score = getattr(doc, "score", None)
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"

            print(f"\n[{rank}] score={score_str} doc_id={doc_id} title={title}", flush=True)
            print(doc.content[:400].replace("\n", " "), flush=True)


def main() -> None:
    print("SCRIPT STARTED", flush=True)

    if not PASSAGES_PATH.exists():
        raise FileNotFoundError(PASSAGES_PATH)

    print(f"Loading documents from {PASSAGES_PATH} ...", flush=True)
    docs = load_documents(PASSAGES_PATH, max_docs=MAX_DOCS, seed=SEED)
    print(f"Loaded {len(docs)} documents", flush=True)

    print("Building in-memory Haystack store ...", flush=True)
    store = build_store(docs)

    print("Running hybrid retrieval test queries ...", flush=True)
    run_queries(store)


if __name__ == "__main__":
    main()
