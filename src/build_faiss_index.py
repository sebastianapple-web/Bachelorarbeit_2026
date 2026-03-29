import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PASSAGES_PATH = Path("corpus/passages.jsonl")
INDEX_PATH = Path("corpus/index.faiss")
META_PATH = Path("corpus/meta.jsonl")

EMB_MODEL = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 64


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def build_embed_text(ex: dict) -> str:
    title = (ex.get("title") or "").strip()
    text = (ex.get("text") or "").strip()

    if title:
        return f"{title}. {text}"
    return text


def main():
    if not PASSAGES_PATH.exists():
        raise FileNotFoundError(PASSAGES_PATH)

    model = SentenceTransformer(EMB_MODEL)

    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    meta_f = META_PATH.open("w", encoding="utf-8")
    embeddings_list = []

    batch = []
    for ex in tqdm(iter_jsonl(PASSAGES_PATH), desc="Reading passages"):
        batch.append(ex)

        if len(batch) >= BATCH_SIZE:
            batch_texts = [build_embed_text(b) for b in batch]
            emb = model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
            embeddings_list.append(emb)

            for b in batch:
                meta_f.write(
                    json.dumps(
                        {
                            "doc_id": b["doc_id"],
                            "title": b.get("title", ""),
                            "text": b["text"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            batch = []

    if batch:
        batch_texts = [build_embed_text(b) for b in batch]
        emb = model.encode(
            batch_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        embeddings_list.append(emb)

        for b in batch:
            meta_f.write(
                json.dumps(
                    {
                        "doc_id": b["doc_id"],
                        "title": b.get("title", ""),
                        "text": b["text"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    meta_f.close()

    if not embeddings_list:
        raise ValueError("No embeddings were created. Check corpus/passages.jsonl")

    embeddings = np.vstack(embeddings_list).astype("float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    print("Embeddings shape:", embeddings.shape)
    print("Wrote index:", INDEX_PATH)
    print("Wrote meta :", META_PATH)


if __name__ == "__main__":
    main()
