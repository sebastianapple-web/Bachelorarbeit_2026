import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

OUT_PATH = Path("corpus/passages.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# None = kompletter Korpus
# z.B. 100_000 / 500_000 / 1_000_000 für begrenzte Läufe
MAX_PASSAGES = 200000


def main():
    # DPR-style Wikipedia passages dataset
    ds = load_dataset("wiki_dpr", "psgs_w100.nq.no_index.no_embeddings", split="train")

    n = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in tqdm(ds, desc="Writing passages"):
            title = ex.get("title", "")
            text = ex.get("text", "")
            pid = ex.get("id", None)

            if not text:
                continue

            row = {
                "doc_id": str(pid) if pid is not None else str(n),
                "title": title,
                "text": text,
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

            if MAX_PASSAGES is not None and n >= MAX_PASSAGES:
                break

    print(f"Wrote {n} passages to {OUT_PATH}")


if __name__ == "__main__":
    main()
