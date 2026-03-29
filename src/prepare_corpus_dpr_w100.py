import json
from pathlib import Path

import ir_datasets
from tqdm import tqdm

OUT_PATH = Path("corpus/passages.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_PASSAGES = 1_000_000  

def main():
    dataset = ir_datasets.load("dpr-w100")

    n = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for doc in tqdm(dataset.docs_iter(), desc="Writing DPR-W100 passages"):
            # doc_id ist String; doc.text enthält Passage
            row = {
                "doc_id": doc.doc_id,
                "title": "",          # DPR-W100 bei ir_datasets: meist nur Text
                "text": doc.text,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n >= MAX_PASSAGES:
                break

    print(f"Wrote {n} passages to {OUT_PATH}")

if __name__ == "__main__":
    main()
