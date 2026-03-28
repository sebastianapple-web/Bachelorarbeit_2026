import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def normalize_question(q: str) -> str:
    return " ".join(q.strip().split())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200, help="Number of questions to sample")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--split", type=str, default="train", choices=["train", "validation"], help="Dataset split")
    p.add_argument("--out", type=str, default="", help="Output path (optional). If empty, auto-named.")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    out_path = Path(args.out) if args.out else Path(f"data_processed/questions_{args.n}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("nq_open", split=args.split)

    candidates = []
    for ex in ds:
        q = ex.get("question", "")
        ans = ex.get("answer", [])
        if not q or not ans:
            continue

        qn = normalize_question(q)
        if not qn:
            continue

        gold = [a for a in ans if isinstance(a, str) and a.strip()]
        if not gold:
            continue

        candidates.append((qn, gold))

    if len(candidates) < args.n:
        raise RuntimeError(f"Not enough candidates: {len(candidates)} < {args.n}")

    random.shuffle(candidates)
    selected = candidates[: args.n]

    with out_path.open("w", encoding="utf-8") as f:
        for i, (q, gold) in enumerate(selected, start=1):
            row = {
                "id": f"q{i:05d}",
                "question": q,
                "gold_answers": gold,
                "source_dataset": "nq_open",
                "split": args.split,
                "seed": args.seed,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {args.n} items to {out_path}")


if __name__ == "__main__":
    main()
