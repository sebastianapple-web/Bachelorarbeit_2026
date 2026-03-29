# src/eval_basic.py
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def norm(s: str) -> str:
    s = str(s or "").strip().lower()

    prefixes = [
        "the answer is ",
        "it is ",
        "it's ",
        "its ",
        "this is ",
        "the ",
        "a ",
        "an ",
    ]
    for p in prefixes:
        if s.startswith(p):
            s = s[len(p):]

    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"“”‘’\(\)\[\]\{\}]", "", s)
    s = re.sub(r"[^\w\s\-\./]", "", s)
    return s.strip()


def gold_match(answer: str, gold_answers: List[str]) -> bool:
    a = norm(answer)
    if not a:
        return False

 
    bad_answers = {
        "",
        "abstain",
        "unknown",
        "not found",
        "not specified",
        "no information available",
        "i dont know",
        "i don't know",
        "none mentioned",
        "not mentioned",
    }
    if a in bad_answers:
        return False

    for g in gold_answers or []:
        g2 = norm(g)
        if not g2:
            continue

        # exakter Match
        if a == g2:
            return True

        # Substring in beide Richtungen
        if a in g2 or g2 in a:
            return True

        # Token-basierter Match für Antworten wie:
        # "the ilium, ischium and pubis meet at the acetabulum"
        # vs. gold = "acetabulum"
        a_tokens = set(a.split())
        g_tokens = set(g2.split())

        if g_tokens and g_tokens.issubset(a_tokens):
            return True

    return False


def parse_abstain(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
    return bool(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to outputs.jsonl")
    ap.add_argument("--out", dest="out", required=True, help="Path to output CSV")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    rows_out: List[Dict[str, Any]] = []

    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)

            parsed = r.get("parsed_output") or {}
            answer = str(parsed.get("answer", "") or "")
            abstain = parse_abstain(parsed.get("abstain", False))
            conf = parsed.get("confidence", None)

            gm = False
            if not abstain:
                gm = gold_match(answer, r.get("gold_answers") or [])

            if abstain:
                label = "abstain"
            else:
                label = "correct" if gm else "incorrect"

            rows_out.append(
                {
                    "run_id": r.get("run_id"),
                    "condition_id": r.get("condition_id"),
                    "question_id": r.get("question_id"),
                    "rag": r.get("rag"),
                    "temp": r.get("temp"),
                    "top_k": r.get("top_k", 0),
                    "abstain": abstain,
                    "confidence": conf,
                    "answer": answer,
                    "gold_answers": " | ".join(r.get("gold_answers") or []),
                    "gold_match": gm,
                    "label_basic": label,
                }
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with out.open("w", encoding="utf-8", newline="") as wf:
        w = csv.DictWriter(wf, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print("WROTE:", out)
    print("ROWS:", len(rows_out))


if __name__ == "__main__":
    main()
