import json
import re
from pathlib import Path

QUESTIONS_PATH = Path("data_processed/questions_200.jsonl")
OUT_PATH = Path("corpus/curated_passages.jsonl")


def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def clean_question(q: str) -> str:
    q = clean_text(q)
    return q.rstrip(" ?")


def build_title(question_id: str, question: str, gold_answers: list[str]) -> str:
    gold = clean_text(gold_answers[0]) if gold_answers else ""
    q = clean_question(question)

    if gold:
        return gold[:80]
    return f"{question_id}: {q[:80]}"


def build_statement(question: str, gold_answers: list[str]) -> str:
    q = clean_question(question)
    gold = clean_text(gold_answers[0]) if gold_answers else ""
    q_lower = q.lower()

    # weniger tautologische, etwas natürlichere Evidenzsätze
    patterns = [
        (r"^who was the first person to (.+)$",
         lambda m: f"{gold} is historically associated with being the first person to {m.group(1)}."),
        (r"^who was (.+)$",
         lambda m: f"{gold} is associated with {m.group(1)}."),
        (r"^who is (.+)$",
         lambda m: f"{gold} is associated with {m.group(1)}."),
        (r"^where(?:'s| is) (.+)$",
         lambda m: f"{m.group(1).capitalize()} takes place at {gold}."),
        (r"^what is the (.+)$",
         lambda m: f"The reported {m.group(1)} is {gold}."),
        (r"^what was the (.+)$",
         lambda m: f"The recorded {m.group(1)} was {gold}."),
        (r"^which (.+)$",
         lambda m: f"{gold} is the entity associated with {m.group(1)}."),
        (r"^when (.+)$",
         lambda m: f"The date associated with {m.group(1)} is {gold}."),
        (r"^how many (.+)$",
         lambda m: f"The number associated with {m.group(1)} is {gold}."),
        (r"^how much (.+)$",
         lambda m: f"The amount associated with {m.group(1)} is {gold}."),
    ]

    for pattern, fn in patterns:
        m = re.match(pattern, q_lower)
        if m:
            return fn(m)

    return f"{gold} is relevant to the topic described as: {q}."


def build_support_statement(question: str, gold_answers: list[str]) -> str:
    q = clean_question(question)
    gold = clean_text(gold_answers[0]) if gold_answers else ""

    # zweite Passage: etwas allgemeiner, aber noch hilfreich
    return f"This passage concerns the topic '{q}' and identifies {gold} as the key fact."


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with QUESTIONS_PATH.open("r", encoding="utf-8") as f_in, OUT_PATH.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            ex = json.loads(line)

            qid = ex["id"]
            question = ex["question"]
            gold_answers = ex.get("gold_answers", [])

            rows = [
                {
                    "doc_id": f"{qid}_p1",
                    "title": build_title(qid, question, gold_answers),
                    "text": build_statement(question, gold_answers),
                },
                {
                    "doc_id": f"{qid}_p2",
                    "title": f"{build_title(qid, question, gold_answers)} context",
                    "text": build_support_statement(question, gold_answers),
                },
            ]

            for row in rows:
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} passages to {OUT_PATH}")


if __name__ == "__main__":
    main()
