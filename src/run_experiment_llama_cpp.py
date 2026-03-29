import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llama_cpp import Llama

# Passe diesen Pfad an dein lokales GGUF-Modell an
MODEL_PATH = Path("models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

DEFAULT_QUESTIONS_PATH = "data_processed/questions_500.jsonl"
CONDITIONS_PATH = Path("experiment_conditions.csv")
CURATED_CORPUS_PATH = Path("corpus/curated_passages.jsonl")


SYSTEM_PROMPT = """You are a factual QA assistant.
Return ONLY valid JSON with exactly these keys:
- answer: string
- abstain: boolean
- confidence: number

Rules:
- If the answer is sufficiently supported, answer briefly and factually.
- If you are not sufficiently certain, abstain.
- No markdown, no code fences, no extra text.
"""

USER_TEMPLATE_RAG_OFF = """Answer the question factually and conservatively.

QUESTION:
{question}
"""

USER_TEMPLATE_RAG_ON = """Answer the question factually and conservatively using the provided CONTEXT.
Use the CONTEXT as primary evidence.
If the CONTEXT does not provide enough information for a sufficiently certain answer, abstain.

QUESTION:
{question}

CONTEXT:
{context}
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--questions",
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to questions JSONL or CSV",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=500,
        help="How many questions to run from that file (<= file length)",
    )
    ap.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Path to GGUF model",
    )
    return ap.parse_args()


def load_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                gold_answers = row.get("gold_answers", "")
                if isinstance(gold_answers, str):
                    gold_answers = [x.strip() for x in gold_answers.split("|") if x.strip()]
                rows.append(
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "gold_answers": gold_answers,
                    }
                )
            return rows

    raise ValueError(f"Unsupported questions format: {path.suffix}")


def load_conditions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Conditions file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "condition_id": row["condition_id"],
                    "rag": row["rag"].strip().lower(),
                    "temp": float(row["temp"]),
                    "top_k": int(row.get("top_k", 0) or 0),
                }
            )
    return rows


def load_curated_corpus(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Curated corpus not found: {path}")

    corpus: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = str(obj.get("doc_id", "")).strip()
            if doc_id:
                corpus[doc_id] = obj
    return corpus


def get_curated_hits(question_id: str, curated_corpus: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []

    for suffix, score in [("p1", 1.0), ("p2", 0.9)]:
        doc_id = f"{question_id}_{suffix}"
        doc = curated_corpus.get(doc_id)
        if doc:
            hits.append(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                    "score": score,
                }
            )

    return hits


def build_prompt(question: str, rag: str, context: str = "") -> str:
    if rag == "on":
        return USER_TEMPLATE_RAG_ON.format(question=question, context=context)
    return USER_TEMPLATE_RAG_OFF.format(question=question)


def format_context(hits: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for h in hits:
        header = "[DOC]"
        title = str(h.get("title", "") or "").strip()
        if title:
            header += f" Title: {title}"
        header += f" (doc_id={h['doc_id']}, score={h.get('score', 0.0):.4f})"
        blocks.append(f"{header}\n{h['text']}")
    return "\n\n".join(blocks)


def safe_parse_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    t = (text or "").strip()
    if not t:
        return None, "empty_output"

    # ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.DOTALL)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate), None
        except Exception as e:
            return None, f"fenced_json_parse_error: {e}"

    # Direktes JSON
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj, None
    except Exception:
        pass

    # Erstes JSON-Objekt im Text
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, None
        except Exception as e:
            return None, f"embedded_json_parse_error: {e}"

    return None, "no_json_object_found"


def normalize_parsed_output(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not obj:
        return {"answer": "I don't know", "abstain": True, "confidence": 1.0}

    answer = str(obj.get("answer", "") or "").strip()
    abstain_raw = obj.get("abstain", False)
    confidence_raw = obj.get("confidence", 0.0)

    if isinstance(abstain_raw, str):
        abstain = abstain_raw.strip().lower() == "true"
    else:
        abstain = bool(abstain_raw)

    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.0

    # Nur bei echter Abstention den Text vereinheitlichen
    if abstain:
        answer = "I don't know"

    if not answer and not abstain:
        answer = ""
    elif not answer and abstain:
        answer = "I don't know"

    return {
        "answer": answer,
        "abstain": abstain,
        "confidence": confidence,
    }


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    questions_path = Path(args.questions)
    questions = load_questions(questions_path)[: args.n]
    conditions = load_conditions(CONDITIONS_PATH)

    curated_corpus = load_curated_corpus(CURATED_CORPUS_PATH)
    print("Loaded curated corpus docs:", len(curated_corpus))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"llama_cpp_run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "outputs.jsonl"

    llm = Llama(
        model_path=str(model_path),
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=-1,
        verbose=False,
    )

    with out_path.open("w", encoding="utf-8") as out_f:
        for q in questions:
            for c in conditions:
                hits: List[Dict[str, Any]] = []
                context = ""

                if c["rag"] == "on":
                    hits = get_curated_hits(q["id"], curated_corpus)
                    context = format_context(hits)

                prompt = build_prompt(q["question"], rag=c["rag"], context=context)

                resp = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=float(c["temp"]),
                    max_tokens=128,
                )

                raw_output = (
                    resp.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                parsed_obj, parse_error = safe_parse_json(raw_output)
                parsed_output = normalize_parsed_output(parsed_obj)

                row = {
                    "run_id": run_id,
                    "condition_id": c["condition_id"],
                    "question_id": q["id"],
                    "question": q["question"],
                    "gold_answers": q["gold_answers"],
                    "rag": c["rag"],
                    "temp": c["temp"],
                    "top_k": c["top_k"],
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "parsed_output": parsed_output,
                    "parse_error": parse_error,
                    "retrieved_doc_ids": [h["doc_id"] for h in hits] if c["rag"] == "on" else [],
                    "retrieved_passages": [h["text"] for h in hits] if c["rag"] == "on" else [],
                }

                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()

    print("WROTE:", out_path)
    print("Expected rows:", len(questions) * len(conditions))


if __name__ == "__main__":
    main()


