import csv
import json
import random
from datetime import datetime
from pathlib import Path

SEED = 42
QUESTIONS_PATH = Path("data_processed/questions_200.jsonl")
CONDITIONS_PATH = Path("experiment_conditions.csv")

#Dry run (test mit 3 Fragen)
DRY_RUN_N = 3

def load_questions(path: Path):
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
    return questions

def load_conditions(path: Path):
    conditions = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["temp"] = float(row["temp"])
            conditions.append(row)
    return conditions

def dummy_generate(question: str, rag: str, temp: float):
    # Dummy: simuliert Output-Format + Konfidenz
    random.seed(hash((question, rag, temp, SEED)) % (2**32))
    abstain = random.random() < 0.2  # 20% abstain
    confidence = random.randint(1, 5)
    answer = "I don't know" if abstain else f"Dummy answer (rag={rag}, temp={temp})"
    return {
        "answer": answer,
        "abstain": abstain,
        "confidence": confidence,
        "citations": [] if rag == "off" else [{"doc_id": "dummy_doc", "quote": "dummy quote"}],
    }

def main():
    random.seed(SEED)

    questions = load_questions(QUESTIONS_PATH)
    conditions = load_conditions(CONDITIONS_PATH)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"dryrun_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "outputs.jsonl"

    with out_path.open("w", encoding="utf-8") as out:
        for q in questions[:DRY_RUN_N]:
            for c in conditions:
                result = dummy_generate(q["question"], c["rag"], c["temp"])
                row = {
                    "run_id": run_id,
                    "question_id": q["id"],
                    "question": q["question"],
                    "gold_answers": q["gold_answers"],
                    "condition_id": c["condition_id"],
                    "rag": c["rag"],
                    "temp": c["temp"],
                    "model_output": result,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Dry-run written to: {out_path}")
    print("Expected rows:", DRY_RUN_N * len(conditions))

if __name__ == "__main__":
    main()




