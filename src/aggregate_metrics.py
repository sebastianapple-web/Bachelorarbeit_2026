# src/aggregate_metrics.py
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_bool(value) -> bool:
    return str(value).strip().lower() == "true"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to eval_basic.csv")
    ap.add_argument("--out", dest="out", required=True, help="Path to agg_metrics.csv")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    S = defaultdict(
        lambda: {
            "n": 0,
            "abstain": 0,
            "answered": 0,
            "correct_ans": 0,
            "incorrect_ans": 0,
        }
    )

    with inp.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rag = str(row["rag"]).strip()
            temp = float(row["temp"])
            key = (rag, temp)

            s = S[key]
            s["n"] += 1

            abstain = parse_bool(row["abstain"])
            label_basic = str(row["label_basic"]).strip().lower()

            if abstain:
                s["abstain"] += 1
            else:
                s["answered"] += 1

                if label_basic == "correct":
                    s["correct_ans"] += 1
                elif label_basic == "incorrect":
                    s["incorrect_ans"] += 1
                else:
                    # Falls ein unerwartetes Label auftaucht, zählen wir es sicherheitshalber als incorrect
                    s["incorrect_ans"] += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as wf:
        fieldnames = [
            "rag",
            "temp",
            "n",
            "abstain_rate",
            "answer_rate",
            "acc_given_answer",
            "error_given_answer",
            "answered",
            "correct_ans",
            "incorrect_ans",
        ]
        w = csv.DictWriter(wf, fieldnames=fieldnames)
        w.writeheader()

        for (rag, temp), s in sorted(S.items(), key=lambda x: (x[0][0], x[0][1])):
            n = s["n"]
            answered = s["answered"]

            abstain_rate = s["abstain"] / n if n else 0.0
            answer_rate = answered / n if n else 0.0
            acc = s["correct_ans"] / answered if answered else 0.0
            err = s["incorrect_ans"] / answered if answered else 0.0

            w.writerow(
                {
                    "rag": rag,
                    "temp": f"{temp:.1f}",
                    "n": n,
                    "abstain_rate": f"{abstain_rate:.6f}",
                    "answer_rate": f"{answer_rate:.6f}",
                    "acc_given_answer": f"{acc:.6f}",
                    "error_given_answer": f"{err:.6f}",
                    "answered": answered,
                    "correct_ans": s["correct_ans"],
                    "incorrect_ans": s["incorrect_ans"],
                }
            )

    print("WROTE:", out)
    print("ROWS:", len(S))


if __name__ == "__main__":
    main()
