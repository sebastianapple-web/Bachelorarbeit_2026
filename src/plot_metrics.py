# src/plot_metrics.py
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_agg(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "rag": row["rag"],
                    "temp": float(row["temp"]),
                    "answer_rate": float(row["answer_rate"]),
                    "error_given_answer": float(row["error_given_answer"]),
                    "acc_given_answer": float(row["acc_given_answer"]),
                    "abstain_rate": float(row["abstain_rate"]),
                }
            )
    return rows


def plot_metric(rows, metric_key: str, out_path: Path, title: str, y_label: str):
    # split by rag
    series = {}
    for r in rows:
        series.setdefault(r["rag"], []).append(r)

    plt.figure()
    for rag, rs in sorted(series.items()):
        rs = sorted(rs, key=lambda x: x["temp"])
        xs = [x["temp"] for x in rs]
        ys = [x[metric_key] for x in rs]
        plt.plot(xs, ys, marker="o", label=rag)

    plt.title(title)
    plt.xlabel("temperature")
    plt.ylabel(y_label)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linewidth=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_acc_vs_coverage(rows, out_path: Path):
    # Scatter: x = answer_rate (coverage), y = acc_given_answer
    series = {}
    for r in rows:
        series.setdefault(r["rag"], []).append(r)

    plt.figure()
    for rag, rs in sorted(series.items()):
        rs = sorted(rs, key=lambda x: x["temp"])
        xs = [x["answer_rate"] for x in rs]
        ys = [x["acc_given_answer"] for x in rs]
        plt.scatter(xs, ys, label=rag)

        # label points by temperature
        for x, y, rr in zip(xs, ys, rs):
            plt.annotate(
                f"{rr['temp']}",
                (x, y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    plt.title("Accuracy vs. Coverage (Answer rate)")
    plt.xlabel("coverage (answer_rate)")
    plt.ylabel("accuracy (acc_given_answer)")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True, linewidth=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to agg_metrics.csv")
    ap.add_argument("--outdir", dest="outdir", required=True, help="Output directory for PNGs")
    args = ap.parse_args()

    inp = Path(args.inp)
    outdir = Path(args.outdir)

    rows = read_agg(inp)

    plot_metric(
        rows,
        metric_key="answer_rate",
        out_path=outdir / "answer_rate_vs_temp.png",
        title="Answer rate vs. temperature",
        y_label="answer_rate",
    )

    plot_metric(
        rows,
        metric_key="error_given_answer",
        out_path=outdir / "error_given_answer_vs_temp.png",
        title="Error rate (given answer) vs. temperature",
        y_label="error_given_answer",
    )

    plot_acc_vs_coverage(
        rows,
        out_path=outdir / "accuracy_vs_coverage.png",
    )

    print("WROTE:", outdir / "answer_rate_vs_temp.png")
    print("WROTE:", outdir / "error_given_answer_vs_temp.png")
    print("WROTE:", outdir / "accuracy_vs_coverage.png")


if __name__ == "__main__":
    main()
