import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
GRAY = "#7f7f7f"


def pct_formatter(x, pos):
    return f"{x:.1f} %"


def pct_formatter_int(x, pos):
    return f"{x:.0f} %"


def setup_style():
    plt.rcParams.update({
        "figure.figsize": (8.8, 5.4),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.1,
        "lines.linewidth": 2.5,
        "lines.markersize": 7,
    })


def style_axis(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, alpha=0.18, linewidth=0.8)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)


def aggregate_behavior(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (rag, temp), sub in df.groupby(["rag", "temp"]):
        n = len(sub)

        abstain_mask = sub["abstain"].astype(str).str.lower() == "true"
        correct_mask = sub["label_basic"].astype(str).str.lower() == "correct"
        incorrect_mask = sub["label_basic"].astype(str).str.lower() == "incorrect"

        abstain = abstain_mask.sum()
        answered = n - abstain
        correct = correct_mask.sum()
        incorrect = incorrect_mask.sum()

        rows.append(
            {
                "rag": rag,
                "temp": float(temp),
                "n": n,
                "abstain_rate": abstain / n * 100,
                "answer_rate": answered / n * 100,
                "correct_rate_total": correct / n * 100,
                "incorrect_rate_total": incorrect / n * 100,
            }
        )

    return pd.DataFrame(rows).sort_values(["rag", "temp"])


def plot_abstention_rate(agg: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots()

    labels = {"off": "ohne RAG", "on": "mit RAG"}
    colors = {"off": BLUE, "on": ORANGE}

    for rag in ["off", "on"]:
        sub = agg[agg["rag"] == rag].sort_values("temp")
        ax.plot(
            sub["temp"],
            sub["abstain_rate"],
            marker="o",
            label=labels[rag],
            color=colors[rag],
        )

    ax.set_title("Enthaltungsrate nach Temperatur", pad=14)
    ax.set_xlabel("Sampling-Temperatur")
    ax.set_ylabel("Enthaltungsrate")
    ax.set_xticks(sorted(agg["temp"].unique()))
    ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(outdir / "antwortverhalten_1_enthaltungsrate.png", bbox_inches="tight")
    plt.close(fig)


def plot_answer_vs_abstain(agg: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots()

    for rag, color_base in [("off", BLUE), ("on", ORANGE)]:
        sub = agg[agg["rag"] == rag].sort_values("temp")
        label_suffix = "ohne RAG" if rag == "off" else "mit RAG"

        ax.plot(
            sub["temp"],
            sub["answer_rate"],
            marker="o",
            linestyle="-",
            color=color_base,
            label=f"Antwortquote {label_suffix}",
        )
        ax.plot(
            sub["temp"],
            sub["abstain_rate"],
            marker="o",
            linestyle="--",
            color=color_base,
            alpha=0.95,
            label=f"Enthaltungsquote {label_suffix}",
        )

    ax.set_title("Antwort- und Enthaltungsquote", pad=14)
    ax.set_xlabel("Sampling-Temperatur")
    ax.set_ylabel("Anteil an allen Fragen")
    ax.set_xticks(sorted(agg["temp"].unique()))
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter_int))
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, ncols=2)

    plt.tight_layout()
    fig.savefig(outdir / "antwortverhalten_2_antwort_vs_enthaltung.png", bbox_inches="tight")
    plt.close(fig)


def plot_stacked_outcomes(agg: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(10.8, 5.8))

    order = []
    labels = []
    for temp in sorted(agg["temp"].unique()):
        order.append(("off", temp))
        order.append(("on", temp))
        labels.append(f"{temp}\nohne RAG")
        labels.append(f"{temp}\nmit RAG")

    correct_vals = []
    incorrect_vals = []
    abstain_vals = []

    for rag, temp in order:
        row = agg[(agg["rag"] == rag) & (agg["temp"] == temp)].iloc[0]
        correct_vals.append(row["correct_rate_total"])
        incorrect_vals.append(row["incorrect_rate_total"])
        abstain_vals.append(row["abstain_rate"])

    x = np.arange(len(order))

    ax.bar(x, correct_vals, color=GREEN, label="korrekt", width=0.72)
    ax.bar(x, incorrect_vals, bottom=correct_vals, color=RED, label="falsch", width=0.72)
    ax.bar(
        x,
        abstain_vals,
        bottom=np.array(correct_vals) + np.array(incorrect_vals),
        color=GRAY,
        label="Enthaltung",
        width=0.72,
    )

    ax.set_title("Antwortverteilung nach Temperatur und Bedingung", pad=14)
    ax.set_xlabel("Temperatur und Bedingung")
    ax.set_ylabel("Anteil an allen Fragen")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter_int))
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, ncols=3)

    plt.tight_layout()
    fig.savefig(outdir / "antwortverhalten_3_antwortverteilung_gestapelt.png", bbox_inches="tight")
    plt.close(fig)


def plot_grouped_outcomes(agg: pd.DataFrame, outdir: Path):
    temps = sorted(agg["temp"].unique())
    x = np.arange(len(temps))
    width = 0.13

    fig, ax = plt.subplots(figsize=(11.2, 5.8))

    off = agg[agg["rag"] == "off"].sort_values("temp")
    on = agg[agg["rag"] == "on"].sort_values("temp")

    ax.bar(x - 2.5 * width, off["correct_rate_total"], width, color=GREEN, alpha=0.95, label="korrekt ohne RAG")
    ax.bar(x - 1.5 * width, off["incorrect_rate_total"], width, color=RED, alpha=0.95, label="falsch ohne RAG")
    ax.bar(x - 0.5 * width, off["abstain_rate"], width, color=GRAY, alpha=0.95, label="Enthaltung ohne RAG")

    ax.bar(x + 0.5 * width, on["correct_rate_total"], width, color=GREEN, alpha=0.55, label="korrekt mit RAG")
    ax.bar(x + 1.5 * width, on["incorrect_rate_total"], width, color=RED, alpha=0.55, label="falsch mit RAG")
    ax.bar(x + 2.5 * width, on["abstain_rate"], width, color=GRAY, alpha=0.55, label="Enthaltung mit RAG")

    ax.set_title("Korrekt, falsch und Enthaltung im Vergleich", pad=14)
    ax.set_xlabel("Sampling-Temperatur")
    ax.set_ylabel("Anteil an allen Fragen")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in temps])
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter_int))
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, ncols=2)

    plt.tight_layout()
    fig.savefig(outdir / "antwortverhalten_4_korrekt_falsch_enthaltung_balken.png", bbox_inches="tight")
    plt.close(fig)


def find_latest_run_dir(base_dir: Path) -> Path:
    candidates = [p for p in base_dir.glob("llama_cpp_run_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Keine Run-Ordner gefunden in: {base_dir}")
    return sorted(candidates)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        default=".",
        help="Basisordner, in dem nach llama_cpp_run_* gesucht wird",
    )
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    run_dir = find_latest_run_dir(base_dir)

    inp = run_dir / "outputs_eval.csv"
    if not inp.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {inp}")

    outdir = run_dir / "plots_response_behavior"
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    setup_style()

    agg = aggregate_behavior(df)

    plot_abstention_rate(agg, outdir)
    plot_answer_vs_abstain(agg, outdir)
    plot_stacked_outcomes(agg, outdir)
    plot_grouped_outcomes(agg, outdir)

    print("RUN_DIR:", run_dir)
    print("INPUT:", inp)
    print("WROTE:", outdir)


if __name__ == "__main__":
    main()
