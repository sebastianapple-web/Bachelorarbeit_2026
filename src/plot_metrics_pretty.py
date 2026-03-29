#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


# -----------------------------
# Farben / Stil (Variante C)
# -----------------------------
BLUE = "#355C7D"       # ohne RAG
ORANGE = "#C06C2B"     # mit RAG
GRAY = "#9A9A9A"       # neutrale dritte Größe

DARK = "#333333"
MID_GRAY = "#BDBDBD"
LIGHT_GRAY = "#E6E6E6"

DPI = 1200
FIG_WIDE = (12, 7)
FIG_STD = (10.5, 6.2)
FIG_TALL = (11, 7.5)


def setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CFCFCF",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": LIGHT_GRAY,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.95,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "legend.frameon": False,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
        }
    )


def percent_label(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f} %"


def pp_label_from_fraction(v: float, digits: int = 1) -> str:
    return f"{v * 100:+.{digits}f} pp"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Datenmodell
# -----------------------------
@dataclass
class Row:
    rag: str  # "ohne RAG" oder "mit RAG"
    temp: float
    answer_rate: float
    error_given_answer: float
    acc_given_answer: float
    abstain_rate: float
    intrinsic_all: float = 0.0
    extrinsic_all: float = 0.0
    correct_abstain_rate: float = 0.0
    wrong_abstain_rate: float = 0.0
    correct_answered: float = 0.0

    @property
    def correct_all(self) -> float:
        return self.answer_rate * self.acc_given_answer

    @property
    def false_all(self) -> float:
        return self.answer_rate * self.error_given_answer


def _to_fraction(value: str | float) -> float:
    v = float(value)
    return v / 100.0 if v > 1.0 else v


def _normalize_rag(raw: str) -> str:
    s = str(raw).strip().lower()

    if s in {
        "1", "true", "yes", "y", "rag", "with_rag", "mit", "mit rag",
        "with", "on", "enabled", "enable"
    }:
        return "mit RAG"

    if s in {
        "0", "false", "no", "n", "no_rag", "ohne", "ohne rag",
        "without", "off", "disabled", "disable"
    }:
        return "ohne RAG"

    if "mit" in s or s == "on":
        return "mit RAG"
    if "ohne" in s or s == "off":
        return "ohne RAG"

    return str(raw).strip()


def read_rows(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "rag",
            "temp",
            "answer_rate",
            "error_given_answer",
            "acc_given_answer",
            "abstain_rate",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Fehlende Spalten in {path}: {', '.join(sorted(missing))}")

        for row in reader:
            rows.append(
                Row(
                    rag=_normalize_rag(row["rag"]),
                    temp=float(row["temp"]),
                    answer_rate=_to_fraction(row["answer_rate"]),
                    error_given_answer=_to_fraction(row["error_given_answer"]),
                    acc_given_answer=_to_fraction(row["acc_given_answer"]),
                    abstain_rate=_to_fraction(row["abstain_rate"]),
                    intrinsic_all=_to_fraction(row.get("intrinsic_all", 0.0)),
                    extrinsic_all=_to_fraction(row.get("extrinsic_all", 0.0)),
                    correct_abstain_rate=_to_fraction(row.get("correct_abstain_rate", 0.0)),
                    wrong_abstain_rate=_to_fraction(row.get("wrong_abstain_rate", 0.0)),
                    correct_answered=_to_fraction(row.get("correct_answered", 0.0)),
                )
            )

    rows.sort(key=lambda r: (r.temp, r.rag))
    return rows


def split_series(rows: List[Row]) -> Dict[str, List[Row]]:
    series: Dict[str, List[Row]] = {"ohne RAG": [], "mit RAG": []}
    for r in rows:
        series.setdefault(r.rag, []).append(r)
    for rag in series:
        series[rag] = sorted(series[rag], key=lambda x: x.temp)
    return series


# -----------------------------
# Achsen / Hilfen
# -----------------------------
def apply_percent_yaxis(ax, ymin: float | None = None, ymax: float | None = None) -> None:
    if ymin is not None or ymax is not None:
        lo, hi = ax.get_ylim()
        ax.set_ylim(ymin if ymin is not None else lo, ymax if ymax is not None else hi)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f} %"))


def apply_percent_xaxis(ax) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.0f} %"))


def apply_pp_yaxis(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.1f} pp"))


def add_external_legend(fig, handles, labels, ncol: int = 2) -> None:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
        borderaxespad=0.0,
    )


def finalize_figure(fig, out_path: Path, top: float = 0.86) -> None:
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------
# Plot 1: Halluzinationsrate
# -----------------------------
def plot_hallucination_rate(series: Dict[str, List[Row]], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=FIG_STD)

    for rag, color in [("ohne RAG", BLUE), ("mit RAG", ORANGE)]:
        xs = [r.temp for r in series[rag]]
        ys = [r.error_given_answer for r in series[rag]]
        ax.plot(xs, ys, marker="o", color=color, label=rag)

    ax.set_xlabel("Sampling-Temperatur")
    ax.set_ylabel("Fehlerrate bei beantworteten Fragen")
    ax.set_xticks([r.temp for r in series["ohne RAG"]])
    apply_percent_yaxis(ax)

    handles, labels = ax.get_legend_handles_labels()
    add_external_legend(fig, handles, labels, ncol=2)
    finalize_figure(fig, outdir / "01_halluzinationsrate_nach_temperatur.png")


# -----------------------------
# Plot 2: Enthaltungsrate
# -----------------------------
def plot_abstention_rate(series: Dict[str, List[Row]], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=FIG_STD)

    for rag, color in [("ohne RAG", BLUE), ("mit RAG", ORANGE)]:
        xs = [r.temp for r in series[rag]]
        ys = [r.abstain_rate for r in series[rag]]
        ax.plot(xs, ys, marker="o", color=color, label=rag)

    ax.set_xlabel("Sampling-Temperatur")
    ax.set_ylabel("Anteil der Enthaltungen")
    ax.set_xticks([r.temp for r in series["ohne RAG"]])
    apply_percent_yaxis(ax)

    handles, labels = ax.get_legend_handles_labels()
    add_external_legend(fig, handles, labels, ncol=2)
    finalize_figure(fig, outdir / "02_enthaltungsrate_nach_temperatur.png")


# -----------------------------
# Plot 3: Interaktion RAG x Temperatur
# -----------------------------
def plot_rag_interaction(series: Dict[str, List[Row]], outdir: Path) -> None:
    temps = [r.temp for r in series["ohne RAG"]]
    no = {r.temp: r for r in series["ohne RAG"]}
    yes = {r.temp: r for r in series["mit RAG"]}

    delta_hall = [no[t].error_given_answer - yes[t].error_given_answer for t in temps]
    delta_correct_all = [yes[t].correct_all - no[t].correct_all for t in temps]
    delta_abstain = [yes[t].abstain_rate - no[t].abstain_rate for t in temps]

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.axhline(0, color=MID_GRAY, linewidth=1.0, zorder=1)

    ax.plot(temps, delta_hall, marker="o", color=BLUE, label="Halluzinationsrate", zorder=3)
    ax.plot(temps, delta_correct_all, marker="o", color=ORANGE, label="Korrekte Antworten", zorder=3)
    ax.plot(temps, delta_abstain, marker="o", color=GRAY, label="Enthaltung", zorder=3)

    ax.set_xlabel("Sampling-Temperatur")
    ax.set_ylabel("Differenz in Prozentpunkten")
    ax.set_xticks(temps)
    apply_pp_yaxis(ax)

    handles, labels = ax.get_legend_handles_labels()
    add_external_legend(fig, handles, labels, ncol=3)
    finalize_figure(fig, outdir / "03_interaktion_rag_temperatur.png")


# -----------------------------
# Plot 4a: Antwortverteilung als 3-Panel-Linienplot
# -----------------------------
def plot_response_distribution_stacked(series: Dict[str, List[Row]], outdir: Path) -> None:
    temps = [r.temp for r in series["ohne RAG"]]
    no = {r.temp: r for r in series["ohne RAG"]}
    yes = {r.temp: r for r in series["mit RAG"]}

    metrics = [
        ([no[t].correct_all for t in temps], [yes[t].correct_all for t in temps]),
        ([no[t].false_all for t in temps], [yes[t].false_all for t in temps]),
        ([no[t].abstain_rate for t in temps], [yes[t].abstain_rate for t in temps]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.8), sharex=True)

    titles = [
    "Korrekte Antworten",
    "Falsche Antworten",
    "Enthaltungen",
]

    for ax, (y_no, y_yes), title in zip(axes, metrics, titles):
        ax.plot(temps, y_no, marker="o", color=BLUE, label="ohne RAG")
        ax.plot(temps, y_yes, marker="o", color=ORANGE, label="mit RAG")
        ax.set_title(title)
        ax.set_xlabel("Sampling-Temperatur")
        ax.set_xticks(temps)
        apply_percent_yaxis(ax)

    axes[0].set_ylabel("Anteil an allen Fragen")

    handles, labels = axes[0].get_legend_handles_labels()
    add_external_legend(fig, handles, labels, ncol=2)
    finalize_figure(fig, outdir / "04a_antwortverteilung_gestapelt.png", top=0.84)


# -----------------------------
# Plot 4b: Gruppierte Antwortverteilung
# -----------------------------
def plot_response_distribution_grouped(series: Dict[str, List[Row]], outdir: Path) -> None:
    temps = [r.temp for r in series["ohne RAG"]]
    x = np.arange(len(temps))
    w = 0.12

    no = {r.temp: r for r in series["ohne RAG"]}
    yes = {r.temp: r for r in series["mit RAG"]}

    fig, ax = plt.subplots(figsize=(13.2, 6.5))

    # ohne RAG
    ax.bar(x - 2.5 * w, [no[t].correct_all for t in temps], w, color=BLUE, alpha=0.90, label="korrekt ohne RAG")
    ax.bar(x - 1.5 * w, [no[t].false_all for t in temps], w, color=BLUE, alpha=0.60, label="falsch ohne RAG")
    ax.bar(x - 0.5 * w, [no[t].abstain_rate for t in temps], w, color=BLUE, alpha=0.35, label="Enthaltung ohne RAG")

    # mit RAG
    ax.bar(x + 0.5 * w, [yes[t].correct_all for t in temps], w, color=ORANGE, alpha=0.90, label="korrekt mit RAG")
    ax.bar(x + 1.5 * w, [yes[t].false_all for t in temps], w, color=ORANGE, alpha=0.60, label="falsch mit RAG")
    ax.bar(x + 2.5 * w, [yes[t].abstain_rate for t in temps], w, color=ORANGE, alpha=0.35, label="Enthaltung mit RAG")

    ax.set_xlabel("Sampling-Temperatur")
    ax.set_ylabel("Anteil an allen Fragen")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in temps])
    apply_percent_yaxis(ax)

    handles, labels = ax.get_legend_handles_labels()
    add_external_legend(fig, handles, labels, ncol=3)
    finalize_figure(fig, outdir / "04b_antwortverteilung_gruppiert.png", top=0.82)


# -----------------------------
# Plot 5: Trade-off
# -----------------------------
def plot_tradeoff(series: Dict[str, List[Row]], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=FIG_STD)

    for rag, color in [("ohne RAG", BLUE), ("mit RAG", ORANGE)]:
        xs = [r.answer_rate for r in series[rag]]
        ys = [r.correct_all for r in series[rag]]
        ax.scatter(xs, ys, s=55, color=color, label=rag, zorder=3)

        for r in series[rag]:
            ax.annotate(
                f"T={r.temp}",
                (r.answer_rate, r.correct_all),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                color=DARK,
            )

    ax.set_xlabel("Antwortquote")
    ax.set_ylabel("Anteil korrekter Antworten an allen Fragen")
    apply_percent_xaxis(ax)
    apply_percent_yaxis(ax)

    handles, labels = ax.get_legend_handles_labels()
    add_external_legend(fig, handles, labels, ncol=2)
    finalize_figure(fig, outdir / "05_tradeoff_korrekt_vs_antwortquote.png")


# -----------------------------
# Plot 6: Dumbbell fuer Halluzinationsrate
# -----------------------------
def plot_dumbbell_hallucination(series: Dict[str, List[Row]], outdir: Path) -> None:
    temps = [r.temp for r in series["ohne RAG"]]
    no = {r.temp: r for r in series["ohne RAG"]}
    yes = {r.temp: r for r in series["mit RAG"]}

    fig, ax = plt.subplots(figsize=FIG_WIDE)

    for t in temps:
        x1 = no[t].error_given_answer
        x2 = yes[t].error_given_answer
        y = t

        ax.plot([x1, x2], [y, y], color=MID_GRAY, linewidth=2.0, zorder=1)
        ax.scatter(x1, y, color=BLUE, s=70, zorder=2)
        ax.scatter(x2, y, color=ORANGE, s=70, zorder=2)

        diff_pp = x1 - x2
        mid = (x1 + x2) / 2
        ax.text(
            mid,
            y - 0.03,
            pp_label_from_fraction(diff_pp),
            ha="center",
            va="center",
            fontsize=9,
            color=DARK,
        )

    ax.scatter([], [], color=BLUE, s=70, label="ohne RAG")
    ax.scatter([], [], color=ORANGE, s=70, label="mit RAG")

    ax.set_xlabel("Fehlerrate bei beantworteten Fragen")
    ax.set_ylabel("Sampling-Temperatur")
    ax.set_yticks(temps)
    ax.invert_yaxis()
    apply_percent_xaxis(ax)

    handles, labels = ax.get_legend_handles_labels()
    add_external_legend(fig, handles, labels, ncol=2)
    finalize_figure(fig, outdir / "06_dumbbell_halluzinationsrate.png")


# -----------------------------
# Plot 7: Heatmap optional
# -----------------------------
def plot_heatmap_hallucination(series: Dict[str, List[Row]], outdir: Path) -> None:
    temps = [r.temp for r in series["ohne RAG"]]
    matrix = np.array(
        [
            [next(x for x in series["ohne RAG"] if x.temp == t).error_given_answer for t in temps],
            [next(x for x in series["mit RAG"] if x.temp == t).error_given_answer for t in temps],
        ]
    )

    fig, ax = plt.subplots(figsize=(10.2, 4.7))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")

    ax.set_xlabel("Sampling-Temperatur")
    ax.set_ylabel("Bedingung")
    ax.set_xticks(np.arange(len(temps)))
    ax.set_xticklabels([str(t) for t in temps])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["ohne RAG", "mit RAG"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(
                j,
                i,
                percent_label(val),
                ha="center",
                va="center",
                color="white" if val > 0.45 else "black",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fehlerrate")
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.1f} %"))

    finalize_figure(fig, outdir / "07_heatmap_halluzinationsrate.png", top=0.96)


# -----------------------------
# Plot 8: Intrinsische vs. extrinsische Halluzinationen
# -----------------------------
def plot_hallucination_types(series: Dict[str, List[Row]], outdir: Path) -> None:
    temps = [r.temp for r in series["ohne RAG"]]
    no = {r.temp: r for r in series["ohne RAG"]}
    yes = {r.temp: r for r in series["mit RAG"]}

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), sharex=True)

    axes[0].plot(temps, [no[t].intrinsic_all for t in temps], marker="o", color=BLUE, label="ohne RAG")
    axes[0].plot(temps, [yes[t].intrinsic_all for t in temps], marker="o", color=ORANGE, label="mit RAG")
    axes[0].set_xlabel("Sampling-Temperatur")
    axes[0].set_ylabel("Anteil an allen Fragen")
    axes[0].set_xticks(temps)
    apply_percent_yaxis(axes[0])

    axes[1].plot(temps, [no[t].extrinsic_all for t in temps], marker="o", color=BLUE, label="ohne RAG")
    axes[1].plot(temps, [yes[t].extrinsic_all for t in temps], marker="o", color=ORANGE, label="mit RAG")
    axes[1].set_xlabel("Sampling-Temperatur")
    axes[1].set_xticks(temps)
    apply_percent_yaxis(axes[1])

    handles, labels = axes[0].get_legend_handles_labels()
    add_external_legend(fig, handles, labels, ncol=2)
    finalize_figure(fig, outdir / "08_intrinsisch_extrinsisch_nach_temperatur.png", top=0.84)


# -----------------------------
# Plot 9: Korrekte vs. unnoetige Enthaltung
# -----------------------------
def plot_abstention_quality(series: Dict[str, List[Row]], outdir: Path) -> None:
    temps = [r.temp for r in series["ohne RAG"]]
    x = np.arange(len(temps))
    w = 0.18

    no = {r.temp: r for r in series["ohne RAG"]}
    yes = {r.temp: r for r in series["mit RAG"]}

    fig, ax = plt.subplots(figsize=FIG_WIDE)

    ax.bar(x - 1.5 * w, [no[t].correct_abstain_rate for t in temps], w, color=BLUE, alpha=0.85, label="korrekte Enthaltung ohne RAG")
    ax.bar(x - 0.5 * w, [no[t].wrong_abstain_rate for t in temps], w, color=BLUE, alpha=0.45, label="unnötige Enthaltung ohne RAG")
    ax.bar(x + 0.5 * w, [yes[t].correct_abstain_rate for t in temps], w, color=ORANGE, alpha=0.85, label="korrekte Enthaltung mit RAG")
    ax.bar(x + 1.5 * w, [yes[t].wrong_abstain_rate for t in temps], w, color=ORANGE, alpha=0.45, label="unnötige Enthaltung mit RAG")

    ax.set_xlabel("Sampling-Temperatur")
    ax.set_ylabel("Anteil an allen Fragen")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in temps])
    apply_percent_yaxis(ax)

    handles, labels = ax.get_legend_handles_labels()
    add_external_legend(fig, handles, labels, ncol=2)
    finalize_figure(fig, outdir / "09_korrekte_vs_unnoetige_enthaltung.png", top=0.84)


# -----------------------------
# Plot 10: Korrekte Antworten
# -----------------------------
def plot_correct_answers(series: Dict[str, List[Row]], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=FIG_STD)

    for rag, color in [("ohne RAG", BLUE), ("mit RAG", ORANGE)]:
        xs = [r.temp for r in series[rag]]
        ys = [r.correct_all for r in series[rag]]
        ax.plot(xs, ys, marker="o", color=color, label=rag)

    ax.set_xlabel("Sampling-Temperatur")
    ax.set_ylabel("Anteil korrekter Antworten an allen Fragen")
    ax.set_xticks([r.temp for r in series["ohne RAG"]])
    apply_percent_yaxis(ax)

    handles, labels = ax.get_legend_handles_labels()
    add_external_legend(fig, handles, labels, ncol=2)
    finalize_figure(fig, outdir / "10_korrekte_antworten_nach_temperatur.png")


# -----------------------------
# Plot 11: Relative Halluzinationsreduktion durch RAG
# -----------------------------
def plot_relative_rag_improvement(series: Dict[str, List[Row]], outdir: Path) -> None:
    temps = [r.temp for r in series["ohne RAG"]]
    no = {r.temp: r for r in series["ohne RAG"]}
    yes = {r.temp: r for r in series["mit RAG"]}

    rel = []
    for t in temps:
        base = no[t].error_given_answer
        rag = yes[t].error_given_answer
        if base == 0:
            rel.append(0.0)
        else:
            rel.append((base - rag) / base)

    x = np.arange(len(temps))

    fig, ax = plt.subplots(figsize=FIG_STD)

    bars = ax.bar(
        x,
        rel,
        width=0.58,
        color=GRAY,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ax.axhline(0, color=MID_GRAY, linewidth=1.0, zorder=2)
    ax.set_xlabel("Sampling-Temperatur")
    ax.set_ylabel("Relative Veränderung der Halluzinationsrate")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in temps])
    apply_percent_yaxis(ax)

    ymin = min(rel)
    ymax = max(rel)
    ax.set_ylim(ymin - 0.04, ymax + 0.02)

    for b, v in zip(bars, rel):
        if v >= 0:
            y = v + 0.01
            va = "bottom"
        else:
            y = v - 0.012
            va = "top"

        ax.text(
            b.get_x() + b.get_width() / 2,
            y,
            f"{v * 100:.1f} %",
            ha="center",
            va=va,
            fontsize=9,
            color=DARK,
            clip_on=False,
            zorder=4,
        )

    finalize_figure(fig, outdir / "11_relative_rag_verbesserung.png", top=0.90)


# -----------------------------
# Plot 12: Genauigkeit unter beantworteten Fragen
# -----------------------------
def plot_accuracy_answered_only(series: Dict[str, List[Row]], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=FIG_STD)

    for rag, color in [("ohne RAG", BLUE), ("mit RAG", ORANGE)]:
        xs = [r.temp for r in series[rag]]
        ys = [r.correct_answered for r in series[rag]]
        ax.plot(xs, ys, marker="o", color=color, label=rag)

    ax.set_xlabel("Sampling-Temperatur")
    ax.set_ylabel("Anteil korrekter Antworten unter beantworteten Fragen")
    ax.set_xticks([r.temp for r in series["ohne RAG"]])
    apply_percent_yaxis(ax)

    handles, labels = ax.get_legend_handles_labels()
    add_external_legend(fig, handles, labels, ncol=2)
    finalize_figure(fig, outdir / "12_genauigkeit_unter_beantworteten_fragen.png")


# -----------------------------
# Summary CSV fuer den Textteil
# -----------------------------
def write_summary_csv(series: Dict[str, List[Row]], outdir: Path) -> None:
    out_path = outdir / "plot_summary_table.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "temp",
                "rag",
                "answer_rate",
                "error_given_answer",
                "acc_given_answer",
                "abstain_rate",
                "intrinsic_all",
                "extrinsic_all",
                "correct_abstain_rate",
                "wrong_abstain_rate",
                "correct_answered",
                "correct_all",
                "false_all",
            ]
        )
        for rag in ["ohne RAG", "mit RAG"]:
            for r in series[rag]:
                writer.writerow(
                    [
                        r.temp,
                        r.rag,
                        r.answer_rate,
                        r.error_given_answer,
                        r.acc_given_answer,
                        r.abstain_rate,
                        r.intrinsic_all,
                        r.extrinsic_all,
                        r.correct_abstain_rate,
                        r.wrong_abstain_rate,
                        r.correct_answered,
                        r.correct_all,
                        r.false_all,
                    ]
                )


# -----------------------------
# Hauptfunktion
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Erzeuge wissenschaftlich saubere Plots fuer Temperatur x RAG."
    )
    parser.add_argument("--in", dest="input_csv", required=True, help="Pfad zu metrics_agg.csv")
    parser.add_argument("--outdir", required=True, help="Ausgabeordner fuer PNG-Dateien")
    parser.add_argument(
        "--with-heatmap",
        action="store_true",
        help="Erzeuge zusaetzlich eine Heatmap als optionale Anhangsgrafik",
    )
    args = parser.parse_args()

    setup_style()

    input_csv = Path(args.input_csv)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    rows = read_rows(input_csv)
    series = split_series(rows)

    if not series.get("ohne RAG") or not series.get("mit RAG"):
        raise ValueError(
            "Es werden Daten fuer beide Bedingungen benoetigt: 'ohne RAG' und 'mit RAG'."
        )

    plot_hallucination_rate(series, outdir)
    plot_abstention_rate(series, outdir)
    plot_rag_interaction(series, outdir)
    plot_response_distribution_stacked(series, outdir)
    plot_response_distribution_grouped(series, outdir)
    plot_tradeoff(series, outdir)
    plot_dumbbell_hallucination(series, outdir)
    plot_hallucination_types(series, outdir)
    plot_abstention_quality(series, outdir)
    plot_correct_answers(series, outdir)
    plot_relative_rag_improvement(series, outdir)
    plot_accuracy_answered_only(series, outdir)

    if args.with_heatmap:
        plot_heatmap_hallucination(series, outdir)

    write_summary_csv(series, outdir)

    print(f"[OK] Plots gespeichert in: {outdir}")


if __name__ == "__main__":
    main()