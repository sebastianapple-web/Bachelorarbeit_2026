#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# =========================================================
# Pfade
# =========================================================
INPUT_CSV = Path("outputs_eval.csv")
OUTDIR = Path("regression_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Hilfsfunktionen
# =========================================================
def parse_abstain(series: pd.Series) -> pd.Series:
    """
    Konvertiert unterschiedliche Bool-Formate robust nach True/False.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
    )

    if mapped.isna().any():
        bad_vals = series[mapped.isna()].dropna().unique()
        raise ValueError(f"Unbekannte Werte in 'abstain': {bad_vals}")

    return mapped.astype(bool)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bereitet die Daten für die beiden logistischen Regressionen vor.
    """
    required_cols = {
        "question_id",
        "rag",
        "temp",
        "abstain",
        "label_basic",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten: {sorted(missing)}")

    out = df.copy()

    # RAG nach 0/1 mappen
    out["rag_num"] = out["rag"].map({"off": 0, "on": 1})
    if out["rag_num"].isna().any():
        bad_vals = out.loc[out["rag_num"].isna(), "rag"].dropna().unique()
        raise ValueError(f"Unerwartete Werte in 'rag': {bad_vals}")

    out["rag_num"] = out["rag_num"].astype(int)

    # Temperatur sicher numerisch
    out["temp"] = pd.to_numeric(out["temp"], errors="raise")

    # Abstain als bool
    out["abstain_bool"] = parse_abstain(out["abstain"])

    # Modell 1: Antwortwahrscheinlichkeit
    # answered = 1, wenn keine Enthaltung
    out["answered"] = (~out["abstain_bool"]).astype(int)

    # Modell 2: Fehlerwahrscheinlichkeit unter beantworteten Fragen
    # incorrect -> 1, correct -> 0, abstain -> NaN
    out["error_given_answer"] = np.where(
        out["label_basic"].eq("incorrect"),
        1,
        np.where(out["label_basic"].eq("correct"), 0, np.nan),
    )

    # question_id als String für Cluster
    out["question_id"] = out["question_id"].astype(str)

    return out


def fit_logit_clustered(formula: str, data: pd.DataFrame, cluster_col: str):
    """
    Logistische Regression mit cluster-robusten Standardfehlern.
    """
    model = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Binomial(),
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": data[cluster_col]},
    )
    return model


def make_result_table(model, model_name: str, n_obs: int) -> pd.DataFrame:
    """
    Erstellt eine Ergebnis-Tabelle mit:
    b, SE, OR, 95%-KI, p, N
    """
    conf = model.conf_int()
    result = pd.DataFrame(
        {
            "Modell": model_name,
            "Prädiktor": model.params.index,
            "b": model.params.values,
            "SE_cluster": model.bse.values,
            "OR": np.exp(model.params.values),
            "KI_unter_95": np.exp(conf[0].values),
            "KI_ober_95": np.exp(conf[1].values),
            "p": model.pvalues.values,
            "N": n_obs,
        }
    )
    return result


def make_compact_table(full_table: pd.DataFrame) -> pd.DataFrame:
    """
    Kompaktere Tabelle für den Fließtext:
    Modell | Prädiktor | OR | 95%-KI | p | N
    """
    compact = full_table.copy()

    compact["OR"] = compact["OR"].round(3)
    compact["KI_unter_95"] = compact["KI_unter_95"].round(3)
    compact["KI_ober_95"] = compact["KI_ober_95"].round(3)
    compact["p"] = compact["p"].apply(
        lambda x: "< .001" if x < 0.001 else f"{x:.3f}"
    )

    compact["95%-KI"] = (
        "[" + compact["KI_unter_95"].astype(str) + ", " + compact["KI_ober_95"].astype(str) + "]"
    )

    compact = compact[["Modell", "Prädiktor", "OR", "95%-KI", "p", "N"]]
    return compact


def make_prediction_table(model, temps: list[float], model_name: str) -> pd.DataFrame:
    """
    Erstellt vorhergesagte Wahrscheinlichkeiten für alle Temperaturstufen
    getrennt nach ohne RAG / mit RAG.
    """
    grid = pd.DataFrame(
        [(t, r) for t in temps for r in [0, 1]],
        columns=["temp", "rag_num"],
    )

    pred = model.get_prediction(grid).summary_frame()

    out = grid.copy()
    out["Modell"] = model_name
    out["rag"] = out["rag_num"].map({0: "ohne RAG", 1: "mit RAG"})
    out["pred_prob"] = pred["mean"].values
    out["ci_lower_95"] = pred["mean_ci_lower"].values
    out["ci_upper_95"] = pred["mean_ci_upper"].values

    return out[["Modell", "temp", "rag", "pred_prob", "ci_lower_95", "ci_upper_95"]]


# =========================================================
# Hauptteil
# =========================================================
def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df = prepare_data(df)

    # -----------------------------
    # Modell 1: Antwortwahrscheinlichkeit
    # answered ~ C(temp) * rag_num
    # -----------------------------
    model_answer = fit_logit_clustered(
        formula="answered ~ C(temp) * rag_num",
        data=df,
        cluster_col="question_id",
    )

    # -----------------------------
    # Modell 2: Fehlerwahrscheinlichkeit unter Antworten
    # error_given_answer ~ C(temp) * rag_num
    # nur beantwortete Fragen
    # -----------------------------
    answered_df = df.loc[df["answered"] == 1].copy()

    model_error = fit_logit_clustered(
        formula="error_given_answer ~ C(temp) * rag_num",
        data=answered_df,
        cluster_col="question_id",
    )

    # -----------------------------
    # Tabellen erzeugen
    # -----------------------------
    full_answer = make_result_table(
        model_answer,
        model_name="Antwortwahrscheinlichkeit",
        n_obs=len(df),
    )
    full_error = make_result_table(
        model_error,
        model_name="Fehlerwahrscheinlichkeit unter Antworten",
        n_obs=len(answered_df),
    )

    full_table = pd.concat([full_answer, full_error], ignore_index=True)
    compact_table = make_compact_table(full_table)

    full_table.to_csv(OUTDIR / "regression_results_full.csv", index=False)
    compact_table.to_csv(OUTDIR / "regression_results_compact.csv", index=False)

    # -----------------------------
    # Vorhergesagte Wahrscheinlichkeiten
    # -----------------------------
    temps = sorted(df["temp"].dropna().unique().tolist())

    pred_answer = make_prediction_table(
        model_answer,
        temps=temps,
        model_name="Antwortwahrscheinlichkeit",
    )
    pred_error = make_prediction_table(
        model_error,
        temps=temps,
        model_name="Fehlerwahrscheinlichkeit unter Antworten",
    )

    pred_answer.to_csv(OUTDIR / "predicted_answer_probability.csv", index=False)
    pred_error.to_csv(OUTDIR / "predicted_error_probability.csv", index=False)

    # -----------------------------
    # Konsolenausgabe
    # -----------------------------
    print("\n=== MODELL 1: Antwortwahrscheinlichkeit ===")
    print(model_answer.summary())

    print("\n=== MODELL 2: Fehlerwahrscheinlichkeit unter Antworten ===")
    print(model_error.summary())

    print("\nDateien geschrieben nach:", OUTDIR.resolve())
    print("- regression_results_full.csv")
    print("- regression_results_compact.csv")
    print("- predicted_answer_probability.csv")
    print("- predicted_error_probability.csv")


if __name__ == "__main__":
    main()
