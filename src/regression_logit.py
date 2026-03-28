# src/regression_logit.py
import argparse
import csv
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to eval_basic.csv")
    ap.add_argument("--out", dest="out", required=True, help="Path to regression_summary.txt")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    df = pd.read_csv(inp)

    # booleans from strings
    df["abstain"] = df["abstain"].astype(str).str.lower().eq("true")
    df["answered"] = (~df["abstain"]).astype(int)

    # incorrect only defined when answered==1
    df["incorrect"] = ((df["label_basic"] == "incorrect") & (df["answered"] == 1)).astype(int)

    # rag as categorical with baseline "off"
    df["rag"] = pd.Categorical(df["rag"], categories=["off", "on"])

    # Model A: answered ~ temp * rag
    m_answered = smf.logit("answered ~ temp * rag", data=df).fit(disp=False)

    # Model B: incorrect ~ temp * rag, subset answered==1
    df_ans = df[df["answered"] == 1].copy()
    m_incorrect = smf.logit("incorrect ~ temp * rag", data=df_ans).fit(disp=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("MODEL A: answered ~ temp * rag\n")
        f.write(m_answered.summary().as_text())
        f.write("\n\n")

        f.write("MODEL B: incorrect (given answered) ~ temp * rag\n")
        f.write(m_incorrect.summary().as_text())
        f.write("\n")

    print("WROTE:", out)


if __name__ == "__main__":
    main()
