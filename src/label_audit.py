"""Step 1: Audit the Health Risk labels against the published NEWS2 rule.

Finds that labels match NEWS2 for 98.7% of patients, and that every mismatch is
a scale-2 (chronic hypercapnic respiratory failure) patient on room air with
SpO2 >= 93. Those patients are over-scored because the label rule applies the
'on oxygen' scale-2 thresholds even when the patient is breathing room air.
Output: results/label_audit.txt, results/label_mismatches.csv
"""
import pandas as pd

import news2
from news2 import RISK_ORDER, news2_table


def main():
    df = pd.read_csv("data/Health_Risk_Dataset.csv")
    t = news2_table(df)
    agree = (df.Risk_Level == t.news2_risk).mean()
    mism = df[df.Risk_Level != t.news2_risk]
    mism_full = pd.concat([mism, t.loc[mism.index]], axis=1)
    mism_full.to_csv("results/label_mismatches.csv", index=False)

    # Hypothesis: labels always use the on-oxygen scale-2 thresholds.
    original = news2.score_spo2
    news2.score_spo2 = lambda s, scale, on_o2: original(s, scale, True) if scale == 2 else original(s, scale, on_o2)
    agree_variant = (df.Risk_Level == news2_table(df).news2_risk).mean()
    news2.score_spo2 = original

    s2_air = df[(df.O2_Scale == 2) & (df.On_Oxygen == 0)]
    crosstab = pd.crosstab(df.Risk_Level, t.news2_risk).reindex(index=RISK_ORDER, columns=RISK_ORDER).fillna(0).astype(int)
    lines = [
        f"Agreement, published NEWS2 vs dataset labels: {agree:.1%}",
        f"Mismatches: {len(mism)} (all scale 2 on room air: {bool(((mism.O2_Scale == 2) & (mism.On_Oxygen == 0)).all())})",
        f"Agreement if scale-2 always uses on-oxygen thresholds: {agree_variant:.1%}",
        f"Scale-2 patients on room air: {len(s2_air)}; with SpO2 >= 93 (over-scored): {(s2_air.Oxygen_Saturation >= 93).sum()}",
        "", "Rows = dataset label, columns = published NEWS2", crosstab.to_string(),
    ]
    open("results/label_audit.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
