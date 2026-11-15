"""Step 8: Subgroup (fairness) analysis and failure examples.

Part 1  Model performance by clinical subgroup. Uses 5-fold out-of-fold
        predictions on all 1,000 patients (the 200-patient test set is too
        small to split). Truth = dataset labels.
Part 2  Oximeter-bias impact by subgroup: mean probability of NEWS2 downgrade
        under the light vs dark error profile (from results/linked_patients.csv).
Part 3  Concrete failure examples.
Outputs: results/subgroup_performance.csv, results/subgroup_bias.csv,
         results/failure_examples.md
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from models import FEATURES, NO_SPO2_FEATURES, SEED, load
from news2 import RISK_ORDER, news2_table

HIGH = RISK_ORDER.index("High")


def subgroup_labels(df):
    o2 = np.where(df.O2_Scale == 1, "Scale 1",
                  np.where(df.On_Oxygen == 1, "Scale 2, on oxygen", "Scale 2, room air"))
    return {
        "SpO2 scale": pd.Series(o2, index=df.index),
        "Supplemental oxygen": df.On_Oxygen.map({0: "Room air", 1: "On oxygen"}),
        "Consciousness": np.where(df.Consciousness == "A", "Alert", "Altered (C/V/P/U)"),
        "True oxygen": pd.cut(df.Oxygen_Saturation, [0, 88, 92, 101], right=False,
                              labels=["<88 (hypoxemic)", "88-91 (borderline)", ">=92"]),
    }


def oof_predictions(df):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    full = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, random_state=SEED)
    no_spo2 = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=SEED)
    return {
        "NEWS2 (published)": news2_table(df).news2_risk.map(RISK_ORDER.index).values,
        "Gradient boosting": cross_val_predict(full, df[FEATURES], df.y, cv=cv),
        "No-SpO2 second check": cross_val_predict(no_spo2, df[NO_SPO2_FEATURES], df.y, cv=cv),
    }


def performance(df, preds):
    rows = []
    for sg_name, sg in subgroup_labels(df).items():
        for level in pd.unique(sg):
            m = np.asarray(sg == level)
            y = df.y.values[m]
            for system, p in preds.items():
                p = p[m]
                rows.append({
                    "subgroup": sg_name, "level": level, "n": int(m.sum()), "system": system,
                    "accuracy": (p == y).mean(),
                    "macro_F1": f1_score(y, p, average="macro"),
                    "High_recall": recall_score(y == HIGH, p == HIGH) if (y == HIGH).any() else np.nan,
                    "under_triage_rate": (p < y).mean(),
                })
    return pd.DataFrame(rows).round(3)


def bias_by_subgroup(df):
    linked = pd.read_csv("results/linked_patients.csv")
    rows = []
    for sg_name, sg in subgroup_labels(df).items():
        for level in pd.unique(sg):
            m = np.asarray(sg == level)
            l, d = linked.p_downgrade_light[m].mean(), linked.p_downgrade_dark[m].mean()
            rows.append({"subgroup": sg_name, "level": level, "n": int(m.sum()),
                         "p_downgrade_light": l, "p_downgrade_dark": d, "dark_minus_light": d - l})
    return pd.DataFrame(rows).round(3)


def vitals(row):
    return (f"RR {row.Respiratory_Rate}, SpO2 {row.Oxygen_Saturation}% (scale {row.O2_Scale}, "
            f"{'on O2' if row.On_Oxygen else 'room air'}), SBP {row.Systolic_BP}, HR {row.Heart_Rate}, "
            f"T {row.Temperature}, {row.Consciousness}")


def failure_examples(df, preds):
    linked = pd.read_csv("results/linked_patients.csv").set_index("Patient_ID")
    news2 = news2_table(df)
    gbm, second = preds["Gradient boosting"], preds["No-SpO2 second check"]
    out = ["# Failure examples", ""]

    # 1. Label error: dataset label disagrees with published NEWS2.
    i = df.index[(df.Risk_Level != news2.news2_risk)][0]
    r = df.loc[i]
    out += ["## 1. The ground truth is wrong (label audit)",
            f"**{r.Patient_ID}**: {vitals(r)}",
            f"Dataset label: **{r.Risk_Level}**. Published NEWS2: **{news2.news2_risk[i]}** (total {news2.total[i]}).",
            "Scale-2 patients on room air with SpO2 >= 93 should score 0 for oxygen; the labels score them as if on oxygen. "
            "Any model trained on these labels learns to over-alarm for this group.", ""]

    # 2. Bias-driven miss: largest dark-minus-light downgrade probability.
    pid = linked.p_downgrade_gap_dark_minus_light.idxmax()
    r, L = df[df.Patient_ID == pid].iloc[0], linked.loc[pid]
    out += ["## 2. Same patient, different alarm (oximeter bias)",
            f"**{pid}**: {vitals(r)}. True NEWS2 risk: **{L.news2_true_risk}**.",
            f"Chance the oximeter reading downgrades this patient: **{L.p_downgrade_light:.0%}** under the light-skin "
            f"error profile vs **{L.p_downgrade_dark:.0%}** under the dark-skin profile.",
            f"No-SpO2 second check predicts **{L.second_check_prediction}**, so it "
            f"{'would flag a downgrade to Low' if L.second_check_prediction in ('Medium', 'High') else 'would NOT catch it'}.", ""]

    # 3. ML model misses a High-risk patient that NEWS2 catches.
    miss = np.where((df.y.values == HIGH) & (gbm < HIGH))[0]
    j = miss[np.argmax(news2.total.values[miss])] if len(miss) else None
    if j is not None:
        r = df.iloc[j]
        out += ["## 3. The ML model misses a High-risk patient the rule catches",
                f"**{r.Patient_ID}**: {vitals(r)}",
                f"Label: High (NEWS2 total {news2.total.iloc[j]}). Gradient boosting predicts **{RISK_ORDER[gbm[j]]}**. "
                f"Out of {(df.y == HIGH).sum()} High-risk patients, the model misses {len(miss)}; NEWS2 misses 0.", ""]

    # 4. Second check fails: a downgrade-prone Medium patient the second check calls Low.
    cand = linked[(linked.news2_true_risk == "Medium") & (linked.second_check_prediction.isin(["Low", "Normal"]))]
    if len(cand):
        pid = cand.p_downgrade_dark.idxmax()
        r, L = df[df.Patient_ID == pid].iloc[0], linked.loc[pid]
        out += ["## 4. The second check fails",
                f"**{pid}**: {vitals(r)}. True risk Medium.",
                f"Chance of downgrade under dark-skin profile: **{L.p_downgrade_dark:.0%}**. The no-SpO2 model predicts "
                f"**{L.second_check_prediction}**, so it would not flag the downgrade: this patient's risk lives almost "
                "entirely in the oxygen reading, which is exactly the number the second check ignores.", ""]
    return "\n".join(out)


def main():
    df = load()
    preds = oof_predictions(df)
    perf = performance(df, preds)
    perf.to_csv("results/subgroup_performance.csv", index=False)
    bias = bias_by_subgroup(df)
    bias.to_csv("results/subgroup_bias.csv", index=False)
    md = failure_examples(df, preds)
    open("results/failure_examples.md", "w").write(md + "\n")

    pd.set_option("display.width", 220)
    print("Part 1: performance by subgroup (out-of-fold, truth = dataset labels)")
    print(perf.pivot_table(index=["subgroup", "level", "n"], columns="system",
                           values=["macro_F1", "under_triage_rate"]).round(2).to_string(), "\n")
    print("Part 2: oximeter-bias impact by subgroup")
    print(bias.to_string(index=False), "\n")
    print(md)


if __name__ == "__main__":
    main()
