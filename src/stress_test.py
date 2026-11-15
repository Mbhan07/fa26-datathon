"""Oximeter-bias stress test on the Health Risk dataset.

True risk  = NEWS2 (published rule) on the recorded SpO2, treated as true SaO2.
Device     = true SpO2 + error sampled from the OpenOximetry error model for a
             skin-tone group and true-oxygen band (results/openox_error_model.json).
We re-score every patient on device readings and count truly high-risk patients
who are downgraded, under each skin-tone error profile.

Second check = gradient boosting model trained WITHOUT SpO2. It flags a patient
when it predicts a higher risk level than the primary system shows.

All ML predictions are out-of-fold (5-fold), so no patient is scored by a model
that saw them in training.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from models import FEATURES, NO_SPO2_FEATURES, SEED, load
from news2 import RISK_ORDER, news2_table

N_SIMS = 200
BANDS = [(70, 80, "70-79"), (80, 85, "80-84"), (85, 90, "85-89"), (90, 101, "90-100")]
GROUPS = ["Light (Monk A-C)", "Medium (Monk D-G)", "Dark (Monk H-J)"]
HIGH = RISK_ORDER.index("High")


def band_of(spo2):
    for lo, hi, name in BANDS:
        if lo <= spo2 < hi:
            return name
    return "70-79"


def sample_errors(spo2_true, group, model, rng):
    """Inverse-CDF sampling from the stored error quantiles (tails clamped)."""
    qs = np.array(model["quantiles"])
    out = np.empty(len(spo2_true))
    u = rng.uniform(qs[0], qs[-1], len(spo2_true))
    for i, (s, ui) in enumerate(zip(spo2_true, u)):
        out[i] = np.interp(ui, qs, model["error_quantiles"][group][band_of(s)])
    return out


def device_frame(df, errors):
    d = df.copy()
    d["Oxygen_Saturation"] = np.clip(np.round(df.Oxygen_Saturation + errors), 50, 100)
    return d


def oof_models(df):
    """Fit 5 fold models (with SpO2) and out-of-fold no-SpO2 predictions."""
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds, second = [], np.empty(len(df), dtype=int)
    for tr, te in skf.split(df, df.y):
        full = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, random_state=SEED)
        full.fit(df.iloc[tr][FEATURES], df.y.iloc[tr])
        no_spo2 = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=SEED)
        no_spo2.fit(df.iloc[tr][NO_SPO2_FEATURES], df.y.iloc[tr])
        second[te] = no_spo2.predict(df.iloc[te][NO_SPO2_FEATURES])
        folds.append((te, full))
    return folds, second


def run(n_sims=N_SIMS):
    df = load()
    model = json.load(open("results/openox_error_model.json"))
    true_risk = news2_table(df).news2_risk.map(RISK_ORDER.index).values
    is_high = true_risk == HIGH
    folds, second = oof_models(df)
    rng = np.random.default_rng(SEED)

    true_spo2 = df.Oxygen_Saturation.values
    hypoxemic = true_spo2 < 88  # Sjoding threshold for true hypoxemia
    rows = []
    for group in GROUPS:
        for sim in range(n_sims):
            dev = device_frame(df, sample_errors(df.Oxygen_Saturation.values, group, model, rng))
            news2_dev = news2_table(dev).news2_risk.map(RISK_ORDER.index).values
            ml_dev = np.empty(len(df), dtype=int)
            for te, full in folds:
                ml_dev[te] = full.predict(dev.iloc[te][FEATURES])
            for system, pred in (("NEWS2", news2_dev), ("ML model", ml_dev)):
                down_high = is_high & (pred < HIGH)
                down_any = pred < true_risk
                flag = second > pred
                masked = hypoxemic & (dev.Oxygen_Saturation.values >= 92)
                med_down = (true_risk == 2) & (pred < 2)
                rows.append({
                    "group": group, "sim": sim, "system": system,
                    "masked_hypoxemia": masked.sum(),
                    "masked_recovered": (masked & flag).sum(),
                    "medium_downgraded": med_down.sum(),
                    "medium_down_recovered": (med_down & flag).sum(),
                    "high_hidden": down_high.sum(),
                    "any_downgraded": down_any.sum(),
                    "high_hidden_recovered": (down_high & (second >= HIGH)).sum(),
                    "any_recovered": (down_any & flag).sum(),
                    "flag_rate_all": flag.mean(),
                })
    return pd.DataFrame(rows), int(is_high.sum())


def summarize(res):
    def ci(s):
        return f"{s.mean():.1f} [{s.quantile(.025):.0f}-{s.quantile(.975):.0f}]"
    g = res.groupby(["system", "group"], sort=False)
    out = pd.DataFrame({
        "Masked hypoxemia": g.masked_hypoxemia.apply(ci),
        "...recovered by 2nd check": g.masked_recovered.apply(ci),
        "Medium->Low downgrades": g.medium_downgraded.apply(ci),
        "...recovered": g.medium_down_recovered.apply(ci),
        "High hidden": g.high_hidden.apply(ci),
        "2nd-check flag rate": g.flag_rate_all.mean().map("{:.1%}".format),
    })
    return out


def gap_ratio(res, system="NEWS2", metric="masked_hypoxemia"):
    """Dark/Light ratio with a 95% interval from paired simulation draws."""
    r = res[res.system == system].pivot(index="sim", columns="group", values=metric)
    ratio = r["Dark (Monk H-J)"] / r["Light (Monk A-C)"].replace(0, np.nan)
    return ratio.mean(), ratio.quantile(.025), ratio.quantile(.975)


if __name__ == "__main__":
    res, n_high = run()
    res.to_csv("results/stress_test_sims.csv", index=False)
    print(f"Patients: 1000 | truly High-risk (NEWS2 on true SpO2): {n_high} | sims per group: {N_SIMS}")
    print("Values: mean [95% simulation interval]\n")
    print(f"Truly hypoxemic patients (SpO2 < 88): {(load().Oxygen_Saturation < 88).sum()}")
    pd.set_option("display.width", 250)
    print(summarize(res).to_string())
    summarize(res).to_csv("results/stress_test_summary.csv")
    m, lo, hi = gap_ratio(res)
    print(f"\nMasked hypoxemia, Dark vs Light error profile: {m:.2f}x [{lo:.2f}-{hi:.2f}]")
