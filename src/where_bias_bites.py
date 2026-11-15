"""Step 5: Where does oximeter bias actually change decisions?

(a) Single-reading view: for a given TRUE oxygen level, probability the
    oximeter shows >= 92% (a common 'reassuring' cutoff), by skin tone.
    Uses only the OpenOximetry error model.
(b) Triage view: among 'oxygen-dependent' Health Risk patients (risk level
    would drop if SpO2 looked normal), downgrade rate by skin-tone error profile.
Output: results/same_oxygen_different_alarm.csv, results/oxygen_dependent.csv
"""
import json

import numpy as np
import pandas as pd

from models import load
from news2 import RISK_ORDER, news2_table, risk_from_components
from stress_test import GROUPS, SEED, band_of, device_frame, sample_errors

N_SIMS = 300


def p_shows_at_least(true_spo2, shown, group, model):
    """P(device >= shown | true SpO2). Quantiles stored 5th-95th, so values
    below 5% or above 95% cannot be resolved and are reported as bounds."""
    qs = np.array(model["quantiles"])
    q = np.array(model["error_quantiles"][group][band_of(true_spo2)])
    need = shown - true_spo2 - 0.5  # device reports integers
    return 1 - np.interp(need, q, qs, left=0.0, right=1.0)


def single_reading(model):
    rows = []
    for t in range(84, 92):
        light = p_shows_at_least(t, 92, GROUPS[0], model)
        dark = p_shows_at_least(t, 92, GROUPS[2], model)
        fmt = lambda p: "<5%" if p < 0.05 else f"{p:.1%}"
        rows.append({"true_SpO2": t, "light_shows_>=92": fmt(light), "dark_shows_>=92": fmt(dark),
                     "ratio": f"{dark / light:.1f}x" if light >= 0.05 else "n/a (light <5%)"})
    return pd.DataFrame(rows)


def oxygen_dependent(model):
    df = load()
    t = news2_table(df)
    true = t.news2_risk.map(RISK_ORDER.index).values
    comps = t.drop(columns=["total", "news2_risk"]).assign(spo2=0).to_dict("records")
    dep = np.array([RISK_ORDER.index(risk_from_components(c)) for c in comps]) < true
    rng = np.random.default_rng(SEED)
    rows = []
    for g in GROUPS:
        for _ in range(N_SIMS):
            d = device_frame(df, sample_errors(df.Oxygen_Saturation.values, g, model, rng))
            p = news2_table(d).news2_risk.map(RISK_ORDER.index).values
            rows.append({"group": g, "dep_downgrade_rate": (dep & (p < true)).sum() / dep.sum()})
    r = pd.DataFrame(rows).groupby("group", sort=False).dep_downgrade_rate
    out = pd.DataFrame({"mean": r.mean(), "lo95": r.quantile(.025), "hi95": r.quantile(.975)}).round(3)
    return int(dep.sum()), pd.Series(np.array(RISK_ORDER)[true[dep]]).value_counts().to_dict(), out


def main():
    model = json.load(open("results/openox_error_model.json"))
    a = single_reading(model)
    a.to_csv("results/same_oxygen_different_alarm.csv", index=False)
    print("(a) Same oxygen, different alarm: P(oximeter shows >= 92%)\n", a.to_string(index=False), "\n")
    n, by_level, b = oxygen_dependent(model)
    b.to_csv("results/oxygen_dependent.csv")
    print(f"(b) Oxygen-dependent patients: {n} {by_level}\nNEWS2 downgrade rate by error profile:\n{b.to_string()}")


if __name__ == "__main__":
    main()
