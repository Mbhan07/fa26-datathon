"""Step 6: Build the per-patient LINKED table (Health Risk x OpenOximetry).

The two datasets share no patients or IDs. They are joined on TRUE OXYGEN LEVEL:
each Health Risk patient's SpO2 (treated as true oxygen, since the dataset has
no blood gas) is matched to the OpenOximetry error distribution for the same
true-oxygen band and each skin-tone group. Simulated device readings are then
re-scored with NEWS2.

One row per Health Risk patient, with columns from both datasets:
  - Health Risk: vitals, true NEWS2 risk
  - OpenOximetry: median and 10th-90th percentile device error for that band,
    per skin-tone group
  - Linked result: probability the device reading downgrades this patient's
    NEWS2 risk level, per skin-tone group (Monte Carlo, N_DRAWS per patient)
  - Second check: out-of-fold no-SpO2 model prediction
Output: results/linked_patients.csv
"""
import json

import numpy as np
import pandas as pd

from models import load
from news2 import RISK_ORDER, news2_table, risk_vec, spo2_score_vec
from stress_test import GROUPS, SEED, band_of, oof_models, sample_errors

N_DRAWS = 1000
SHORT = {"Light (Monk A-C)": "light", "Medium (Monk D-G)": "medium", "Dark (Monk H-J)": "dark"}


def device_risk_matrix(df, comps, group, model, rng, n_draws=N_DRAWS, offset=0.0):
    """Simulate n_draws device readings per patient (plus an optional extra
    overestimation offset, used by the tipping-point analysis) and re-score
    NEWS2. Returns (risk matrix, device SpO2 matrix), each patients x draws."""
    other = comps.drop(columns=["spo2"])
    other_total, other_max = other.sum(axis=1).values, other.max(axis=1).values
    n = len(df)
    true_spo2 = np.repeat(df.Oxygen_Saturation.values, n_draws)
    errors = sample_errors(true_spo2, group, model, rng).reshape(n, n_draws) + offset
    device = np.clip(np.round(df.Oxygen_Saturation.values[:, None] + errors), 50, 100)
    s = spo2_score_vec(device, df.O2_Scale.values[:, None], df.On_Oxygen.values[:, None])
    risk = risk_vec(other_total[:, None] + s, np.maximum(other_max[:, None], s))
    return risk, device


def p_downgrade(df, comps, true_risk, group, model, rng):
    """Probability, per patient, that the simulated device reading lowers the NEWS2 risk level."""
    risk, device = device_risk_matrix(df, comps, group, model, rng)
    return (risk < true_risk[:, None]).mean(axis=1), (device >= 92).mean(axis=1)


def main():
    df = load()
    model = json.load(open("results/openox_error_model.json"))
    t = news2_table(df)
    comps = t.drop(columns=["total", "news2_risk"])
    true_risk = t.news2_risk.map(RISK_ORDER.index).values
    _, second = oof_models(df)
    rng = np.random.default_rng(SEED)

    out = df[["Patient_ID", "Respiratory_Rate", "Oxygen_Saturation", "O2_Scale", "On_Oxygen",
              "Systolic_BP", "Heart_Rate", "Temperature", "Consciousness"]].copy()
    out["true_oxygen_band"] = df.Oxygen_Saturation.map(band_of)
    out["news2_total"] = t.total
    out["news2_true_risk"] = t.news2_risk
    qs = model["quantiles"]
    i10, i50, i90 = qs.index(0.1), qs.index(0.5), qs.index(0.9)
    for g in GROUPS:
        k = SHORT[g]
        q = out.true_oxygen_band.map(lambda b: model["error_quantiles"][g][b])
        out[f"err_median_{k}"] = q.map(lambda v: v[i50])
        out[f"err_p10_p90_{k}"] = q.map(lambda v: f"{v[i10]:+.1f} to {v[i90]:+.1f}")
        out[f"p_downgrade_{k}"], out[f"p_shows_92plus_{k}"] = p_downgrade(df, comps, true_risk, g, model, rng)
    out["p_downgrade_gap_dark_minus_light"] = out.p_downgrade_dark - out.p_downgrade_light
    out["second_check_prediction"] = np.array(RISK_ORDER)[second]
    out.round(3).to_csv("results/linked_patients.csv", index=False)

    print(f"Linked table: {len(out)} patients x {out.shape[1]} columns -> results/linked_patients.csv")
    print("Mean P(downgrade) by error profile:",
          {k: round(out[f'p_downgrade_{k}'].mean(), 3) for k in SHORT.values()})
    at_risk = out.p_downgrade_light.gt(0) | out.p_downgrade_dark.gt(0)
    print(f"Patients with any chance of downgrade: {at_risk.sum()}")
    hyp = out[out.Oxygen_Saturation < 88]
    print(f"Truly hypoxemic (SpO2<88, n={len(hyp)}): mean P(device shows >=92) "
          f"light {hyp.p_shows_92plus_light.mean():.3f} vs dark {hyp.p_shows_92plus_dark.mean():.3f}")
    cols = ["Patient_ID", "Oxygen_Saturation", "news2_true_risk", "err_median_light", "err_median_dark",
            "p_downgrade_light", "p_downgrade_dark", "second_check_prediction"]
    print("\nExample rows (largest dark-vs-light gap):")
    print(out.sort_values("p_downgrade_gap_dark_minus_light", ascending=False)[cols].head(5).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
