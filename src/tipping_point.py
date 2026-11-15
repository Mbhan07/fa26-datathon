"""Step 7: Tipping-point analysis.

OpenOximetry measures oximeter error in HEALTHY VOLUNTEERS in a lab. Bias in
sick hospital patients may be larger. Question: how much extra overestimation
for darker skin (delta points, added to the measured dark-skin error) before
NEWS2 triage fails patients differently by skin tone?

Light-skin error stays as measured. For each delta we simulate the ward
N_DRAWS times under each profile and compare per-draw counts of
Medium -> Low downgrades (Medium is the level that triggers urgent review).
Tipping point = smallest delta where the 95% interval of (dark - light)
excludes zero.
Output: results/tipping_point.csv
"""
import json

import numpy as np
import pandas as pd

from linked_table import device_risk_matrix
from models import load
from news2 import RISK_ORDER, news2_table
from stress_test import GROUPS, SEED

N_DRAWS = 400
DELTAS = np.arange(0, 6.5, 0.5)
LIGHT, DARK = GROUPS[0], GROUPS[2]


def ward_counts(risk, true_risk, true_spo2):
    med_down = ((true_risk[:, None] == 2) & (risk < 2)).sum(axis=0)
    any_down = (risk < true_risk[:, None]).sum(axis=0)
    return med_down, any_down


def main():
    df = load()
    model = json.load(open("results/openox_error_model.json"))
    t = news2_table(df)
    comps = t.drop(columns=["total", "news2_risk"])
    true_risk = t.news2_risk.map(RISK_ORDER.index).values
    true_spo2 = df.Oxygen_Saturation.values
    rng = np.random.default_rng(SEED)

    light_risk, _ = device_risk_matrix(df, comps, LIGHT, model, rng, N_DRAWS)
    light_med, light_any = ward_counts(light_risk, true_risk, true_spo2)

    rows = []
    for d in DELTAS:
        dark_risk, _ = device_risk_matrix(df, comps, DARK, model, rng, N_DRAWS, offset=d)
        dark_med, dark_any = ward_counts(dark_risk, true_risk, true_spo2)
        gap = dark_med - light_med
        rows.append({
            "extra_bias_points": d,
            "medium_to_low_light": light_med.mean(),
            "medium_to_low_dark": dark_med.mean(),
            "gap_mean": gap.mean(), "gap_lo95": np.quantile(gap, .025), "gap_hi95": np.quantile(gap, .975),
            "any_downgrade_dark": dark_any.mean(),
            "pct_medium_hidden_dark": dark_med.mean() / (true_risk == 2).sum(),
        })
    res = pd.DataFrame(rows).round(2)
    res.to_csv("results/tipping_point.csv", index=False)
    print(f"True Medium-risk patients: {(true_risk == 2).sum()} | draws per scenario: {N_DRAWS}\n")
    print(res.to_string(index=False))
    tip = res[res.gap_lo95 > 0]
    if len(tip):
        d = tip.extra_bias_points.iloc[0]
        print(f"\nTipping point: +{d} extra points of dark-skin overestimation -> "
              f"dark-skin profile hides {tip.gap_mean.iloc[0]:.0f} more Medium-risk patients per 1,000 "
              f"(95% interval {tip.gap_lo95.iloc[0]:.0f} to {tip.gap_hi95.iloc[0]:.0f}).")
    else:
        print("\nNo tipping point within tested range.")


if __name__ == "__main__":
    main()
