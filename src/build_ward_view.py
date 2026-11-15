"""Step 9: Generate the interactive 50-bed ward view (docs/index.html).

Each bed is a real Health Risk patient (seeded sample, always including the
failure-example patients). Every bed gets ONE fixed random percentile draw u,
reused across all scenarios (common random numbers): the displayed SpO2 is
  true SpO2 + error_quantile(u | skin-tone profile, true-oxygen band) + extra bias.
Because u is fixed, raising the bias only ever pushes readings up, as a real
systematic bias would. Displayed risk is NEWS2 re-scored on the displayed SpO2.
P(downgrade) per patient comes from the Monte Carlo in results/linked_patients.csv
(extra bias 0) and is recomputed here for each extra-bias level.
Output: docs/index.html (self-contained; host with GitHub Pages from /docs).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from linked_table import device_risk_matrix
from models import load
from news2 import RISK_ORDER, news2_table, risk_vec, spo2_score_vec
from stress_test import GROUPS, SEED, band_of, oof_models

N_BEDS = 50
ALWAYS = ["P0600", "P0440"]
OFFSETS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
PROFILES = {"light": GROUPS[0], "dark": GROUPS[2]}
N_DRAWS = 1000


def main():
    df = load()
    model = json.load(open("results/openox_error_model.json"))
    qs = np.array(model["quantiles"])
    t = news2_table(df)
    comps = t.drop(columns=["total", "news2_risk"])
    true_risk = t.news2_risk.map(RISK_ORDER.index).values
    _, second = oof_models(df)

    rng = np.random.default_rng(SEED)
    pool = df.index[~df.Patient_ID.isin(ALWAYS)]
    idx = list(df.index[df.Patient_ID.isin(ALWAYS)]) + list(rng.choice(pool, N_BEDS - len(ALWAYS), replace=False))
    rng.shuffle(idx)
    ward = df.loc[idx].reset_index(drop=True)
    wc = comps.loc[idx].reset_index(drop=True)
    wtrue = true_risk[idx]
    u = rng.uniform(qs[0], qs[-1], N_BEDS)

    other = wc.drop(columns=["spo2"])
    o_tot, o_max = other.sum(axis=1).values, other.max(axis=1).values
    scen = {}
    for name, group in PROFILES.items():
        for off in OFFSETS:
            extra = off if name == "dark" else 0.0  # extra bias is a dark-skin scenario; light stays as measured
            err = np.array([np.interp(ui, qs, model["error_quantiles"][group][band_of(s)])
                            for ui, s in zip(u, ward.Oxygen_Saturation)]) + extra
            shown = np.clip(np.round(ward.Oxygen_Saturation.values + err), 50, 100)
            s = spo2_score_vec(shown, ward.O2_Scale.values, ward.On_Oxygen.values)
            risk = risk_vec(o_tot + s, np.maximum(o_max, s))
            prisk, _ = device_risk_matrix(ward, wc, group, model, np.random.default_rng(SEED), N_DRAWS, offset=extra)
            scen[f"{name}|{off:.1f}"] = {
                "shown": shown.astype(int).tolist(),
                "risk": risk.astype(int).tolist(),
                "p_down": np.round((prisk < wtrue[:, None]).mean(axis=1), 3).tolist(),
                "exp_hidden": round(float((prisk < wtrue[:, None]).mean(axis=1).sum()), 1),
                "exp_med": round(float(((wtrue[:, None] == 2) & (prisk < 2)).mean(axis=1).sum()), 1),
            }

    beds = []
    for i, r in ward.iterrows():
        beds.append({
            "id": r.Patient_ID, "rr": int(r.Respiratory_Rate), "spo2": int(r.Oxygen_Saturation),
            "scale": int(r.O2_Scale), "o2": int(r.On_Oxygen), "sbp": int(r.Systolic_BP),
            "hr": int(r.Heart_Rate), "temp": float(r.Temperature), "avpu": r.Consciousness,
            "true": int(wtrue[i]), "second": int(second[idx[i]]),
        })
    payload = {"beds": beds, "scenarios": scen, "offsets": OFFSETS, "levels": RISK_ORDER}
    html = Path("src/ward_template.html").read_text().replace("__DATA__", json.dumps(payload))
    Path("docs").mkdir(exist_ok=True)
    Path("docs/index.html").write_text(html)

    for name in PROFILES:
        for off in (0.0, 2.0):
            r = np.array(scen[f"{name}|{off:.1f}"]["risk"])
            print(f"{name:5s} +{off}: beds downgraded {int((r < wtrue).sum())}, "
                  f"Medium hidden {int(((wtrue == 2) & (r < 2)).sum())}")
    print("Ward true risk mix:", pd.Series(np.array(RISK_ORDER)[wtrue]).value_counts().to_dict())
    print("Wrote docs/index.html")


if __name__ == "__main__":
    main()
