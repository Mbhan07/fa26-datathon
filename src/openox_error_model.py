"""Fit a pulse-oximeter error model from the OpenOximetry Repository (PhysioNet).

The OpenOximetry data are under the PhysioNet Restricted Health Data License,
so they are NOT included in this repo. To reproduce:
  1. Get access at https://physionet.org/content/openox-repo/
  2. Put bloodgas.csv, pulseoximeter.csv, encounter.csv, patient.csv and
     spectrophotometer.csv in one folder.
  3. python src/openox_error_model.py /path/to/that/folder

Only aggregate statistics are printed and saved (openox_error_model.json).
Cells with fewer than MIN_PATIENTS participants are suppressed.

Device error = SpO2 (pulse oximeter) - SaO2 (arterial blood gas co-oximetry).
Positive error means the oximeter OVERESTIMATES oxygen, which hides hypoxemia.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_PATIENTS = 5
SAO2_BANDS = [70, 80, 85, 90, 101]  # band edges: 70-79, 80-84, 85-89, 90-100
SAO2_LABELS = ["70-79", "80-84", "85-89", "90-100"]
MONK_GROUPS = {  # Monk Skin Tone scale, lightest (A) to darkest (J)
    **{k: "Light (Monk A-C)" for k in "ABC"},
    **{k: "Medium (Monk D-G)" for k in "DEFG"},
    **{k: "Dark (Monk H-J)" for k in "HIJ"},
}
GROUP_ORDER = ["Light (Monk A-C)", "Medium (Monk D-G)", "Dark (Monk H-J)"]


def load(folder):
    f = Path(folder)
    bg = pd.read_csv(f / "bloodgas.csv", low_memory=False)
    po = pd.read_csv(f / "pulseoximeter.csv", low_memory=False)
    enc = pd.read_csv(f / "encounter.csv", low_memory=False)
    pat = pd.read_csv(f / "patient.csv", low_memory=False)
    return bg, po, enc, pat


def build_pairs(bg, po, enc, pat):
    # Arterial samples only (type 1 = ABG), with a co-oximetry saturation.
    bg = bg[bg["type"].astype(str).str.strip().isin(["1", "1.0", "ABG"])]
    bg = bg.dropna(subset=["so2"])[["encounter_id", "sample", "so2"]]
    po = po.dropna(subset=["saturation"])[["encounter_id", "sample_number", "device", "saturation"]]
    pairs = po.merge(bg, left_on=["encounter_id", "sample_number"],
                     right_on=["encounter_id", "sample"], how="inner")
    pairs = pairs.merge(enc[["encounter_id", "patient_id", "monk_dorsal", "monk_fingernail", "fitzpatrick"]],
                        on="encounter_id", how="left")
    pairs = pairs.merge(pat[["patient_id", "race"]], on="patient_id", how="left")

    pairs = pairs.rename(columns={"saturation": "spo2", "so2": "sao2"})
    pairs = pairs[(pairs.sao2 >= 70) & (pairs.sao2 <= 100) & (pairs.spo2 > 0) & (pairs.spo2 <= 100)]
    pairs["error"] = pairs.spo2 - pairs.sao2
    tone = pairs.monk_dorsal.fillna(pairs.monk_fingernail).astype(str).str.strip().str.upper()
    pairs["tone_group"] = tone.map(MONK_GROUPS)
    pairs["sao2_band"] = pd.cut(pairs.sao2, SAO2_BANDS, right=False, labels=SAO2_LABELS)
    return pairs


def suppress(df, n_col="patients"):
    df = df.copy()
    small = df[n_col].fillna(0) < MIN_PATIENTS
    df[n_col] = df[n_col].astype(object)
    for c in df.columns:
        if c != n_col:
            df.loc[small, c] = np.nan
    df.loc[small, n_col] = f"<{MIN_PATIENTS}"
    return df


def summarize(pairs):
    p = pairs.dropna(subset=["tone_group"])
    out = {}

    overall = p.groupby("tone_group").agg(
        patients=("patient_id", "nunique"), pairs=("error", "size"),
        mean_error=("error", "mean"), sd_error=("error", "std"),
    ).reindex(GROUP_ORDER)
    out["overall"] = suppress(overall.round(2))

    by_band = p.groupby(["tone_group", "sao2_band"], observed=True).agg(
        patients=("patient_id", "nunique"), pairs=("error", "size"),
        mean_error=("error", "mean"), sd_error=("error", "std"),
        p10=("error", lambda e: e.quantile(0.10)), p50=("error", "median"),
        p90=("error", lambda e: e.quantile(0.90)),
    )
    out["by_band"] = suppress(by_band.round(2))

    # Occult hypoxemia (Sjoding definition): SaO2 < 88 while SpO2 reads 92-96.
    window = p[(p.spo2 >= 92) & (p.spo2 <= 96)]
    occult = window.groupby("tone_group").agg(
        patients=("patient_id", "nunique"), readings_92_96=("error", "size"),
        occult_rate=("sao2", lambda s: (s < 88).mean()),
    ).reindex(GROUP_ORDER)
    out["occult"] = suppress(occult.round(3))

    # Empirical error quantiles (5th..95th) per tone group x SaO2 band, used to
    # simulate device readings in the Health Risk stress test.
    qs = np.round(np.arange(0.05, 1.0, 0.05), 2)
    model = {}
    for (g, b), grp in p.groupby(["tone_group", "sao2_band"], observed=True):
        if grp.patient_id.nunique() >= MIN_PATIENTS and len(grp) >= 30:
            model.setdefault(g, {})[str(b)] = [round(float(v), 2) for v in grp.error.quantile(qs)]
    out["quantile_model"] = {"quantiles": qs.tolist(), "error_quantiles": model}
    return out


def main(folder):
    pairs = build_pairs(*load(folder))
    print(f"Matched SpO2-SaO2 pairs: {len(pairs):,} | participants: {pairs.patient_id.nunique()} | "
          f"devices: {pairs.device.nunique()} | missing skin tone: {pairs.tone_group.isna().mean():.1%}\n")
    s = summarize(pairs)
    print("1) Mean device error (SpO2 - SaO2) by skin tone\n", s["overall"].to_string(), "\n")
    print("2) Error by skin tone and true oxygen level\n", s["by_band"].to_string(), "\n")
    print("3) Occult hypoxemia: SaO2 < 88% while oximeter reads 92-96%\n", s["occult"].to_string(), "\n")
    with open("openox_error_model.json", "w") as fh:
        json.dump(s["quantile_model"], fh, indent=1)
    print("Saved aggregate error model -> openox_error_model.json (safe to share: quantiles only)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
