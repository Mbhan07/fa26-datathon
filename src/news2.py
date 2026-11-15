"""NEWS2 early warning score, implemented by hand from the Royal College of
Physicians NEWS2 specification (2017). Used as the published-rule baseline."""
import numpy as np
import pandas as pd

RISK_ORDER = ["Normal", "Low", "Medium", "High"]


def score_rr(rr):
    if rr <= 8: return 3
    if rr <= 11: return 1
    if rr <= 20: return 0
    if rr <= 24: return 2
    return 3


def score_spo2(spo2, scale, on_oxygen):
    if scale == 2:
        if spo2 <= 83: return 3
        if spo2 <= 85: return 2
        if spo2 <= 87: return 1
        if spo2 <= 92 or not on_oxygen: return 0  # 88-92, or >=93 on air
        if spo2 <= 94: return 1
        if spo2 <= 96: return 2
        return 3
    if spo2 <= 91: return 3
    if spo2 <= 93: return 2
    if spo2 <= 95: return 1
    return 0


def score_oxygen(on_oxygen):
    return 2 if on_oxygen else 0


def score_sbp(sbp):
    if sbp <= 90: return 3
    if sbp <= 100: return 2
    if sbp <= 110: return 1
    if sbp <= 219: return 0
    return 3


def score_hr(hr):
    if hr <= 40: return 3
    if hr <= 50: return 1
    if hr <= 90: return 0
    if hr <= 110: return 1
    if hr <= 130: return 2
    return 3


def score_consciousness(c):
    return 0 if c == "A" else 3


def score_temp(t):
    if t <= 35.0: return 3
    if t <= 36.0: return 1
    if t <= 38.0: return 0
    if t <= 39.0: return 1
    return 2


def news2_components(row):
    return {
        "rr": score_rr(row.Respiratory_Rate),
        "spo2": score_spo2(row.Oxygen_Saturation, row.O2_Scale, row.On_Oxygen),
        "oxygen": score_oxygen(row.On_Oxygen),
        "sbp": score_sbp(row.Systolic_BP),
        "hr": score_hr(row.Heart_Rate),
        "consciousness": score_consciousness(row.Consciousness),
        "temp": score_temp(row.Temperature),
    }


def risk_from_components(comp):
    total = sum(comp.values())
    if total >= 7: return "High"
    if total >= 5 or max(comp.values()) == 3: return "Medium"
    if total >= 1: return "Low"
    return "Normal"


def spo2_score_vec(spo2, scale, on_oxygen):
    """Vectorized score_spo2 (same rules) for fast Monte Carlo re-scoring."""
    spo2, scale, on_oxygen = np.asarray(spo2), np.asarray(scale), np.asarray(on_oxygen).astype(bool)
    s1 = np.select([spo2 <= 91, spo2 <= 93, spo2 <= 95], [3, 2, 1], 0)
    s2 = np.select([spo2 <= 83, spo2 <= 85, spo2 <= 87, (spo2 <= 92) | ~on_oxygen, spo2 <= 94, spo2 <= 96],
                   [3, 2, 1, 0, 1, 2], 3)
    return np.where(scale == 2, s2, s1)


def risk_vec(total, max_component):
    """Vectorized risk band: 0 Normal, 1 Low, 2 Medium, 3 High."""
    return np.select([total >= 7, (total >= 5) | (max_component == 3), total >= 1], [3, 2, 1], 0)


def news2_table(df):
    """Return a DataFrame with each component score, the total, and the risk band."""
    comps = pd.DataFrame([news2_components(r) for r in df.itertuples()], index=df.index)
    comps["total"] = comps.sum(axis=1)
    comps["news2_risk"] = [risk_from_components(c) for c in comps.drop(columns="total").to_dict("records")]
    return comps
