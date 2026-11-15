"""Train and evaluate risk models on a held-out test set.

Models: majority-class baseline, NEWS2 (published rule), logistic regression,
gradient boosting, and a 'no-SpO2' gradient boosting model used as the
second-check. Tuning uses cross-validation on the training split only.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from news2 import RISK_ORDER, news2_table

ACVPU = {"A": 0, "C": 1, "V": 2, "P": 3, "U": 4}
FEATURES = ["Respiratory_Rate", "Oxygen_Saturation", "O2_Scale", "Systolic_BP",
            "Heart_Rate", "Temperature", "Consciousness_ord", "On_Oxygen"]
NO_SPO2_FEATURES = [f for f in FEATURES if f != "Oxygen_Saturation"]
SEED = 42


def load(path="data/Health_Risk_Dataset.csv"):
    df = pd.read_csv(path)
    df["Consciousness_ord"] = df.Consciousness.map(ACVPU)
    df["y"] = df.Risk_Level.map({r: i for i, r in enumerate(RISK_ORDER)})
    return df


def metrics(y_true, y_pred):
    high = RISK_ORDER.index("High")
    return {
        "macro_F1": f1_score(y_true, y_pred, average="macro"),
        "High_recall": recall_score(y_true == high, y_pred == high),
        "QW_kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


def build_models():
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    logreg = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        {"logisticregression__C": [0.1, 1, 10]}, cv=cv, scoring="f1_macro")
    gbm_grid = {"max_depth": [2, 3, None], "learning_rate": [0.05, 0.1]}
    gbm = GridSearchCV(HistGradientBoostingClassifier(random_state=SEED), gbm_grid, cv=cv, scoring="f1_macro")
    gbm_no_spo2 = GridSearchCV(HistGradientBoostingClassifier(random_state=SEED), gbm_grid, cv=cv, scoring="f1_macro")
    return logreg, gbm, gbm_no_spo2


def run():
    df = load()
    train, test = train_test_split(df, test_size=0.2, stratify=df.y, random_state=SEED)
    logreg, gbm, gbm_no_spo2 = build_models()

    logreg.fit(train[FEATURES], train.y)
    gbm.fit(train[FEATURES], train.y)
    gbm_no_spo2.fit(train[NO_SPO2_FEATURES], train.y)

    preds = {
        "Majority class": DummyClassifier().fit(train[FEATURES], train.y).predict(test[FEATURES]),
        "NEWS2 (published rule)": news2_table(test).news2_risk.map(RISK_ORDER.index).values,
        "Logistic regression": logreg.predict(test[FEATURES]),
        "Gradient boosting": gbm.predict(test[FEATURES]),
        "Gradient boosting, no SpO2": gbm_no_spo2.predict(test[NO_SPO2_FEATURES]),
    }
    results = pd.DataFrame({k: metrics(test.y.values, v) for k, v in preds.items()}).T.round(3)
    return df, train, test, {"logreg": logreg, "gbm": gbm, "gbm_no_spo2": gbm_no_spo2}, preds, results


if __name__ == "__main__":
    _, _, test, models, preds, results = run()
    print(f"Held-out test set: {len(test)} patients\n")
    print(results.to_string())
    results.to_csv("results/model_results.csv")
    for name in ("logreg", "gbm", "gbm_no_spo2"):
        print(name, models[name].best_params_)
