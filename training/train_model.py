# training/train_model.py — long‑term classifier & regressor
"""
Train both:
1) XGBoost classifier → will index be up ≥5 % in 30 days?
2) XGBoost regressor  → forecast close price 30 days ahead

Features: enriched tech indicators + sector sentiment.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# ── Imports from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.utils.indicators import (
    add_features,
    add_index_features,
    build_feature_columns,
)

# ── File paths
DATA_DIR = Path("training/data")
MAIN_CSV = DATA_DIR / "NIFTY500.csv"
if not MAIN_CSV.exists():
    raise FileNotFoundError(f"{MAIN_CSV} not found")

INDEX_PATHS = {
    "NIFTY50": DATA_DIR / "NIFTY50.csv",
    "NIFTYMIDCAP150": DATA_DIR / "NIFTYMIDCAP150.csv",
    "NIFTYSMALLCAP250": DATA_DIR / "NIFTYSMALLCAP250.csv",
    "NIFTYBANK": DATA_DIR / "NIFTYBANK.csv",
    "NIFTYIT": DATA_DIR / "NIFTYIT.csv",
    "NIFTYFMCG": DATA_DIR / "NIFTYFMCG.csv",
}

# ── Load main CSV
print(f"✅ Loading {MAIN_CSV}")
df = pd.read_csv(MAIN_CSV)

# Clean price columns
for c in ["Open", "High", "Low", "Close"]:
    df[c] = (
        df[c]
        .astype(str)
        .replace("-", np.nan)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Basic cleanup
if "Index Name" in df.columns:
    df.drop(columns="Index Name", inplace=True)

df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y")
df.sort_values("Date", inplace=True)

if "Volume" not in df.columns:
    df["Volume"] = 0.0

# ── Feature engineering
df = add_features(df)
df = add_index_features(df, INDEX_PATHS)

# Calendar features
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month

# ── Targets
HORIZON = 30            # days ahead
THRESH  = 0.05          # +5 %
df["TargetCls"] = ((df["Close"].shift(-HORIZON) - df["Close"]) / df["Close"] > THRESH).astype(int)
df["FutureReturn_30d"] = (df["Close"].shift(-30) - df["Close"]) / df["Close"]

# Final NA drop
df.dropna(inplace=True)

# ── Feature matrix
FEATURES = build_feature_columns(list(INDEX_PATHS.keys())) + ["DayOfWeek", "Month"]
X_raw = df[FEATURES]

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), columns=FEATURES, index=df.index)

y_cls = df["TargetCls"]
y_reg = df["FutureReturn_30d"]

# ── Train‑test split (time‑based)
X_train, X_test, ycls_train, ycls_test, yreg_train, yreg_test = train_test_split(
    X, y_cls, y_reg, test_size=0.2, shuffle=False
)

# ──────────────────────────────────
# 1)  Long‑term CLASSIFIER
# ──────────────────────────────────
scale_pos = (ycls_train == 0).sum() / (ycls_train == 1).sum()

def focal_loss(alpha=0.25, gamma=2.0):
    def fl_obj(y_pred, dtrain):
        a, g = alpha, gamma
        y_true = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        grad = a * (y_true - p) * (g * (1 - p)**(g - 1) * p * np.log(p + 1e-8) + (1 - p)**g)
        hess = a * ((g * (1 - p)**(g - 1) * p * (1 - p)) -
                    (g * (g - 1) * (1 - p)**(g - 2) * p**2 * np.log(p + 1e-8)))
        return -grad, -hess
    return fl_obj

xgb_cls_base = xgb.XGBClassifier(
    objective=focal_loss(),
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos,
)

param_dist = {
    "n_estimators": [300, 500, 700],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [1.0, 10.0, 100.0],
}

tscv = TimeSeriesSplit(n_splits=5)
cls_search = RandomizedSearchCV(
    xgb_cls_base,
    param_dist,
    n_iter=25,
    cv=tscv,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

sample_wts = compute_sample_weight(class_weight="balanced", y=ycls_train)
print("🔍  Tuning classifier …")
cls_search.fit(X_train, ycls_train, sample_weight=sample_wts)
cls_model = cls_search.best_estimator_
print("🔑  Best CLS params:", cls_search.best_params_)

# ── Evaluate classifier
ycls_pred = cls_model.predict(X_test)
print("\n📊  Classifier (30‑day up ≥5 %)")
print("Accuracy:", round(accuracy_score(ycls_test, ycls_pred), 4))
print(classification_report(ycls_test, ycls_pred, digits=4))

# ──────────────────────────────────
# 2)  Long‑term REGRESSOR
# ──────────────────────────────────
xgb_reg = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=600,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

print("\n🛠  Training regressor …")
xgb_reg.fit(X_train, yreg_train)
yreg_pred = xgb_reg.predict(X_test)
print("MAE :", round(mean_absolute_error(yreg_test, yreg_pred), 2))
print("R²  :", round(r2_score(yreg_test, yreg_pred), 4))

# ── Persist everything
MODEL_DIR = Path("backend/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(cls_model,  MODEL_DIR / "xgb_cls.pkl")
joblib.dump(xgb_reg,    MODEL_DIR / "xgb_reg.pkl")
joblib.dump(scaler,     MODEL_DIR / "scaler.pkl")
print("✅ Models & scaler saved to", MODEL_DIR)

meta = {
    "trained_on"   : datetime.utcnow().strftime("%Y-%m-%d"),
    "rows"         : len(df),
    "horizon_days" : HORIZON,
    "threshold_up" : THRESH,
    "features"     : FEATURES,
    "cls_best_params": cls_search.best_params_,
    "reg_params"     : xgb_reg.get_params(),
    "cls_accuracy"   : accuracy_score(ycls_test, ycls_pred),
    "reg_mae"        : mean_absolute_error(yreg_test, yreg_pred),
}
with open(MODEL_DIR / "model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n🎉  Training pipeline complete!")

from xgboost import plot_importance
plot_importance(cls_model, max_num_features=15)

import shap
explainer = shap.Explainer(cls_model)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)


