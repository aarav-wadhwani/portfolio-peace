# training/train_model.py — Enhanced long-term prediction model
"""
Multi-class classification and improved regression for NIFTY 500 predictions.

Models:
1) XGBoost multi-class classifier → Bearish/Neutral/Bullish (30 days)
2) XGBoost regressor → 30-day return forecast
3) Ensemble approach for better generalization
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings('ignore')

# ── Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.utils.indicators import (
    add_features,
    add_index_features,
    build_feature_columns,
)

# ── Configuration
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

# Model parameters
HORIZON_DAYS = 30
BEARISH_THRESHOLD = -0.03  # -3%
BULLISH_THRESHOLD = 0.03   # +3%
RANDOM_STATE = 42

# ── Helper Functions ──────────────────────────────────────────────
def create_multiclass_target(returns: pd.Series) -> pd.Series:
    """
    Create 3-class target:
    0: Bearish (< -3%)
    1: Neutral (-3% to +3%)
    2: Bullish (> +3%)
    """
    conditions = [
        returns < BEARISH_THRESHOLD,
        (returns >= BEARISH_THRESHOLD) & (returns <= BULLISH_THRESHOLD),
        returns > BULLISH_THRESHOLD
    ]
    choices = [0, 1, 2]
    return pd.Series(np.select(conditions, choices), index=returns.index)

def print_class_distribution(y: pd.Series, label: str = ""):
    """Print class distribution"""
    print(f"\n{label} Class Distribution:")
    counts = y.value_counts().sort_index()
    for cls, count in counts.items():
        pct = count / len(y) * 100
        class_name = ['Bearish', 'Neutral', 'Bullish'][cls]
        print(f"  {class_name} ({cls}): {count:,} ({pct:.1f}%)")

class ExpandingWindowSplit:
    """Time series split with expanding window"""
    def __init__(self, min_train_size=252, test_size=63, gap=5):
        self.min_train_size = min_train_size  # 1 year minimum
        self.test_size = test_size            # 3 months test
        self.gap = gap                         # Gap between train and test
        
    def split(self, X):
        n_samples = len(X)
        splits = []
        
        # Start from minimum training size
        train_end = self.min_train_size
        
        while train_end + self.gap + self.test_size <= n_samples:
            train_idx = np.arange(0, train_end)
            test_start = train_end + self.gap
            test_end = min(test_start + self.test_size, n_samples)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
            
            # Move forward by half the test size for next split
            train_end += self.test_size // 2
            
        return splits

# ── Load and Prepare Data ────────────────────────────────────────
print("📊 Loading NIFTY 500 data...")
df = pd.read_csv(MAIN_CSV)

# Clean price columns
for col in ["Open", "High", "Low", "Close"]:
    df[col] = (
        df[col]
        .astype(str)
        .replace("-", np.nan)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Basic cleanup
if "Index Name" in df.columns:
    df.drop(columns="Index Name", inplace=True)

df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y")
df.sort_values("Date", inplace=True)

# Add volume if missing
if "Volume" not in df.columns:
    df["Volume"] = 1e6  # Default volume

print(f"✅ Loaded {len(df):,} rows from {df['Date'].min()} to {df['Date'].max()}")

# ── Feature Engineering ───────────────────────────────────────────
print("\n🔧 Engineering features...")
df = add_features(df)
df = add_index_features(df, INDEX_PATHS)

# ── Create Targets ────────────────────────────────────────────────
df["future_return"] = (df["Close"].shift(-HORIZON_DAYS) - df["Close"]) / df["Close"]
df["target_class"] = create_multiclass_target(df["future_return"])

# Drop rows with NaN targets
df = df.dropna(subset=["target_class", "future_return"])

print(f"✅ Final dataset: {len(df):,} rows with {len(df.columns)} columns")
print_class_distribution(df["target_class"], "Overall")

# ── Feature Selection ─────────────────────────────────────────────
# Get feature columns
feature_cols = build_feature_columns(list(INDEX_PATHS.keys()))

# Verify all features exist
missing_features = [col for col in feature_cols if col not in df.columns]
if missing_features:
    print(f"\n⚠️  Missing features: {missing_features}")
    feature_cols = [col for col in feature_cols if col in df.columns]

print(f"\n📊 Using {len(feature_cols)} features")

# ── Prepare Data for Modeling ────────────────────────────────────
X = df[feature_cols].fillna(0)  # Fill any remaining NaNs
y_class = df["target_class"].astype(int)
y_reg = df["future_return"]

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# ── Train-Test Split ──────────────────────────────────────────────
# Use last 20% for testing (time-based)
split_idx = int(len(X_scaled) * 0.8)
X_train = X_scaled.iloc[:split_idx]
X_test = X_scaled.iloc[split_idx:]
y_class_train = y_class.iloc[:split_idx]
y_class_test = y_class.iloc[split_idx:]
y_reg_train = y_reg.iloc[:split_idx]
y_reg_test = y_reg.iloc[split_idx:]

print(f"\n📊 Train: {len(X_train):,} samples")
print(f"📊 Test:  {len(X_test):,} samples")
print_class_distribution(y_class_train, "Training")
print_class_distribution(y_class_test, "Testing")

# ──────────────────────────────────────────────────────────────────
# 1) Multi-class Classification Models
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("🎯 Training Multi-class Classifiers")
print("="*60)

# Calculate class weights for balanced training
class_weights = len(y_class_train) / (3 * np.bincount(y_class_train))
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# Model 1: XGBoost with custom parameters
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='mlogloss'
)

# Model 2: LightGBM for comparison
lgb_model = LGBMClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    objective='multiclass',
    num_class=3,
    class_weight=class_weight_dict
)

# Time series cross-validation
cv_splitter = ExpandingWindowSplit(min_train_size=252, test_size=63)
cv_scores_xgb = []
cv_scores_lgb = []

print("\n📈 Cross-validation in progress...")
for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train)):
    # Get fold data
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_class_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_class_train.iloc[val_idx]
    
    # Calculate sample weights
    sample_weights = np.array([class_weights[y] for y in y_fold_train])
    
    # Train XGBoost
    xgb_model.fit(
        X_fold_train, y_fold_train,
        sample_weight=sample_weights,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=False
    )
    xgb_pred = xgb_model.predict(X_fold_val)
    cv_scores_xgb.append(accuracy_score(y_fold_val, xgb_pred))
    
    # Train LightGBM
    lgb_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        callbacks=[{'log_evaluation': False}]
    )
    lgb_pred = lgb_model.predict(X_fold_val)
    cv_scores_lgb.append(accuracy_score(y_fold_val, lgb_pred))
    
    print(f"  Fold {fold+1}: XGB={cv_scores_xgb[-1]:.3f}, LGB={cv_scores_lgb[-1]:.3f}")

print(f"\n📊 CV Results:")
print(f"  XGBoost: {np.mean(cv_scores_xgb):.3f} (+/- {np.std(cv_scores_xgb):.3f})")
print(f"  LightGBM: {np.mean(cv_scores_lgb):.3f} (+/- {np.std(cv_scores_lgb):.3f})")

# Train final models on full training set
print("\n🎯 Training final models...")

# Calculate sample weights for full training set
sample_weights_train = np.array([class_weights[y] for y in y_class_train])

# Train XGBoost
xgb_model.fit(
    X_train, y_class_train,
    sample_weight=sample_weights_train,
    eval_set=[(X_test, y_class_test)],
    verbose=False
)

# Train LightGBM
lgb_model.fit(
    X_train, y_class_train,
    eval_set=[(X_test, y_class_test)],
    callbacks=[{'log_evaluation': False}]
)

# Create ensemble
ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model)
    ],
    voting='soft',
    weights=[0.6, 0.4]  # Give more weight to XGBoost
)

# Train ensemble
ensemble_model.fit(X_train, y_class_train)

# ── Evaluate Classification Models ────────────────────────────────
print("\n" + "="*60)
print("📊 Classification Results")
print("="*60)

models = {
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
    'Ensemble': ensemble_model
}

best_model = None
best_score = 0

for name, model in models.items():
    print(f"\n🔍 {name} Performance:")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Metrics
    acc = accuracy_score(y_class_test, y_pred)
    mcc = matthews_corrcoef(y_class_test, y_pred)
    
    print(f"  Accuracy: {acc:.3f}")
    print(f"  MCC: {mcc:.3f}")
    
    # Per-class metrics
    print("\n  Classification Report:")
    report = classification_report(
        y_class_test, y_pred,
        target_names=['Bearish', 'Neutral', 'Bullish'],
        digits=3
    )
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_class_test, y_pred)
    print("\n  Confusion Matrix:")
    print("  " + " "*10 + "Predicted")
    print("  " + " "*10 + "Bear  Neut  Bull")
    for i, row in enumerate(cm):
        label = ['Bear', 'Neut', 'Bull'][i]
        print(f"  Actual {label}: {row}")
    
    # Track best model
    if acc > best_score:
        best_score = acc
        best_model = model

print(f"\n✅ Best classifier: {type(best_model).__name__} with accuracy {best_score:.3f}")

# ──────────────────────────────────────────────────────────────────
# 2) Regression Models
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("🎯 Training Regression Models")
print("="*60)

# XGBoost Regressor
xgb_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# LightGBM Regressor
lgb_reg = LGBMRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Train regressors
print("📈 Training regressors...")
xgb_reg.fit(X_train, y_reg_train)
lgb_reg.fit(X_train, y_reg_train)

# Evaluate
print("\n📊 Regression Results:")
for name, model in [('XGBoost', xgb_reg), ('LightGBM', lgb_reg)]:
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_reg_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    r2 = r2_score(y_reg_test, y_pred)
    
    print(f"\n{name}:")
    print(f"  MAE:  {mae:.4f} ({mae*100:.2f}%)")
    print(f"  RMSE: {rmse:.4f} ({rmse*100:.2f}%)")
    print(f"  R²:   {r2:.4f}")

# ──────────────────────────────────────────────────────────────────
# 3) Feature Importance Analysis
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("📊 Feature Importance Analysis")
print("="*60)

# Get feature importances from XGBoost
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
for i, row in feature_importance.head(20).iterrows():
    print(f"  {row['feature']:.<40} {row['importance']:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 20 Features - XGBoost Classifier')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# ──────────────────────────────────────────────────────────────────
# 4) Advanced Analysis
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("📊 Advanced Analysis")
print("="*60)

# Analyze prediction confidence
y_proba_best = best_model.predict_proba(X_test)
max_proba = np.max(y_proba_best, axis=1)

print("\n🔍 Prediction Confidence Analysis:")
print(f"  Mean confidence: {np.mean(max_proba):.3f}")
print(f"  Median confidence: {np.median(max_proba):.3f}")
print(f"  Min confidence: {np.min(max_proba):.3f}")
print(f"  Max confidence: {np.max(max_proba):.3f}")

# High confidence predictions
high_conf_mask = max_proba > 0.6
high_conf_pred = best_model.predict(X_test[high_conf_mask])
high_conf_actual = y_class_test[high_conf_mask]

if len(high_conf_actual) > 0:
    high_conf_acc = accuracy_score(high_conf_actual, high_conf_pred)
    print(f"\n  High confidence (>60%) predictions: {sum(high_conf_mask)} ({sum(high_conf_mask)/len(X_test)*100:.1f}%)")
    print(f"  High confidence accuracy: {high_conf_acc:.3f}")

# Performance by market regime
volatility_regime = pd.qcut(df.loc[X_test.index, 'volatility_20'], q=3, labels=['Low', 'Medium', 'High'])
print("\n🔍 Performance by Volatility Regime:")
for regime in ['Low', 'Medium', 'High']:
    mask = volatility_regime == regime
    if sum(mask) > 0:
        regime_pred = best_model.predict(X_test[mask])
        regime_actual = y_class_test[mask]
        regime_acc = accuracy_score(regime_actual, regime_pred)
        print(f"  {regime} volatility: {regime_acc:.3f} (n={sum(mask)})")

# ──────────────────────────────────────────────────────────────────
# 5) Save Models and Metadata
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("💾 Saving Models")
print("="*60)

MODEL_DIR = Path("backend/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Save models
joblib.dump(best_model, MODEL_DIR / "xgboost_stock_predictor.pkl")
joblib.dump(xgb_reg, MODEL_DIR / "xgb_regressor.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(feature_cols, MODEL_DIR / "feature_columns.pkl")

print(f"✅ Models saved to {MODEL_DIR}")

# Save metadata
metadata = {
    "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": {
        "rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": len(feature_cols),
        "date_range": f"{df['Date'].min()} to {df['Date'].max()}"
    },
    "model_config": {
        "horizon_days": HORIZON_DAYS,
        "bearish_threshold": BEARISH_THRESHOLD,
        "bullish_threshold": BULLISH_THRESHOLD,
        "classifier_type": type(best_model).__name__,
        "regressor_type": "XGBRegressor"
    },
    "performance": {
        "classifier": {
            "accuracy": float(best_score),
            "mcc": float(matthews_corrcoef(y_class_test, best_model.predict(X_test))),
            "class_distribution": {
                "bearish": int((y_class_test == 0).sum()),
                "neutral": int((y_class_test == 1).sum()),
                "bullish": int((y_class_test == 2).sum())
            }
        },
        "regressor": {
            "mae": float(mean_absolute_error(y_reg_test, xgb_reg.predict(X_test))),
            "rmse": float(np.sqrt(mean_squared_error(y_reg_test, xgb_reg.predict(X_test)))),
            "r2": float(r2_score(y_reg_test, xgb_reg.predict(X_test)))
        }
    },
    "feature_importance": feature_importance.head(20).to_dict('records')
}

with open(MODEL_DIR / "model_meta.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Metadata saved")

# ──────────────────────────────────────────────────────────────────
# 6) Generate Prediction Examples
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("🎯 Sample Predictions on Test Set")
print("="*60)

# Get last 10 predictions
sample_idx = X_test.index[-10:]
sample_X = X_test.loc[sample_idx]
sample_dates = df.loc[sample_idx, 'Date']

# Predictions
sample_pred_class = best_model.predict(sample_X)
sample_pred_proba = best_model.predict_proba(sample_X)
sample_pred_return = xgb_reg.predict(sample_X)

print("\nRecent Predictions:")
print("-" * 80)
print(f"{'Date':<12} {'Predicted':<10} {'Confidence':<12} {'Return%':<10} {'Actual':<10}")
print("-" * 80)

for i, idx in enumerate(sample_idx):
    date = sample_dates.iloc[i].strftime('%Y-%m-%d')
    pred_class = ['Bearish', 'Neutral', 'Bullish'][sample_pred_class[i]]
    confidence = np.max(sample_pred_proba[i])
    pred_return = sample_pred_return[i] * 100
    actual_class = ['Bearish', 'Neutral', 'Bullish'][y_class_test.loc[idx]]
    
    print(f"{date:<12} {pred_class:<10} {confidence:<12.3f} {pred_return:<10.2f} {actual_class:<10}")

# ──────────────────────────────────────────────────────────────────
# 7) SHAP Analysis (Optional - comment out if not needed)
# ──────────────────────────────────────────────────────────────────
try:
    import shap
    print("\n" + "="*60)
    print("🔍 SHAP Analysis")
    print("="*60)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test[:100])  # Use subset for speed
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test[:100],
        feature_names=feature_cols,
        show=False,
        max_display=20,
        class_names=['Bearish', 'Neutral', 'Bullish']
    )
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ SHAP analysis saved to shap_summary.png")
    
except ImportError:
    print("\n⚠️  SHAP not installed. Skipping SHAP analysis.")

print("\n" + "="*60)
print("🎉 Training Pipeline Complete!")
print("="*60)
print(f"\nModels saved to: {MODEL_DIR}")
print(f"Feature importance plot: feature_importance.png")
print(f"\nBest classifier accuracy: {best_score:.3f}")
print(f"Regressor R²: {r2_score(y_reg_test, xgb_reg.predict(X_test)):.3f}")