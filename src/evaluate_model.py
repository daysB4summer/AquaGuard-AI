
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "water_data.csv"
MODEL_PATH = BASE_DIR / "models" / "aquaguard_v4.joblib"
REPORT_PATH = BASE_DIR / "reports" / "v4_test_results.csv"


# ==================================================
# 2. LOAD DATA
# ==================================================

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

# Preserve chronological order
df = df.sort_values("timestamp").reset_index(drop=True)

print("Total observations:", len(df))


# ==================================================
# 3. RECREATE CHRONOLOGICAL TEST SPLIT
# ==================================================

split = int(len(df) * 0.8)

test = df.iloc[split:].copy()

# Exclude the first seven days (168 hourly observations)
gap = 7 * 24

test_safe = test.iloc[gap:].copy()

print("Independent test observations:", len(test_safe))
print("Test period starts:", test_safe["timestamp"].min())
print("Test period ends:", test_safe["timestamp"].max())


# ==================================================
# 4. LOAD SAVED RANDOM FOREST V4
# ==================================================

saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
medians = saved_model["medians"]
features = saved_model["features"]
threshold = saved_model["threshold"]

print("\nRandom Forest v4 loaded successfully.")
print("Features:", len(features))
print("Threshold:", threshold)


# ==================================================
# 5. RECREATE TRAINING FEATURES
# ==================================================

X_test = test_safe.copy()

X_test["gw_missing"] = (
    X_test["gw_lvl_site2_anomaly_score"]
    .isna()
    .astype(int)
)

X_test["vigor_missing"] = (
    X_test["326416_ws_vigor"]
    .isna()
    .astype(int)
)

# Select features in their original training order
X_test = X_test[features]

# Apply saved training medians
X_test = X_test.fillna(medians)

y_test = test_safe["fault_d7"]


# ==================================================
# 6. MAKE PREDICTIONS
# ==================================================

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)


# ==================================================
# 7. CALCULATE EVALUATION METRICS
# ==================================================

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

print("\n" + "=" * 50)
print("AQUAGUARD AI — RANDOM FOREST V4 TEST RESULTS")
print("=" * 50)

print(f"Accuracy:          {accuracy:.3f}")
print(f"Balanced Accuracy: {balanced_accuracy:.3f}")
print(f"Precision:         {precision:.3f}")
print(f"Recall:            {recall:.3f}")
print(f"F1-Score:          {f1:.3f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==================================================
# 8. SAVE RESULTS
# ==================================================

results = pd.DataFrame([{
    "model": "Random Forest v4",
    "dataset": "Independent chronological test",
    "observations": len(y_test),
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
}])

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(REPORT_PATH, index=False)

print("\nResults saved to:", REPORT_PATH)
