
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
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "water_data.csv"
MODEL_PATH = BASE_DIR / "models" / "aquaguard_v4.joblib"


def test_evaluation_reproducibility():
    # Recreate the chronological test split
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    split = int(len(df) * 0.8)
    test = df.iloc[split:].copy()
    test_safe = test.iloc[168:].copy()

    assert len(test_safe) == 1580

    # Load saved model and preprocessing settings
    saved = joblib.load(MODEL_PATH)

    model = saved["model"]
    features = saved["features"]
    medians = saved["medians"]
    threshold = saved["threshold"]

    # Recreate missingness indicators
    X_test = test_safe.copy()

    X_test["gw_missing"] = (
        X_test["gw_lvl_site2_anomaly_score"].isna().astype(int)
    )
    X_test["vigor_missing"] = (
        X_test["326416_ws_vigor"].isna().astype(int)
    )

    X_test = X_test[features].fillna(medians)
    y_test = test_safe["fault_d7"]

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    # Check published evaluation metrics
    assert round(accuracy_score(y_test, predictions), 3) == 0.871
    assert round(balanced_accuracy_score(y_test, predictions), 3) == 0.659
    assert round(precision_score(y_test, predictions), 3) == 0.881
    assert round(recall_score(y_test, predictions), 3) == 0.977
    assert round(f1_score(y_test, predictions), 3) == 0.927

    # Check confusion matrix
    expected_matrix = [[90, 174], [30, 1286]]
    assert confusion_matrix(y_test, predictions).tolist() == expected_matrix
