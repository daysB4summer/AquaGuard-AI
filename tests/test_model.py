
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "aquaguard_v4.joblib"
DATA_PATH = BASE_DIR / "data" / "processed" / "water_data.csv"


def test_model_file_exists():
    assert MODEL_PATH.exists(), "Saved model file is missing"


def test_dataset_exists():
    assert DATA_PATH.exists(), "Processed dataset is missing"


def test_model_loads_correctly():
    saved = joblib.load(MODEL_PATH)

    assert isinstance(saved["model"], RandomForestClassifier)
    assert len(saved["features"]) == 20
    assert 0 <= saved["threshold"] <= 1


def test_required_columns_exist():
    df = pd.read_csv(DATA_PATH)

    assert "timestamp" in df.columns
    assert "fault_d7" in df.columns
    assert "326416_ws_vigor" in df.columns
    assert "gw_lvl_site2_anomaly_score" in df.columns


def test_target_is_binary():
    df = pd.read_csv(DATA_PATH)

    assert set(df["fault_d7"].dropna().unique()).issubset({0, 1})



def test_model_can_make_predictions():
    saved = joblib.load(MODEL_PATH)

    model = saved["model"]
    features = saved["features"]
    medians = saved["medians"]

    df = pd.read_csv(DATA_PATH)

    df["gw_missing"] = (
        df["gw_lvl_site2_anomaly_score"].isna().astype(int)
    )
    df["vigor_missing"] = (
        df["326416_ws_vigor"].isna().astype(int)
    )

    X = df[features].head(10).fillna(medians)

    predictions = model.predict(X)

    assert len(predictions) == 10
    assert set(predictions).issubset({0, 1})
