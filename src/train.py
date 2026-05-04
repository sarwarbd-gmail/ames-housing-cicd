"""
train.py  –  Train an XGBoost regression model on the Ames Housing dataset.

Run locally:
    python src/train.py

XGBoost note:
    This file uses sklearn GradientBoostingRegressor so it runs without
    installing xgboost.  To switch to real XGBoost, replace the two lines
    marked  # <<  with the XGBoost equivalents shown in the comments.
"""
import os, json, pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor   # << swap for XGBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── Hyperparameters ──────────────────────────────────────────────────────────
# BEFORE (original)
N_ESTIMATORS  = 100
MAX_DEPTH     = 4
LEARNING_RATE = 0.1

# AFTER (student's improvement)
N_ESTIMATORS  = 200
MAX_DEPTH     = 5
LEARNING_RATE = 0.05

# AFTER (local improvement)
N_ESTIMATORS  = 150
MAX_DEPTH     = 3
LEARNING_RATE = 0.02


# ────────────────────────────────────────────────────────────────────────────

# Columns we actually use (keeps the script simple and readable)
NUMERIC_COLS = [
    "GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageCars",
    "FullBath",  "TotRmsAbvGrd", "YearBuilt", "OverallQual",
]
CAT_COLS = ["Neighborhood", "HouseStyle"]
TARGET   = "SalePrice"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Keep only the columns we need
    keep = NUMERIC_COLS + CAT_COLS + [TARGET]
    df   = df[keep].copy()

    # Fill numeric nulls with median
    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical nulls with mode
    for col in CAT_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def train(data_path: str = "data/AmesHousing.csv"):
    print(f"Loading data from {data_path} ...")
    df  = load_and_clean(data_path)
    df  = encode(df)

    X   = df.drop(columns=[TARGET])
    y   = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)} rows  |  Val: {len(X_val)} rows")

    # << For XGBoost replace the next two lines with:
    # from xgboost import XGBRegressor
    # model = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
    #                      learning_rate=LEARNING_RATE, random_state=42)
    model = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mae   = round(mean_absolute_error(y_val, preds), 2)
    r2    = round(r2_score(y_val, preds), 4)

    metrics = {
        "mae":          mae,
        "r2":           r2,
        "n_estimators": N_ESTIMATORS,
        "max_depth":    MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
    }

    os.makedirs("model",   exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    pickle.dump(model, open("model/model.pkl", "wb"))
    json.dump(metrics, open("metrics/metrics.json", "w"), indent=2)

    print(f"\nMAE : ${mae:,.0f}")
    print(f"R²  : {r2}")
    print("\nSaved → model/model.pkl")
    print("Saved → metrics/metrics.json")
    return metrics


if __name__ == "__main__":
    train()
