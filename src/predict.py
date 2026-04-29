"""
predict.py  –  Load the trained model and predict a house sale price.

Run locally:
    python src/predict.py

In production this becomes a FastAPI endpoint:
    POST /predict
    {"GrLivArea": 1500, "TotalBsmtSF": 800, "1stFlrSF": 800,
     "GarageCars": 2, "FullBath": 2, "TotRmsAbvGrd": 7,
     "YearBuilt": 2005, "OverallQual": 7,
     "Neighborhood": 5, "HouseStyle": 1}
"""
import pickle
import pandas as pd

FEATURE_COLS = [
    "GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageCars",
    "FullBath",  "TotRmsAbvGrd", "YearBuilt", "OverallQual",
    "Neighborhood", "HouseStyle",
]


def predict(features: dict) -> float:
    model = pickle.load(open("model/model.pkl", "rb"))
    X     = pd.DataFrame([features])[FEATURE_COLS]
    price = model.predict(X)[0]
    return round(float(price), 2)


if __name__ == "__main__":
    # Three sample houses
    examples = [
        {
            "label":        "Small older house",
            "GrLivArea":    1050, "TotalBsmtSF": 600,  "1stFlrSF": 600,
            "GarageCars":   1,    "FullBath":    1,     "TotRmsAbvGrd": 5,
            "YearBuilt":    1965, "OverallQual": 5,
            "Neighborhood": 10,   "HouseStyle":  1,
        },
        {
            "label":        "Mid-size modern house",
            "GrLivArea":    1800, "TotalBsmtSF": 900,  "1stFlrSF": 900,
            "GarageCars":   2,    "FullBath":    2,     "TotRmsAbvGrd": 7,
            "YearBuilt":    2005, "OverallQual": 7,
            "Neighborhood": 15,   "HouseStyle":  2,
        },
        {
            "label":        "Large luxury house",
            "GrLivArea":    3200, "TotalBsmtSF": 1600, "1stFlrSF": 1600,
            "GarageCars":   3,    "FullBath":    3,     "TotRmsAbvGrd": 10,
            "YearBuilt":    2015, "OverallQual": 9,
            "Neighborhood": 20,   "HouseStyle":  3,
        },
    ]

    print(f"\n{'House':<25}  {'Predicted Price':>16}")
    print("-" * 43)
    for e in examples:
        label = e.pop("label")
        price = predict(e)
        print(f"{label:<25}  ${price:>15,.0f}")
