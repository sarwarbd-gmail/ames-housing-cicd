"""
tests/test_model.py  –  Unit tests for training pipeline.
Run with:  pytest tests/ -v
"""
import sys, os, json, pickle, unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.train import load_and_clean, encode, NUMERIC_COLS, CAT_COLS, TARGET


def make_sample_csv(path="data/sample.csv", n=50):
    """Creates a tiny CSV with the same columns as AmesHousing.csv."""
    np.random.seed(0)
    df = pd.DataFrame({
        "GrLivArea":    np.random.randint(700,  3000, n),
        "TotalBsmtSF":  np.random.randint(300,  1500, n),
        "1stFlrSF":     np.random.randint(300,  1500, n),
        "GarageCars":   np.random.randint(0,    4,    n),
        "FullBath":     np.random.randint(1,    4,    n),
        "TotRmsAbvGrd": np.random.randint(4,    12,   n),
        "YearBuilt":    np.random.randint(1950, 2020, n),
        "OverallQual":  np.random.randint(3,    10,   n),
        "Neighborhood": np.random.choice(["NAmes","CollgCr","OldTown"], n),
        "HouseStyle":   np.random.choice(["1Story","2Story","SLvl"], n),
        "SalePrice":    np.random.randint(80000, 400000, n),
    })
    os.makedirs("data", exist_ok=True)
    df.to_csv(path, index=False)
    return df


class TestLoadAndClean(unittest.TestCase):
    def setUp(self):
        make_sample_csv()
        self.df = load_and_clean("data/sample.csv")

    def test_no_nulls(self):
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_expected_columns(self):
        for col in NUMERIC_COLS + CAT_COLS + [TARGET]:
            self.assertIn(col, self.df.columns)

    def test_price_positive(self):
        self.assertTrue((self.df[TARGET] > 0).all())


class TestEncode(unittest.TestCase):
    def setUp(self):
        make_sample_csv()
        raw      = load_and_clean("data/sample.csv")
        self.df  = encode(raw)

    def test_cat_cols_are_numeric(self):
        for col in CAT_COLS:
            self.assertTrue(pd.api.types.is_numeric_dtype(self.df[col]),
                            f"{col} should be numeric after encoding")

    def test_row_count_unchanged(self):
        raw = load_and_clean("data/sample.csv")
        self.assertEqual(len(self.df), len(raw))


class TestMetricsFile(unittest.TestCase):
    """Checks the metrics file written by train() has required keys."""
    def test_metrics_keys(self):
        # Only runs if metrics/metrics.json already exists (after training)
        path = "metrics/metrics.json"
        if not os.path.exists(path):
            self.skipTest("metrics.json not found – run train.py first")
        m = json.load(open(path))
        for key in ["mae", "r2"]:
            self.assertIn(key, m)
        self.assertGreater(m["r2"], 0.5, "R² should be > 0.5")


if __name__ == "__main__":
    unittest.main(verbosity=2)
