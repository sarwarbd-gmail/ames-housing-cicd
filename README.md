# Ames Housing — CI/CD with GitHub Actions
### A hands-on MLOps demo for students

---

## Project Overview

This project demonstrates a complete CI/CD pipeline for a machine learning model using:
- **Dataset**: Ames Housing (`AmesHousing.csv`)
- **Model**: XGBoost / GradientBoosting regression to predict house sale prices
- **CI/CD**: GitHub Actions (3-job pipeline: Test → Train → Deploy)

---

## Folder Structure

```
ames-housing-cicd/
├── data/
│   └── AmesHousing.csv          ← place your dataset here
├── src/
│   ├── train.py                 ← trains the model, saves model.pkl
│   └── predict.py               ← loads model.pkl, predicts sale price
├── tests/
│   └── test_model.py            ← unit tests (run before training)
├── .github/
│   └── workflows/
│       └── ci_cd.yml            ← GitHub Actions pipeline definition
├── requirements.txt             ← Python dependencies
└── README.md                    ← this file
```

---

## Prerequisites

- Python 3.9 or higher
- Git installed
- A GitHub account (free)

---

## Step-by-Step Instructions

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Add the dataset

Place your `AmesHousing.csv` file inside the `data/` folder.

The dataset is available at:
https://www.kaggle.com/datasets/prevek18/ames-housing-dataset

### Step 3 — Run tests locally

```bash
pytest tests/ -v
```

All 5 tests should pass. If they do, your environment is ready.

### Step 4 — Train the model locally

```bash
python src/train.py
```

Output example:
```
Train: 2328 rows  |  Val: 582 rows
MAE : $22,000
R²  : 0.89
Saved → model/model.pkl
Saved → metrics/metrics.json
```

### Step 5 — Run predictions locally

```bash
python src/predict.py
```

Output example:
```
House                      Predicted Price
Small older house              $  115,000
Mid-size modern house          $  195,000
Large luxury house             $  380,000
```

---

## Setting Up CI/CD on GitHub

### Step 1 — Create a GitHub repository

1. Go to github.com → click **New repository**
2. Name it `ames-housing-cicd`
3. Click **Create repository**

### Step 2 — Push your code

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/ames-housing-cicd.git
git branch -M main
git push -u origin main
```

### Step 3 — Watch the pipeline

1. Go to your GitHub repo
2. Click the **Actions** tab
3. See the pipeline: **Test → Train → Deploy**

### Step 4 — Update the model and redeploy

Edit `src/train.py` — change the hyperparameters at the top:

```python
N_ESTIMATORS  = 200   # was 100
MAX_DEPTH     = 5     # was 4
LEARNING_RATE = 0.05  # was 0.1
```

Then push:

```bash
git add src/train.py
git commit -m "Improved model hyperparameters"
git push
```

GitHub Actions automatically runs: tests → retrain → redeploy.

---

## CI/CD Pipeline Explained

```
git push to main
    │
    ▼
Job 1: TEST
    └── pytest tests/ -v
    └── If any test fails → pipeline stops here, no training, no deploy

    │ (only if tests pass)
    ▼
Job 2: TRAIN
    └── python src/train.py
    └── Saves model.pkl + metrics.json as a GitHub artifact

    │ (only if training passes, only on main branch)
    ▼
Job 3: DEPLOY
    └── Downloads the trained model artifact
    └── Deploys to production (Docker / AWS Lambda / Hugging Face Spaces)
```

Pull Requests trigger Jobs 1 and 2 only — deploy is skipped until code merges to main.

---

## Switching to Real XGBoost

In `src/train.py`, replace the two marked lines:

```python
# REMOVE:
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=..., max_depth=..., ...)

# ADD:
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                     learning_rate=LEARNING_RATE, random_state=42)
```

Everything else stays exactly the same.

---

## Features Used for Training

| Feature | Description |
|---|---|
| GrLivArea | Above grade living area (sq ft) |
| TotalBsmtSF | Total basement area (sq ft) |
| 1stFlrSF | First floor area (sq ft) |
| GarageCars | Garage capacity (number of cars) |
| FullBath | Number of full bathrooms |
| TotRmsAbvGrd | Total rooms above ground |
| YearBuilt | Year the house was built |
| OverallQual | Overall material and finish quality (1-10) |
| Neighborhood | Physical location (label-encoded) |
| HouseStyle | Style of dwelling (label-encoded) |

---

## Learning Objectives

After completing this demo, students will understand:

1. How to structure a data science project for CI/CD
2. How to write unit tests for ML code
3. How GitHub Actions automates test → train → deploy
4. How a Pull Request safely tests changes before they reach production
5. How to update a model and trigger automatic redeployment
