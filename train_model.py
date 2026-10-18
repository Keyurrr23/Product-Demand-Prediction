"""Train a Random Forest Regressor for product demand prediction."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "demand_data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

def create_sample_data(path: Path, rows: int = 1500) -> pd.DataFrame:
    """Create reproducible DEMO data if no real dataset exists."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    products = rng.choice(["P001", "P002", "P003", "P004", "P005"], rows)
    categories = {"P001":"Grocery", "P002":"Home", "P003":"Electronics",
                  "P004":"Fashion", "P005":"Grocery"}
    prices = {"P001": 80, "P002": 450, "P003": 1200, "P004": 650, "P005": 35}
    product_price = np.array([prices[p] for p in products], dtype=float)
    price = np.maximum(5, product_price * rng.normal(1.0, 0.08, rows))
    promotion = rng.integers(0, 2, rows)
    stock = rng.integers(20, 500, rows)
    day_of_week = dates.dayofweek
    month = dates.month
    season = np.where(np.isin(month, [10, 11, 12]), 18, 0)
    product_effect = np.select(
        [products == "P001", products == "P002", products == "P003",
         products == "P004", products == "P005"],
        [35, 12, 5, 14, 45], default=10
    )
    noise = rng.normal(0, 7, rows)
    demand = (product_effect + 0.10 * stock - 0.018 * price
              + 12 * promotion + season
              + np.where(day_of_week >= 5, 8, 0) + noise)
    demand = np.maximum(0, np.round(demand)).astype(int)
    df = pd.DataFrame({
        "date": dates, "product_id": products,
        "category": [categories[p] for p in products],
        "price": np.round(price, 2), "promotion": promotion,
        "stock_available": stock, "day_of_week": day_of_week,
        "month": month, "demand_quantity": demand
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

def main():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print(f"Using dataset: {DATA_PATH}")
    else:
        df = create_sample_data(DATA_PATH)
        print("Created DEMO synthetic dataset. Replace it with real historical demand data before making business decisions.")

    required = ["product_id", "category", "price", "promotion",
                "stock_available", "day_of_week", "month", "demand_quantity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[required].dropna()
    X = df.drop(columns=["demand_quantity"])
    y = df["demand_quantity"].astype(float)
    categorical = ["product_id", "category"]
    numeric = ["price", "promotion", "stock_available", "day_of_week", "month"]
    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(
            n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    print(f"Rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"MAE:  {mean_absolute_error(y_test, pred):.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2_score(y_test, pred):.3f}")
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model pipeline to {MODEL_PATH}")

if __name__ == "__main__":
    main()
