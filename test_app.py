import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import app as app_module

def setup_module():
    # Small deterministic fitted pipeline keeps tests independent of training command.
    X = pd.DataFrame({
        "product_id": ["P001", "P002", "P001", "P003", "P002", "P004"],
        "category": ["Grocery", "Home", "Grocery", "Electronics", "Home", "Fashion"],
        "price": [80, 450, 75, 1200, 460, 650],
        "promotion": [0, 1, 1, 0, 0, 1],
        "stock_available": [100, 150, 120, 80, 90, 200],
        "day_of_week": [0, 5, 2, 3, 6, 1],
        "month": [1, 10, 4, 7, 12, 8],
    })
    y = np.array([40, 25, 55, 8, 13, 30])
    categorical = ["product_id", "category"]
    numeric = ["price", "promotion", "stock_available", "day_of_week", "month"]
    pre = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric)
    ])
    pipe = Pipeline([("preprocess", pre),
                     ("regressor", RandomForestRegressor(n_estimators=5, random_state=1))])
    pipe.fit(X, y)
    joblib.dump(pipe, app_module.MODEL_PATH)
    app_module.model = None

def test_health():
    response = app_module.app.test_client().get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"

def test_home_page():
    response = app_module.app.test_client().get("/")
    assert response.status_code == 200
    assert b"Product Demand Predictor" in response.data

def test_api_prediction():
    response = app_module.app.test_client().post("/api/predict", json={
        "product_id": "P001", "category": "Grocery", "price": 80,
        "promotion": 1, "stock_available": 100, "day_of_week": 5, "month": 10
    })
    assert response.status_code == 200
    assert "predicted_demand_quantity" in response.json

def test_api_validation():
    response = app_module.app.test_client().post("/api/predict", json={"price": -5})
    assert response.status_code == 400
    assert "error" in response.json
