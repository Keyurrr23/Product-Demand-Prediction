from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
app = Flask(__name__)
model = None

def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("model.pkl is missing. Run: python train_model.py")
        model = joblib.load(MODEL_PATH)
    return model

def parse_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    product_id = str(payload.get("product_id", "")).strip()
    category = str(payload.get("category", "")).strip()
    if not product_id or not category:
        raise ValueError("product_id and category are required.")
    try:
        price = float(payload["price"])
        promotion = int(payload["promotion"])
        stock = int(payload["stock_available"])
        day = int(payload["day_of_week"])
        month = int(payload["month"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("price, promotion, stock_available, day_of_week and month must be valid numbers.")
    if price <= 0:
        raise ValueError("price must be greater than 0.")
    if promotion not in (0, 1):
        raise ValueError("promotion must be 0 or 1.")
    if stock < 0:
        raise ValueError("stock_available cannot be negative.")
    if day < 0 or day > 6:
        raise ValueError("day_of_week must be 0 (Monday) through 6 (Sunday).")
    if month < 1 or month > 12:
        raise ValueError("month must be between 1 and 12.")
    return pd.DataFrame([{
        "product_id": product_id, "category": category, "price": price,
        "promotion": promotion, "stock_available": stock,
        "day_of_week": day, "month": month
    }])

@app.get("/")
def home():
    return render_template("index.html", result=None, error=None)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.exists()})

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html", result=None, error=None)
    payload = request.form.to_dict()
    try:
        features = parse_payload(payload)
        prediction = max(0.0, float(load_model().predict(features)[0]))
        result = round(prediction, 2)
        return render_template("index.html", result=result, error=None)
    except (ValueError, FileNotFoundError) as exc:
        return render_template("index.html", result=None, error=str(exc)), 400

@app.post("/api/predict")
def api_predict():
    try:
        payload = request.get_json(silent=True)
        features = parse_payload(payload)
        prediction = max(0.0, float(load_model().predict(features)[0]))
        return jsonify({"predicted_demand_quantity": round(prediction, 2),
                        "unit": "units", "model": "RandomForestRegressor"})
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    # 0.0.0.0 is needed for Docker/VM access; use a production WSGI server in deployment.
    app.run(host="0.0.0.0", port=5000, debug=False)
