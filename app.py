from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import time
import threading
from collections import deque

# ======================
# Prometheus
# ======================
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app

# ======================
# Evidently
# ======================
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


app = Flask(__name__)

# ======================
# Model loading
# ======================
MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# ======================
# Prometheus Metrics
# ======================
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions",
    ["label"]
)

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Model inference latency"
)

PREDICTION_PROB = Histogram(
    "prediction_probability",
    "Prediction probability for diabetes",
    buckets=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
)

DATA_DRIFT_SCORE = Gauge(
    "evidently_data_drift_score",
    "Overall data drift score (0=no drift, 1=drift)"
)

# ======================
# Rolling window for live data
# ======================
RECENT_WINDOW = deque(maxlen=1000)

# ======================
# Load reference data (VERY IMPORTANT)
# ======================
# This should be your TRAINING dataset (CSV)
# Same columns as FEATURES
reference_df = pd.read_csv("./data/diabetes_cleaned.csv")

# ======================
# Routes
# ======================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    prob = None

    if request.method == "POST":
        start_time = time.time()

        try:
            vals = [float(request.form.get(f, 0)) for f in FEATURES]
            arr = np.array(vals).reshape(1, -1)
            arr_scaled = scaler.transform(arr)

            pred = model.predict(arr_scaled)[0]
            prob = model.predict_proba(arr_scaled)[0, 1]

            label = "positive" if pred == 1 else "negative"
            result = "Positive for diabetes" if pred == 1 else "Negative for diabetes"

            # ======================
            # Prometheus updates
            # ======================
            PREDICTIONS_TOTAL.labels(label=label).inc()
            PREDICTION_PROB.observe(prob)
            INFERENCE_LATENCY.observe(time.time() - start_time)

            # ======================
            # Save data for drift detection
            # ======================
            row = dict(zip(FEATURES, vals))
            RECENT_WINDOW.append(row)

        except Exception as e:
            result = f"Error: {e}"

    return render_template(
        "index.html",
        features=FEATURES,
        result=result,
        prob=prob
    )

# ======================
# Prometheus /metrics endpoint
# ======================
app.wsgi_app = DispatcherMiddleware(
    app.wsgi_app,
    {"/metrics": make_wsgi_app()}
)

# ======================
# Evidently background worker
# ======================
def run_drift_monitor(interval=300):
    """
    Runs every 5 minutes
    """
    while True:
        try:
            if len(RECENT_WINDOW) >= 50:
                current_df = pd.DataFrame(list(RECENT_WINDOW))

                report = Report(metrics=[DataDriftPreset()])
                report.run(
                    reference_data=reference_df,
                    current_data=current_df
                )

                report_dict = report.as_dict()

                # Extract dataset drift score
                drift = report_dict["metrics"][0]["result"]["dataset_drift"]
                score = float(drift["data"]["drift_score"])

                DATA_DRIFT_SCORE.set(score)

        except Exception as e:
            print("Evidently error:", e)

        time.sleep(interval)

# ======================
# Start background thread
# ======================
threading.Thread(
    target=run_drift_monitor,
    daemon=True
).start()

# ======================
# Main
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)