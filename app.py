from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import time
import threading
from collections import deque
import math

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

# ======================
# ML metrics
# ======================
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

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
    "Prediction probability",
    buckets=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
)

DATA_DRIFT_SCORE = Gauge(
    "evidently_data_drift_score",
    "Overall data drift score"
)

# ===== Model performance metrics =====
MODEL_ACCURACY = Gauge("model_accuracy", "Model accuracy")
MODEL_PRECISION = Gauge("model_precision", "Model precision")
MODEL_RECALL = Gauge("model_recall", "Model recall")
MODEL_AUC = Gauge("model_auc", "Model AUC")

CONF_TP = Gauge("model_confusion_tp", "True positives")
CONF_FP = Gauge("model_confusion_fp", "False positives")
CONF_TN = Gauge("model_confusion_tn", "True negatives")
CONF_FN = Gauge("model_confusion_fn", "False negatives")

# ======================
# Rolling window
# ======================
RECENT_WINDOW = deque(maxlen=1000)

# ======================
# Reference data
# ======================
reference_df = pd.read_csv("./data/diabetes_cleaned.csv")

# ======================
# Per-feature drift gauges
# ======================
FEATURE_DRIFT_GAUGES = {}

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

            pred = int(model.predict(arr_scaled)[0])
            prob = float(model.predict_proba(arr_scaled)[0, 1])

            label = "positive" if pred == 1 else "negative"
            result = "Positive for diabetes" if pred == 1 else "Negative for diabetes"

            PREDICTIONS_TOTAL.labels(label=label).inc()
            PREDICTION_PROB.observe(prob)
            INFERENCE_LATENCY.observe(time.time() - start_time)

            row = dict(zip(FEATURES, vals))
            row["prediction"] = pred
            row["probability"] = prob
            row["true_label"] = None  # filled later
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
# Receive ground-truth labels
# ======================
@app.route("/label", methods=["POST"])
def receive_label():
    data = request.get_json()
    if not data:
        return jsonify({"error": "invalid payload"}), 400

    items = data if isinstance(data, list) else [data]
    added = 0

    for item in items:
        try:
            row = {}
            for f in FEATURES:
                row[f] = float(item["features"][f])
            row["prediction"] = int(item["prediction"])
            row["probability"] = float(item["probability"])
            row["true_label"] = int(item["true_label"])
            RECENT_WINDOW.append(row)
            added += 1
        except Exception as e:
            print("Label error:", e)

    return jsonify({"added": added})

# ======================
# Prometheus endpoint
# ======================
app.wsgi_app = DispatcherMiddleware(
    app.wsgi_app,
    {"/metrics": make_wsgi_app()}
)

# ======================
# Data drift monitor
# ======================
def run_drift_monitor(interval=300):
    while True:
        try:
            if len(RECENT_WINDOW) >= 50:
                current_df = pd.DataFrame(list(RECENT_WINDOW))[FEATURES]

                report = Report(metrics=[DataDriftPreset()])
                report.run(reference_data=reference_df, current_data=current_df)
                result = report.as_dict()

                dataset_drift = result["metrics"][0]["result"]["dataset_drift"]["data"]
                DATA_DRIFT_SCORE.set(float(dataset_drift["drift_score"]))

        except Exception as e:
            print("Drift error:", e)

        time.sleep(interval)

# ======================
# Model performance monitor
# ======================
def run_perf_monitor(interval=60):
    while True:
        try:
            labeled = [r for r in RECENT_WINDOW if r["true_label"] is not None]
            if len(labeled) >= 10:
                y_true = [r["true_label"] for r in labeled]
                y_pred = [r["prediction"] for r in labeled]
                y_prob = [r["probability"] for r in labeled]

                MODEL_ACCURACY.set(accuracy_score(y_true, y_pred))
                MODEL_PRECISION.set(precision_score(y_true, y_pred, zero_division=0))
                MODEL_RECALL.set(recall_score(y_true, y_pred, zero_division=0))

                if len(set(y_true)) > 1:
                    MODEL_AUC.set(roc_auc_score(y_true, y_prob))

                CONF_TP.set(sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1))
                CONF_FP.set(sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1))
                CONF_TN.set(sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==0))
                CONF_FN.set(sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0))

        except Exception as e:
            print("Perf error:", e)

        time.sleep(interval)

# ======================
# Start background threads
# ======================
threading.Thread(target=run_drift_monitor, daemon=True).start()
threading.Thread(target=run_perf_monitor, daemon=True).start()

# ======================
# Main
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)