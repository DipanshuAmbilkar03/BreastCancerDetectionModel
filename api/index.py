from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__, template_folder="../templates", static_folder="../static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "model", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "..", "model", "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

FEATURES = [
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'compactness_mean',
    'concavity_mean',
    'concave points_mean',
    'radius_worst',
    'texture_worst',
    'perimeter_worst',
    'area_worst',
    'concavity_worst',
    'concave points_worst'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            values = [float(request.form[f]) for f in FEATURES]
            X = np.array(values).reshape(1, -1)
            X_scaled = scaler.transform(X)
            result = model.predict(X_scaled)[0]
            proba = model.predict_proba(X)[0]
            print("Benign:", proba[0], "Malignant:", proba[1])

            prediction = "Malignant (Cancerous)" if result == 1 else "Benign (Non-Cancerous)"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", features=FEATURES, prediction=prediction)


