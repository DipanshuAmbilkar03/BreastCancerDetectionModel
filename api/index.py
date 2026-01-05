from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# Load model
model = pickle.load(open("../model/model.pkl", "rb"))
scaler = pickle.load(open("../model/scaler.pkl", "rb"))

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
        values = [float(request.form[f]) for f in FEATURES]
        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        result = model.predict(X_scaled)[0]
        prediction = "Malignant" if result == 1 else "Benign"
    return render_template("index.html", prediction=prediction)

# Vercel entry point
def handler(event, context):
    return app(event, context)
