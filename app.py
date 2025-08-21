from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import json
import os
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------- 1) Load data ----------
csv_path = Path("Crop_recommendation.csv")  
df = pd.read_csv(csv_path)

model = joblib.load("model.pkl")
sc = joblib.load("standscaler.pkl")
mx = joblib.load("minmaxscaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

with open("feature_order.json", "r") as f:
    FEATURE_ORDER = json.load(f)  

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

def _parse_float(name: str, value: str):
    """Parse a single float with a helpful error if empty or bad."""
    try:
        return float(value)
    except Exception:
        raise ValueError(f"Invalid number for '{name}'")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        form = request.form

        form_map = {
            "Nitrogen": "n",
            "Phosporus": "p",   # (your form has this spelling)
            "Potassium": "k",
            "Temperature": "temperature",
            "Humidity": "humidity",
            "pH": "ph",
            "Rainfall": "rainfall",
        }


        feat_dict = {}
        for form_key, canon_key in form_map.items():
            if form_key not in form:
                raise ValueError(f"Missing form field '{form_key}'")
            feat_dict[canon_key] = _parse_float(form_key, form[form_key])

        # construct the feature vector in the exact order used during training
        feature_list = [feat_dict[k] for k in FEATURE_ORDER]
        single = np.array(feature_list, dtype=float).reshape(1, -1)

        # same preprocessing as training: MinMax -> StandardScaler
        x_mx = mx.transform(single)
        x_scaled = sc.transform(x_mx)

        # predict & decode to label name
        y_pred = model.predict(x_scaled)[0]
        crop_name = label_encoder.inverse_transform([y_pred])[0]

        result = f"{crop_name} is the best crop to be cultivated right there."

    except Exception as e:
        result = f"Error: {e}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    # make sure we are loading the intended files
    for f in ["model.pkl", "standscaler.pkl", "minmaxscaler.pkl", "label_encoder.pkl", "feature_order.json"]:
        assert os.path.exists(f), f"Missing file: {f}"
    app.run(debug=True)


