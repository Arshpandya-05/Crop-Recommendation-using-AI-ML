
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------- 1) Load data ----------
csv_path = Path("Crop_recommendation.csv")  
df = pd.read_csv(csv_path)

# normalize column names to handle case differences
df.columns = [c.strip().lower() for c in df.columns]

# expected feature columns (order matters & must match the form)
FEATURES = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]

# allow a couple of common alternate names
rename_map = {
    "nitrogen": "n",
    "phosphorus": "p",
    "phosporus": "p",   # typo sometimes present
    "potassium": "k",
}
df = df.rename(columns=rename_map)

missing = [c for c in FEATURES + ["label"] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

X = df[FEATURES].astype(float).values
y_raw = df["label"].astype(str).values

# ---------- 2) Encode labels (save encoder so Flask can inverse-transform) ----------
le = LabelEncoder()
y = le.fit_transform(y_raw)

# ---------- 3) Fit scalers ----------
mx = MinMaxScaler()
X_mx = mx.fit_transform(X)

sc = StandardScaler()
X_scaled = sc.fit_transform(X_mx)

# ---------- 4) Train model ----------
clf = RandomForestClassifier(
    n_estimators=300, random_state=42, n_jobs=-1, class_weight=None
)
Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
clf.fit(Xtr, ytr)

# quick sanity check
pred = clf.predict(Xte)
acc = accuracy_score(yte, pred)
print(f"Validation accuracy: {acc:.3f}")

# ---------- 5) Save everything ----------
joblib.dump(mx, "minmaxscaler.pkl")
joblib.dump(sc, "standscaler.pkl")
joblib.dump(clf, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

# also save the feature order so Flask can build vectors consistently
with open("feature_order.json", "w") as f:
    json.dump(FEATURES, f)

print("✅ Saved: model.pkl, standscaler.pkl, minmaxscaler.pkl, label_encoder.pkl, feature_order.json")
