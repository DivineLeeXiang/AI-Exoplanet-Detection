from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

csv_path = BASE_DIR / "cumulative_2020.12.30_14.14.11.csv"

df = pd.read_csv(csv_path, comment="#")

# Keep only confirmed planets and false positives
df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]

# Use four features
features = [
    "koi_period",
    "koi_depth",
    "koi_duration",
    "koi_model_snr"
]

X = df[features]

y = df["koi_disposition"].map({
    "FALSE POSITIVE": 0,
    "CONFIRMED": 1
})

# Remove missing values
mask = X.notnull().all(axis=1)
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print()
print(classification_report(y_test, pred))

MODEL_DIR = BASE_DIR.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(model, MODEL_DIR / "exoplanet_model.pkl")

print("\nModel updated successfully!")