from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "detection"
    / "models"
    / "exoplanet_model.pkl"
)

model = joblib.load(MODEL_PATH)


def classify_candidate(period, depth, duration, snr):

    features = pd.DataFrame([{
        "koi_period": period,
        "koi_depth": depth,
        "koi_duration": duration,
        "koi_model_snr": snr
    }])

    prediction = model.predict(features)[0]

    confidence = model.predict_proba(features)[0][prediction] * 100

    return prediction, confidence