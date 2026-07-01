import os

from src.load_data import load_lightcurve
from src.preprocess import preprocess_lightcurve
from src.detect import detect_transit
from src.estimate import estimate_parameters
from src.classify import classify_candidate


def analyze_star(filepath):

    # Load FITS data
    data = load_lightcurve(filepath)

    time = data["TIME"]
    flux = data["FLUX"]
    quality = data["QUALITY"]

    # Preprocess
    time, flux = preprocess_lightcurve(time, flux, quality)

    # Detect transit
    result = detect_transit(time, flux)

    # Estimate parameters
    params = estimate_parameters(result, flux)

    # Store filename
    params["file"] = os.path.basename(filepath)

    # AI Classification
    prediction, confidence = classify_candidate(
    params["period"],
    params["depth"],
    params["duration"],
    params["snr"]
)
    params["prediction"] = int(prediction)

    if prediction == 1:
        params["status"] = "Likely Exoplanet"
    else:
        params["status"] = "False Positive"

    params["confidence"] = round(confidence, 2)

    return {
        "params": params,
        "time": time,
        "flux": flux,
        "result": result
    }