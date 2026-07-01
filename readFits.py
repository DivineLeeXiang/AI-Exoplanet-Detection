from src.load_data import load_lightcurve
from src.preprocess import remove_nan, remove_bad_quality, normalize_flux

file = r"C:\AI Exoplanet Detection\data\raw\tess2026015060000-s1751-0000000018941472-0301-a_fast-lc.fits"

data = load_lightcurve(file)

time = data["TIME"]
flux = data["PDCSAP_FLUX"]
quality = data["QUALITY"]

time, flux, quality = remove_nan(time, flux, quality)
time, flux = remove_bad_quality(time, flux, quality)
flux = normalize_flux(flux)

print(len(time))
print(flux[:5])

from src.detect import detect_transit

result = detect_transit(time, flux)

best = result.power.argmax()

print("Best Period:", result.period[best])
print("Detection Power:", result.power[best])

from src.visualize import plot_periodogram

plot_periodogram(result)