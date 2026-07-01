import numpy as np

def remove_nan(time, flux, quality):
    mask = np.isfinite(time) & np.isfinite(flux)
    return time[mask], flux[mask], quality[mask]


def remove_bad_quality(time, flux, quality):
    mask = (quality == 0)
    return time[mask], flux[mask]


def normalize_flux(flux):
    return flux / np.median(flux)

from scipy.signal import savgol_filter

def detrend_flux(flux, window_length=301, polyorder=2):
    trend = savgol_filter(flux, window_length, polyorder)
    return flux / trend

def preprocess_lightcurve(time, flux, quality):
    time, flux, quality = remove_nan(time, flux, quality)
    time, flux = remove_bad_quality(time, flux, quality)
    flux = normalize_flux(flux)
    flux = detrend_flux(flux)

    return time, flux