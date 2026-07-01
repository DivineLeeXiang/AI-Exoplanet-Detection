import numpy as np

# Assumption: Sun-like host star
R_SUN_REARTH = 109.2


def estimate_depth(flux):
    return np.median(flux) - np.min(flux)


def estimate_snr(flux, depth):
    """
    Estimate Signal-to-Noise Ratio.
    """

    noise = np.std(flux)

    if noise == 0:
        return 0.0

    return depth / noise


def estimate_radius(depth):
    """
    Estimate planet radius assuming a Sun-sized star.
    """

    if depth <= 0:
        return np.nan

    return np.sqrt(depth) * R_SUN_REARTH


def estimate_semi_major_axis(period_days):
    """
    Kepler's Third Law
    assuming a Sun-like star.
    """

    period_years = period_days / 365.25

    return period_years ** (2 / 3)


def estimate_parameters(result, flux):

    best = np.argmax(result.power)

    period = result.period[best]
    power = result.power[best]
    duration = result.duration[best]

    depth = estimate_depth(flux)

    snr = estimate_snr(flux, depth)

    radius = estimate_radius(depth)

    semi_major = estimate_semi_major_axis(period)

    return {
        "period": period,
        "power": power,
        "depth": depth,
        "duration": duration,
        "snr": snr,
        "radius_rearth": radius,
        "semi_major_axis_au": semi_major
    }