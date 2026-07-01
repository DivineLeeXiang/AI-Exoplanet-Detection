from astropy.timeseries import BoxLeastSquares
import numpy as np

from astropy.timeseries import BoxLeastSquares
import numpy as np


def detect_transit(time, flux):

    model = BoxLeastSquares(time, flux)

    periods = np.linspace(0.5, 10, 5000)

    durations = np.linspace(0.02, 0.30, 20)

    result = model.power(periods, durations)

    return result

