from ts2vg import HorizontalVG
import numpy as np


def generate_hvg_series(time_series: np.ndarray) -> np.ndarray:
    """
    Function for forming hvg series
        :param time_series: numpy array containing time series
        :return: numpy array containing hvg
    """

    hvg_series = []
    for series in time_series:
        g = HorizontalVG(directed=None).build(series)
        hvg_series.append(list(g.degrees))

    return np.array(hvg_series)
