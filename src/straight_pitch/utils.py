from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from numpy.typing import ArrayLike, NDArray

from math import ceil
from scipy import signal as sps


# based on dio.py in python-WORLD (MIT)
# https://github.com/tuanad121/Python-WORLD/blob/e65cf3010b68efae2898a538b629c35c5c528db1/world/dio.py#L330
def decimate(x: ArrayLike, q: int, n: int = 8, axis: int = -1) -> NDArray:
    """MATLAB decimate clone (no `'fir'` option supported)
    :param x: signal
    :param q: decimation factor
    :param n: filter order
    :param axis: The axis of x to which the filter is applied. Default is -1.

    :returns: resampled signal
    """

    if q <= 0:
        raise ValueError("q must be a positive integer")

    if n <= 0:
        raise TypeError("n must be a positive integer")

    system = sps.dlti(*sps.cheby1(n, 0.05, 0.8 / q))

    # zero_phase = True

    y = sps.filtfilt(
        system.num,
        system.den,
        x,
        axis=axis,
        padlen=3 * (max(len(system.den), len(system.num)) - 1),
    )

    # make it the same as matlab
    nd = len(y)
    n_out = ceil(nd / q)
    n_beg = q - (q * n_out - nd)
    return y[n_beg - 1 :: q]
