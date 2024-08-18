from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from functools import lru_cache

from numpy.typing import NDArray

from math import pi
import numpy as np

from .utils import fftfilt


@lru_cache(512)
def build_filter(f0, f0_eta, nw, nh, wbias, npad, mlt):
    n = np.arange(0.5, nw)
    # w1 = np.exp(pi * (2j * f0 * n - (f0_eta * n) ** 2))
    w1 = np.exp(-pi * (f0_eta * n) ** 2)
    w = np.empty(2 * nw)
    w[:nw], w[nw:] = w1[::-1], w1

    n1 = np.arange(0.5, nh)
    h1 = 1 - f0_eta * n1
    h = np.empty(2 * nh)
    h[:nh], h[nh:] = h1[::-1], h1

    w2 = np.exp(2j * pi * mlt * f0 * np.arange(-wbias, wbias + 1))
    return np.pad(np.convolve(w, h) * w2, (npad, npad))


def build_filters(f0, f0_eta, nw, nh, mlt):
    wbiases = nw + nh - 1
    wbias_max = wbiases.max()
    npads = wbias_max - wbiases
    return (
        np.stack(
            [
                build_filter(*args, mlt)
                for args in zip(f0, f0_eta, nw, nh, wbiases, npads)
            ]
        ),
        wbias_max,
    )


def multanalytFineCSPB(
    x: NDArray, f0: NDArray, eta: float, mlt: int
) -> NDArray[np.complex128]:
    """Dual waveleta analysis using cardinal spline manipulation

    Parameters
    ----------
    x
        n-sample input signal (2kHz sampling rate is sufficient.)
    f0
        size-m normalized fundamental frequency vector
    eta
        temporal stretch factor

    Returns
    -------
        pm: mxn-array wavelet transform using isometric Gabor function
    """

    f0_eta = f0 / eta

    nw = np.ceil(3.5 / f0_eta).astype(int)
    nh = np.ceil(1 / f0_eta).astype(int)
    W, wbias = build_filters(f0, f0_eta, nw, nh, mlt)

    f0min = np.min(f0)
    c = np.sqrt(f0 / f0min).reshape(-1, 1)

    return fftfilt(W, np.pad(x, (0, 2 * wbias)))[:, wbias:-wbias] * c
