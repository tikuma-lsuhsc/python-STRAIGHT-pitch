from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from functools import lru_cache

from numpy.typing import ArrayLike, NDArray
from numbers import Real

from math import ceil, floor, log2, pi, sqrt
import numpy as np
from scipy import signal as sps

from matplotlib import pyplot as plt


@lru_cache(512)
def get_filter(f0i, f0_eta_i, nwmaxi, nhmaxi, mlt):
    n = np.arange(0.5, nwmaxi)
    w = np.exp(pi * (2j * f0i * n - (f0_eta_i * n) ** 2))
    w = np.exp(-pi * (f0_eta_i * n) ** 2)

    n1 = np.arange(0.5, nhmaxi)
    h = 1 - f0_eta_i * n1

    wbias = nwmaxi + nhmaxi - 1
    return np.convolve(
        np.concatenate([w[-1::-1], w]), np.concatenate([h[-1::-1], h])
    ) * np.exp(2j * pi * mlt * f0i * np.arange(-wbias, wbias + 1))


def run_filter(x, nx, f0i, f0_eta_i, nwmaxi, nhmaxi, mlt):
    wwd = get_filter(f0i, f0_eta_i, nwmaxi, nhmaxi, mlt)
    wbias = nwmaxi + nhmaxi
    pmtmp1 = sps.lfilter(wwd, 1, x[: nx + 2 * wbias])
    return pmtmp1[wbias:-wbias]


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

    # If you have any questions,  mailto:kawahara@hip.atr.co.jp

    # Copyright (c) ATR Human Information Processing Research Labs. 1996
    # Invented and coded by Hideki Kawahara
    # 30/Oct./1996
    # 07/Dec./2002 waitbar was added
    # 10/Aug./2005 modified by Takahashi on waitbar
    # 10/Sept./2005 modified by Kawahara on waitbar

    # 10/Sept./2005

    f0min = np.min(f0)
    f0_eta = f0 / eta

    nwmax = np.ceil(3.5 / f0_eta).astype(int)
    nhmax = np.ceil(1 / f0_eta).astype(int)

    wl = 2 * (nwmax[0] + nhmax[0])
    tx = np.pad(x, (0, wl))

    # if imgi==1:
    #     hpg=waitbar(0,['wavelet analysis for initial F0 ' ...
    #         'and P/N estimation with HM#:' num2str(mlt)])

    # 07/Dec./2002 by H.K.#10/Aug./2005
    nx = len(x)
    pm = np.zeros((len(f0), nx), complex)
    for i, (f0i, f0_eta_i, nwmaxi, nhmaxi) in enumerate(zip(f0, f0_eta, nwmax, nhmax)):
        pm[i] = run_filter(tx, nx, f0i, f0_eta_i, nwmaxi, nhmaxi, mlt)

    pm = pm * np.sqrt(f0 / f0min).reshape(-1, 1)

    return pm
