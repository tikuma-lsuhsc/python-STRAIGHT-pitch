from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from numpy.typing import ArrayLike, NDArray
from numbers import Real

from math import ceil, floor, log2, pi, sqrt
import numpy as np
from scipy import signal as sps

from matplotlib import pyplot as plt


def multanalytFineCSPB(
    x: NDArray,
    fs: float,
    f0floor: float,
    nvc: int,
    nvo: int,
    mu: float,
    mlt: int,
    imgi: bool = False,
):
    """Dual waveleta analysis using cardinal spline manipulation

    Parameters
    ----------
    x
        input signal (2kHz sampling rate is sufficient.)
    fs
        sampling frequency (Hz)
    f0floor
        lower bound for pitch search (60Hz suggested)
    nvc
        number of total voices for wavelet analysis
    nvo
        number of voices in an octave
    mu
        temporal stretch factor
    mlt
        harmonic ID#
    imgi, optional
        True to turn on display, by default False

    Returns
    -------
        pm: wavelet transform using iso-metric Gabor function
    """

    # If you have any questions,  mailto:kawahara@hip.atr.co.jp

    # Copyright (c) ATR Human Information Processing Research Labs. 1996
    # Invented and coded by Hideki Kawahara
    # 30/Oct./1996
    # 07/Dec./2002 waitbar was added
    # 10/Aug./2005 modified by Takahashi on waitbar
    # 10/Sept./2005 modified by Kawahara on waitbar

    # 10/Sept./2005
    t0 = 1 / f0floor
    lmx = round(6 * t0 * fs * mu)
    wl = 2 ** ceil(log2(lmx))

    nx = len(x)
    tx = np.pad(x, (0, wl))
    gent = (np.arange(wl) - wl / 2) / fs

    pm = np.zeros((nvc, nx), complex)
    mpv = 1
    # if imgi==1:
    #     hpg=waitbar(0,['wavelet analysis for initial F0 ' ...
    #         'and P/N estimation with HM#:' num2str(mlt)])

    # 07/Dec./2002 by H.K.#10/Aug./2005
    for ii in range(nvc):
        tb = gent * mpv
        t = tb[np.abs(tb) < 3.5 * mu * t0]
        wd1 = np.exp(-pi * (t / t0 / mu) ** 2)
        # wd1 = wd1 * np.exp((2j * pi * mlt) * t / t0)
        wd2 = np.maximum(0, 1 - np.abs(t / t0 / mu))
        wd2 = wd2[wd2 > 0]
        wwd = np.convolve(wd2, wd1)
        wwd = wwd[np.abs(wwd) > 0.00001]
        wbias = (len(wwd) - 1) // 2
        wwd = wwd * np.exp(
            (2j * pi * mlt)
            * t[np.round(np.arange(len(wwd)) - wbias + len(t) / 2).astype(int)]
            / t0
        )
        pmtmp1 = sps.lfilter(wwd, 1, tx)  # fftfilt(wwd,tx)
        pm[ii, :] = pmtmp1[wbias : wbias + nx] * np.sqrt(mpv)
        mpv = mpv * (2.0 ** (1 / nvo))
        # if imgi==1
        #     waitbar(ii/nvc)

        # ,hpg)
        # 07/Dec./2002 by H.K.#10/Aug./2005

    # if imgi==1:
    #     close(hpg)
    # 07/Dec./2002 by H.K.#10/Aug./2005

    return pm
