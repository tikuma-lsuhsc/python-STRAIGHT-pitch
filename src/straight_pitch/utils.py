from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from numpy.typing import ArrayLike, NDArray

import numpy as np
from math import ceil, log2, floor
from scipy import signal as sps
from scipy import fft as spfft

next_fast_len = spfft.next_fast_len


def fftfilt(
    b: ArrayLike, x: ArrayLike, n: int | None = None, axis: int = -1
) -> NDArray:
    """FFT-based FIR filtering using overlap-add method (a partial port of MATLAB fftfilt)

    Parameters
    ----------
    b
        Filter coefficients, specified as a 1D real array. If b has a greater dimension,
        `fftfilt` applies the filter along the last dimension unless `axis` argument is
        specified
    x
        Input data, specified as a 1D real array. If `x` has a greater dimension, `fftfilt`
        applies the filter along the last dimension unless `axis` argument is specified.
        If `b` is also multidimensional, the non-filtering dimensions must match and
        filtering is applied along the specified axis.
    n, optional
        FFT length, by default None to choose an FFT length and a data block length that
        guarantee efficient execution time.
    axis, optional
        Filtering axis, by default -1 to choose the last axis. This applies to both
        `b` and `x` only if multidimensional.

    Returns
    -------
        Output data
    """

    # if dimension mismatches, reshape if 1D
    if x.ndim > 1 and b.ndim == 1:
        new_shape = np.ones_like(x.shape)
        new_shape[axis] = -1
        b = b.reshape(new_shape)
    elif b.ndim > 1 and x.ndim == 1:
        new_shape = np.ones_like(b.shape)
        new_shape[axis] = -1
        x = x.reshape(new_shape)
    elif x.ndim != b.ndim:
        raise ValueError(
            f"The shapes of `b` {b.shape} and `x` {x.shape} must match if multidimensional"
        )

    is_real = not (np.iscomplexobj(x) or np.iscomplexobj(b))
    fft = spfft.rfft if is_real else spfft.fft
    ifft = spfft.irfft if is_real else spfft.ifft

    nb: int = b.shape[axis]  # filter length
    nx: int = x.shape[axis]  # data length
    L: int = None  # data block size
    nblks: int = None  # number of data blocks

    if n is None:
        # minimize the number of blocks L times the number of flops per FFT (nfft*log nfft)
        nfft_max = next_fast_len(nb + nx - 1, real=is_real)
        if nx > nb:
            nfft = next_fast_len(2 * nb - 1, real=is_real)
            L = nb
            nblks = ceil(nx / L)
            c = nblks * nfft * floor(log2(nfft))
            nfft_next = next_fast_len(nfft + 1, real=is_real)
            while nfft_next <= nfft_max:
                L_next = nfft - nb + 1
                nblks_next = ceil(nx / L_next)
                c_next = nblks_next * nfft_next * log2(nfft_next)
                if c_next > c:
                    break
                c = c_next
                nfft = nfft_next
                L = L_next
                nblks = nblks_next
                nfft_next = next_fast_len(nfft + 1, real=is_real)
        else:
            nfft = nfft_max
            L = nx
    else:
        n = max(n, nb)
        nfft = next_fast_len(n)

    if L is None:
        L = nfft - nb + 1

    if L >= nx:
        # single-block op
        return ifft(fft(b, nfft) * fft(x, nfft))[:nx]

    # multiblock op

    if nblks is None:
        nblks = ceil(nx / L)

    y = np.zeros_like(x, dtype=x.dtype if is_real or np.iscomplexobj(x) else b.dtype)
    out_slice = [slice(None)] * x.ndim
    ifft_slice = [slice(None)] * x.ndim
    in_slice = [slice(None)] * x.ndim
    nout_full = L + nb - 1  # full output length

    # overlap-add all the blocks except for the last
    for i in range(nblks):
        ni = i * L
        in_slice[axis] = slice(ni, min(ni + L, nx))
        nout = min(nout_full, nx - ni)
        ifft_slice[axis] = slice(0, nout)
        out_slice[axis] = slice(ni, ni + nout)
        y[tuple(out_slice)] += ifft(fft(x[tuple(in_slice)], nfft) * fft(b, nfft))[
            tuple(ifft_slice)
        ]

    return y


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
