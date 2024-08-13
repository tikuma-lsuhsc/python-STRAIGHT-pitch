from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from numpy.typing import ArrayLike, NDArray
from numbers import Real

from math import ceil, floor, log2, pi, sqrt
import numpy as np
from scipy import signal as sps
from scipy.fft import rfft, next_fast_len


def refineF06(x, fs, f0raw, fftl, eta, nhmx, shiftm, nl, nu, imgi=1):
    """F0 estimation refinement
    [f0r,ecr]=refineF06(x,fs,f0raw,fftl,nhmx,shiftm,nl,nu,imgi)
            x		: input waveform
            fs		: sampling frequency (Hz)
            f0raw	: F0 candidate (Hz)
            fftl	: FFT length
            eta		: temporal stretch factor
            nhmx	: highest harmonic number
            shiftm	: frame shift period (ms)
            nl		: lower frame number
            nu		: uppter frame number
            imgi	: display indicator, 1: display on (default), 0: off

    Example of usage (with STRAIGHT)

    global xold fs f0shiftm f0raw

    dn=floor(fs/(800*3*2))
    [f0raw,ecr]=refineF02(decimate(xold,dn),fs/dn,f0raw,512,1.1,3,f0shiftm,1,len(f0raw))
    """
    # 	Designed and coded by Hideki Kawahara
    # 	28/July/1999
    # 	29/July/1999 test version using power weighting
    # 	30/July/1999 GcBs is added (bug fix)
    # 	07/August/1999 small bug fix
    #   07/Dec./2002 wqitbar was added
    #   13.May/2005 minor vulnerability fix
    # 	10/Aug./2005 modified by Takahashi on waitbar
    # 	10/Sept./2005 modified by Kawahara on waitbar
    #   16/Sept./2005 minor bug fix
    #   26/Sept./2005 bug fix

    f0i = f0raw
    f0i[f0i == 0] = 160
    fax = np.arange(fftl) / fftl * fs
    nfr = len(f0i)  # 07/August/1999

    shiftl = shiftm / 1000 * fs
    x = np.pad(x, (fftl, fftl))

    tt = (np.arange(fftl) - fftl / 2) / fs
    th = np.arange(fftl) / fftl * 2 * pi
    rr = np.exp(-1j * th)

    f0t = 100
    w1 = np.maximum(0, 1 - np.abs(tt * f0t / eta))
    w1 = w1[w1 > 0]
    wg = np.exp(-pi * (tt * f0t / eta) ** 2)
    wgg = wg[np.abs(wg) > 0.0002]
    wo = sps.lfilter(wgg, 1, np.pad(w1, (0, len(wgg))))

    xo = np.arange(len(wo)) / (len(wo) - 1)
    nlo = len(wo) - 1

    if nl * nu < 0:
        nl = 1
        nu = nfr

    bx = np.arange(fftl // 2 + 1, dtype=int)
    pif = np.zeros((fftl // 2 + 1, nfr))
    dpif = np.zeros((fftl // 2 + 1, nfr))
    pwm = np.zeros((fftl // 2 + 1, nfr))
    rmsValue = x.std()  # 26/Sept./2005 by HK

    # if imgi==1 hpg=waitbar(0,'F0 refinement using F0 adaptive analysis') end # 07/Dec./2002 by H.K.#10/Aug./2005
    for kk in range(nl, nu + 1):
        if f0i[kk] < 40:
            f0i[kk] = 40

        f0t = f0i[kk]
        xi = np.arange(0, 1, 1 / nlo * f0t / 100)
        wa = np.interp(xo, wo, xi, "*linear")
        wal = len(wa)
        bb = np.arange(wal)
        bias = round(fftl - wal / 2 + (kk - 1) * shiftl)
        if (
            x[bb + bias].std() * x[bb + bias - 1].std() * (bb + bias + 1).std() == 0
        ):  # 26/Sept./2005 by HK
            x[bb + bias] = np.random.randn(len(bb)) * rmsValue / 100000

        dcl = np.mean(x[bb + bias])
        ff0 = rfft((x[bb + bias - 1] - dcl) * wa, fftl)
        ff1 = rfft((x[bb + bias] - dcl) * wa, fftl)
        ff2 = rfft((x[bb + bias + 1] - dcl) * wa, fftl)
        fd = ff2 * rr - ff1
        fd0 = ff1 * rr - ff0
        crf = fax + (ff1.real * fd.imag - ff1.imag * fd.real) / (
            np.abs(ff1) ** 2
        ) * fs / (2 * pi)
        crf0 = fax + (ff0.real * fd0.imag - ff0.imag * fd0.real) / (
            np.abs(ff0) ** 2
        ) * fs / (pi * 2)
        pif[:, kk] = crf[bx] * 2 * pi
        dpif[:, kk] = (crf[bx] - crf0[bx]) * 2 * pi
        pwm[:, kk] = np.abs(ff1[bx])  # 29/July/1999
        # if imgi==1 waitbar((kk-nl)/(nu-nl)) end # ,hpg) # 07/Dec./2002 by H.K.#10/Aug./2005

    # if imgi==1 close(hpg) end

    slp = (np.concatenate([pif[1:, :], pif[fftl / 2, :]], axis=0) - pif) / (
        fs / fftl * 2 * pi
    )
    dslp = (
        (np.concatenate([dpif[1:, :], dpif[fftl / 2, :]], axis=0) - dpif)
        / (fs / fftl * 2 * pi)
        * fs
    )
    mmp = slp * 0

    [c1, c2] = znrmlcf2(shiftm)
    fxx = (np.arange(fftl / 2 + 1) + 0.5) / fftl * fs * 2 * pi

    # --- calculation of relative noise level

    # if imgi==1 hpg=waitbar(0,'P/N calculation') end # 07/Dec./2002 by H.K.#10/Aug./2005
    for ii in np.arange(fftl / 2 + 1):
        c2 = c2 * (fxx[ii] / 2 / pi) ^ 2
        mmp[ii, :] = (dslp[ii, :] / sqrt(c2)) ** 2 + (slp[ii, :] / sqrt(c1)) ** 2
        # if imgi==1 and rem(ii,10)==0: waitbar(ii/(fftl/2+1))end  # 07/Dec./2002 by H.K.#10/Aug./2005

    # if imgi==1 close(hpg) end # 07/Dec./2002 by H.K.#10/Aug./2005

    # --- Temporal smoothing

    sml = round(1.5 * fs / 1000 / 2 / shiftm) * 2 + 1  # 3 ms, and odd number
    smb = round((sml - 1) / 2)  # bias due to filtering

    # if imgi==1 hpg=waitbar(0,'P/N smoothing') end # 07/Dec./2002 by H.K.#10/Aug./2005

    # This smoothing is modified (30 Nov. 2000).
    win = sps.get_window("hann", sml)
    win2 = win**2
    xin = (
        np.pad(mmp, [(0, fftl // 2 + 1), (0, sml * 2)]).T
        + np.max(mmp(~mmp.isnan() & mmp.isinf())) * 0.0000001
    )
    smmp = sps.lfilter(win2 / np.sum(win2), 1, xin, axis=0).T  # 26/Sept./2005 by H.K.
    # if imgi==1 waitbar(0.2) end#10/Aug./2005
    smmp = 1.0 / sps.lfilter(win / np.sum(win), 1, 1 / smmp)
    # if imgi==1 waitbar(0.4) end
    smmp = smmp[
        :, np.maximum(1, np.arange(nfr) + sml - 2)
    ]  # fixed by H.K. on 10/Dec./2002

    # --- Power adaptive weighting (29/July/1999)

    spwm = sps.lfilter(
        win / np.sum(win), 1, np.pad(pwm, [(0, fftl // 2 + 1), (0, sml * 2)]) + 0.00001
    )
    # if imgi==1 waitbar(0.6) end#10/Aug./2005
    spfm = sps.lfilter(
        win / np.sum(win),
        1,
        np.pad(pwm * pif, [(0, fftl // 2 + 1), (0, sml * 2)]) + 0.00001,
    )
    # if imgi==1 waitbar(0.8) end#10/Aug./2005
    spif = spfm / spwm
    spif = spif[:, np.arange(nfr) + smb]

    idx = np.maximum(0, f0i / fs * fftl)
    fqv = np.zeros((nhmx, nfr))
    vvv = np.zeros((nhmx, nfr))

    iidx = np.arange(nfr) * (fftl / 2 + 1) + 1
    for ii in range(nhmx):
        iidx = idx + iidx
        vvv[ii, :] = (
            smmp(floor(iidx))
            + (iidx - floor(iidx)) * (smmp(floor(iidx) + 1) - smmp(floor(iidx)))
        ) / (ii * ii)
        fqv[ii, :] = (
            (
                spif(floor(iidx))
                + (iidx - floor(iidx)) * (spif(floor(iidx) + 1) - spif(floor(iidx)))
            )
            / 2
            / pi
            / ii
        )  # 29/July/199

    # if imgi==1 waitbar(1) end#10/Aug./2005
    vvvf = 1.0 / np.sum(1.0 / vvv)
    f0r = np.sum(fqv / sqrt(vvv)) / np.sum(1.0 / sqrt(vvv)) * (f0raw > 0)
    ecr = np.sqrt(1.0 / vvvf) * (f0raw > 0) + (f0raw <= 0)
    # if imgi==1 close(hpg) end#10/Aug./2005

    # keyboard

    return f0r, ecr


# --------------------
def znrmlcf2(f):

    n = 100
    x = np.arange(0, 3, 1 / n)
    g = GcBs(x, 0)
    dg = np.concatenate([np.diff(g) * n, 0])
    dgs = dg / 2 / pi / f
    xx = 2 * pi * f * x
    c1 = np.sum((xx * dgs) ** 2) / n
    c2 = np.sum((xx**2 * dgs) ** 2) / n

    return c1, c2


# ---------------------
def GcBs(x, k):

    tt = x + 0.0000001
    p = (
        tt**k
        * np.exp(-pi * tt**2)
        * (np.sin(pi * tt + 0.0001) / (pi * tt + 0.0001)) ** 2
    )

    return p
