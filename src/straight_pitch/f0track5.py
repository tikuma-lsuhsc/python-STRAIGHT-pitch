from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from numpy.typing import ArrayLike, NDArray
from numbers import Real

from math import ceil, floor, log2, pi, sqrt
import numpy as np
from scipy import signal as sps


def f0track5(f0v, vrv, dfv, pwt, pwh, aav, shiftm:int, imgi=1):
    """F0 trajectory tracker

    [f0,irms,df,amp]=f0track2(f0v,vrv,dfv,shiftm,imgi)
            f0	: extracted F0 (Hz)
            irms	: relative interfering energy in rms

    f0v	: fixed point frequency vector
    vrv	: relative interfering energy vector
    dfv	: fixed point slope vector
    pwt	: total power
    pwh	: power in higher frequency range
    aav	: amplitude list for fixed points
    shiftm	: frame update period (s)
    imgi	: display indicator, 1: display on (default), 0: off

    This is a very primitive and conventional algorithm.
    """
    # 	coded by Hideki Kawahara
    # 	copyright(c) Wakayama University/CREST/ATR
    # 	10/April/1999 first version
    # 	17/May/1999 relative fq jump thresholding
    # 	01/August/1999 parameter tweeking
    #   07/Dec./2002 waitbar was added
    #   13/Jan./2005 bug fix on lines 58, 97 (Thanx Ishikasa-san)
    # 	30/April/2005 modification for Matlab v7.0 compatibility
    # 	10/Aug./2005 modified  by Takahashi on waitbar
    # 	10/Sept./2005 modified by Kawahara on waitbar

    vrv = np.sqrt(vrv)
    mm = min(vrv.shape[1], len(pwt))

    f0 = np.zeros(mm)
    irms = np.ones(mm)
    df = np.ones(mm)
    amp = np.zeros(mm)
    von = 0

    ixx = np.argmin(vrv, axis=0)
    mxvr = np.take_along_axis(vrv, np.expand_dims(ixx, axis=0), axis=0).squeeze(axis=0)

    hth = 0.12  # highly confident voiced threshould (updated on 01/August/1999)
    lth = 0.9  # threshold to loose confidence
    bklm = 0.100  # back track length for voicing decision
    lalm = 0.010  # look ahead length for silence decision
    bkls = bklm / shiftm
    lals = lalm / shiftm
    htr = 10 * np.log10(pwh / pwt)

    thf0j = 0.04 * 10**-1.5 * sqrt(shiftm)  #  4 # of F0 is the limit of jump
    f0ref = 0
    htrth = -2.0  # was -3 mod 2002.6.3

    # if imgi==1 hpg=waitbar(0,'F0 tracking') end # 07/Dec./2002 by H.K.#10/Aug./2005

    ii = 0
    while ii < mm:
        if (von == 0) and (mxvr[ii] < hth) and (htr[ii] < htrth):
            von = 1
            f0ref = f0v[ixx[ii], ii]  # start value for search
            for jj in range(ii, max(0, ii - bkls), -1):
                gom = np.abs((f0v[:, jj] - f0ref) / f0ref)
                jxx = np.argmin(gom)
                gomi = gom[jxx]
                gomi = gomi + (f0ref > 10000) + (f0v[jxx, jj] > 10000)
                if (
                    ((gomi > thf0j) or (vrv[jxx, jj] > lth) or (htr[jj] > htrth))
                    and (f0v[jxx, jj] < 1000)
                ) and htr[jj] > -18:
                    # print(['break pt1 at ' num2str[jj]])
                    break

                if gomi > thf0j:
                    # print(['break pt2 at ' num2str[jj]])
                    break

                f0[jj] = f0v[jxx, jj]
                irms[jj] = vrv[jxx, jj]
                df[jj] = dfv[jxx, jj]
                amp[jj] = aav[jxx, jj]
                f0ref = f0[jj]

            f0ref = f0v[ixx[ii], ii]

        if (f0ref > 0) and (f0ref < 10000):
            gom = np.abs((f0v[:, ii] - f0ref) / f0ref)
            jxx = np.argmin(gom)
            gomi = gom[jxx]
        else:
            gomi = 10

        if (von == 1) and (mxvr[ii] > hth):
            for jj in range(ii, min(mm, ii + lals)):
                ii = jj
                gom = np.abs((f0v[:, ii] - f0ref) / f0ref)
                jxx = np.argmin(gom)
                gomi = gom[jxx]
                gomi = gomi + (f0ref > 10000) + (f0v[jxx, ii] > 10000)
                if (gomi < thf0j) and ((htr[ii] < htrth) or (f0v[jxx, ii] >= 1000)):
                    f0[ii] = f0v[jxx, ii]
                    irms[ii] = vrv[jxx, ii]
                    df[ii] = dfv[jxx, ii]
                    amp[ii] = aav[jxx, ii]
                    f0ref = f0[ii]

                if (
                    (gomi > thf0j)
                    or (vrv[jxx, ii] > lth)
                    or ((htr[ii] > htrth) and (f0v[jxx, ii] < 1000))
                ):
                    von = f0ref = 0
                    break

        elif (
            (von == 1)
            and (gomi < thf0j)
            and ((htr[ii] < htrth) or (f0v[jxx, ii] >= 1000))
        ):
            f0[ii] = f0v[jxx, ii]
            irms[ii] = vrv[jxx, ii]
            df[ii] = dfv[jxx, ii]
            amp[ii] = aav[jxx, ii]
            f0ref = f0[ii]
        else:
            von = 0

        # if imgi==1 waitbar(ii/mm) end #,hpg) # 07/Dec./2002 by H.K.#10/Aug./2005
        ii += 1

    # if imgi==1 close(hpg) end#10/Aug./2005

    return f0, irms, df, amp
