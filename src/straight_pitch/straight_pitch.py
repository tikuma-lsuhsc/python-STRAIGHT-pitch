from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from numpy.typing import ArrayLike, NDArray
from numbers import Real

from math import ceil, floor, log2
import numpy as np
from scipy import signal as sps
from scipy.fft import next_fast_len

from .fixpF0VexMltpBG4 import fixpF0VexMltpBG4
from .f0track5 import f0track5
from .refineF06 import refineF06
from .utils import decimate, fftfilt

from matplotlib import pyplot as plt

# def exstraightsource(x,fs,optionalParams=None):
#   Source information extraction for STRAIGHT
#   [f0raw,ap,analysisParams]=exstraightsource(x,fs,optionalParams)
#   Input parameters
#   x   : input signal. if it is multi channel, only the first channel is used
#   fs  : sampling frequency (Hz)
#   optionalParams : Optional parameters for analysis
#   Output parameters
#   f0raw   : fundamental frequency (Hz)
#   ap  : amount of aperiodic component in the time frequency represntation
#       : represented in dB
#   analysisParams : Analysis parameters actually used
#
#   Usage:
#   Case 1: The simplest method
#   [f0raw,ap]=exstraightsource(x,fs)
#   Case 2: You can get to know what parameters were used.
#   [f0raw,ap,analysisParams]=exstraightsource(x,fs)
#   CAse 3: You can have full control of STRAIGHT synthesis.
#       Please use case 2 to find desired parameters to modify.
#   [f0raw,ap,analysisParams]=exstraightsource(x,fs,optionalParams)

#   Notes on programing style
#   This routine is based on the current (2005.1.31) implementation of
#   STRAIGHT that consist of many legacy fragments. They were intentionally
#   kept for maintaining historic record. Revised functions written in a
#   reasonable stylistic practice will be made available soon.

#   Designed and coded by Hideki Kawahara
#   15/January/2005
#   01/February/2005 extended for user control
# 	30/April/2005 modification for Matlab v7.0 compatibility


def straight_pitch(
    x: ArrayLike, fs: Real, optionalParams: dict | None = None
) -> NDArray:

    x = np.asarray(x)

    if x.ndim > 1:
        raise ValueError("x must be a 1D array.")

    if fs <= 0:
        raise ValueError(f"{fs=} must be strictly positive.")

    prm = zinitializeParameters()
    if optionalParams is not None:
        prm.update(optionalParams)

    #   Initialize default parameters
    f0floor = prm["F0searchLowerBound"]
    f0ceil = prm["F0searchUpperBound"]

    # nvo # Number of channels in one octave
    nvo = prm["NofChannelsInOctave"]
    nvc = ceil(log2(f0ceil / f0floor) * nvo)  # number of channels
    Fxx = f0floor * 2 ** (np.arange(nvc) / nvo)  # channel center frequencies in Hz

    # remove the low-frequency noise
    x = cleaninglownoise(x, fs, f0floor)

    # ---- F0 extraction based on a fixed-point method via filterbank
    f0v, vrv, dfv, aav, frame_step = fixpF0VexMltpBG4(
        x,
        Fxx / fs,
        prm["IFWindowStretch"],
        round(prm["F0frameUpdateInterval"] * 1e-3 * fs),
        prm["IFsmoothingLengthRelToFc"],
        round(prm["IFminimumSmoothingLength"] * 1e-3 * fs),
        prm["IFexponentForNonlinearSum"],
        prm["IFnumberOfHarmonicForInitialEstimate"],
    )

    plt.plot(f0v * fs, vrv, ".")
    plt.show()

    # ---- post processing for V/UV decision and F0 tracking
    flp: float = 125
    fhp: float = 3000
    pwt, pwh = zplotcpower(x, fs, frame_step)
    htr = 10 * np.log10(pwh / pwt)

    plt.plot(htr, ".-")
    plt.show()

    # 11/Sept./2005
    f0raw = f0track5(f0v, vrv, htr, frame_step / fs)
    f0t = f0raw
    f0t[f0t == 0] = np.nan

    # if imageOn
    #     avf0 = np.mean(f0raw[f0raw > 0])
    #     tt = np.arange(len(f0t))
    #     subplot(615)
    #     plot(tt*f0shiftm,f0t,'g')
    #     grid on
    #     if ~isnan(avf0)
    #         axis([1 max(tt)*f0shiftm ...
    #                 min(avf0/sqrt(2),0.95*min(f0raw(f0raw>0)))  ...
    #                 max(avf0*sqrt(2),1.05*max(f0raw(f0raw>0)))])
    #     end
    #     ylabel('F0 (Hz)')
    #     hold on
    # end

    # ---- F0 refinement

    # paramaters for F0 refinement
    # fftlf0r =   # fftlf0r=1024 # FFT length for F0 refinement
    # tstretch =   # tstretch=1.1 # time window stretching factor
    # nhmx =   # nhmx=3 # number of harmonic components for F0 refinement
    # iPeriodicityInterval = prm[
    #     "periodicityFrameUpdateInterval"
    # ]  # frame update interval for periodicity index (ms)

    # default frame length for pitch extraction (ms)
    framem = prm["F0defaultWindowLength"]

    framel = round(framem * fs / 1000)

    fftl = 1024  # default FFT length
    if fftl < framel:
        fftl = next_fast_len(framel)

    nstp = 1  # start position of F0 refinement (samples)
    nedp = len(f0raw)  # last position of F0 refinement (samples)
    dn = floor(fs / (f0ceil * 3 * 2))  # fix by H.K. at 28/Jan./2003
    [f0raw, ecr] = refineF06(
        decimate(x, dn),
        fs / dn,
        f0raw,
        prm["refineFftLength"],
        prm["refineTimeStretchingFactor"],
        prm["refineNumberofHarmonicComponent"],
        frame_step,
        nstp,
        nedp,
    )
    # 31/Aug./2004# 11/Sept.2005

    # if imageOn
    #     f0t=f0raw
    #     f0t(f0t==0)=f0t(f0t==0)*NaN
    #     tt=1:len(f0t)
    #     subplot(615)
    #     plot(tt*f0shiftm,f0t,'k')
    #     hold off
    #     drawnow
    # end
    # ----------- 31/July/1999
    ecrt = ecr
    ecrt[f0raw == 0] = np.nan

    # if imageOn
    #     tirms=irms
    #     tirms(f0raw==0)=tirms(f0raw==0)*NaN
    #     tirms(f0raw>0)=-20*log10(tirms(f0raw>0))
    #     subplot(616)
    # hrms=plot(tt*f0shiftm,tirms,'g',tt*f0shiftm,20*log10(ecrt),'r') #31/July/1999
    #     set(hrms,'LineWidth',2)
    # hold on
    #     plot(tt*f0shiftm,-10*log10(vrv),'k.')
    #     grid on
    # hold off
    #     axis([1 max(tt)*f0shiftm -10 60])
    #     xlabel('time (ms)')
    # ylabel('C/N (dB)')
    #     drawnow
    # end

    # -------------------------------------------------------------------------------------
    f0raw[f0raw <= 0] = 0  # safeguard 31/August/2004
    f0raw[f0raw > f0ceil] += f0ceil  # safeguard 31/August/2004

    return f0raw, prm

    # if nargout == 1 return end

    # #----- aperiodicity estimation
    # [apvq,dpvq,~,~]=aperiodicpartERB2(x,fs,f0raw,f0shiftm,iPeriodicityInterval,fftl/2+1,imageOn) # 10/April/2002$11/Sept./2005
    # apv=10*log10(apvq) # for compatibility
    # dpv=10*log10(dpvq) # for compatibility
    # #- ---------
    # #   Notes on aperiodicity estimation: The previous implementation of
    # #   aperiodicity estimation was sensitive to low frequency noise. It is a
    # #   bad news, because environmental noise usually has its power in the low
    # #   frequency region. The following corrction uses the C/N information
    # #   which is the byproduct of fixed point based F0 estimation.
    # #   by H.K. 04/Feb./2003
    # #- ---------
    # dpv=correctdpv(apv,dpv,iPeriodicityInterval,f0raw,ecrt,f0shiftm,fs) # Aperiodicity correction 04/Feb./2003 by H.K.

    # # if imageOn
    # #     bv=boundmes2(apv,dpv,fs,f0shiftm,iPeriodicityInterval,fftl/2+1)
    # #     figure
    # #     semilogy((0:len(bv)-1)*f0shiftm,0.5./10.0.^(bv))
    # #     grid on
    # #     set(gcf,'PaperPosition', [0.634517 0.634517 19.715 28.4084])
    # # end

    # ap=aperiodiccomp(apv,dpv,iPeriodicityInterval,f0raw,f0shiftm,imageOn) # 11/Sept./2005

    # switch nargout
    #     case 2
    #     case 3
    #         analysisParams=prm
    #     otherwise
    #         disp('Number of output parameters has to be 2 or 3!')
    # end

    # return f0raw,ap,analysisParams


###---- internal functions


def cleaninglownoise_filter(fs, f0floor, flm=50):
    flp = round(fs * flm / 1000)
    wlp = sps.firwin(flp * 2, f0floor / (fs / 2))
    wlp[flp] = wlp[flp] - 1
    wlp = -wlp
    return wlp, flp


# #--------------------------------------------
def cleaninglownoise(x, fs, f0floor, flm=50):

    wlp, flp = cleaninglownoise_filter(fs, f0floor, flm)

    return fftfilt(wlp, np.pad(x, (0, 2 * flp)))[flp:-flp]


def zplotcpower(
    x: NDArray, flp: float, fhp: float, shiftm: int = 1
) -> tuple[NDArray, NDArray, NDArray]:
    """signal power calculator

    Parameters
    ----------
    x
        ac signal
    flp
        lowpass filter cutoff in fractional frequency for total power calculation
    fhp
        highpass filter cutoff in fractional frequency for highpass power calculation
    shiftm
        downsampling factor

    Returns
    -------
        htr - pwh/pw
        pwh - high-frequency power
        pw  - total power
    """
    # lowpass filter to collect power
    nfl = round(1 / flp)
    nf = 2 * nfl + 1
    w = sps.firwin(nf, flp, pass_zero="lowpass", fs=1, window="hann")
    b = sps.firwin(nf, fhp, pass_zero="highpass", fs=1, window="hann")

    # squaring to get instantaneous power and pad zeros to account for the group delay
    y = np.pad(x**2, (0, 2 * nfl))
    pw = fftfilt(w, y)[nfl:-nfl:shiftm]

    xh = fftfilt(b, np.pad(x, (0, 4 * nfl)))
    pwh = fftfilt(w, xh**2)[2 * nfl : -2 * nfl : shiftm]

    #
    htr = np.zeros_like(pw)
    tf = (pw > 0.0) & (pwh > 0.0)
    htr[tf] = pwh[tf] / pw[tf]

    return htr, pwh, pw


###------
def zinitializeParameters():
    return dict(
        F0searchLowerBound=40,  # f0floor
        F0searchUpperBound=800,  # f0ceil
        F0defaultWindowLength=80,  # default frame length for pitch extraction (ms)
        F0frameUpdateInterval=1,  # shiftm # F0 calculation interval (ms)
        NofChannelsInOctave=24,  # nvo=24 # Number of channels in one octave
        IFWindowStretch=1.2,  # mu=1.2 # window stretch from isometric window
        DisplayPlots=0,  # imgi=1 # image display indicator (1: display image)
        IFsmoothingLengthRelToFc=1,  #  smp=1 # smoothing length relative to fc (ratio)
        IFminimumSmoothingLength=5,  #  minm=5 # minimum smoothing length (ms)
        IFexponentForNonlinearSum=0.5,  # pc=0.5 # exponent to represent nonlinear summation
        IFnumberOfHarmonicForInitialEstimate=1,  # nc=1 # number of harmonic component to use (1,2,3)
        refineFftLength=1024,  # fftlf0r=1024 # FFT length for F0 refinement
        refineTimeStretchingFactor=1.1,  # tstretch=1.1 # time window stretching factor
        refineNumberofHarmonicComponent=3,  # nhmx=3 # number of harmonic components for F0 refinement
        periodicityFrameUpdateInterval=5,  # frame update interval for periodicity index (ms)return
        note=" ",
    )  # Any text to be printed on the source information plot
