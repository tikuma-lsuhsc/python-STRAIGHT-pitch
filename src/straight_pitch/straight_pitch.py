from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from numpy.typing import ArrayLike, NDArray
from numbers import Real

from math import ceil, floor, log2
import numpy as np
from scipy import signal as sps

from .fixpF0VexMltpBG4 import fixpF0VexMltpBG4
from .f0track5 import f0track5
from .refineF06 import refineF06

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


def straight_pitch(x: ArrayLike, fs: Real, optionalParams: dict | None = None)->NDArray:

    if fs<=0:
        raise ValueError(f'{fs=} must be strictly positive.')

    prm = zinitializeParameters()
    if optionalParams is not None:
        prm.update(optionalParams)

    #   Initialize default parameters
    f0floor = prm["F0searchLowerBound"]  # f0floor
    f0ceil = prm["F0searchUpperBound"]  # f0ceil
    framem = prm[
        "F0defaultWindowLength"
    ]  # default frame length for pitch extraction (ms)
    f0shiftm = prm["F0frameUpdateInterval"]  # shiftm # F0 calculation interval (ms)

    fftl = 1024  # default FFT length

    framel = framem * fs / 1000

    if fftl < framel:
        fftl = 2 ^ ceil(log2(framel))

    x = np.asarray(x)

    if x.ndim > 1:
        raise ValueError("x must be a 1D array.")

    nvo = prm["NofChannelsInOctave"]  # nvo=24 # Number of channels in one octave
    mu = prm["IFWindowStretch"]  # mu=1.2 # window stretch from isometric window
    imageOn = prm["DisplayPlots"]  # imgi=1 # image display indicator (1: display image)
    smp = prm[
        "IFsmoothingLengthRelToFc"
    ]  #  smp=1 # smoothing length relative to fc (ratio)
    minsm = prm["IFminimumSmoothingLength"]  #  minm=5 # minimum smoothing length (ms)
    pcf0 = prm[
        "IFexponentForNonlinearSum"
    ]  # pc=0.5 # exponent to represent nonlinear summation
    nh = prm[
        "IFnumberOfHarmonicForInitialEstimate"
    ]  # nc=1 # number of harmonic component to use (1,2,3)
    fname = prm["note"]  # =' ' # Any text to be printed on the source information plot

    nvc = ceil(log2(f0ceil / f0floor) * nvo)  # number of channels

    # paramaters for F0 refinement
    fftlf0r = prm["refineFftLength"]  # fftlf0r=1024 # FFT length for F0 refinement
    tstretch = prm[
        "refineTimeStretchingFactor"
    ]  # tstretch=1.1 # time window stretching factor
    nhmx = prm[
        "refineNumberofHarmonicComponent"
    ]  # nhmx=3 # number of harmonic components for F0 refinement
    iPeriodicityInterval = prm[
        "periodicityFrameUpdateInterval"
    ]  # frame update interval for periodicity index (ms)

    # ---- F0 extraction based on a fixed-point method in the frequency domain

    f0v, vrv, dfv, _, aav = fixpF0VexMltpBG4(
        x, fs, f0floor, nvc, nvo, mu, imageOn, f0shiftm, smp, minsm, pcf0, nh
    )
    # if imageOn
    #     title([fname '  ' datestr(now,0)])
    #     drawnow

    # ---- post processing for V/UV decision and F0 tracking
    pwt, pwh = zplotcpower(x, fs, f0shiftm, imageOn)

    f0raw, irms, *_ = f0track5(
        f0v, vrv, dfv, pwt, pwh, aav, f0shiftm, imageOn
    )  # 11/Sept./2005
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
    nstp = 1  # start position of F0 refinement (samples)
    nedp = len(f0raw)  # last position of F0 refinement (samples)
    dn = floor(fs / (f0ceil * 3 * 2))  # fix by H.K. at 28/Jan./2003
    [f0raw, ecr] = refineF06(
        sps.decimate(x, dn),
        fs / dn,
        f0raw,
        fftlf0r,
        tstretch,
        nhmx,
        f0shiftm,
        nstp,
        nedp,
        imageOn,
    )  # 31/Aug./2004# 11/Sept.2005

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
def zplotcpower(x, fs, shiftm, imageOn):

    flm = 8  # 01/August/1999
    fl = round(flm * fs / 1000)
    w = sps.get_window("hann", 2 * fl + 1)
    w = w / w.sum()
    nn = len(x)

    flpm = 40
    flp = round(flpm * fs / 1000)
    wlp = sps.firwin(flp * 2, 70 / (fs / 2))
    wlp[flp] -= 1
    wlp = -wlp

    tx = np.pad(x, (0, 2 * len(wlp)))
    ttx = sps.lfilter(wlp, 1, tx)  # fftfilt(wlp,tx)
    ttx = ttx[flp : nn + flp]
    tx = np.pad(ttx, (0, 2 * len(w)))

    pw = sps.lfilter(w, 1, tx**2)  # fftfilt(w,tx**2)
    pw = pw[fl : nn + fl]
    mpw = np.max(pw)
    pw = pw[np.round(np.arange(0, nn, shiftm * fs / 1000)).astype(int)]
    pw[pw < mpw / 10000000] += mpw / 10000000  # safeguard 15/Jan./2003

    b = sps.firwin(2 * fl + 1, [0.0001, 3000 / (fs / 2)])
    b[fl] -= -1
    xh = sps.lfilter(b, 1, tx)  # fftfilt(b,tx)
    xh = xh[fl : nn + fl]
    tx = np.pad(xh, (0, 10 * len(w)))
    pwh = sps.lfilter(w, 1, tx**2)  # fftfilt(w,tx**2)
    pwh = pwh[fl : nn + fl]
    pwh = pwh[np.round(np.arange(0, nn, shiftm * fs / 1000)).astype(int)]
    pwh[pwh < mpw / 10000000] += mpw / 10000000
    # safeguard 15/Jan./2003

    # if imageOn
    #     subplot(614)
    #     tt=1:len(pw)
    #     hhg=plot(tt*shiftm,10*log10(pw),'b')
    #     hold on
    #     plot(tt*shiftm,10*log10(pwh),'r')
    #     grid on
    #     hold off
    #     set(hhg,'LineWidth',2)
    #     mp=max(10*log10(pw))
    #     axis([0 max(tt)*shiftm mp-70 mp+5])
    #     ylabel('level (dB)')
    #     title('thick line: total power thin line:high fq. power (>3kHz) ')

    return pw, pwh


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


###--------
# def replaceSuppliedParameters(prmin):
#     prm=zinitializeParameters
#     if isfield(prmin,'F0searchLowerBound')==1
#         prm['F0searchLowerBound=prmin.F0searchLowerBound']
#     end
#     if isfield(prmin,'F0searchUpperBound')==1
#         prm['F0searchUpperBound=prmin.F0searchUpperBound']
#     end
#     if isfield(prmin,'F0defaultWindowLength')==1
#         prm['F0defaultWindowLength=prmin.F0defaultWindowLength']
#     end
#     if isfield(prmin,'F0frameUpdateInterval')==1
#         prm['F0frameUpdateInterval=prmin.F0frameUpdateInterval']
#     end
#     if isfield(prmin,'NofChannelsInOctave')==1
#         prm['NofChannelsInOctave=prmin.NofChannelsInOctave']
#     end
#     if isfield(prmin,'IFWindowStretch')==1
#         prm['IFWindowStretch=prmin.IFWindowStretch']
#     end
#     if isfield(prmin,'DisplayPlots')==1
#         prm['DisplayPlots=prmin.DisplayPlots']
#     end
#     if isfield(prmin,'IFsmoothingLengthRelToFc')==1
#         prm['IFsmoothingLengthRelToFc=prmin.IFsmoothingLengthRelToFc']
#     end
#     if isfield(prmin,'IFminimumSmoothingLength')==1
#         prm['IFminimumSmoothingLength=prmin.IFminimumSmoothingLength']
#     end
#     if isfield(prmin,'IFexponentForNonlinearSum')==1
#         prm['IFexponentForNonlinearSum=prmin.IFexponentForNonlinearSum']
#     end
#     if isfield(prmin,'IFnumberOfHarmonicForInitialEstimate')==1
#         prm['IFnumberOfHarmonicForInitialEstimate=prmin.IFnumberOfHarmonicForInitialEstimate']
#     end
#     if isfield(prmin,'refineFftLength')==1
#         prm['refineFftLength=prmin.refineFftLength']
#     end
#     if isfield(prmin,'refineTimeStretchingFactor')==1
#         prm['refineTimeStretchingFactor=prmin.refineTimeStretchingFactor']
#     end
#     if isfield(prmin,'refineNumberofHarmonicComponent')==1
#         prm['refineNumberofHarmonicComponent=prmin.refineNumberofHarmonicComponent']
#     end
#     if isfield(prmin,'periodicityFrameUpdateInterval')==1
#         prm['periodicityFrameUpdateInterval=prmin.periodicityFrameUpdateInterval']
#     end
#     if isfield(prmin,'note')==1
#         prm['note=prmin.note']
#     end

#     return prm
