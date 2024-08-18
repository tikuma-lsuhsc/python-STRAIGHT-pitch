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

import ffmpegio as ff

from straight_pitch.fixpF0VexMltpBG41 import fixpF0VexMltpBG4
from straight_pitch.f0track5 import f0track5
from straight_pitch.refineF06 import refineF06
from straight_pitch.utils import decimate
from straight_pitch.straight_pitch import zinitializeParameters

from matplotlib import pyplot as plt

fs, x = ff.audio.read("test/vaiueo2d.wav", sample_fmt="dbl", ac=1)
x = x[:,0]

prm = zinitializeParameters()

#   Initialize default parameters
f0floor = prm["F0searchLowerBound"]
f0ceil = prm["F0searchUpperBound"]

# default frame length for pitch extraction (ms)
framem = prm["F0defaultWindowLength"]

fftl = 1024  # default FFT length
framel = round(framem * fs / 1000)

if fftl < framel:
    fftl = next_fast_len(framel)

# nvo # Number of channels in one octave
nvo = prm["NofChannelsInOctave"]
nvc = ceil(log2(f0ceil / f0floor) * nvo)  # number of channels

# ---- F0 extraction based on a fixed-point method in the frequency domain
f0v, vrv, dfv, _, aav = fixpF0VexMltpBG4(
    x,
    fs,
    f0floor,
    nvc,
    nvo,
    prm["IFWindowStretch"],
    prm["F0frameUpdateInterval"],
    prm["IFsmoothingLengthRelToFc"],
    prm["IFminimumSmoothingLength"],
    prm["IFexponentForNonlinearSum"],
    prm["IFnumberOfHarmonicForInitialEstimate"],
    prm["DisplayPlots"],
)
# if imageOn
#     title([fname '  ' datestr(now,0)])
#     drawnow

# ---- post processing for V/UV decision and F0 tracking
pwt, pwh = zplotcpower(x, fs, prm["F0frameUpdateInterval"], prm["DisplayPlots"])

# paramaters for F0 refinement
# fftlf0r =   # fftlf0r=1024 # FFT length for F0 refinement
# tstretch =   # tstretch=1.1 # time window stretching factor
# nhmx =   # nhmx=3 # number of harmonic components for F0 refinement
# iPeriodicityInterval = prm[
#     "periodicityFrameUpdateInterval"
# ]  # frame update interval for periodicity index (ms)

f0raw, irms, *_ = f0track5(
    f0v, vrv, dfv, pwt, pwh, aav, prm["F0frameUpdateInterval"], prm["DisplayPlots"]
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
    decimate(x, dn),
    fs / dn,
    f0raw,
    prm["refineFftLength"],
    prm["refineTimeStretchingFactor"],
    prm["refineNumberofHarmonicComponent"],
    prm["F0frameUpdateInterval"],
    nstp,
    nedp,
    prm["DisplayPlots"],
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
