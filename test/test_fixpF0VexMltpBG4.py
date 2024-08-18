from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from functools import lru_cache
from numpy.typing import NDArray

from math import ceil, floor, pi, sqrt
import numpy as np
from scipy import signal as sps

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


from straight_pitch.multanalytFineCSPB import multanalytFineCSPB
from straight_pitch.utils import decimate, fftfilt
from straight_pitch.fixpF0VexMltpBG4 import (
    cleaninglownoise,
    zwvlt2ifq,
    zsmoothmapB,
    zifq2gpm2,
    zfixpfreq3,
    znrmlcf2,
)

from matplotlib import pyplot as plt

fs, x = ff.audio.read("test/vaiueo2d.wav", sample_fmt="dbl", ac=1)
x = x[:, 0]

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
# f0v, vrv, dfv, _, aav = fixpF0VexMltpBG4(
#     x,
#     fs,
#     f0floor,
#     nvc,
#     nvo,
#     prm["IFWindowStretch"],
#     prm["F0frameUpdateInterval"],
#     prm["IFsmoothingLengthRelToFc"],
#     prm["IFminimumSmoothingLength"],
#     prm["IFexponentForNonlinearSum"],
#     prm["IFnumberOfHarmonicForInitialEstimate"],
#     prm["DisplayPlots"],
# )

two_pi = 2 * pi


mu: float = prm["IFWindowStretch"]
shiftm: float = prm["F0frameUpdateInterval"]
smp: float = prm["IFsmoothingLengthRelToFc"]
minm: float = prm["IFminimumSmoothingLength"]
pc: float = prm["IFexponentForNonlinearSum"]
nc: int = prm["IFnumberOfHarmonicForInitialEstimate"]


# remove the low-frequency noise
x = cleaninglownoise(x, fs, f0floor)

# f0 search grid
Fxx = f0floor * 2 ** (np.arange(nvc) / nvo)

fxx = Fxx / fs
omegaxx = two_pi * fxx.reshape(-1, 1)

# minimum number of samples per cycles
dn = max(1, floor(1 / (fxx[-1] * 6.3)))

# compute up-to-3 gabor spectrograms -> frequency map (in Hz)

dn1 = dn * 3  # 1st harmonic decimation factor

xd1 = decimate(x, dn1)
pm1 = multanalytFineCSPB(xd1, fxx * dn1, mu, 1)
pm1[pm1 == 0] = np.max(np.abs(pm1)) / 10e7

# plt.imshow(np.abs(pm1), origin="lower", aspect="auto")
# plt.show()
# exit()

pif1 = zwvlt2ifq(pm1) / dn1  # in radian/samples
mm = pif1.shape[1]  # number of frames

# plt.imshow(
#     pif1 * fs / two_pi / Fxx.reshape(-1,1),
#     origin="lower",
#     aspect="auto",
#     extent=[0, pm1.shape[1], Fxx[0], Fxx[-1]],
#     # vmin=-15,vmax=15
# )
# plt.yscale('log')
# plt.colorbar()
# plt.show()
# exit()

if nc > 1:
    # decimate so the highest pitch presents only one cycle
    xd = decimate(x, dn)

    # estimate second harmonic instantaneous frequency (enhancement)
    # - a bug?
    pm2 = multanalytFineCSPB(xd, fxx * dn, mu, 2)
    pif2 = zwvlt2ifq(pm2) / dn
    pif2 = pif2[:, ::3]
    pm2 = pm2[:, ::3]
    mm = min(mm, pif2.shape[1])

    if nc > 2:
        # estimate third harmonic instantaneous frequency (enhancement)
        pm3 = multanalytFineCSPB(xd, fxx * dn, mu, 3)[:, ::3]
        pif3 = zwvlt2ifq(pm3)[:, ::3] / dn
        mm = min(mm, pif3.shape[1])
        pm3 = pm3[:, :mm]

    pm2 = pm2[:, :mm]

pif1 = pif1[:, :mm]

# plt.imshow(pif1-Fxx.reshape(-1,1), origin="lower", aspect="auto")
# plt.colorbar()
# plt.show()
# exit()

# combine pif1 & pif2 & pif3 (weighted average with pm's as the weights, which can be exponentially tweaked by the factor `pc`)
pm1abs = np.abs(pm1)
if nc == 1:
    pif2 = pif1
else:
    pif2 = pif2[:, :mm]
    a1 = pm1abs**pc
    a2 = np.abs(pm2) ** pc
    pif2_num = pif1 * a1 + pif2 / 2 * a2
    pif2_den = a1 + a2

    if nc == 3:
        pif3 = pif3[:, :mm]
        a3 = np.abs(pm3) ** pc
        pif2_num += pif3 / 3 * a3
        pif2_den += a3

    pif2 = pif2_num / pif2_den

dn = dn * 3  # =dn1

# plt.scatter(np.tile(Fxx.reshape(-1, 1), (1, mm)).reshape(-1), pif1.reshape(-1))
# plt.xscale("log")
# plt.subplots(5, 1)
# plt.subplot(5, 1, 1)
# plt.imshow(pm1.real, origin="lower", aspect="auto")
# plt.subplot(5, 1, 2)
# plt.imshow(pm1.imag, origin="lower", aspect="auto")
# plt.subplot(5, 1, 3)
# plt.imshow(pm1abs, origin="lower", aspect="auto")
# plt.subplot(5, 1, 4)
# plt.imshow(
#     np.angle(pm1[:, 1:] / pm1[:, :-1]) * fsn / pi / 2,
#     origin="lower",
#     aspect="auto",
#     vmin=0,
#     vmax=800,
# )
# plt.colorbar()
# plt.subplot(5, 1, 5)
# plt.imshow(pif1, origin="lower", aspect="auto", vmin=0, vmax=800)
# plt.colorbar()
# plt.show()
# exit()

# ~ dwc(t,lambda)/dlambda for eq (10)
slp = zifq2gpm2(pif2, omegaxx)

# d2wc/dt dlambda for eq (10)
nn, mm = pif2.shape
dpif = np.empty_like(pif2)
dpif[:, :-1] = (pif2[:, 1:] - pif2[:, :-1]) / dn
dpif[:, -1] = dpif[:, -1]
dslp = zifq2gpm2(dpif, omegaxx)

# plt.subplots(2, 1)
# plt.subplot(2, 1, 1)
# plt.imshow(slp, origin="lower", aspect="auto")
# plt.subplot(2, 1, 2)
# plt.imshow(dslp, origin="lower", aspect="auto")
# plt.show()
# exit()

# fxx=f0floor*2.0.^((0:nn-1)/nvo)'*2*pi;
mmp = np.empty_like(dslp)
c1, c2 = znrmlcf2()
# c2 = c2b * (Fxx / two_pi) ** 2
mmp = slp**2 / c1 + dslp**2 / c2.reshape(-1, 1)  # sig2_tilde - eq(10)

# for ii=1:nn
# 	c2=c2b*(Fxx(ii)/2/pi)^2;
# 	cff=damp(ii,:)/Fxx(ii)*2*pi*0;
# 	mmp(ii,:)=(dslp(ii,:)./(1+cff.^2)/sqrt(c2)).^2+(slp(ii,:)./sqrt(1+cff.^2)/sqrt(c1)).^2;
# end;

# smooth the relative noise variance map eq (11)
smap = zsmoothmapB(mmp, fxx, smp, minm * 1e-3 * fs, 0.4) if smp != 0 else mmp

# plt.subplots(2,1)
# plt.subplot(2,1,1)
# plt.imshow(-np.log10(mmp), origin="lower", aspect="auto")
# plt.subplot(2,1,2)
# plt.imshow(-np.log10(smap), origin="lower", aspect="auto")
# plt.show()
# exit()

# -----

# plt.plot(omegaxx * fs / two_pi, pif2 * fs / two_pi, ".-")
# plt.show()
# exit()

# select f0 candidates after selecting the output frames
m = slice(None, None, round(shiftm * 1e-3 * fs / dn))
res = [
    zfixpfreq3(omegaxx.reshape(-1), *args)
    for args in zip(pif2[:, m].T, smap[:, m].T, dpif[:, m].T / 2 / pi, pm1abs[:, m].T)
]
nfmax = max(x[0].shape[0] for x in res)
print(nfmax)

fixpp, fixvv, fixdf, fixav = np.stack(
    [np.pad(v, [(0, 0), (0, nfmax - v.shape[1])], constant_values=np.nan) for v in res],
    -1,
)

# return fixpp / two_pi, fixvv, fixdf, fixav

plt.subplots(3, 1)
plt.subplot(3, 1, 1)
plt.specgram(xd1, Fs=fs / dn1)
plt.ylim([0, 400])
plt.subplot(3, 1, 2)
plt.plot(fixpp.T * fs / two_pi, ".")
plt.subplot(3, 1, 3)
plt.plot(-10 * np.log10(fixvv.T), ".")
# plt.plot(fixpp * fs / dn / pi, np.log10(fixvv))
plt.show()
