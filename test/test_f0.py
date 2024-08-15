import ffmpegio as ff
from straight_pitch import straight_pitch
from straight_pitch.straight_pitch import zinitializeParameters
from straight_pitch.fixpF0VexMltpBG4 import cleaninglownoise, fixpF0VexMltpBG4

from matplotlib import pyplot as plt

from math import ceil, floor, log2
import numpy as np

from straight_pitch.multanalytFineCSPB import (
    multanalytFineCSPB as multanalytFineCSPB1,
    get_filter as get_filter1,
)
from straight_pitch.multanalytFineCSPB1 import multanalytFineCSPB, get_filter

fs, x = ff.audio.read("test/vaiueo2d.wav", sample_fmt="dbl", ac=1)


# plt.specgram(x,Fs=fs)
# plt.show()

prm = zinitializeParameters()
f0floor = prm["F0searchLowerBound"]
f0ceil = prm["F0searchUpperBound"]
framem = prm["F0defaultWindowLength"]
fftl = 1024  # default FFT length
framel = round(framem * fs / 1000)

# nvo # Number of channels in one octave
nvo = prm["NofChannelsInOctave"]
nvc = ceil(log2(f0ceil / f0floor) * nvo)  # number of channels

# # ---- F0 extraction based on a fixed-point method in the frequency domain
f0v, *_ = fixpF0VexMltpBG4(
    x[:, 0],
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

plt.plot(f0v)
# plt.imshow(f0v, origin="lower", aspect="auto")

plt.show()

# [f0raw,vuv,auxouts,prmouts]=MulticueF0v14(x,fs)
# f0rawFixp, analysisParams = straight_pitch(x[:, 0], fs)

# [ap,analysisParams]=exstraightAPind(x,fs,f0raw)
# [n3sgram,prmP]=exstraightspec(x,f0raw.*vuv,fs)

# [n3sgramFixp,prmP]=exstraightspec(x,f0rawFixp,fs)

# [sy,prmS] = exstraightsynth(f0raw.*vuv, n3sgram, ap, fs)
# [syFixp,prmS] = exstraightsynth(f0rawFixp,n3sgramFixp,apFixp,fs)
