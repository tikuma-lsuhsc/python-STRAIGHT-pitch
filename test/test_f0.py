import ffmpegio as ff
from straight_pitch import straight_pitch
from straight_pitch.straight_pitch import zinitializeParameters
from straight_pitch.fixpF0VexMltpBG4 import cleaninglownoise, fixpF0VexMltpBG4

from matplotlib import pyplot as plt

from math import ceil, floor, log2
import numpy as np

from straight_pitch.multanalytFineCSPB1 import multanalytFineCSPB as multanalytFineCSPB1
from straight_pitch.multanalytFineCSPB import multanalytFineCSPB

fs, x = ff.audio.read("test/vaiueo2d.wav", sample_fmt="dbl", ac=1)

# have a voiced frame at the beginning
# n0 = round(0.3 * fs)
x = x[:, 0]

f0floor = 40
f0ceil = 800
nvo = 24
nvc = ceil(log2(f0ceil / f0floor) * nvo)  # number of channels
f0 = f0floor * 2 ** (np.arange(nvc) / nvo)
eta = 1.2
mlt = 1
# Pm = multanalytFineCSPB1(x, mlt * f0 / fs, eta)

Pm = multanalytFineCSPB(x, fs, f0floor, nvc, nvo, eta, mlt)
plt.imshow(np.abs(Pm), origin="lower", aspect="auto")
plt.show()
# plt.specgram(x,Fs=fs)
# plt.show()

# prm = zinitializeParameters()
# f0floor = prm["F0searchLowerBound"]
# f0ceil = prm["F0searchUpperBound"]
# framem = prm["F0defaultWindowLength"]
# fftl = 1024  # default FFT length
# framel = round(framem * fs / 1000)

# # nvo # Number of channels in one octave
# nvo = prm["NofChannelsInOctave"]
# nvc = ceil(log2(f0ceil / f0floor) * nvo)  # number of channels

# # ---- F0 extraction based on a fixed-point method in the frequency domain
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


# [f0raw,vuv,auxouts,prmouts]=MulticueF0v14(x,fs)
# f0rawFixp, analysisParams = straight_pitch(x[:, 0], fs)

# [ap,analysisParams]=exstraightAPind(x,fs,f0raw)
# [n3sgram,prmP]=exstraightspec(x,f0raw.*vuv,fs)

# [n3sgramFixp,prmP]=exstraightspec(x,f0rawFixp,fs)

# [sy,prmS] = exstraightsynth(f0raw.*vuv, n3sgram, ap, fs)
# [syFixp,prmS] = exstraightsynth(f0rawFixp,n3sgramFixp,apFixp,fs)
