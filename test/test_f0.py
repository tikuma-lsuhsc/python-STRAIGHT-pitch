import ffmpegio as ff
from straight_pitch import straight_pitch

from matplotlib import pyplot as plt


fs, x = ff.audio.read("test/vaiueo2d.wav", sample_fmt="dbl", ac=1)

# [f0raw,vuv,auxouts,prmouts]=MulticueF0v14(x,fs)
f0rawFixp, analysisParams = straight_pitch(x[:, 0], fs)

# [ap,analysisParams]=exstraightAPind(x,fs,f0raw)
# [n3sgram,prmP]=exstraightspec(x,f0raw.*vuv,fs)

# [n3sgramFixp,prmP]=exstraightspec(x,f0rawFixp,fs)

# [sy,prmS] = exstraightsynth(f0raw.*vuv, n3sgram, ap, fs)
# [syFixp,prmS] = exstraightsynth(f0rawFixp,n3sgramFixp,apFixp,fs)
