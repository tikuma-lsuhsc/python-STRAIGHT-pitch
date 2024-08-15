from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from functools import lru_cache
from numpy.typing import NDArray

from math import ceil, floor, pi, sqrt
import numpy as np
from scipy import signal as sps

from .multanalytFineCSPB import multanalytFineCSPB
from .utils import decimate, fftfilt

two_pi = 2 * pi


def fixpF0VexMltpBG4(
    x: NDArray,
    fs: float,
    f0floor: float,
    nvc: int,
    nvo: int = 24,
    mu: float = 1.2,
    shiftm: float = 1,
    smp: float = 1,
    minm: float = 5,
    pc: float = 0.5,
    nc: int = 1,
    imgi: bool = False,
):
    """Fixed point analysis to extract F0

    Parameters
    ----------
    x
        input signal
    fs
        sampling frequency (Hz)
    f0floor
        lowest frequency for F0 search
    nvc
        total number of filter channels
    nvo, optional
        number of channels per octave, by default 24
    mu, optional
        temporal stretching factor, by default 1.2
    shiftm, optional
        frame shift in ms, by default 1
    smp, optional
        smoothing length relative to fc (ratio), by default 1
    minm, optional
        minimum smoothing length (ms), by default 5
    pc, optional
        exponent to represent nonlinear summation, by default 0.5
    nc, optional
        number of harmonic component to use (1,2,3), by default 1
    imgi, optional
        image display indicator, by default False

    Returns
    -------
        f0v: NDArray
        vrv: NDArray
        dfv: NDArray
        nf: NDArray
    """
    # 	Designed and coded by Hideki Kawahara
    # 	28/March/1999
    # 	04/April/1999 revised to multi component version
    # 	07/April/1999 bi-reciprocal smoothing for multi component compensation
    # 	01/May/1999 first derivative of Amplitude is taken into account
    # 	17/Dec./2000 display bug fix
    # 	19/Sep./2002 bug fix (mu information was discarded.)
    #   07/Dec./2002 waitbar was added
    # 	30/April/2005 modification for Matlab v7.0 compatibility
    # 	10/Aug./2005 modified by Takahashi on waitbar
    # 	10/Sept./2005 mofidied by Kawahara on waitbar
    #   11/Sept./2005 fixed waitbar problem

    # f0floor=40
    # nvo=12
    # nvc=52
    # mu=1.1

    # remove the low-frequency noise
    x = cleaninglownoise(x, fs, f0floor)

    # f0 search grid
    Fxx = f0floor * 2 ** (np.arange(nvc) / nvo)

    fxx = Fxx / fs
    omegaxx = two_pi * fxx.reshape(-1, 1)

    # minimum number of samples per cycles
    dn = max(1, floor(1 / (fxx[-1] * 6.3)))

    # decimate so the highest pitch presents only one cycle
    xd = decimate(x, dn)

    # compute up-to-3 gabor spectrograms -> frequency map (in Hz)

    dn1 = dn * 3  # 1st harmonic decimation factor
    pm1 = multanalytFineCSPB(decimate(x, dn1), fxx / dn1, mu, 1)
    #### safe guard added on 15/Jan./2003
    pm1[pm1 == 0] = np.max(np.abs(pm1)) / 10e7
    #### safe guard end
    pif1 = zwvlt2ifq(pm1, fs / dn1)
    mm = pif1.shape[1]  # number of frames

    if nc == 1:
        pif2 = pif1

    if nc > 1:
        pm2 = multanalytFineCSPB(xd, fxx / dn, mu, 2)
        pif2 = zwvlt2ifq(pm2, fs / dn)
        pif2 = pif2[:, ::3]
        pm2 = pm2[:, ::3]
        mm = min(mm, pif2.shape[1])

    if nc > 2:
        pm3 = multanalytFineCSPB(xd, fxx / dn, mu, 3)
        pif3 = zwvlt2ifq(pm3, fs / dn)
        pif3 = pif3[:, ::3]
        pm3 = pm3[:, ::3]
        mm = min(mm, pif3.shape[1])

    pif1 = pif1[:, :mm]

    # combine pif2 & pif3
    pm1abs = np.abs(pm1)
    if nc > 1:
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

    pif2 = pif2 * two_pi
    dn = dn * 3  # =dn1
    fsn = fs / dn  # sampling rate after decimation

    slp = zifq2gpm2(pif2, omegaxx)
    nn, mm = pif2.shape
    dpif = np.empty_like(pif2)
    dpif[:, :-1] = (pif2[:, 1:] - pif2[:, :-1]) * fsn
    dpif[:, -1] = dpif[:, -1]
    dslp = zifq2gpm2(dpif, omegaxx)

    # damp = np.zeros_like(pm1)
    # damp[:, :-1] = (pm1abs[:, 1:] - pm1abs[:, :-1]) * fsn
    # damp[:, -1] = damp[:, -1]
    # damp = damp / pm1abs

    # [c1,c2]=znormwght(1000)
    mmp = np.zeros_like(dslp)
    c1, c2b = znrmlcf2(1)
    c2 = c2b * (omegaxx / two_pi) ** 2
    # cff = damp / omegaxx * two_pi * 0
    # mmp = (dslp / (1 + cff**2) / sqrt(c2)) ** 2 + (
    #     slp / np.sqrt(1 + cff**2) / np.sqrt(c1)
    # ) ** 2
    mmp = (dslp / np.sqrt(c2)) ** 2 + (slp / np.sqrt(c1)) ** 2

    smap = zsmoothmapB(mmp, fxx, smp, minm * 1e-3 * fs, 0.4) if smp != 0 else mmp

    fixpp = np.zeros((round(nn / 3), mm))
    fixvv = np.full_like(fixpp, 100000000)
    fixdf = np.full_like(fixpp, 100000000)
    fixav = np.full_like(fixpp, 1000000000)
    nf = np.zeros(mm, dtype=int)
    # if imgi==1 hpg=waitbar(0,'Fixed pints calculation') end # 07/Dec./2002 by H.K.#10/Aug./2005
    for ii in range(mm):
        ff, vv, df, aa = zfixpfreq3(
            fxx, pif2[:, ii], smap[:, ii], dpif[:, ii] / 2 / pi, pm1[:, ii]
        )
        kk = len(ff)
        fixpp[:kk, ii] = ff
        fixvv[:kk, ii] = vv
        fixdf[:kk, ii] = df
        fixav[:kk, ii] = aa
        nf[ii] = kk
        # if imgi==1 and rem(ii,10)==0 waitbar(ii/mm) end
    # 07/Dec./2002 by H.K.#10/Aug./2005

    # if imgi==1 close(hpg) end # 07/Dec./2002 by H.K.#10/Aug./2005
    fixpp[fixpp == 0] += 1000000

    # keyboard
    # [vvm,ivv]=min(fixvv)
    #
    # for ii=1:mm
    # 	ff00[ii]=fixpp(ivv[ii],ii)
    # 	esgm[ii]=fixvv(ivv[ii],ii)
    # end
    n = np.arange(np.max(nf))
    m = np.arange(0, mm, round(shiftm / dn * fs / 1000)).astype(int)
    ix = np.ix_(n, m)
    f0v = fixpp[ix] / (2 * pi)
    vrv = fixvv[ix]
    dfv = fixdf[ix]
    aav = fixav[ix]
    nf = nf[m]

    # if imgi==1
    # 	cnmap(fixpp,smap,fs,dn,nvo,f0floor,shiftm)

    # ff00=ff00(round(1:shiftm/dn*fs/1000:mm))
    # esgm=sqrt(esgm(round(1:shiftm/dn*fs/1000:mm)))
    # keyboard

    return f0v, vrv, dfv, nf, aav


# ------------------------------------------------------------------

# def cnmap(fixpp,smap,fs,dn,nvo,f0floor,shiftm):

# 	# 	This function had a bug in map axis.
# 	#	17/Dec./2000 bug fix by Hideki Kawahara.

# 	dt=dn/fs
# 	[nn,mm]=size(smap)
# 	aa=figure
# 	set(aa,'PaperPosition',[0.3 0.25 8 10.9])
# 	set(aa,'Position',[30 130 520 680])
# 	subplot(211)
# 	imagesc([0 (mm-1)*dt*1000],[1 nn],20*log10(smap(:,round(1:shiftm/dn*fs/1000:mm))))
# 	axis('xy')
# 	hold on
# 	tx=((1:shiftm/dn*fs/1000:mm)-1)*dt*1000
# 	plot(tx,(nvo*log(fixpp(:,round(1:shiftm/dn*fs/1000:mm))/f0floor/2/pi)/log(2)+0.5)','ko')
# 	plot(tx,(nvo*log(fixpp(:,round(1:shiftm/dn*fs/1000:mm))/f0floor/2/pi)/log(2)+0.5)','w.')
# 	hold off
# 	xlabel('time (ms)')
# 	ylabel('channel #')
# 	colormap(jet)

# 	okid=1
# 	return okid

# #------------------------------------------------------------------

# ##function pm=zmultanalytFineCSPm(x,fs,f0floor,nvc,nvo,mu,mlt)

# #       Dual waveleta analysis using cardinal spline manipulation
# #               pm=multanalytFineCSP(x,fs,f0floor,nvc,nvo)
# #       Input parameters
# #
# #               x       : input signal (2kHz sampling rate is sufficient.)
# #               fs      : sampling frequency (Hz)
# #               f0floor : lower bound for pitch search (60Hz suggested)
# #               nvc     : number of total voices for wavelet analysis
# #               nvo     : number of voices in an octave
# #				mu		: temporal stretch factor
# #       Outpur parameters
# #               pm      : wavelet transform using iso-metric Gabor function
# #
# #       If you have any questions,  mailto:kawahara@hip.atr.co.jp
# #
# #       Copyright (c) ATR Human Information Processing Research Labs. 1996
# #       Invented and coded by Hideki Kawahara
# #       30/Oct./1996

# #t0=1/f0floor
# #lmx=round(6*t0*fs*mu)
# #wl=2^ceil(log(lmx)/log(2))
# #x=x(:)'
# #nx=length(x)
# #tx=[x,zeros(1,wl)]
# #gent=((1:wl)-wl/2)/fs

# #nvc=18

# #wd=zeros(nvc,wl)
# #wd2=zeros(nvc,wl)
# #ym=zeros(nvc,nx)
# #pm=zeros(nvc,nx)
# #mpv=1
# #mu=1.0
# #for ii=1:nvc
# #  t=gent*mpv
# #  t=t(abs(t)<3.5*mu*t0)
# #  wbias=round((length(t)-1)/2)
# #  wd1=exp(-pi*(t/t0/mu)**2)
# #.*exp(i*2*pi*t/t0)
# #  wd2=max(0,1-abs(t/t0/mu))
# #  wd2=wd2(wd2>0)
# #  wwd=conv(wd2,wd1)
# #  wwd=wwd(abs(wwd)>0.0001)
# #  wbias=round((length(wwd)-1)/2)
# #  wwd=wwd.*exp(i*2*pi*mlt*t(round((1:length(wwd))-wbias+length(t)/2))/t0)
# #  pmtmp1=fftfilt(wwd,tx)
# #  pm[ii,:]=pmtmp1(wbias+1:wbias+nx)*sqrt(mpv)
# #  mpv=mpv*(2.0^(1/nvo))
# #  keyboard
# #end
# #[nn,mm]=size(pm)
# #pm=pm(:,1:mlt:mm)


# #----------------------------------------------------------------
def zwvlt2ifq(pm: NDArray, fs: float) -> NDArray:
    """Wavelet to instantaneous frequency map

    Parameters
    ----------
    pm
        symmetrical Gabor wavelet (2D complex)
    fs
        sampling rate

    Returns
    -------
        frequency map in Hz (same dimensions as `pm`)
    """

    # 	Coded by Hideki Kawahara
    # 	02/March/1999

    pm = pm / np.abs(pm)
    pif = np.abs(pm - np.concatenate([pm[:, :1], pm[:, :-1]], axis=1))
    pif = fs / pi * np.asin(pif / 2)
    pif[:, 1] = pif[:, 2]

    return pif


# #----------------------------------------------------------------


def zifq2gpm2(pif: NDArray, omegax: NDArray) -> NDArray:
    """Geometric parameter of instantaneous frequency

    Parameters
    ----------
    pif
        2D instantaneous frequency map in Hz [omegax bins x frames]
    omegax
        frequency bins in radian/s

    Returns
    -------
        slp: Geometric parameter slp (first order coefficients)
        # pbl		: second order coefficient
    """

    # 	Coded by Hideki Kawahara
    # 	02/March/1999

    # mean of weighted difference (partial wrt omegax bins)
    dpif = np.diff(omegax * pif, 1, axis=0) / np.diff(omegax, axis=0)
    slp = np.empty_like(pif)
    slp[1:-1] = (dpif[:-1] + dpif[1:]) / 2
    slp[0] = slp[1]
    slp[-1] = slp[-2]
    # slp=((pif(2:nn-1,:)-pif(1:nn-2,:))/(1-1/c) + (pif(3:nn,:)-pif(2:nn-1,:))/(c-1))/2
    # slp=[slp(1,:);slp;slp(nn-2,:)];
    slp = slp / omegax

    # second parameter
    # c=2.0^(1/nvo);
    # g=[1/c/c 1/c 1;1 1 1;c*c c 1];
    # h=inv(g);
    # pbl=pif(1:nn-2,:)*h(2,1)+pif(2:nn-1,:)*h(2,2)+pif(3:nn,:)*h(2,3);
    # pbl=[pbl(1,:);pbl;pbl(nn-2,:)];
    # pbl=pbl/fx(:);

    return slp  # , pbl


# #------------------------------------------

# #def znormwght(n): c1,c2

# #zz=0:1/n:3
# #hh=[diff(zGcBs(zz,0)) 0]*n
# #c1=sum((zz.*hh)**2)/n
# #c2=sum((2*pi*zz**2.*hh)**2)/n

# #-------------------------------------------


@lru_cache(2)
def zGcBs(n: int, k: float) -> tuple[NDArray, NDArray]:
    x = np.arange(3 * n + 1) / n
    f = np.exp(-pi * x**2) * np.sinc(x) ** 2
    return x, ((x**k * f) if k != 0 else f)


# #--------------------------------------------


@lru_cache(512)
def zsmoothmapB_filters(f: float, mu: float, pex: float) -> tuple[NDArray, NDArray]:
    """design smoothing FIR filter for a given frequency

    Parameters
    ----------
    f
        normalized frequency
    mu
        smoothing length relative to fc (ratio)
    pex
        ???

    Returns
    -------
        wd1 and wd2
    """
    n = np.arange(3.5 * mu / f)
    b = -pi * (n * f / mu) ** 2
    nb = len(n)

    def build(c):
        hp = np.exp(b * c**-2)
        h = np.empty(2 * nb - 1)
        h[:nb-1] = hp[-1:0:-1]
        h[nb-1:] = hp
        return h

    return build(1 - pex), build(1 + pex), nb


def zsmoothmapB(
    map: NDArray, f: NDArray, mu: float, mlim: float, pex: float
) -> NDArray:
    """smooth the frequency(?) map

    Parameters
    ----------
    map
        input map
    f
        monotonically increasing frequency bins in normalized frequency
    mu
        smoothing length relative to fc (ratio)
    mlim
        minimum smoothing length in samples
    pex
        ???

    Returns
    -------
        smoothed map
    """

    # set the shortest filter length, and mark the frequencies beyond it
    fmax = mu / mlim
    imax = np.argmax(f > fmax) if f[-1] > fmax else len(f)

    if imax == 0:
        raise ValueError(f"{mlim=} is set too large.")

    # preallocate the output to match the input shape
    smap = np.empty_like(map)

    # zero-pad the input
    map = np.pad(map, [(0, 0), (0, round(7 * mu / f[0] + 1))])
    n = smap.shape[1]

    # perform the smoothing
    for i, (fi, mapi) in enumerate(zip(f, map)):
        if i < imax:
            wd1, wd2, wbias = zsmoothmapB_filters(fi, mu, pex)
        tmp = np.zeros(n + 2 * wbias)
        tmp[: -2 * wbias] = 1 / fftfilt(wd1, mapi)[wbias:-wbias]
        smap[i, :] = 1 / fftfilt(wd2, tmp)[wbias:-wbias]

    return smap


# #--------------------------------------------
# #def zfixpfreq2(fxx,pif2,mmp,dfv): ff,vv,df
# #
# #nn=length(fxx)
# #iix=(1:nn)'
# #cd1=pif2-fxx
# #cd2=[diff(cd1);cd1(nn)-cd1(nn-1)]
# #cdd1=[cd1(2:nn);cd1(nn)]
# #fp=(cd1.*cdd1<0).*(cd2<0)
# #ixx=iix(fp>0)
# #ff=pif2[ixx]+(pif2[ixx+1]-pif2[ixx]).*cd1[ixx]./(cd1[ixx]-cdd1[ixx])
# #vv=mmp[ixx]
# #vv=mmp[ixx]+(mmp[ixx+1]-mmp[ixx]).*(ff-fxx[ixx])./(fxx[ixx+1]-fxx[ixx])
# #df=dfv[ixx]+(dfv[ixx+1]-dfv[ixx]).*(ff-fxx[ixx])./(fxx[ixx+1]-fxx[ixx])


# #--------------------------------------------
def zfixpfreq3(fxx, pif2, mmp, dfv, pm):

    aav = np.abs(pm)
    nn = len(fxx)
    iix = np.arange(nn)
    cd1 = pif2 - fxx
    cd2 = np.concatenate([np.diff(cd1), [cd1[-1] - cd1[-2]]])
    cdd1 = np.concatenate([cd1[1:], [cd1[-1]]])
    fp = (cd1 * cdd1 < 0) * (cd2 < 0)
    ixx = iix[fp > 0]
    ff = pif2[ixx] + (pif2[ixx + 1] - pif2[ixx]) * cd1[ixx] / (cd1[ixx] - cdd1[ixx])
    # vv=mmp[ixx]
    vv = mmp[ixx] + (mmp[ixx + 1] - mmp[ixx]) * (ff - fxx[ixx]) / (
        fxx[ixx + 1] - fxx[ixx]
    )
    df = dfv[ixx] + (dfv[ixx + 1] - dfv[ixx]) * (ff - fxx[ixx]) / (
        fxx[ixx + 1] - fxx[ixx]
    )
    aa = aav[ixx] + (aav[ixx + 1] - aav[ixx]) * (ff - fxx[ixx]) / (
        fxx[ixx + 1] - fxx[ixx]
    )

    return ff, vv, df, aa


# #--------------------------------------------
def znrmlcf2(f):

    n = 100
    x, g = zGcBs(n, 0)
    dgs = np.empty_like(g)
    dgs[:-1] = np.diff(g) * n
    dgs[-1] = 0
    c1 = np.sum((x * dgs) ** 2) / n * 2
    c2 = np.sum((2 * pi * f * x**2.0 * dgs) ** 2) / n * 2

    return c1, c2


# #--------------------------------------------
def cleaninglownoise(x, fs, f0floor, flm=50):

    flp = round(fs * flm / 1000)
    wlp = sps.firwin(flp * 2, f0floor / (fs / 2))
    wlp[flp] = wlp[flp] - 1
    wlp = -wlp

    tx = np.pad(x, (0, len(wlp)))
    ttx = sps.lfilter(wlp, 1, tx)
    x = ttx[flp:-flp]

    return x
