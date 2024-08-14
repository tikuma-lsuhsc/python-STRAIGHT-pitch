from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from numpy.typing import NDArray

from math import ceil, floor, pi, sqrt
import numpy as np
from scipy import signal as sps

from .multanalytFineCSPB import multanalytFineCSPB
from .utils import decimate


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

    fxx = f0floor * 2.0 ** (np.arange(nvc) / nvo)
    fxh = fxx[-1]  # max(fxx)

    # minimum number of samples per cycles
    dn = max(1, floor(fs / (fxh * 6.3)))

    # decimate so the highest pitch presents only one cycle
    xd = decimate(x, dn)

    if nc > 2:
        pm3 = multanalytFineCSPB(
            xd, fs / dn, f0floor, nvc, nvo, mu, 3, imgi
        )  # error crrect 2002.9.19 (mu was fixed 1.1)
        pif3 = zwvlt2ifq(pm3, fs / dn)
        pif3 = pif3[:, ::3]
        pm3 = pm3[:, ::3]

    if nc > 1:
        pm2 = multanalytFineCSPB(xd, fs / dn, f0floor, nvc, nvo, mu, 2, imgi)
        # error crrect 2002.9.19(mu was fixed 1.1)
        pif2 = zwvlt2ifq(pm2, fs / dn)
        pif2 = pif2[:, ::3]
        pm2 = pm2[:, ::3]

    pm1 = multanalytFineCSPB(
        decimate(x, dn * 3), fs / (dn * 3), f0floor, nvc, nvo, mu, 1, imgi
    )
    # error crrect 2002.9.19(mu was fixed 1.1)
    #### safe guard added on 15/Jan./2003
    mxpm1 = np.max(np.abs(pm1))
    eeps = mxpm1 / 10000000
    pm1[pm1 == 0] = eeps
    #### safe guard end
    pif1 = zwvlt2ifq(pm1, fs / (dn * 3))
    # keyboard

    mm = pif1.shape[1]
    if nc > 1:
        mm = min(mm, pif2.shape[1])

    if nc > 2:
        mm = min(mm, pif3.shape[1])

    if nc == 2:
        for ii in range(mm):
            pif2[:, ii] = (
                pif1[:, ii] * (np.abs(pm1[:, ii])) ** pc
                + pif2[:, ii] / 2 * (np.abs(pm2[:, ii])) ** pc
            ) / ((np.abs(pm1[:, ii])) ** pc + (np.abs(pm2[:, ii])) ** pc)

    if nc == 3:
        for ii in range(mm):
            pif2[:, ii] = (
                pif1[:, ii] * (np.abs(pm1[:, ii])) ** pc
                + pif2[:, ii] / 2 * (np.abs(pm2[:, ii])) ** pc
                + pif3[:, ii] / 3 * (np.abs(pm3[:, ii])) ** pc
            ) / (
                (np.abs(pm1[:, ii])) ** pc
                + (np.abs(pm2[:, ii])) ** pc
                + (np.abs(pm3[:, ii])) ** pc
            )

    if nc == 1:
        pif2 = pif1

    # pif2=zwvlt2ifq(pm,fs/dn)*2*pi
    pif2 = pif2 * 2 * pi
    dn = dn * 3

    slp, _ = zifq2gpm2(pif2, f0floor, nvo)
    nn, mm = pif2.shape
    dpif = (pif2[:, 1:] - pif2[:, :-1]) * fs / dn
    dpif = np.concatenate([dpif, dpif[:, -1:]], axis=1)
    dslp, _ = zifq2gpm2(dpif, f0floor, nvo)

    damp = (np.abs(pm1[:, 1:]) - np.abs(pm1[:, :-1])) * fs / dn
    damp = np.concatenate([damp, damp[:, -1:]], axis=1)
    damp = damp / np.abs(pm1)

    # [c1,c2]=znormwght(1000)
    fxx = f0floor * 2.0 ** (np.arange(nn) / nvo) * 2 * pi
    mmp = 0 * dslp
    c1, c2b = znrmlcf2(1)
    # if imgi==1 hpg=waitbar(0,'P/N map calculation') end # 07/Dec./2002 by H.K.#10/Aug./2005

    for ii in range(nn):
        # 	[c1,c2]=znrmlcf2(fxx[ii]/2/pi) # This is OK, but the next Eq is much faster.
        c2 = c2b * (fxx[ii] / 2 / pi) ** 2
        cff = damp[ii, :] / fxx[ii] * 2 * pi * 0
        mmp[ii, :] = (dslp[ii, :] / (1 + cff**2) / sqrt(c2)) ** 2 + (
            slp[ii, :] / np.sqrt(1 + cff**2) / np.sqrt(c1)
        ) ** 2
        # if imgi==1 waitbar(ii/nn) end #,hpg) # 07/Dec./2002 by H.K.#10/Aug./2005

    # if imgi==1 close(hpg) end
    # 10/Aug./2005

    smap = zsmoothmapB(mmp, fs / dn, f0floor, nvo, smp, minm, 0.4) if smp != 0 else mmp

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
def zwvlt2ifq(pm, fs):
    # 	Wavelet to instantaneous frequency map
    # 	fqv=wvlt2ifq(pm,fs)

    # 	Coded by Hideki Kawahara
    # 	02/March/1999

    pm = pm / (np.abs(pm))
    pif = np.abs(pm - np.concatenate([pm[:, :1], pm[:, :-1]], axis=1))
    pif = fs / pi * np.asin(pif / 2)
    pif[:, 1] = pif[:, 2]

    return pif


# #----------------------------------------------------------------


def zifq2gpm2(pif, f0floor, nvo):
    """Instantaneous frequency 2 geometric parameters

    [slp,pbl]=ifq2gpm(pif,f0floor,nvo)
    slp		: first order coefficient
    pbl		: second order coefficient
    """
    # 	Coded by Hideki Kawahara
    # 	02/March/1999

    fx = f0floor * 2.0 ** (np.arange(pif.shape[0]) / nvo) * 2 * pi

    c = 2.0 ** (1 / nvo)
    g = np.array([[1 / (c * c), 1 / c, 1], [1, 1, 1], [c * c, c, 1]])
    h = np.linalg.inv(g)

    # slp=pif[:-2]*h(1,1)+pif[1:-1]*h(1,2)+pif[2:]*h(1,3)
    slp = ((pif[1:-1] - pif[:-2]) / (1 - 1 / c) + (pif[2:] - pif[1:-1]) / (c - 1)) / 2
    slp = np.vstack([slp[0, :], slp, slp[-2, :]])

    pbl = pif[:-2] * h[1, 0] + pif[1:-1] * h[1, 1] + pif[2:] * h[1, 2]
    pbl = np.vstack([pbl[0, :], pbl, pbl[-2, :]])

    for ii in range(pif.shape[0]):
        slp[ii, :] = slp[ii, :] / fx[ii]
        pbl[ii, :] = pbl[ii, :] / fx[ii]

    slp = slp / fx.reshape(-1, 1)
    pbl = pbl / fx.reshape(-1, 1)

    return slp, pbl


# #------------------------------------------

# #def znormwght(n): c1,c2

# #zz=0:1/n:3
# #hh=[diff(zGcBs(zz,0)) 0]*n
# #c1=sum((zz.*hh)**2)/n
# #c2=sum((2*pi*zz**2.*hh)**2)/n

# #-------------------------------------------


def zGcBs(x, k):

    tt = x + 0.0000001
    p = (
        tt**k
        * np.exp(-pi * tt**2)
        * (np.sin(pi * tt + 0.0001) / (pi * tt + 0.0001)) ** 2
    )
    return p


# #--------------------------------------------
def zsmoothmapB(map, fs, f0floor, nvo, mu, mlim, pex):

    nvc, mm = map.shape
    # mu=0.4
    t0 = 1 / f0floor
    lmx = round(6 * t0 * fs * mu)
    wl = 2 ** ceil(np.log2(lmx))
    gent = (np.arange(wl) - wl / 2) / fs

    smap = map
    mpv = 1
    zt = 0 * gent
    iiv = np.arange(mm)
    for ii in range(nvc):
        t = gent * mpv  # t0*mu/mpv*1000
        t = t[np.abs(t) < 3.5 * mu * t0]
        wbias = round((len(t) - 1) / 2)
        wd1 = np.exp(-pi * (t / (t0 * (1 - pex)) / mu) ** 2)
        wd2 = np.exp(-pi * (t / (t0 * (1 + pex)) / mu) ** 2)
        wd1 = wd1 / np.sum(wd1)
        wd2 = wd2 / np.sum(wd2)
        tm = sps.lfilter(wd1, 1, np.concatenate([map[ii, :], zt], axis=0))
        tm = sps.lfilter(wd2, 1, np.concatenate([1.0 / tm[iiv + wbias], zt], axis=0))
        smap[ii, :] = 1.0 / tm[iiv + wbias]
        if t0 * mu / mpv * 1000 > mlim:
            mpv = mpv * (2.0 ** (1 / nvo))

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
    x = np.arange(0, 3, 1 / n)
    g = zGcBs(x, 0)
    dg = np.concatenate([np.diff(g) * n, [0]])
    dgs = dg / 2 / pi / f
    xx = 2 * pi * f * x
    c1 = np.sum((xx * dgs) ** 2) / n * 2
    c2 = np.sum((xx**2.0 * dgs) ** 2) / n * 2

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
