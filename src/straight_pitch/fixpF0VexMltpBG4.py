from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from functools import lru_cache
from numpy.typing import NDArray

from math import ceil, floor, pi
import numpy as np
from scipy import signal as sps

from .multanalytFineCSPB import multanalytFineCSPB
from .utils import decimate, fftfilt

two_pi = 2 * pi


def fixpF0VexMltpBG4(
    x: NDArray,
    fxx: NDArray,
    mu: float = 1.2,
    shiftm: int = 8,
    smp: float = 1,
    minm: int = 40,
    pc: float = 0.5,
    nc: int = 1,
) -> tuple[NDArray, NDArray, NDArray, NDArray, int]:
    """Fixed point analysis to extract F0

    Parameters
    ----------
    x
        input signal
    fxx
        channel center frequencies in normalized frequency
    mu, optional
        temporal stretching factor, by default 1.2
    shiftm, optional
        frame shift in samples, by default 8
    smp, optional
        smoothing length relative to fc (ratio), by default 1
    minm, optional
        minimum smoothing length (samples), by default 40
    pc, optional
        exponent to represent nonlinear summation, by default 0.5
    nc, optional
        number of harmonic component to use (1,2,3), by default 1

    Returns
    -------
        f0v: F0 candidates in normalized frequency
        vrv: relative noise energy of the candidates' channels
        dfv: derivative of F0 candidates
        av: filterbank output of the candidates' channels
        step: actual frame shift used

        Each is a nb_frames x nb_candidate 2D array. If a frame does not have
        all nb_candidate candidates, unused elements are filled with NaN's.
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

    # frequency bin conversions
    omegaxx = two_pi * fxx.reshape(-1, 1)  # in radians/sample

    # decimate signal so the signal at `f0ceil` Hz gets ~6 samples
    dn = max(1, floor(1 / (fxx[-1] * 6.3)))  # for higher-harmonic analyses
    dn1 = dn * 3  # for the 1st harmonic analysis

    # output downsampling factor so the frames are separated by approximately `shiftm` seconds
    frame_step = round(shiftm / dn1)

    # decimate so the highest pitch presents only one cycle
    xd1 = decimate(x, dn1)
    pm1 = multanalytFineCSPB(xd1, fxx * dn1, mu, 1)
    pm1abs = np.abs(pm1)
    pif1 = zwvlt2ifq(pm1) / dn1
    mm = pif1.shape[1]  # number of frames

    if nc > 1:
        print("nc>1 has not been validated!!")
        # decimate so the highest pitch presents only one cycle
        xd = decimate(x, dn)

        # estimate second harmonic instantaneous frequency (enhancement)
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

    pif1 = pif1[:, :mm]

    # combine pif1 & pif2 & pif3 (weighted average with pm's as the weights, which can be exponentially tweaked by the factor `pc`)
    pm1abs = np.abs(pm1)
    if nc == 1:
        pif = pif1
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

        pif = pif2_num / pif2_den

    # -----
    # Estimate relative noise energy

    # ~ dwc(t,lambda)/dlambda for eq (10)
    slp = zifq2gpm2(pif, omegaxx)

    # d2wc/dt dlambda for eq (10)
    dpif = np.empty_like(pif)
    dpif[:, :-1] = (pif[:, 1:] - pif[:, :-1]) / dn1
    dpif[:, -1] = dpif[:, -1]
    dslp = zifq2gpm2(dpif, omegaxx)

    mmp = np.empty_like(dslp)
    c1, c2 = znrmlcf2()
    mmp = slp**2 / c1 + dslp**2 / c2.reshape(-1, 1)  # sig2_tilde - eq(10)

    # smooth the relative noise energy map eq (11)
    smap = zsmoothmapB(mmp, fxx, smp, minm, 0.4) if smp != 0 else mmp

    # -----
    # Select candidates

    m = slice(None, None, frame_step)
    res = [
        zfixpfreq3(omegaxx.reshape(-1), *args)
        for args in zip(
            pif1[:, m].T, smap[:, m].T, dpif[:, m].T / 2 / pi, pm1abs[:, m].T
        )
    ]
    nfmax = max(x[0].shape[0] for x in res)
    fixpp, fixvv, fixdf, fixav = np.stack(
        [
            np.pad(v, [(0, 0), (0, nfmax - v.shape[1])], constant_values=np.nan)
            for v in res
        ],
        -1,
    )

    return fixpp.T / two_pi, fixvv.T, fixdf.T, fixav.T, frame_step * dn1


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
def zwvlt2ifq(pm: NDArray) -> NDArray:
    """Wavelet to instantaneous frequency map

    Parameters
    ----------
    pm
        symmetrical Gabor wavelet (2D complex)

    Returns
    -------
        frequency map in rad/sample (same dimensions as `pm`)
    """

    # f0 approximated by approximation:
    # - half-sample rotation at pif-Hz from x-axis UP travels long the y-axis by
    #   the observed sample-to-sample differential
    # - why not atan2?

    # pmabs = np.abs(pm)
    # pm[pm == 0] = np.max(pmabs) / 10e7
    # pm = pm / pmabs
    # dpmdn = pm[:, 1:] - pm[:, :-1]
    # pif = np.empty_like(pm, dtype=pm.dtype.type(0).real.dtype)
    # pif[:, 1:] = 2 * np.asin(np.abs(dpmdn) / 2)

    # MODIFIED APPROACH WIH ATAN2

    tf = pm[:, :-1] != 0

    y = np.empty((pm.shape[0], pm.shape[1] - 1), dtype=pm.dtype)
    y[tf] = pm[:, 1:][tf] / pm[:, :-1][tf]
    y[~tf] = 0

    pif = np.empty_like(pm, dtype=pm.dtype.type(0).real.dtype)
    pif[:, 1:] = np.abs(np.atan2(y.imag, y.real))
    pif[:, 0] = pif[:, 1]

    return pif


# #----------------------------------------------------------------


def zifq2gpm2(pif: NDArray, omegax: NDArray) -> NDArray:
    """Compute derivative of the instantaneous frequency wrt frequency (for Eurospeech 1999, eq (10))

    Parameters
    ----------
    pif
        2D instantaneous frequency map in Hz [omegax bins x frames]
    omegax
        frequency bins in radian/Sample

    Returns
    -------
        slp: Geometric parameter slp (first order coefficients)
        # pbl		: second order coefficient
    """

    # 	Coded by Hideki Kawahara
    # 	02/March/1999

    # mean of weighted difference (partial wrt omegax bins)
    dpif = np.diff(omegax * pif, axis=0) / np.diff(omegax, axis=0)
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
        h[: nb - 1] = hp[-1:0:-1]
        h[nb - 1 :] = hp
        return h

    return build(1 - pex), build(1 + pex), nb


def zsmoothmapB(
    map: NDArray, f: NDArray, mu: float, mlim: float, pex: float
) -> NDArray:
    """smooth the frequency(?) map (Eurospeech 1999, eq (11))

    Parameters
    ----------
    map
        input relative noise energy
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
    tmp = np.empty(n + 2 * ceil(3.5 * mu / f[0]))

    # perform the smoothing
    for i, (fi, mapi) in enumerate(zip(f, map)):
        if i < imax:
            wd1, wd2, wbias = zsmoothmapB_filters(fi, mu, pex)
        ntmp = n + 2 * wbias
        tmp[:ntmp] = 0.0
        tmp[:n] = 1 / fftfilt(wd1, mapi[:ntmp])[wbias:-wbias]
        smap[i, :] = 1 / fftfilt(wd2, tmp[:ntmp])[wbias:-wbias]

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
# #ff=pif2[ixx0]+(pif2[ixx+1]-pif2[ixx0]).*cd1[ixx0]./(cd1[ixx0]-cdd1[ixx0])
# #vv=mmp[ixx0]
# #vv=mmp[ixx0]+(mmp[ixx+1]-mmp[ixx0]).*(ff-fxx[ixx0])./(fxx[ixx+1]-fxx[ixx0])
# #df=dfv[ixx0]+(dfv[ixx+1]-dfv[ixx0]).*(ff-fxx[ixx0])./(fxx[ixx+1]-fxx[ixx0])


# #--------------------------------------------
def zfixpfreq3(
    omegaxx: NDArray, pif2: NDArray, mmp: NDArray, dfv: NDArray, aav: NDArray
) -> NDArray:
    """select f0 candidates

    :param omegaxx: frequencies of the frequency bins
    :param pif2: instantaneous frequency map (freq bins x time samples)
    :param mmp: relative noise power (freq bins x time samples)
    :param dfv: time derivative of pif2 (freq bins x time samples)
    :param aav: magnitude of H1 filterbank output (freq bins x time samples)
    :return: structured array with 4 columns:

             - ff - f0 candidates
             - vv - relative noise powers ?
             - df - time derivatives of pif2 ?
             = aa - H1 filterbank output amplitudes ?

             The attributes are linearly interpolated at ff values
    """

    # find hi->lo zero-crossing of pif2 - omegaxx
    e = pif2 - omegaxx
    ixx0 = np.where((e[:-1] > 0) & (e[1:] < 0))[0]

    # if no match, return empty arrays
    if not len(ixx0):
        return np.empty((4, 0))

    # grab only the values at ixx0 and the ones next
    ixx1 = ixx0 + 1
    f0, f1 = omegaxx[ixx0], omegaxx[ixx1]
    if0, if1 = pif2[ixx0], pif2[ixx1]
    e0, e1 = e[ixx0], e[ixx1]
    mmp0, mmp1 = mmp[ixx0], mmp[ixx1]
    df0, df1 = dfv[ixx0], dfv[ixx1]
    aa0, aa1 = aav[ixx0], aav[ixx1]

    # find f0 candidates which crosses the omegaxx plane (linear interpolation wrt the error)
    ff = if0 + (if1 - if0) * e0 / (e0 - e1)

    # find f0 candidate attributes (via linear interpolation)
    m = (ff - f0) / (f1 - f0)
    vv = mmp0 + (mmp1 - mmp0) * m
    df = df0 + (df1 - df0) * m
    aa = aa0 + (aa1 - aa0) * m

    return np.stack([ff, vv, df, aa], axis=0)


# #--------------------------------------------
def znrmlcf2() -> tuple[float, float]:
    """compute the reciprocal of ca and cb in Eurospeech 1999 Eq. (10)

    :return: 1/ca and 1/cb values
    """

    # define the frequency repsonse of a(n arbitrary) "normalized filter" for eq (5)
    n = 100
    dx = 1 / n
    nx = 3 * n
    x = np.arange(1, nx + 1) * dx
    # g = np.exp(-pi * x**2) * np.sinc(x) ** 2
    dg = (
        # fmt:off
        2 * (-pi * x**3 * np.sinc(x) + x * np.cos(x) - np.sin(x))
        * np.exp(-pi * x**2) * np.sinc(x) / x**2
        # fmt:on
    )

    # denominators of ca and cb (numerical integration)
    xx = two_pi * x
    c1 = np.sum((xx * dg) ** 2) * (2 * dx)
    c2 = np.sum((xx**2 * dg) ** 2) * (2 * dx)

    # xx=2*pi*x;
    # c1=sum((xx.*dgs).^2)/n*2;
    # c2=sum((xx.^2.*dgs).^2)/n*2;

    return c1, c2
