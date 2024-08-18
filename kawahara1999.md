Filter impulse response:
$$
w_s(t,\lambda) = w(t,\lambda) ⋆ h(t,\lambda)
$$

Gabor function:
$$
w(t,\lambda) = \exp\left(-\frac{\lambda^2t^2}{4\pi \eta^2} + j\lambda t\right)
$$

Second-order cardinal B-spline basis function tuned to target $F_0$:
$$
h(t,\lambda) = \begin{cases}
1-\left| \frac{\lambda t}{2\pi\eta} \right| & \text{if } \left| t \right| > \frac{2\pi\eta}{\lambda}\\
0 & \text{otherwise}
\end{cases}
$$

Here, $\lambda = 2 \pi F_0$ is the target fundamental frequency in rad/second, and $\eta$ is the time-stretching factor. Expanding $\lambda$ and substitute $t$ for $n/F_s$, and $f_0=F_0/F_s$ yields

$$
\begin{align}
w_s(n,f_0) &= w(n,f_0) ⋆ h(n,f_0) \\
w(n,f_0) &= \exp\left[-\pi \left(\frac{f_0}{\eta} n\right)^2 + j 2 \pi f_0 n \right] \\
h(n,f_0) &= \begin{cases}
1-\frac{f_0}{\eta}|n| & \text{if } \left| n \right|  > \eta / f_0 \\
0 & \text{otherwise}
\end{cases}
\end{align}
$$

From STRAIGHT Matlab code, the support for $w(t,\lambda)$ is limited to $|n| < 3.5 \eta /f_{0,min}$ OR $|n| < 3 \eta /f_{0,min} $ for $f_{0,min}$ is the lowest target fundamental frequency.

The discrete-time samples are placed so that $w_(n,f_0)$ is even-symmetric

$$
f_{0,i} = f_{0,min} 2^{i/nvo}
$$

In MATLAB, each computation is weighted by the square root of the normalized target frequency
$$
\sqrt {mpv_i} = \sqrt{2^{i/nvo}} = \sqrt{f_{0,i}/f_{0,min}}
$$


    t0 = 1 / f0floor
    lmx = round(6 * t0 * fs * mu)
    wl = 2 ** ceil(log2(lmx))

    nx = len(x)
    tx = np.pad(x, (0, wl))
    gent = (np.arange(wl) - wl / 2) / fs

    tb = gent * mpv
    t = tb[np.abs(tb) < 3.5 * mu * t0]
    wd1 = np.exp(-pi * (t / t0 / mu) ** 2)
        # wd1 = wd1 * np.exp((2j * pi * mlt) * t / t0)
    wd2 = np.maximum(0, 1 - np.abs(t / t0 / mu))
    wd2 = wd2[wd2 > 0]
    wwd = np.convolve(wd2, wd1)
    wwd = wwd[np.abs(wwd) > 0.00001]
    wbias = (len(wwd) - 1) // 2
    wwd = wwd * np.exp(
            (2j * pi * mlt)
            * t[np.round(np.arange(len(wwd)) - wbias + len(t) / 2).astype(int)]
            / t0
        )
    
$$
t_b = t 2^{i/n_{vo}}
$$

$$
w_{d1}(t_b) = \exp \left[ -\pi \left(\frac{t_b}{t_0 \eta}\right)^2 \right]
$$

$$
w_{d2}(t_b) = 
\begin{cases} 
1 - \left| \frac{t_b}{t_0 \eta} \right| & \text{if } |\frac{t_b}{t_0 \eta}| < 1 \\
0 & \text{otherwise} \\
\end{cases}
$$

$$
w_{wd}(t_b) = (w_{d2} \star w_{d1}) \exp \left[ j 2\pi m \frac{ t_b} {t_0} \right]
$$

Expand $t_b$, $t_0 = 1/F_{0, min}$, $F_{0,i} = 2^{i/n_{vo}} F_{0, min}$:

$$
w_{d1}(t, F_{0,i}) = \exp \left[ -\pi \left(\frac{F_{0,i} t}{\eta}\right)^2 \right]
$$

$$
w_{d2}(t, F_{0,i}) = 
\begin{cases} 
1 - \frac{F_{0,i}}{\eta} |t|  & \text{if } \frac{F_{0,i}}{\eta} |t| < 1 \\
0 & \text{otherwise} \\
\end{cases}
$$

$$
w_{wd}(t, F_{0,i}) = (w_{d2} \star w_{d1}) \exp \left[ j 2\pi m t F_{0,i} \right]
$$

Convert to discrete-time expression: $t = n/F_s$ and $f_{0,i} = F_{0,i}/F_s$:

$$
w_{d1}(n, f_{0,i}) = \exp \left[ -\pi \left(\frac{f_{0,i}}{\eta}n\right)^2 \right]
$$

$$
w_{d2}(n, f_{0,i}) = 
\begin{cases} 
1 - \frac{f_{0,i}}{\eta}|n| & \text{if } \frac{f_{0,i} }{\eta}|n| < 1 \\
0 & \text{otherwise} \\
\end{cases}
$$

$$
w_{wd}(n, f_{0,i}) = (w_{d2} \star w_{d1}) e^{j 2\pi m f_{0,i}\ n }
$$





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


$t_{gen} = n/F_s$
$m_{pv}(F) = F/F_{min}$
$t = t_{gen} m_{pv} = n f/F_{min}$
$t_0 = 1/F_{min}$
$w_{d1}(F,t) = \exp \left[ -\pi\left(\frac{t}{t_0 \mu (1-p_{ex})}\right)^2 \right]$
$w_{d2}(F,t) = \exp \left[ -\pi\left(\frac{t}{t_0 \mu (1+p_{ex})}\right)^2 \right]$

$w_{d}(f,n) = \exp \left[ -\pi\left(\frac{n f}{\mu (1 \mp p_{ex})}\right)^2 \right]$

    np.abs(t) < 3.5 * mu * t0

$|t| < 3.5 \mu t_0$
$|n| < 3.5 \mu/f$

    # update IF
    t0 * mu / mpv * 1000 > mlim

$t_0 * \mu / m_{pv} * 1000 > m_{lim}$
$f < \frac{\mu}{m_{lim} F_s } 10^{3}$

$f_{max} = \frac{\mu}{m_{lim}}$
$m_{lim}$ minimum smoothing length in samples

    """
    :param fxx: frequencies of the frequency bins
    :param pif2: instantaneous frequency map (freq bins x time samples)
    :param mmp: relative noise power (freq bins x time samples)
    :param dfv: time derivative of pif2 (freq bins x time samples)
    :param aav: magnitude of H1 filterbank output (freq bins x time samples)
    :return: (ff, vv, df, aa) - f0 candidates and their attributes

             - ff - f0 candidates
             - vv - relative noise powers ?
             - df - time derivatives of pif2 ?
             = aa - H1 filterbank output amplitudes ?

             The attributes are linearly interpolated at ff values
    """
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

$$
e_i = p_i - f_i
$$

test

$$
\begin{aligned}
f &= f_0 + \frac{(f_1 - f_0) e_0} {(e_0 -e_1)} \\
  &= f_0 + \frac{(f_1 - f_0) (p_0 - f_0)} {(p_0 - f_0 - p_1 + f_1)}
\end{aligned}
$$
