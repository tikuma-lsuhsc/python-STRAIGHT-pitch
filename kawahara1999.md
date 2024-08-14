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

