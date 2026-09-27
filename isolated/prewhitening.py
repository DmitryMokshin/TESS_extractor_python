"""
Frequency analysis by iterative prewhitening, with realistic significance and
errors -- the part that a single Lomb-Scargle periodogram + naive FAP does not
give (see Peridogram_compute.py for the full run).

What is done differently from `periodogram.py`:

* a slow polynomial trend is removed first and then refitted together with the
  sinusoids (otherwise a trend over the sector leaks into the lowest
  frequencies: a "period" of ~T/2 is just 2/T);
* frequencies are extracted one by one (prewhitening): after each new frequency
  ALL frequencies, amplitudes, phases and the trend are refitted by nonlinear
  least squares -- close frequencies (spacing < 1.5/T) and spectral-window
  sidelobes no longer shift each other's peaks;
* significance is S/N >= 4 against the actual noise level (Breger et al. 1993),
  estimated two ways: the mean residual amplitude within +/-0.5 c/d and a
  red+white noise model A0/(1+(f/fc)^g)+Cw (Bowman et al. 2019). The naive FAP
  assumes white noise and independent frequencies; for Be stars with strong
  red noise it gives meaningless values like 1e-147;
* formal errors: Montgomery & O'Donoghue (1999), multiplied by sqrt(D) for
  correlated residuals (Schwarzenberg-Czerny 1991);
* amplitudes are in the units of the input (mmag or ppt), not LS power.

Also: amplitude spectrum, spectral window and a dynamic (sliding-window)
amplitude spectrum.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

POLY_DEG = 2   # degree of the slow trend fitted together with the sinusoids


# --------------------------------------------------------------------------- #
# Spectra
# --------------------------------------------------------------------------- #

def amp_spectrum(t, y, freqs, chunk=400):
    """Amplitude spectrum (DFT), in units of y: A(f) = 2/N |sum y exp(-2 pi i f t)|."""
    t = np.asarray(t, float)
    y = np.asarray(y, float) - np.mean(y)
    out = np.empty(freqs.size)
    for k in range(0, freqs.size, chunk):
        ph = np.exp(-2j * np.pi * np.outer(freqs[k:k + chunk], t))
        out[k:k + chunk] = 2.0 / t.size * np.abs(ph @ y)
    return out


def window_function(t, freqs, f0):
    """Spectral window centred on f0 (amplitude spectrum of a pure sinusoid)."""
    return amp_spectrum(t, np.sin(2 * np.pi * f0 * np.asarray(t, float)), freqs)


def detrend_poly(t, y, deg=POLY_DEG):
    x = t - t.mean()
    return y - np.polyval(np.polyfit(x, y, deg), x)


def red_noise_fit(freqs, amp, smooth=1.0):
    """
    Noise model A(f) = A0 / (1 + (f/fc)^g) + Cw (Bowman et al. 2019), fitted to
    the amplitude spectrum smoothed by a running MEDIAN of width `smooth` c/d
    (x1.0645 converts median to mean for a noise spectrum). The median is
    robust to strong peaks not yet removed. Returns (callable model, params).
    """
    from scipy.ndimage import median_filter
    df = freqs[1] - freqs[0]
    n = max(3, int(round(smooth / df)) | 1)
    sm = 1.0645 * median_filter(amp, size=n, mode="reflect")
    m = freqs > 2 * df
    x, yv = freqs[m], sm[m]

    def mdl(p, xx):
        a0, fc, g, cw = np.exp(p[0]), np.exp(p[1]), p[2], np.exp(p[3])
        return a0 / (1 + (xx / fc) ** g) + cw

    hi = x > np.percentile(x, 80)
    lo_b, hi_b = [-20, np.log(0.02), 0.5, -20], [10, np.log(10), 6, 10]
    p0 = [np.log(yv[x < 1].mean() + 1e-9) if np.any(x < 1) else 0.0, 0.0, 2.0,
          np.log(np.median(yv[hi]) + 1e-9)]
    p0 = list(np.clip(p0, np.array(lo_b) + 1e-6, np.array(hi_b) - 1e-6))
    r = least_squares(lambda p: np.log(mdl(p, x)) - np.log(yv), p0, bounds=(lo_b, hi_b))
    return (lambda f: mdl(r.x, f)), r.x


def local_noise(freqs_grid, amp_res, f, box=1.0):
    """Mean residual amplitude within +/- box/2 around f (Breger et al. 1993)."""
    m = (freqs_grid > max(freqs_grid[0], f - box / 2)) & (freqs_grid < f + box / 2)
    return float(amp_res[m].mean())


# --------------------------------------------------------------------------- #
# Multi-sine model and prewhitening
# --------------------------------------------------------------------------- #

def _model(p, t, nf, t0):
    x = t - t0
    y = sum(p[k] * x ** k for k in range(POLY_DEG + 1)) + np.zeros_like(t)
    for k in range(nf):
        f, a, ph = p[POLY_DEG + 1 + 3 * k: POLY_DEG + 4 + 3 * k]
        y = y + a * np.sin(2 * np.pi * (f * t + ph))
    return y


def fit_all(t, y, freqs, amps, phases, fmin=0.0, fmax=np.inf, anchors=None, halfwidth=None):
    """
    Nonlinear least-squares fit of polynomial trend + sum of sinusoids.
    If `anchors` is given, each frequency may move at most `halfwidth` from the
    value at which it was found (keeps close frequencies from merging into a
    pair with huge anti-phased amplitudes). Returns (freqs, amps, phases, residuals).
    """
    t0 = t.mean()
    npoly = POLY_DEG + 1
    p0 = list(np.polyfit(t - t0, y, POLY_DEG)[::-1])
    lo, hi = [-np.inf] * npoly, [np.inf] * npoly
    for i, (f, a, ph) in enumerate(zip(freqs, amps, phases)):
        p0 += [f, a, ph]
        if anchors is not None:
            lo += [max(fmin, anchors[i] - halfwidth), -np.inf, -np.inf]
            hi += [min(fmax, anchors[i] + halfwidth), np.inf, np.inf]
        else:
            lo += [fmin, -np.inf, -np.inf]
            hi += [fmax, np.inf, np.inf]
    p0 = np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
    r = least_squares(lambda p: _model(p, t, len(freqs), t0) - y, p0, bounds=(lo, hi),
                      method="trf", x_scale="jac")
    p = r.x
    fr, am, ph = p[npoly::3].copy(), p[npoly + 1::3].copy(), p[npoly + 2::3].copy()
    neg = am < 0
    am[neg] *= -1
    ph[neg] += 0.5
    return fr, am, ph % 1, y - _model(p, t, len(freqs), t0)


def _phase_guess(t, y, f):
    X = np.c_[np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)]
    s, c = np.linalg.lstsq(X, y - y.mean(), rcond=None)[0]
    return np.hypot(s, c), (np.arctan2(c, s) / (2 * np.pi)) % 1


def run_lengths_d(res):
    """Mean length of runs of same-sign residuals (Schwarzenberg-Czerny 1991)."""
    s = np.sign(res)
    return max(1.0, res.size / (np.count_nonzero(s[1:] != s[:-1]) + 1))


def errors_mo99(t, res, amps):
    """Montgomery & O'Donoghue (1999) errors x sqrt(D). Returns (sigma_f, sigma_A, sigma_phase, D, sigma_res)."""
    n, T = t.size, t.max() - t.min()
    s_n = res.std()
    d = run_lengths_d(res)
    sf = np.sqrt(6.0) / (np.pi * T) * s_n / (amps * np.sqrt(n)) * np.sqrt(d)
    sa = np.sqrt(2.0 / n) * s_n * np.sqrt(d) * np.ones_like(amps)
    return sf, sa, sa / amps / (2 * np.pi), d, s_n


def prewhiten(t, y, fmax, fmin=None, oversample=10, nmax=15, snr_stop=4.0, box=1.0):
    """
    Iterative prewhitening of (t [d], y [mmag or ppt]).

    At each step the residual amplitude spectrum is compared with a fitted
    red+white noise model, and the local maximum with the HIGHEST S/N is taken
    (not simply the highest peak -- Be stars have strong low-frequency red
    noise). Peaks closer than 1/T to already found frequencies are skipped.
    After adding a frequency, trend + all sinusoids are refitted together.
    Stops when the best S/N < snr_stop.

    Returns a list of dicts sorted by frequency:
        {"frequency", "frequency_err", "period", "period_err", "amplitude",
         "amplitude_err", "phase", "snr_local", "snr_red", "note"}
    and a dict with residuals and spectra for plotting.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    T = t.max() - t.min()
    fmin = 0.5 / T if fmin is None else fmin
    df = 1.0 / (oversample * T)
    grid = np.arange(fmin, fmax, df)
    fr, am, ph, anchors = [], [], [], []
    res = detrend_poly(t, y)
    for _ in range(nmax):
        A = amp_spectrum(t, res, grid)
        rn, _ = red_noise_fit(grid, A)
        snr = A / rn(grid)
        peak = np.r_[False, (A[1:-1] > A[:-2]) & (A[1:-1] > A[2:]), False]
        for f0 in fr:
            peak &= np.abs(grid - f0) > 1.0 / T
        snr[~peak] = 0
        j = int(np.argmax(snr))
        if snr[j] < snr_stop:
            break
        fine = np.linspace(grid[j] - df, grid[j] + df, 41)
        fnew = fine[np.argmax(amp_spectrum(t, res, fine))]
        a0, p0 = _phase_guess(t, res, fnew)
        fr.append(fnew); am.append(a0); ph.append(p0); anchors.append(fnew)
        f_, a_, p_, res = fit_all(t, y, fr, am, ph, fmin=fmin, fmax=fmax,
                                  anchors=anchors, halfwidth=0.25 / T)
        fr, am, ph = list(f_), list(a_), list(p_)

    fr, am, ph = np.array(fr), np.array(am), np.array(ph)
    A_res = amp_spectrum(t, res, grid)
    extra = dict(residuals=res, grid=grid, amp_residuals=A_res, T=T, D=np.nan, sigma_res=res.std())
    if fr.size == 0:
        return [], extra
    o = np.argsort(fr)
    fr, am, ph = fr[o], am[o], ph[o]
    sf, sa, sp, d, s_n = errors_mo99(t, res, am)
    rn, _ = red_noise_fit(grid, A_res)
    extra.update(D=d, sigma_res=s_n, noise_model=rn)

    out = []
    for i in range(fr.size):
        note = []
        if fr[i] < 2.0 / T:
            note.append("<2/T: trend, not a period")
        for k in range(fr.size):
            if k != i and abs(fr[i] - fr[k]) < 1.0 / T:
                note.append(f"unresolved from {fr[k]:.4f} (<1/T)")
            elif k != i and abs(fr[i] - fr[k]) < 1.5 / T:
                note.append(f"marginal vs {fr[k]:.4f} (<1.5/T)")
        out.append(dict(frequency=float(fr[i]), frequency_err=float(sf[i]),
                        period=float(1 / fr[i]), period_err=float(sf[i] / fr[i] ** 2),
                        amplitude=float(am[i]), amplitude_err=float(sa[i]), phase=float(ph[i]),
                        snr_local=float(am[i] / local_noise(grid, A_res, fr[i], box)),
                        snr_red=float(am[i] / rn(fr[i])), note="; ".join(note)))
    return out, extra


def multisine_model(t_eval, t, y, frequencies):
    """Linear LSQ of trend + sinusoids at FIXED frequencies; returns model at t_eval."""
    t0 = t.mean()

    def X(tq):
        cols = [(tq - t0) ** k for k in range(POLY_DEG + 1)]
        for f in frequencies:
            cols += [np.cos(2 * np.pi * f * tq), np.sin(2 * np.pi * f * tq)]
        return np.column_stack(cols)

    c, *_ = np.linalg.lstsq(X(t), y, rcond=None)
    return X(np.asarray(t_eval, float)) @ c


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def format_frequency_table(peaks, units="mmag"):
    lines = [f"{'nu[1/d]':>9} {'err':>7} {'P[d]':>9} {'err':>7} {'A[' + units + ']':>9} {'err':>6} "
             f"{'S/N_loc':>7} {'S/N_red':>7}  note"]
    for p in peaks:
        lines.append(f"{p['frequency']:9.4f} {p['frequency_err']:7.4f} {p['period']:9.4f} {p['period_err']:7.4f} "
                     f"{p['amplitude']:9.2f} {p['amplitude_err']:6.2f} {p['snr_local']:7.1f} {p['snr_red']:7.1f}  "
                     f"{p['note']}")
    return "\n".join(lines)


def save_frequency_table(path, peaks, header_lines=(), units="mmag"):
    with open(path, "w") as fo:
        for h in header_lines:
            fo.write(f"# {h}\n")
        fo.write(format_frequency_table(peaks, units) + "\n")
    return path


def plot_amplitude_spectrum(t, y, peaks, extra, fmax, units="mmag", zoom=None, title=None,
                            out_path=None, figsize=(7, 5)):
    """
    Top: amplitude spectrum of the detrended data with 4x noise model and the
    found frequencies. Bottom (if zoom=(f1, f2)): zoom with the spectral window
    centred on the strongest frequency inside the zoom.
    """
    grid = np.arange(extra["grid"][0], fmax, extra["grid"][1] - extra["grid"][0])
    A = amp_spectrum(t, detrend_poly(t, y), grid)
    nrows = 2 if zoom else 1
    fig, axes = plt.subplots(nrows, 1, figsize=figsize, squeeze=False)
    ax = axes[0, 0]
    ax.plot(grid, A, "k-", lw=0.6)
    if "noise_model" in extra:
        ax.plot(grid, 4 * extra["noise_model"](grid), "--", color="0.45", lw=0.8, label="4 x noise")
        ax.legend(fontsize=7)
    for p in peaks:
        ax.axvline(p["frequency"], color="C0", lw=0.5, alpha=0.7)
    ax.set_xlim(0, fmax)
    ax.set_xlabel("Frequency, 1/d")
    ax.set_ylabel(f"Amplitude, {units}")
    if title:
        ax.set_title(title)
    if zoom:
        ax2 = axes[1, 0]
        z = (grid > zoom[0]) & (grid < zoom[1])
        ax2.plot(grid[z], A[z], "k-", lw=0.9, label="data")
        inside = [p for p in peaks if zoom[0] < p["frequency"] < zoom[1]]
        if inside:
            f0 = max(inside, key=lambda p: p["amplitude"])["frequency"]
            W = window_function(t, grid[z], f0)
            ax2.plot(grid[z], W / W.max() * A[z].max(), ":", color="0.4", label=f"window at {f0:.3f}")
        for p in inside:
            ax2.axvline(p["frequency"], color="C0", lw=0.8)
        ax2.set_xlabel("Frequency, 1/d")
        ax2.set_ylabel(f"Amplitude, {units}")
        ax2.legend(fontsize=7)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.show()
    return out_path or fig


def dynamic_spectrum(t, y, window=10.0, step=0.25, frange=(0.8, 2.5), df=0.004, min_fill=0.6):
    """
    Dynamic (sliding-window) amplitude spectrum: the amplitude spectrum is
    computed in a window of `window` days moved by `step` days. Shows whether a
    frequency is present during the whole run, and whether its frequency and
    amplitude change. Frequency resolution inside the window is ~1/window, so
    frequencies closer than that are NOT separated (they beat).
    Returns (centers, freqs, amplitude[len(centers), len(freqs)]).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    centers = np.arange(t.min() + window / 2, t.max() - window / 2 + 1e-9, step)
    freqs = np.arange(frange[0], frange[1], df)
    dyn = np.full((centers.size, freqs.size), np.nan)
    cadence = np.median(np.diff(t))
    for i, c in enumerate(centers):
        m = np.abs(t - c) < window / 2
        if m.sum() < min_fill * window / cadence:       # too many gaps in the window
            continue
        dyn[i] = amp_spectrum(t[m], detrend_poly(t[m], y[m]), freqs)
    return centers, freqs, dyn


def plot_dynamic_spectrum(centers, freqs, dyn, peaks=(), units="mmag", window=10.0, title=None,
                          out_path=None, figsize=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.ma.masked_invalid(dyn), origin="lower", aspect="auto", cmap="Greys",
                   interpolation="nearest",
                   extent=[freqs[0], freqs[-1], centers[0] - (centers[1] - centers[0]) / 2,
                           centers[-1] + (centers[1] - centers[0]) / 2])
    for p in peaks:
        if freqs[0] < p["frequency"] < freqs[-1]:
            ax.axvline(p["frequency"], color="C0", ls=":", lw=0.7)
    ax.set_xlabel("Frequency, 1/d")
    ax.set_ylabel(f"Window centre ({window:g} d)")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, label=f"Amplitude, {units}")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.show()
    return out_path or fig


def plot_prewhitening_fit(t, y, peaks, units="mmag", title=None, out_path=None, figsize=(8, 3),
                          data_color="purple", fit_color="green", invert=True):
    """
    Light curve with the model built from the prewhitened frequencies (+ trend),
    the analogue of `plot_periodogram_approximation` but with frequencies from
    `prewhiten` instead of the N highest LS peaks (which may include sidelobes).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    tt = np.linspace(t.min(), t.max(), 6000)
    model = multisine_model(tt, t, y, [p["frequency"] for p in peaks])
    for k in np.where(np.diff(t) > 0.1)[0]:          # do not draw the model across gaps
        model[(tt > t[k]) & (tt < t[k + 1])] = np.nan
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, y, ".", ms=1.5, color=data_color, label="light curve")
    ax.plot(tt, model, "-", color=fit_color, lw=1.0, label=f"{len(peaks)}-frequency fit + trend")
    if invert:
        ax.invert_yaxis()
    ax.set_xlabel("BJD - 2457000")
    ax.set_ylabel(f"Delta, {units}")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7, framealpha=1)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.show()
    return out_path or fig
