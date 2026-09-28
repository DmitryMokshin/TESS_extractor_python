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

def complex_spectrum(t, y, freqs, chunk=400):
    """
    Complex DFT, in units of y: A(f) = 2/N sum y exp(-2 pi i f t). `amp_spectrum`
    is `np.abs(complex_spectrum(...))` -- this is the complex version ROADMAP.md
    Этап 8's CLEAN (`clean_periodogram`) needs, to read off both the amplitude
    AND phase of the strongest residual peak at each iteration.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float) - np.mean(y)
    out = np.empty(freqs.size, dtype=complex)
    for k in range(0, freqs.size, chunk):
        ph = np.exp(-2j * np.pi * np.outer(freqs[k:k + chunk], t))
        out[k:k + chunk] = 2.0 / t.size * (ph @ y)
    return out


def amp_spectrum(t, y, freqs, chunk=400):
    """Amplitude spectrum (DFT), in units of y: A(f) = 2/N |sum y exp(-2 pi i f t)|."""
    return np.abs(complex_spectrum(t, y, freqs, chunk))


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


# --------------------------------------------------------------------------- #
# Error checks: bootstrap and split-half (ROADMAP.md Этап 8)
# --------------------------------------------------------------------------- #

def bootstrap_frequency_errors(t, y, peaks, extra, n_boot=200, halfwidth=None, rng=None):
    """
    Residual-bootstrap empirical std of each found frequency/amplitude/phase --
    an independent, formula-free check on `errors_mo99`'s Montgomery & O'Donoghue
    (1999) sigma_f (ROADMAP.md Этап 8, "ошибки частот бутстрепом по остаткам").
    Resamples `extra["residuals"]` (the final `prewhiten` residuals) WITH
    replacement, adds them back onto the fitted model, and refits the SAME
    frequencies each time via `fit_all`, anchored within `halfwidth` (default
    0.5/T) so bootstrap noise can't swap or merge close frequencies.

    Returns `{"frequency": std_array, "amplitude": std_array, "phase": std_array}`,
    one value per peak, same order as `peaks`.
    """
    if not peaks:
        return {"frequency": np.array([]), "amplitude": np.array([]), "phase": np.array([])}
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    rng = rng or np.random.default_rng()
    if halfwidth is None:
        halfwidth = 0.5 / extra["T"]
    residuals = extra["residuals"]
    model_t = y - residuals
    freqs0 = [p["frequency"] for p in peaks]
    amps0 = [p["amplitude"] for p in peaks]
    phases0 = [p["phase"] for p in peaks]

    fr_boot = np.empty((n_boot, len(peaks)))
    am_boot = np.empty((n_boot, len(peaks)))
    ph_boot = np.empty((n_boot, len(peaks)))
    for i in range(n_boot):
        y_boot = model_t + rng.choice(residuals, size=t.size, replace=True)
        fr_b, am_b, ph_b, _ = fit_all(t, y_boot, freqs0, amps0, phases0,
                                      anchors=freqs0, halfwidth=halfwidth)
        fr_boot[i], am_boot[i], ph_boot[i] = fr_b, am_b, ph_b

    return {"frequency": fr_boot.std(axis=0), "amplitude": am_boot.std(axis=0),
            "phase": ph_boot.std(axis=0)}


def split_half_frequency_errors(t, y, peaks, halfwidth=None):
    """
    Fit the found frequencies independently on the first and second half of the
    time series (split at the median time) -- an independent, formula-free error
    estimate (ROADMAP.md Этап 8, "...и по двум половинам ряда"): how much do the
    frequencies/amplitudes/phases differ between two halves of the same run? Not
    a replacement for `errors_mo99`'s sigma_f, a cross-check on it.

    `halfwidth` (default 1/T_half, generous -- unlike the bootstrap's tighter
    anchor, this must let each half's fit actually reflect its own resolution,
    not be clamped back toward the full-series value) bounds how far a
    frequency may move from its full-series value on each half.

    Returns `(first, second)`, each `{"frequency": array, "amplitude": array,
    "phase": array}`, one value per peak (same order as `peaks`). A caller
    compares e.g. `abs(first["frequency"] - second["frequency"])` against
    `sigma_f * sqrt(2)` (two independent formal errors) or similar.
    """
    if not peaks:
        empty = {"frequency": np.array([]), "amplitude": np.array([]), "phase": np.array([])}
        return empty, dict(empty)
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    freqs0 = [p["frequency"] for p in peaks]
    amps0 = [p["amplitude"] for p in peaks]
    phases0 = [p["phase"] for p in peaks]

    tmid = np.median(t)
    results = []
    for mask in (t <= tmid, t > tmid):
        t_h, y_h = t[mask], y[mask]
        hw = halfwidth if halfwidth is not None else 1.0 / (t_h.max() - t_h.min())
        fr_h, am_h, ph_h, _ = fit_all(t_h, y_h, freqs0, amps0, phases0,
                                      anchors=freqs0, halfwidth=hw)
        results.append({"frequency": fr_h, "amplitude": am_h, "phase": ph_h})
    return results[0], results[1]


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


def find_combination_frequencies(peaks, n_top=5, n_sigma=3.0):
    """
    For the `n_top` strongest "parent" frequencies, check every other (weaker)
    frequency against their sum/difference -- same search `ss397_tess.py` used
    to do inline, but with the tolerance now `n_sigma * sqrt(sigma_f_i^2 +
    sigma_f_j^2 + sigma_f_k^2)` (each peak's own `frequency_err` from
    `errors_mo99`) instead of a flat `0.5/T` (ROADMAP.md Этап 8, "комбинационные
    частоты с допуском по ошибкам, а не просто 0.5/T").

    Returns a list of dicts `{child, parent1, parent2, kind ('+'/'-'),
    predicted, tolerance}` -- `child`/`parent1`/`parent2` are indices into
    `peaks`.
    """
    n = len(peaks)
    if n == 0:
        return []
    amps = np.array([p["amplitude"] for p in peaks])
    freqs = np.array([p["frequency"] for p in peaks])
    sigmas = np.array([p["frequency_err"] for p in peaks])
    top = np.argsort(amps)[::-1][:n_top]

    combinations = []
    for ii, i in enumerate(top):
        for j in top[ii:]:
            for kind, predicted in [("+", freqs[i] + freqs[j]), ("-", abs(freqs[i] - freqs[j]))]:
                for k in range(n):
                    if k in (i, j) or amps[k] >= min(amps[i], amps[j]):
                        continue
                    tolerance = n_sigma * np.sqrt(sigmas[i] ** 2 + sigmas[j] ** 2 + sigmas[k] ** 2)
                    if abs(freqs[k] - predicted) < tolerance:
                        combinations.append({"child": int(k), "parent1": int(i), "parent2": int(j),
                                             "kind": kind, "predicted": float(predicted),
                                             "tolerance": float(tolerance)})
    return combinations


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


def frequency_stability(t, y, frequencies, window=10.0, step=1.0, min_fill=0.5):
    """
    Track each of the given (already found, FIXED) `frequencies`' amplitude and
    phase in a sliding window of `window` days moved by `step` days -- shows
    whether a real signal's amplitude/phase drifts or modulates over the run
    (ROADMAP.md Этап 8, "стабильность амплитуд и фаз: подгонка найденных частот
    в скользящих окнах"). Complementary to `dynamic_spectrum` (which scans a
    frequency RANGE per window to find where power sits, for when you don't yet
    know the frequency); this tracks specific, already-known frequencies
    precisely via a linear fit (`_phase_guess`) instead of a spectrum scan.

    Same gap-handling convention as `dynamic_spectrum`: a window with fewer than
    `min_fill` of the expected point count (given the median cadence) is
    skipped (NaN). Each window is locally detrended (`detrend_poly`, same
    default degree as `dynamic_spectrum` uses) before the fit, same reasoning:
    a window's own slow local drift shouldn't bias its amplitude/phase estimate.

    Returns `(centers, amplitudes[len(centers), len(frequencies)],
    phases[len(centers), len(frequencies)])`.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    frequencies = np.asarray(frequencies, dtype=float)
    centers = np.arange(t.min() + window / 2, t.max() - window / 2 + 1e-9, step)
    amplitudes = np.full((centers.size, frequencies.size), np.nan)
    phases = np.full((centers.size, frequencies.size), np.nan)
    cadence = np.median(np.diff(t))
    for i, c in enumerate(centers):
        m = np.abs(t - c) < window / 2
        if m.sum() < min_fill * window / cadence:        # too many gaps in the window
            continue
        t_w = t[m]
        y_w = detrend_poly(t_w, y[m])
        for j, f in enumerate(frequencies):
            amp, phase = _phase_guess(t_w, y_w, f)
            amplitudes[i, j] = amp
            phases[i, j] = phase
    return centers, amplitudes, phases


def plot_frequency_stability(centers, amplitudes, phases, frequencies, units="mmag", window=10.0,
                             title=None, out_path=None, figsize=(8, 5)):
    """
    Amplitude and phase of each tracked frequency over time -- one line per
    frequency, two panels (amplitude, phase), same sliding-window convention as
    `dynamic_spectrum`/`plot_dynamic_spectrum`.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    for j, f in enumerate(frequencies):
        ax1.plot(centers, amplitudes[:, j], ".-", ms=3, lw=0.8, label=f"{f:.4f} 1/d")
        ax2.plot(centers, phases[:, j], ".", ms=3)
    ax1.set_ylabel(f"Amplitude, {units}")
    ax1.legend(fontsize=7, ncol=min(len(frequencies), 4) or 1)
    ax2.set_ylabel("Phase")
    ax2.set_xlabel(f"Window centre ({window:g} d)")
    if title:
        ax1.set_title(title)
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


# --------------------------------------------------------------------------- #
# CLEAN (Roberts, Lehar & Dreher 1987) -- ROADMAP.md Этап 8
# --------------------------------------------------------------------------- #

def clean_periodogram(t, y, freqs, gain=0.2, n_iter=200, stop_fraction=0.05):
    """
    CLEAN deconvolution (Roberts, Lehar & Dreher 1987) for an unevenly-sampled
    time series -- ROADMAP.md Этап 8, "опция CLEAN... для сравнения с
    программами коллег". Formulated in the TIME domain (subtract a scaled,
    phase-matched sinusoid from the residual light curve at each iteration)
    rather than the classical "subtract a scaled dirty beam from the spectrum"
    formulation: by linearity of the DFT the two are equivalent, but the
    time-domain version needs no separate complex spectral-window machinery --
    subtracting the right sinusoid from the data automatically removes the
    correctly phase-aligned amount from every frequency in the spectrum,
    sidelobes included.

    At each iteration: find the strongest peak (by magnitude) in the residual's
    complex spectrum (`complex_spectrum`) over `freqs`, subtract `gain` of a
    matching sinusoid from the residual light curve, and record a CLEAN
    component (frequency, amplitude, phase in the module's own
    `sin(2*pi*(f*t+phase))` convention). Stops after `n_iter` iterations or
    once the strongest residual peak drops below `stop_fraction` of the very
    first (dirty) peak.

    Returns `(components, residual, dirty_amp)`: `components` is a list of
    dicts `{frequency, amplitude, phase}` (one per CLEAN iteration -- not
    merged by frequency, several may land near the same real peak),
    `residual` the final residual light curve (same shape as `y`), `dirty_amp`
    the original (pre-CLEAN) amplitude spectrum over `freqs`, for a
    before/after comparison plot (see `plot_clean_spectrum`).
    """
    t = np.asarray(t, float)
    residual = np.asarray(y, float) - np.mean(y)
    dirty_amp = amp_spectrum(t, residual, freqs)
    a0 = None
    components = []
    for _ in range(n_iter):
        spec = complex_spectrum(t, residual, freqs)
        idx = int(np.argmax(np.abs(spec)))
        amp = float(np.abs(spec[idx]))
        if a0 is None:
            a0 = amp
        if amp < stop_fraction * a0:
            break
        f0 = float(freqs[idx])
        angle = float(np.angle(spec[idx]))
        residual = residual - gain * amp * np.cos(2 * np.pi * f0 * t + angle)
        phase = (angle / (2 * np.pi) + 0.25) % 1
        components.append({"frequency": f0, "amplitude": gain * amp, "phase": phase})
    return components, residual, dirty_amp


def plot_clean_spectrum(freqs, dirty_amp, components, units="mmag", title=None, out_path=None, figsize=(7, 4)):
    """Dirty (original) amplitude spectrum with CLEAN components overlaid as
    stems -- same panel layout as `plot_amplitude_spectrum`."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freqs, dirty_amp, "-", color="0.6", lw=0.7, label="dirty spectrum")
    if components:
        cf = [c["frequency"] for c in components]
        ca = [c["amplitude"] for c in components]
        ax.stem(cf, ca, linefmt="C3-", markerfmt="C3.", basefmt=" ")
        ax.plot([], [], "C3-", label="CLEAN components")  # legend entry for the stems
    ax.set_xlabel("Frequency, 1/d")
    ax.set_ylabel(f"Amplitude, {units}")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.show()
    return out_path or fig
