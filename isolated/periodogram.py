"""
Lomb-Scargle periodogram analysis for a single star/sector light curve:
period search plus false-alarm-probability significance, saving the result
to a .dat file / quick-look PNG, and approximating the light curve as a
sum of sinusoids at the strongest found frequencies (CLEAN-fit-style
overlay).

Builds on the same cleaning step as `stats.find_period`
(`stats.cleaned_jds_mags`), just keeping the full periodogram (and the
`astropy.timeseries.LombScargle` instance, needed for the FAP levels)
instead of only the single best period.

The periodogram itself is astropy's standard generalized (floating-mean)
Lomb-Scargle; only the significance (FAP) defaults to the classical/naive
method (independent-frequency Bonferroni correction on an exponential
per-frequency distribution) rather than Baluev's analytic approximation --
pass `fap_method="baluev"` to any of the FAP-computing functions below to
go back to that.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

from .config import STAR_DIRECTORY
from .geometry import get_nospace_star_name
from .stats import cleaned_jds_mags
from .lightcurve_tools import find_sampling


def suggest_period_range(mjd, nyquist_factor=2.0, max_period_fraction=1.0):
    """
    Suggest a (min_period, max_period) in days for `compute_ls_periodogram`'s
    `period_range`, from the light curve's own time sampling instead of a
    fixed guess.

    min_period = nyquist_factor * find_sampling(mjd) -- the *median* cadence
    (see `lightcurve_tools.find_sampling`), not the minimum gap between
    points. After cleaning, `np.diff(mjd)` is a mix of the normal cadence
    and a handful of much larger gaps left behind by excised windows (see
    `exclude_frame_windows`): the minimum gap is fragile (a single
    coincidentally-close pair of points sends it towards 0 and the
    frequency grid towards nonsense), and the *mean* gap is biased upward
    by the excision gaps. The median stays robust to both, as long as the
    excised gaps are a minority of all the gaps -- true for a normal
    handful of excised windows. `nyquist_factor=2` (the default) is the
    classic Nyquist limit: you need at least 2 samples per cycle to resolve
    a period at all; raise it for extra margin against timing jitter.

    max_period = max_period_fraction * (mjd.max() - mjd.min()) -- the total
    time baseline (post-cleaning), since a period you don't see repeat
    within your observing window isn't trustworthy. Lower it (e.g. 0.5) to
    require at least 2 full cycles inside the baseline.
    """
    mjd = np.asarray(mjd, dtype=float)
    sampling = find_sampling(mjd)
    baseline = mjd.max() - mjd.min()
    return nyquist_factor * sampling, max_period_fraction * baseline


def compute_ls_periodogram_from_series(t, y, period_range=(0.08, 10.0), samples_per_peak=10):
    """
    Lomb-Scargle periodogram of an already-prepared (t, y) series -- no
    cleaning/detrending here, unlike `compute_ls_periodogram` (which calls
    this internally after its own DataFrame-based cleaning step). For a
    series that was already cleaned/detrended elsewhere, e.g. per-sector by
    `multisector.stitch_light_curves` before stitching several sectors
    together (ROADMAP.md Этап 7) -- `cleaned_jds_mags`'s single-file
    box-smooth cleaning doesn't apply to an already-combined multi-sector
    series.

    Returns (freq, power, ls) -- see `compute_ls_periodogram`.
    """
    ls = LombScargle(t, y)
    min_freq = 1.0 / period_range[1]
    max_freq = 1.0 / period_range[0]
    freq, power = ls.autopower(minimum_frequency=min_freq, maximum_frequency=max_freq,
                                samples_per_peak=samples_per_peak)
    return freq, power, ls


def compute_ls_periodogram(df_lc, jd_box=0.1, sigma_tol=10, n_out=20,
                            period_range=(0.08, 10.0), samples_per_peak=10, detrend_deg=0):
    """
    Lomb-Scargle periodogram of a light curve's magnitudes (median-subtracted,
    sigma-clipped the same way as `stats.find_period`).

    Returns (freq, power, ls): ascending in frequency (cycles/day), and
    `ls` the astropy `LombScargle` instance (needed for
    false_alarm_probability/false_alarm_level -- see `find_periodogram_peaks`
    and `save_periodogram`).

    `detrend_deg` > 0 subtracts a polynomial of that degree in time first.
    Without it a slow trend over the sector leaks into the lowest frequencies
    (the highest "peak" then sits at 1/T or 2/T -- an artefact, not a period).
    Default 0 keeps the old behaviour.
    """
    cleaned_jds, cleaned_mags = cleaned_jds_mags(df_lc, jd_box, sigma_tol, n_out)
    y = cleaned_mags - np.median(cleaned_mags)
    if detrend_deg > 0:
        x = cleaned_jds - cleaned_jds.mean()
        y = y - np.polyval(np.polyfit(x, y, detrend_deg), x)

    return compute_ls_periodogram_from_series(cleaned_jds, y, period_range, samples_per_peak)


def find_periodogram_peaks(freq, power, ls, fap_levels=(0.1, 0.01, 0.001), max_peaks=10, fap_method="naive"):
    """
    Local-maxima peaks of the periodogram, each with its frequency, period,
    power and false-alarm probability. `fap_method="naive"` (the default)
    is the classical method: treats each sampled frequency as an
    independent trial with an exponential null-hypothesis power
    distribution (Bonferroni correction over the actual number of
    frequency samples) -- as opposed to `"baluev"`, which instead estimates
    an effective number of independent frequencies analytically. Lower FAP
    == more significant either way.

    Returns up to `max_peaks` peaks sorted by ascending FAP (most
    significant first):
        [{"frequency": ..., "period": ..., "power": ..., "fap": ...}, ...]
    """
    is_peak = np.r_[False, (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:]), False]
    idx = np.nonzero(is_peak)[0]
    if len(idx) == 0:
        return []

    faps = ls.false_alarm_probability(power[idx], method=fap_method)
    peaks = [{"frequency": float(freq[i]), "period": float(1.0 / freq[i]),
              "power": float(power[i]), "fap": float(fap)}
             for i, fap in zip(idx, faps)]
    peaks.sort(key=lambda p: p["fap"])
    return peaks[:max_peaks]


def save_periodogram(star_name, sector, freq, power, ls, cut_width, cut_height=None,
                      fap_levels=(0.1, 0.01, 0.001), fap_method="naive", peaks=None,
                      star_directory=STAR_DIRECTORY):
    """
    Save the periodogram (frequency, power columns) as a .dat file next to
    the star's other cached data, with the FAP significance thresholds and
    the most significant peaks written into the header for a quick look:
        {star_directory}/{star}/{cut_width}x{cut_height}/periodogram_sector_{sector}.dat

    Returns the path written to.
    """
    if cut_height is None:
        cut_height = cut_width
    if peaks is None:
        peaks = find_periodogram_peaks(freq, power, ls, fap_levels, fap_method=fap_method)

    levels = ls.false_alarm_level(fap_levels, method=fap_method)
    header_lines = [
        "Lomb-Scargle periodogram",
        f"star: {star_name}, sector: {sector}",
        f"FAP method: {fap_method}",
        "",
        *[f"FAP {fap:g} -> power {level:.6g}" for fap, level in zip(fap_levels, levels)],
        "",
        "Most significant peaks (frequency[1/d]  period[d]  power  FAP):",
        *[f"  {p['frequency']:.6f}  {p['period']:.6f}  {p['power']:.6f}  {p['fap']:.3e}" for p in peaks],
        "",
        "frequency[1/d] power",
    ]

    out_dir = os.path.join(star_directory, get_nospace_star_name(star_name), f"{cut_width}x{cut_height}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"periodogram_sector_{sector}.dat")
    np.savetxt(out_path, np.column_stack([freq, power]), header="\n".join(header_lines), fmt="%.6f %.6e")
    return out_path


def plot_periodogram(star_name, sector, freq, power, ls, fap_levels=(0.1, 0.01, 0.001), fap_method="naive",
                      peaks=None, out_path=None, max_peaks_for_plotting=5, figsize=(7, 3),
                      line_color="navy", fap_color="magenta", peak_color="darkred"):
    """
    Plot frequency (linear axis, cycles/day) vs power, with FAP threshold
    lines and the strongest peaks marked -- same layout as a classical
    Lomb-Scargle/CLEAN periodogram plot. If `out_path` is given the figure
    is also saved there (and still shown) -- format is whatever its
    extension says (e.g. ".eps" for a publication-ready vector figure;
    note EPS has no transparency, which is why nothing here uses alpha).

    `figsize`/`line_color`/`fap_color`/`peak_color` default to a compact,
    print-friendly look; override them to match a particular journal style.
    """
    if peaks is None:
        peaks = find_periodogram_peaks(freq, power, ls, fap_levels, fap_method=fap_method)
    levels = ls.false_alarm_level(fap_levels, method=fap_method)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freq, power, lw=0.8, color=line_color)
    ax.set_xlabel("Frequency, 1/d")
    ax.set_ylabel("LS power")
    ax.set_title(f"{star_name}, sector {sector}")

    line_styles = ["--", "-.", ":"]
    for fap, level, line_style in zip(fap_levels, levels, line_styles):
        ax.axhline(level, color=fap_color, ls=line_style, lw=0.8, label=f"FAP = {fap:g}")

    y_top = max(power.max(), levels.max()) * 1.35  # headroom for the rotated peak labels
    ax.set_ylim(0, y_top)
    label_y_gap = y_top * 0.06

    for p in peaks[:max_peaks_for_plotting]:
        label_y = p["power"] + label_y_gap
        ax.plot([p["frequency"], p["frequency"]], [0, label_y], color=peak_color, ls=":", lw=0.6)
        ax.text(p["frequency"], label_y, f"{p['frequency']:.3f} 1/d", fontsize=7,
                rotation=90, va="bottom", ha="center")
    ax.legend(loc="upper right", fontsize=7, framealpha=1)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.show()
        return out_path
    plt.show()
    return fig


def fit_multi_sine(jds, mags, frequencies):
    """
    Least-squares fit of `mags` as a constant offset plus a sum of
    sinusoids at fixed `frequencies` (cycles/day):
        mag(t) = mean + sum_k [a_k*cos(2*pi*f_k*t) + b_k*sin(2*pi*f_k*t)]
    Since the frequencies are fixed (e.g. from periodogram peaks), this is
    a plain linear least-squares problem, same idea as reconstructing a
    CLEAN fit from its found frequencies.

    Returns (predict, coeffs): `predict(t)` evaluates the fit at arbitrary
    times (scalar or array); `coeffs` is the raw solution
    [mean, a_1, b_1, a_2, b_2, ...].
    """
    frequencies = np.asarray(frequencies, dtype=float)

    def design_matrix(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        cols = [np.ones_like(t)]
        for f in frequencies:
            cols.append(np.cos(2 * np.pi * f * t))
            cols.append(np.sin(2 * np.pi * f * t))
        return np.column_stack(cols)

    coeffs, *_ = np.linalg.lstsq(design_matrix(jds), mags, rcond=None)

    def predict(t):
        return design_matrix(t) @ coeffs

    return predict, coeffs


def fit_periodogram_approximation(df_lc, peaks, n_frequencies=5, jd_box=0.1, sigma_tol=10, n_out=20):
    """
    Approximate the (cleaned) light curve as a constant plus a sum of
    sinusoids at the `n_frequencies` strongest periodogram peaks (see
    `find_periodogram_peaks`, already sorted most-significant-first). If
    fewer peaks than `n_frequencies` were found, all of them are used.

    Returns (cleaned_jds, cleaned_mags, predict): `predict(t)` evaluates
    the fitted approximation at arbitrary JDs -- see
    `plot_periodogram_approximation` for a ready-made plot of this.
    """
    cleaned_jds, cleaned_mags = cleaned_jds_mags(df_lc, jd_box, sigma_tol, n_out)
    frequencies = [p["frequency"] for p in peaks[:n_frequencies]]
    predict, coeffs = fit_multi_sine(cleaned_jds, cleaned_mags, frequencies)
    return cleaned_jds, cleaned_mags, predict


def plot_periodogram_approximation(star_name, sector, df_lc, peaks, n_frequencies=5,
                                    jd_box=0.1, sigma_tol=10, n_out=20, out_path=None, figsize=(8, 3),
                                    data_color="purple", fit_color="green"):
    """
    Plot the (cleaned) light curve together with its multi-sine
    approximation built from the `n_frequencies` strongest periodogram
    peaks -- data as points, fit as a dashed line, the same idea as a
    CLEAN-fit overlay on the TESS flux. If `out_path` is given the figure
    is also saved there -- format is whatever its extension says (e.g.
    ".eps" for a publication-ready vector figure; note EPS has no
    transparency, which is why nothing here uses alpha).

    `figsize`/`data_color`/`fit_color` default to a compact, print-friendly
    look; override them to match a particular journal style.
    """
    cleaned_jds, cleaned_mags, predict = fit_periodogram_approximation(
        df_lc, peaks, n_frequencies, jd_box, sigma_tol, n_out)

    dense_t = np.linspace(cleaned_jds.min(), cleaned_jds.max(), 3000)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cleaned_jds, cleaned_mags, ".", ms=2, color=data_color, label="light curve")
    ax.plot(dense_t, predict(dense_t), "--", color=fit_color, lw=1.2, label=f"{n_frequencies}-frequency fit")
    ax.invert_yaxis()
    ax.set_xlabel("JD")
    ax.set_ylabel("TESS magnitude")
    ax.set_title(f"{star_name}, sector {sector}" if star_name else "periodogram approximation")
    ax.legend(fontsize=7, framealpha=1)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.show()
        return out_path
    plt.show()
    return fig
