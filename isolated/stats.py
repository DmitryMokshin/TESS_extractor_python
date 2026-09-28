"""
Per-light-curve statistics.

Ported from process-func.jl:
    calc_noise_rms, find_period, calc_periodicity, calc_asymmetry
"""
from __future__ import annotations

import numpy as np
from astropy.timeseries import LombScargle

from .geometry import get_true_jd
from .lightcurve_tools import delete_nans, clean_flux_sigma, box_smooth, find_sampling


def cleaned_jds_mags(df_lc, jd_box=0.1, sigma_tol=10, n_out=20):
    """NaN-dropped, sigma-clipped (jds, mags) pair shared by the stats below
    and by the periodogram module."""
    jds, mags = delete_nans(get_true_jd(df_lc["BTJD" if "BTJD" in df_lc.columns else "MJD"].to_numpy()),
                            df_lc["MAG"].to_numpy())
    mags = mags.copy()
    clean_flux_sigma(jds, mags, jd_box, sigma_tol, n_out)
    return delete_nans(jds, mags)


def calc_noise_rms(df_lc, jd_box=0.1, sigma_tol=10, n_out=20) -> float:
    """Direct port of `calc_noise_rms`."""
    cleaned_jds, cleaned_mags = cleaned_jds_mags(df_lc, jd_box, sigma_tol, n_out)

    smooth = box_smooth(cleaned_jds, cleaned_mags, jd_box)
    anomalous = cleaned_mags - smooth
    noout = np.sort(anomalous)[: len(anomalous) - n_out]
    return float(np.sqrt(np.mean((noout - np.mean(noout)) ** 2)))


def find_period(df_lc, jd_box=0.1, sigma_tol=10, n_out=20, period_range=(0.08, 10.0)) -> float:
    """
    Direct port of `find_period`: Lomb-Scargle period of maximum power within
    `period_range` days (Julia: `findmaxperiod(pgram, [0.08, 10])[1]`).
    """
    cleaned_jds, cleaned_mags = cleaned_jds_mags(df_lc, jd_box, sigma_tol, n_out)
    y = cleaned_mags - np.median(cleaned_mags)

    ls = LombScargle(cleaned_jds, y)
    min_freq = 1.0 / period_range[1]
    max_freq = 1.0 / period_range[0]
    freq, power = ls.autopower(minimum_frequency=min_freq, maximum_frequency=max_freq)

    return float(1.0 / freq[np.argmax(power)])


def calc_periodicity(df_lc, jd_box=0.1, sigma_tol=10, n_out=20) -> float:
    """Direct port of `calc_periodicity`."""
    cleaned_jds, cleaned_mags = cleaned_jds_mags(df_lc, jd_box, sigma_tol, n_out)

    period = find_period(df_lc, jd_box, sigma_tol, n_out)
    phased_jds = np.mod(cleaned_jds, period)

    n_mags = len(cleaned_mags)
    boxed_mags = np.empty(n_mags)
    for i in range(n_mags):
        boxed_mags[i] = np.median(cleaned_mags[np.abs(phased_jds - phased_jds[i]) < period * 0.01])

    mean_mag = np.mean(cleaned_mags)
    mag_dispersion = np.sqrt(np.mean((cleaned_mags - mean_mag) ** 2))
    diff = cleaned_mags - boxed_mags
    mean_phased = np.mean(diff)
    mag_phased_dispersion = np.sqrt(np.mean((diff - mean_phased) ** 2))

    noise = calc_noise_rms(df_lc, jd_box, sigma_tol, n_out)

    return (mag_phased_dispersion - noise) / (mag_dispersion - noise)


def calc_asymmetry(df_lc, jd_box=0.1, sigma_tol=10, n_out=20) -> float:
    """Direct port of `calc_asymmetry`."""
    cleaned_jds, cleaned_mags = cleaned_jds_mags(df_lc, jd_box, sigma_tol, n_out)

    order = np.argsort(cleaned_mags)
    n_mags = len(cleaned_mags)
    n_10 = round(n_mags / 10)

    top_bottom = np.concatenate([cleaned_mags[order[-n_10:]], cleaned_mags[order[:n_10]]])
    mean_10 = np.mean(top_bottom)

    median_mag = np.median(cleaned_mags)
    sigma = np.sqrt(np.mean((cleaned_mags - median_mag) ** 2))

    return (mean_10 - median_mag) / sigma
