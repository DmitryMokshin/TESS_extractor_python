"""
Light-curve cleaning / analysis helpers and the batch "get everything for
these stars" driver.

Ported from main.jl:
    delete_nans, box_smooth, clean_flux_sigma!, clean_flux!, save_lc_figure,
    find_sampling, find_ACF, get_all_data
"""
from __future__ import annotations

import os
import traceback

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button
from astropy.time import Time
from tqdm.auto import tqdm

from .config import STAR_DIRECTORY, TESS_MAX_SECTORS
from .geometry import get_nospace_star_name, get_true_jd
from .tess_point import find_tess_sectors
from .data_io import load_light_curve


def delete_nans(jds, fluxs):
    """Direct port of `delete_nans`."""
    jds = np.asarray(jds, dtype=float)
    fluxs = np.asarray(fluxs, dtype=float)
    mask = ~np.isnan(fluxs)
    return jds[mask], fluxs[mask]


def box_smooth(jds, fluxs, jd_box):
    """Direct port of `box_smooth`: running median in a +/-jd_box window."""
    jds = np.asarray(jds, dtype=float)
    fluxs = np.asarray(fluxs, dtype=float)
    smooth = np.empty_like(fluxs)
    for i, jd in enumerate(jds):
        smooth[i] = np.median(fluxs[np.abs(jds - jd) < jd_box])
    return smooth


def clean_flux_sigma(jds, fluxs, jd_box, sigma_tol, n_out):
    """
    Direct port of `clean_flux_sigma!`. Modifies `fluxs` in place (setting
    outliers to NaN), same as the Julia `!`-function.
    """
    smooth = box_smooth(jds, fluxs, jd_box)
    anomalous = fluxs - smooth
    noout = np.sort(anomalous)[: len(anomalous) - n_out]
    # Julia `varm(x, m)` == mean((x .- m).^2), i.e. population variance about m
    sigma = np.sqrt(np.mean((noout - np.mean(noout)) ** 2))
    for i in range(len(fluxs)):
        if abs(anomalous[i]) > sigma * sigma_tol:
            fluxs[i] = np.nan


def clean_flux(mjd, flux, pred_tol=0.1, jd_tol=0.1):
    """
    Direct port of `clean_flux!`: flags points that deviate from a linear
    (2-point) prediction, or that follow a time gap, as NaN. Modifies `flux`
    in place.
    """
    n = len(flux)
    i = 2  # Julia i_flux starts at 3 (1-based) -> 0-based index 2
    while i < n:
        pred_point = 2 * flux[i - 1] - flux[i - 2]
        if abs(flux[i] - pred_point) > pred_tol:
            flux[i] = np.nan
            i += 3
            if i >= n:
                break
        if (mjd[i] - mjd[i - 1]) > jd_tol:
            flux[i] = np.nan
        i += 1

    if abs(flux[0] - 2 * flux[1] + flux[2]) > pred_tol:
        flux[0] = np.nan
        flux[1] = np.nan

    if abs(flux[-1] - 2 * flux[-2] + flux[-3]) > pred_tol:
        flux[-1] = np.nan
        flux[-2] = np.nan


def exclude_frame_windows(df_lc, windows):
    """
    Split off rows falling inside any of the given frame-number windows, e.g.
        windows = [[2500, 2700], [10300, 10650]]
    Same 1-based, inclusive convention as `save_cutout_video`'s frame_range,
    so a bad stretch spotted in a rendered video clip can be excised with
    the exact same numbers.

    If `df_lc` has a `FRAME` column (ROADMAP.md Этап 1; written by
    `load_light_curve`/`prf_photometry.load_prf_light_curve`), windows are
    matched against it directly, so this also works correctly on a curve
    that already had some rows dropped (frame numbers no longer equal row
    position there). Older cached files without `FRAME` fall back to the
    original assumption: `df_lc` is the untouched, full-cadence light curve
    (one row per cutout frame, in frame order) -- row i (0-based) is frame
    i + 1.

    Returns (clean_df, trash_df): clean_df is everything outside the
    windows, trash_df is just the excised rows (nothing is discarded, so it
    can still be inspected -- see `save_trash_light_curve`). Both are copies
    with the index reset; `df_lc` is left untouched.
    """
    n = len(df_lc)
    frame_no = df_lc["FRAME"].to_numpy() if "FRAME" in df_lc.columns else np.arange(1, n + 1)
    keep = np.ones(n, dtype=bool)
    for start, end in windows:
        keep &= ~((frame_no >= start) & (frame_no <= end))
    return df_lc.loc[keep].reset_index(drop=True), df_lc.loc[~keep].reset_index(drop=True)


def windows_to_log(df_lc, windows, reason):
    """
    Log entries (`FRAME, BTJD, REASON` columns) for the rows
    `exclude_frame_windows` would drop for `windows` -- lets a manual window
    pick (`pick_exclusion_windows`) be recorded in the same log format as
    `cleaning.auto_clean_light_curve`'s automatic rules, so both can be
    concatenated (`pd.concat`) into one combined cleaning log
    (ROADMAP.md Этап 5).
    """
    _clean, trash = exclude_frame_windows(df_lc, windows)
    time_col = "BTJD" if "BTJD" in df_lc.columns else "MJD"
    frame_no = trash["FRAME"] if "FRAME" in trash.columns else trash.index + 1
    return pd.DataFrame({"FRAME": frame_no, "BTJD": trash[time_col], "REASON": reason})


class WindowPicker:
    """Holds the figure/state for `pick_exclusion_windows`. `.windows` is the
    running list of [start_frame, end_frame] (1-based, inclusive) windows,
    same convention as `save_cutout_video`'s frame_range."""

    def __init__(self, df_lc, star_name="", sector=None):
        self.df_lc = df_lc
        time_col = "BTJD" if "BTJD" in df_lc.columns else "MJD"
        self.mjds = df_lc[time_col].to_numpy()
        self.frame_no = df_lc["FRAME"].to_numpy() if "FRAME" in df_lc.columns else None
        self.windows = []
        self._spans = []

        self.fig, self.ax = plt.subplots(figsize=(15, 5))
        self.fig.subplots_adjust(bottom=0.22)
        self.ax.plot(self.mjds, df_lc["MAG"].to_numpy(), lw=0.8)
        self.ax.invert_yaxis()
        self.ax.set_xlabel(time_col)
        self.ax.set_ylabel("TESS magnitude")
        title = f"{star_name}, sector {sector}" if star_name else "light curve"
        self.ax.set_title(f"{title} -- drag to mark a bad window, Undo/Done below")

        ax_undo = self.fig.add_axes([0.70, 0.05, 0.1, 0.06])
        ax_done = self.fig.add_axes([0.81, 0.05, 0.1, 0.06])
        self.undo_button = Button(ax_undo, "Undo")
        self.done_button = Button(ax_done, "Done")
        self.undo_button.on_clicked(self._on_undo)
        self.done_button.on_clicked(lambda event: plt.close(self.fig))

        # keep a reference so the widget isn't garbage-collected
        self.span = SpanSelector(self.ax, self._on_select, "horizontal", useblit=True,
                                  props=dict(alpha=0.3, facecolor="red"), interactive=False)

    def _on_select(self, xmin, xmax):
        pos_start = int(np.searchsorted(self.mjds, xmin, side="left"))
        pos_end = int(np.searchsorted(self.mjds, xmax, side="right")) - 1
        pos_start, pos_end = max(0, pos_start), min(len(self.mjds) - 1, pos_end)
        if pos_end < pos_start:
            return
        if self.frame_no is not None:
            self.windows.append([int(self.frame_no[pos_start]), int(self.frame_no[pos_end])])
        else:
            self.windows.append([pos_start + 1, pos_end + 1])
        self._spans.append(self.ax.axvspan(xmin, xmax, color="red", alpha=0.3))
        self.fig.canvas.draw_idle()

    def _on_undo(self, event):
        if self.windows:
            self.windows.pop()
            self._spans.pop().remove()
            self.fig.canvas.draw_idle()


def pick_exclusion_windows(df_lc, star_name="", sector=None):
    """
    Interactive picker: drag across bad-looking stretches of the light curve
    to mark them (shown as a red band); "Undo" removes the last mark,
    "Done" (or just closing the window) finishes.

    Returns a list of [start_frame, end_frame] windows -- 1-based, inclusive,
    same convention as `save_cutout_video`'s frame_range -- ready to pass to
    `exclude_frame_windows`, or to double-check with `save_cutout_video`.
    """
    picker = WindowPicker(df_lc, star_name, sector)
    plt.show()
    return picker.windows


def save_clean_light_curve(df_lc, star_name, sector, cut_width, cut_height=None, star_directory=STAR_DIRECTORY):
    """
    Save a cleaned light curve (e.g. the output of `exclude_frame_windows`)
    next to the original one written by `load_light_curve`, same naming but
    with a "_clean" suffix: light_curve_sector_{sector}_clean.csv.

    Returns the path written to.
    """
    if cut_height is None:
        cut_height = cut_width
    out_dir = os.path.join(star_directory, get_nospace_star_name(star_name), f"{cut_width}x{cut_height}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"light_curve_sector_{sector}_clean.csv")
    df_lc.to_csv(out_path, index=False)
    return out_path


def save_trash_light_curve(df_trash, star_name, sector, windows, cut_width, cut_height=None,
                            star_directory=STAR_DIRECTORY):
    """
    Save the rows excised by `exclude_frame_windows` next to the original
    light curve, so they can be inspected later instead of being thrown
    away. Same naming as `save_clean_light_curve`, but "_trash" plus the
    frame windows that were cut, e.g.
        light_curve_sector_43_trash_frames_2500-2700_10300-10650.csv

    `windows` should be the same list passed to `exclude_frame_windows`.
    Returns the path written to.
    """
    if cut_height is None:
        cut_height = cut_width
    out_dir = os.path.join(star_directory, get_nospace_star_name(star_name), f"{cut_width}x{cut_height}")
    os.makedirs(out_dir, exist_ok=True)
    windows_tag = "_".join(f"{start}-{end}" for start, end in windows)
    out_path = os.path.join(out_dir, f"light_curve_sector_{sector}_trash_frames_{windows_tag}.csv")
    df_trash.to_csv(out_path, index=False)
    return out_path


def find_sampling(jds):
    """Direct port of `find_sampling`: median cadence."""
    jds = np.asarray(jds, dtype=float)
    return float(np.median(jds[1:] - jds[:-1]))


def find_acf(jds, fluxs, oversample=1.5):
    """Direct port of `find_ACF`: autocorrelation function on a uniformly
    resampled (linear-interpolated) light curve."""
    jds = np.asarray(jds, dtype=float)
    fluxs = np.asarray(fluxs, dtype=float)

    sampling = find_sampling(jds)
    time_length = jds[-1] - jds[0]
    lags = np.arange(sampling, time_length + 1e-9, sampling / oversample)
    n_lags = len(lags)

    times = jds - jds[0]
    interpolated = np.interp(lags, times, fluxs)  # Dierckx Spline1D(k=1) == linear interpolation

    mean_flux = np.mean(interpolated)
    flux_dispersion = np.sum((interpolated - mean_flux) ** 2)
    acf = np.zeros(n_lags)
    for i_lag in range(1, n_lags + 1):
        acf[i_lag - 1] = np.sum((interpolated[: len(interpolated) - i_lag] - mean_flux) *
                                 (interpolated[i_lag:] - mean_flux))

    return lags, acf / flux_dispersion


def local_point_to_point_sigma(t, y, window=0.25):
    """
    Robust point-to-point scatter -- the MAD of consecutive differences
    (scaled to a Gaussian sigma), in a sliding +/-window/2 window -- as an
    estimate of the local noise level. Used both to flag noisy stretches
    (points with `sigma > kappa * median(sigma)` are the bad ones) and as a
    per-point error proxy.

    Not a Julia port -- a shared replacement for three near-identical copies
    that used to live in `ss397_localize.py` (`local_p2p_ppt`,
    `local_noise_mask`) and `ss397_tess.py` (`local_p2p`); also the same
    metric ROADMAP.md's automatic-cleaning stage (Этап 5) wants for its
    "local scatter of neighbouring points > kappa * median" rule.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.diff(y)
    lo = np.searchsorted(t, t - window / 2)
    hi = np.searchsorted(t, t + window / 2)
    sigma = np.array([
        1.4826 * np.median(np.abs(d[a:min(b, d.size)] - np.median(d[a:min(b, d.size)]))) / np.sqrt(2)
        if min(b, d.size) - a > 10 else np.nan
        for a, b in zip(lo, hi)
    ])
    sigma[np.isnan(sigma)] = np.nanmedian(sigma)
    return sigma


def save_lc_figure(star_name, sector, cut_size, d_mag_r=5.0, aperture_radius=3, day_step=2,
                    jd_box=0.3, sigma_tol=5, n_out=10, star_directory=STAR_DIRECTORY, out_dir="plots/png"):
    """
    Direct port of `save_lc_figure` (Makie -> matplotlib). Saves a PNG light
    curve plot for one star/sector, same as the Julia function's
    `plots/png/{cut}x{cut}/{star}_sector_{sector}.png`.
    """
    df_lc = load_light_curve(star_name, sector, cut_size, d_mag_r=d_mag_r, aperture_radius=aperture_radius,
                              rewrite_file=False, star_directory=star_directory)
    nospace_star_name = get_nospace_star_name(star_name)

    jds = get_true_jd(df_lc["BTJD" if "BTJD" in df_lc.columns else "MJD"].to_numpy())

    nonan_jd, nonan_mag = delete_nans(jds, df_lc["MAG"].to_numpy())
    clean_flux_sigma(nonan_jd, nonan_mag, jd_box, sigma_tol, n_out)
    nonan_jd, nonan_mag = delete_nans(nonan_jd, nonan_mag)

    fig, ax = plt.subplots(figsize=(15, 4.75))
    ax.plot(nonan_jd, nonan_mag, lw=1)
    ax.invert_yaxis()
    ax.set_title(f"{star_name}, sector {sector}")

    jd_start = (round(jds[0]) // day_step) * day_step
    jd_end = (round(jds[-1]) // day_step) * day_step
    tick_positions = np.arange(jd_start, jd_end + day_step, day_step)
    tick_labels = [Time(t, format="jd").datetime.strftime("%d/%m/%y") for t in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_xlim(jd_start - day_step / 2, jd_end + day_step / 2)

    target_dir = os.path.join(out_dir, f"{cut_size}x{cut_size}")
    os.makedirs(target_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(target_dir, f"{nospace_star_name}_sector_{sector}.png"), dpi=150)
    plt.close(fig)


def get_all_data(star_names, cut_size, rewrite_files=False, aperture_radius=3, d_mag_r=5.0,
                  star_directory=STAR_DIRECTORY, log_path="get_all_data.log", **save_lc_kwargs):
    """
    Direct port of `get_all_data`: for each star, resolve TESS sectors, pull
    the light curve and save a plot, logging progress/errors to a file
    (matching the Julia `get_all_data.log`).
    """
    with open(log_path, "w") as log:
        for star_name in tqdm(star_names, desc="Stars", unit="star"):
            try:
                gaia_data = None
                from .data_io import load_star_gaia_data
                gaia_data = load_star_gaia_data(star_name, star_directory)
                sectors = find_tess_sectors(float(gaia_data["ra"]), float(gaia_data["dec"]), TESS_MAX_SECTORS)
            except Exception:
                log.write(traceback.format_exc())
                sectors = []

            if not sectors:
                continue

            log.write("Resolved in TESS sectors " + ", ".join(str(s) for s in sectors) + "\n")
            for sector in tqdm(sectors, desc=f"{star_name}: sectors", unit="sector", leave=False):
                log.write(f"Processing sector {sector}: ")
                try:
                    load_light_curve(star_name, sector, cut_size, rewrite_file=rewrite_files,
                                      rewrite_gaia_stars_file=rewrite_files, aperture_radius=aperture_radius,
                                      d_mag_r=d_mag_r, star_directory=star_directory)
                    log.write("light curve loaded")
                    save_lc_figure(star_name, sector, cut_size, star_directory=star_directory, **save_lc_kwargs)
                    log.write(", plot saved\n")
                except Exception:
                    log.write(traceback.format_exc())
            log.write("\n")
            log.flush()
