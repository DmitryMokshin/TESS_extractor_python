"""
Diagnosing CCD-wide artifacts (scattered light streaks/banding from
Earth/Moon, straps, etc.) that show up as bright bands across a cutout, so
they can be told apart from a real flare on the target star and excised
from the light curve.

Workflow:
    1. `pick_stripe_pixels` -- scrub through frames in the normal `plot_cuts`
       viewer and click a handful of background pixels that sit on the
       artifact but not on any star.
    2. `stray_light_monitor_curve` -- turn those pixels into a per-frame
       "how bright is the artifact right now" diagnostic curve.
    3. Either cut the affected frames precisely: feed the monitor curve to
       the already-existing `pick_exclusion_windows` (or just eyeball it)
       to get [start_frame, end_frame] windows, then pass the *same*
       windows to `exclude_frame_windows` on the real light curve -- a
       spike in the monitor curve at frame N means frame N of the real
       light curve is contaminated too, since both are indexed by the same
       per-sector frame/cadence sequence.
    or 4. Or roughly correct instead of cutting: `correct_stray_light`
       subtracts a median-based estimate of the artifact from the target's
       flux, frame by frame.

Note this is *frame*-level cleaning (correcting/excising specific cadences
because of a CCD-wide artifact seen in the cutout images), as opposed to
point-level light-curve cleaning like removing zero/garbage flux values.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .config import STAR_DIRECTORY
from .data_io import load_tess_cutouts, load_star_gaia_data, load_gaia_stars_in_view_data, find_target_row
from .geometry import calc_tess_magnitude
from .viewer import plot_cuts, get_frame


def pick_stripe_pixels(star_name, sector, cut_width, cut_height=None, star_directory=STAR_DIRECTORY,
                        start_cut=500, **plot_cuts_kwargs):
    """
    Interactive helper for marking background pixels that sit on a
    scattered-light streak (or similar CCD-wide artifact) but not on a
    star. Opens the normal `plot_cuts` viewer -- scrub through frames with
    its slider/buttons to find where the artifact is visible, then
    left-click on the cutout to mark a pixel (red '+'); right-click undoes
    the last mark. Close the window when done.

    Returns a list of (x, y) pixel coordinates (1-based, rounded to the
    nearest pixel -- same convention as the cutout axes). Feed them to
    `stray_light_monitor_curve`.
    """
    viewer = plot_cuts(star_name, sector, cut_width, cut_height, star_directory=star_directory,
                        start_cut=start_cut, **plot_cuts_kwargs)
    picked = []
    markers = []

    def on_click(event):
        if event.inaxes is not viewer.ax_cut or event.xdata is None:
            return
        if event.button == 1:
            x, y = round(event.xdata), round(event.ydata)
            picked.append((x, y))
            markers.append(viewer.ax_cut.plot(x, y, "r+", ms=12, mew=2)[0])
        elif event.button == 3 and picked:
            picked.pop()
            markers.pop().remove()
        viewer.fig.canvas.draw_idle()

    viewer.fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    return picked


def stray_light_monitor_curve(star_name, sector, cut_width, cut_height, pixels, star_directory=STAR_DIRECTORY):
    """
    Build a per-frame diagnostic light curve from the flux at hand-picked
    background pixels (see `pick_stripe_pixels`): the median flux at those
    pixels, converted to a TESS magnitude the same way as a real light
    curve. Frames affected by the artifact spike in this curve; unaffected
    frames sit near a flat baseline.

    `pixels` : list of (x, y) 1-based pixel coordinates, e.g. from
        `pick_stripe_pixels`.

    Returns a DataFrame with "MJD"/"MAG" columns, same one-row-per-frame
    convention as `load_light_curve`'s output, so it plugs straight into
    `pick_exclusion_windows` / `exclude_frame_windows`.
    """
    if cut_height is None:
        cut_height = cut_width
    cut_fits = load_tess_cutouts(star_name, cut_width, cut_height, star_directory)[sector]
    flux_cuts = cut_fits[1].data["FLUX"]
    mjds = cut_fits[1].data["TIME"]
    n_cuts = flux_cuts.shape[0]

    xs = [x - 1 for x, y in pixels]
    ys = [y - 1 for x, y in pixels]

    monitor_flux = np.array([np.median(get_frame(flux_cuts, i + 1)[xs, ys]) for i in range(n_cuts)])

    return pd.DataFrame({"MJD": mjds, "FLUX": monitor_flux, "MAG": calc_tess_magnitude(monitor_flux)})


def plot_stray_light_diagnostics(star_name, sector, cut_width, cut_height, df_lc, df_monitor,
                                  windows=None, star_directory=STAR_DIRECTORY, out_path=None):
    """
    Build a 4-panel figure to help judge whether an excursion in `df_lc` is
    a genuine flare on the target or a CCD-wide artifact picked up by
    `df_monitor` (see `stray_light_monitor_curve`):

      1. Target light curve over time, candidate windows shaded (if given).
      2. Monitor (stray-light) curve over the same time axis -- do the
         bumps in panel 1 line up with bumps here?
      3. Target excess vs. monitor excess scatter -- each point is one
         frame; a tight, positive trend means the target's brightness
         tracks the empty-background signal, which a real flare has no
         reason to do.
      4. The cutout image at the quietest frame vs. the frame of peak
         monitor flux (within `windows` if given, else the whole sector),
         same log-flux scale, target position marked with an 'x' -- is the
         extra light spread across bands, or sitting on the star?

    `windows` : optional list of [start_frame, end_frame] (1-based,
        inclusive, same convention as `exclude_frame_windows`) to shade and
        to restrict the "peak frame" search to -- pass what
        `pick_exclusion_windows` gave you to focus the diagnostic on the
        stretch you're actually unsure about.

    If `out_path` is given the figure is saved there and closed instead of
    shown.
    """
    if cut_height is None:
        cut_height = cut_width

    mjd = df_lc["MJD"].to_numpy()
    target_flux = df_lc["FLUX"].to_numpy()
    monitor_flux = df_monitor["FLUX"].to_numpy()
    n_cuts = len(monitor_flux)

    target_excess = target_flux - np.median(target_flux)
    monitor_excess = monitor_flux - np.median(monitor_flux)

    if windows:
        in_window = np.zeros(n_cuts, dtype=bool)
        for start, end in windows:
            in_window[start - 1:end] = True
        candidate_idx = np.nonzero(in_window)[0]
    else:
        candidate_idx = np.arange(n_cuts)

    peak_i = candidate_idx[np.argmax(monitor_flux[candidate_idx])]
    quiet_i = int(np.argmin(np.abs(monitor_flux - np.median(monitor_flux))))

    cut_fits = load_tess_cutouts(star_name, cut_width, cut_height, star_directory)[sector]
    flux_cuts = cut_fits[1].data["FLUX"]
    gaia_data = load_star_gaia_data(star_name, star_directory)
    frame_stars_df = load_gaia_stars_in_view_data(star_name, cut_fits, star_directory=star_directory)
    match = find_target_row(frame_stars_df, gaia_data, sector)
    star_px = (float(match["px_x"]), float(match["px_y"]))

    log_quiet = np.log10(np.abs(get_frame(flux_cuts, quiet_i + 1))).T
    log_peak = np.log10(np.abs(get_frame(flux_cuts, peak_i + 1))).T
    vmin, vmax = min(log_quiet.min(), log_peak.min()), max(log_quiet.max(), log_peak.max())

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.3], hspace=0.5, wspace=0.35)
    ax_lc = fig.add_subplot(gs[0, :])
    ax_mon = fig.add_subplot(gs[1, :], sharex=ax_lc)
    ax_corr = fig.add_subplot(gs[2, 0])
    ax_quiet = fig.add_subplot(gs[2, 1])
    ax_peak = fig.add_subplot(gs[2, 2])

    for ax, y, ylabel in ((ax_lc, df_lc["MAG"].to_numpy(), "target MAG"),
                           (ax_mon, df_monitor["MAG"].to_numpy(), "monitor MAG")):
        ax.plot(mjd, y, lw=0.8)
        ax.invert_yaxis()
        ax.set_ylabel(ylabel)
        if windows:
            for start, end in windows:
                ax.axvspan(mjd[start - 1], mjd[end - 1], color="red", alpha=0.15)
    ax_lc.set_title(f"{star_name}, sector {sector}")
    ax_mon.set_xlabel("MJD")

    ax_corr.scatter(monitor_excess, target_excess, s=6, alpha=0.4)
    ax_corr.scatter(monitor_excess[candidate_idx], target_excess[candidate_idx], s=10, color="red",
                     label="candidate window" if windows else None)
    ax_corr.axhline(0, color="gray", lw=0.5)
    ax_corr.axvline(0, color="gray", lw=0.5)
    ax_corr.set_xlabel("monitor flux excess")
    ax_corr.set_ylabel("target flux excess")
    if windows:
        ax_corr.legend(fontsize=7)

    ax_quiet.imshow(log_quiet, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_quiet.plot(*star_px, marker="x", c="magenta", ms=10)
    ax_quiet.set_title(f"quiet frame ({quiet_i + 1})", fontsize=9)

    im = ax_peak.imshow(log_peak, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_peak.plot(*star_px, marker="x", c="magenta", ms=10)
    ax_peak.set_title(f"peak-monitor frame ({peak_i + 1})", fontsize=9)
    fig.colorbar(im, ax=(ax_quiet, ax_peak), label="lg TESS flux", fraction=0.05)

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    plt.show()
    return fig


def correct_stray_light(df_lc, df_monitor, aperture_radius=3, plot=False, star_name="", sector=None):
    """
    Roughly correct the target's light curve for the artifact diagnosed by
    `stray_light_monitor_curve`: at each frame, take the monitor's flux
    excess over its own quiescent baseline (the median monitor flux across
    the whole sector) and subtract that excess -- scaled by the aperture's
    pixel count (pi * aperture_radius**2) -- from the target's flux.
    Frames where the monitor is at/below baseline are left untouched.

    This is a deliberately rough, per-frame correction of the *photometry*
    (not the raw cutout pixels), distinct from the row-dropping light-curve
    cleaning in `lightcurve_tools` (which throws bad points away instead of
    correcting them). For a precise cut instead of an approximate
    correction, use `pick_exclusion_windows` / `exclude_frame_windows` on
    `df_monitor` instead.

    `aperture_radius` should match the one used for `df_lc`'s photometry
    (see `load_light_curve`'s `aperture_radius`).

    Returns a corrected copy of df_lc (FLUX and MAG columns updated,
    df_lc/df_monitor left untouched). If plot=True, also shows the
    original vs. corrected target curve alongside the monitor curve for a
    quick visual check.
    """
    baseline = np.median(df_monitor["FLUX"].to_numpy())
    excess = np.clip(df_monitor["FLUX"].to_numpy() - baseline, 0, None)
    n_aperture_px = np.pi * aperture_radius ** 2

    df_corrected = df_lc.copy()
    df_corrected["FLUX"] = df_lc["FLUX"].to_numpy() - excess * n_aperture_px
    df_corrected["MAG"] = calc_tess_magnitude(df_corrected["FLUX"].to_numpy())

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax1.plot(df_lc["MJD"], df_lc["MAG"], lw=0.8, label="original")
        ax1.plot(df_corrected["MJD"], df_corrected["MAG"], lw=0.8, label="corrected")
        ax1.invert_yaxis()
        ax1.set_ylabel("TESS magnitude")
        ax1.set_title(f"{star_name}, sector {sector}" if star_name else "stray-light correction")
        ax1.legend(fontsize=8)

        ax2.plot(df_monitor["MJD"], df_monitor["MAG"], lw=0.8, color="tab:red")
        ax2.invert_yaxis()
        ax2.set_xlabel("MJD")
        ax2.set_ylabel("stray-light\nmonitor (MAG)")

        fig.tight_layout()
        plt.show()

    return df_corrected
