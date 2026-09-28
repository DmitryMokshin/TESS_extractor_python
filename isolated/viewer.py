"""
Interactive TESS-cutout viewer.

Ported (as a best-effort matplotlib equivalent, *not* a pixel-for-pixel GUI
clone -- Makie's reactive `Observable`/`Slider`/`Button` machinery has no
direct matplotlib analogue) from the interactive parts of main.jl:
    plot_cut, plot_cuts

Usage:
    from isolated.viewer import plot_cuts
    viewer = plot_cuts("V501 Aur", 43, 30, 30)
    plt.show()

`viewer` exposes `.fig` and `.i_cut` (a mutable [int] "observable"-like box)
so other code (e.g. the comet overlay script) can add more artists to
`viewer.ax_cut` the same way the Julia code re-used `fig[1:2,1:2]`.

`save_cutout_video` renders the same scene (minus the interactive widgets)
straight to an mp4/gif file, for sharing with people who don't have this
package installed -- roughly the Python equivalent of the Julia code's
commented-out `record(fig, "58.mp4", 400:500; framerate=10)`.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.animation import FuncAnimation
from tqdm.auto import tqdm

from .config import STAR_DIRECTORY
from .data_io import load_tess_cutouts, load_star_gaia_data, load_gaia_stars_in_view_data, load_light_curve, \
    find_target_row
from .psf import get_tesscut_prf_supersampled
from .photometry import find_background_prf_gaia_mags, fit_flat_background
from .geometry import calc_tess_magnitude


class CutoutViewer:
    """Holds the figure + widgets so callbacks and external code can reach them."""

    def __init__(self):
        self.fig = None
        self.ax_cut = None
        self.ax_lc = None
        self.i_cut = [1]  # 1-based, like the Julia `Observable(500)`


def get_frame(flux_cuts, i_cut_1based):
    """flux_cuts: astropy FLUX column, shape (n_cuts, height, width) -> (width, height) at frame i (1-based)."""
    return flux_cuts[i_cut_1based - 1].T


def _load_cutout_scene(star_name, sector, cut_width, cut_height, aperture_radius, d_mag_r, star_directory,
                        rewrite_gaia_stars_file, rewrite_light_curve_file):
    """
    Shared data-loading step for both `plot_cuts` (interactive) and
    `save_cutout_video` (rendered to a file): TESS cutout + light curve +
    Gaia star positions + PRF background-pixel mask.
    """
    cut_fits = load_tess_cutouts(star_name, cut_width, cut_height, star_directory, sector=sector)[sector]
    flux_cuts = cut_fits[1].data["FLUX"]
    n_cuts = flux_cuts.shape[0]

    df_lc = load_light_curve(star_name, sector, cut_width, cut_height, d_mag_r=d_mag_r,
                              aperture_radius=aperture_radius, star_directory=star_directory,
                              rewrite_file=rewrite_light_curve_file,
                              rewrite_gaia_stars_file=rewrite_gaia_stars_file)
    phot_flux = df_lc["FLUX"].to_numpy()
    mjds = df_lc["BTJD" if "BTJD" in df_lc.columns else "MJD"].to_numpy()

    df_star = load_star_gaia_data(star_name, star_directory)
    star_mag = float(df_star["phot_rp_mean_mag"])
    frame_stars_df = load_gaia_stars_in_view_data(star_name, cut_fits, d_mag_r, rewrite_gaia_stars_file,
                                                   star_directory)

    match = find_target_row(frame_stars_df, df_star, sector)
    star_px = (float(match["px_x"]), float(match["px_y"]))

    stars_x = frame_stars_df["px_x"].to_numpy()
    stars_y = frame_stars_df["px_y"].to_numpy()
    stars_mag = frame_stars_df["phot_rp_mean_mag"].to_numpy()
    keep = stars_mag < star_mag + d_mag_r
    stars_x, stars_y, stars_mag = stars_x[keep], stars_y[keep], stars_mag[keep]

    prf = get_tesscut_prf_supersampled(cut_fits)
    bkg_pixels = find_background_prf_gaia_mags(get_frame(flux_cuts, n_cuts // 4), prf, stars_x, stars_y, stars_mag)

    return dict(flux_cuts=flux_cuts, n_cuts=n_cuts, mjds=mjds, phot_flux=phot_flux, star_px=star_px,
                stars_x=stars_x, stars_y=stars_y, bkg_pixels=bkg_pixels)


def plot_cuts(star_name, sector, cut_width, cut_height=None, aperture_radius=5, d_mag_r=5.0,
              star_directory=STAR_DIRECTORY, start_cut=500,
              rewrite_gaia_stars_file=False, rewrite_light_curve_file=False):
    """
    Best-effort matplotlib port of `plot_cuts`: a log-flux heatmap of the
    cutout with a frame slider/prev/next buttons, Gaia star markers, a
    toggleable background-pixel overlay, and the light curve with a moving
    cursor line.

    Pass rewrite_gaia_stars_file=True to force a fresh Gaia query instead of
    reusing a cached gaia_stars_in_view_sector_{sector}.csv (useful if that
    cache was written by an older version of this package, or looks wrong).
    """
    if cut_height is None:
        cut_height = cut_width

    scene = _load_cutout_scene(star_name, sector, cut_width, cut_height, aperture_radius, d_mag_r, star_directory,
                                rewrite_gaia_stars_file, rewrite_light_curve_file)
    flux_cuts, n_cuts, mjds, phot_flux = scene["flux_cuts"], scene["n_cuts"], scene["mjds"], scene["phot_flux"]
    star_px, stars_x, stars_y, bkg_pixels = scene["star_px"], scene["stars_x"], scene["stars_y"], scene["bkg_pixels"]

    viewer = CutoutViewer()
    fig = plt.figure(figsize=(9, 9))
    viewer.fig = fig

    ax_cut = fig.add_axes([0.08, 0.42, 0.62, 0.5])
    ax_lc = fig.add_axes([0.08, 0.22, 0.85, 0.14])
    ax_slider = fig.add_axes([0.08, 0.1, 0.7, 0.03])
    ax_prev = fig.add_axes([0.08, 0.03, 0.1, 0.05])
    ax_next = fig.add_axes([0.2, 0.03, 0.1, 0.05])
    ax_check = fig.add_axes([0.4, 0.02, 0.25, 0.07])
    viewer.ax_cut = ax_cut
    viewer.ax_lc = ax_lc

    def log_frame(i_cut_1based):
        cut = get_frame(flux_cuts, i_cut_1based)
        bkg_cut = fit_flat_background(cut, bkg_pixels)
        return np.log10(np.abs(cut - bkg_cut))

    im = ax_cut.imshow(log_frame(start_cut).T, origin="lower", cmap="viridis",
                        extent=(0.5, cut_width + 0.5, 0.5, cut_height + 0.5))
    fig.colorbar(im, ax=ax_cut, label="lg TESS flux")
    star_scatter = ax_cut.scatter(stars_x, stars_y, s=15, c="lightgray", edgecolors="k", linewidths=0.3)
    target_scatter = ax_cut.scatter(*star_px, marker="x", c="magenta", s=60, label=star_name)
    bkg_x = [p[0] for p in bkg_pixels]
    bkg_y = [p[1] for p in bkg_pixels]
    bkg_scatter = ax_cut.scatter(bkg_x, bkg_y, marker="x", c="red", s=10, visible=False)
    ax_cut.set_xlim(0.5, cut_width + 0.5)
    ax_cut.set_ylim(0.5, cut_height + 0.5)
    ax_cut.set_title(f"{star_name}, sector {sector}")
    ax_cut.legend(loc="upper right", fontsize=8)

    mags = calc_tess_magnitude(phot_flux)
    ax_lc.plot(mjds, mags, lw=0.8)
    ax_lc.invert_yaxis()
    ax_lc.set_xlabel("BTJD")
    ax_lc.set_ylabel("TESS magnitude")
    cursor_line = ax_lc.axvline(mjds[start_cut - 1], color="red")
    cursor_point, = ax_lc.plot([mjds[start_cut - 1]], [mags[start_cut - 1]], "ro", ms=4)

    slider = Slider(ax_slider, "frame", 1, n_cuts, valinit=start_cut, valstep=1)
    prev_button = Button(ax_prev, "Prev")
    next_button = Button(ax_next, "Next")
    check = CheckButtons(ax_check, ["Background pixels"], [False])

    def update(i_cut_1based):
        i_cut_1based = int(np.clip(i_cut_1based, 1, n_cuts))
        viewer.i_cut[0] = i_cut_1based
        im.set_data(log_frame(i_cut_1based).T)
        cursor_line.set_xdata([mjds[i_cut_1based - 1], mjds[i_cut_1based - 1]])
        cursor_point.set_data([mjds[i_cut_1based - 1]], [mags[i_cut_1based - 1]])
        fig.canvas.draw_idle()

    def on_slider(val):
        update(int(val))

    def on_prev(event):
        slider.set_val(((viewer.i_cut[0] - 2) % n_cuts) + 1)

    def on_next(event):
        slider.set_val((viewer.i_cut[0] % n_cuts) + 1)

    def on_check(label):
        bkg_scatter.set_visible(not bkg_scatter.get_visible())
        fig.canvas.draw_idle()

    slider.on_changed(on_slider)
    prev_button.on_clicked(on_prev)
    next_button.on_clicked(on_next)
    check.on_clicked(on_check)

    # keep references so callbacks/widgets aren't garbage-collected
    viewer._widgets = (slider, prev_button, next_button, check)
    viewer._artists = (im, star_scatter, target_scatter, bkg_scatter, cursor_line, cursor_point)

    return viewer


def _mp4_available() -> bool:
    try:
        from matplotlib.animation import FFMpegWriter
        return FFMpegWriter.isAvailable()
    except Exception:
        return False


def save_cutout_video(star_name, sector, cut_width, cut_height=None, aperture_radius=5, d_mag_r=5.0,
                       star_directory=STAR_DIRECTORY, frame_range=None, fps=10, output_path=None,
                       show_background_pixels=False, dpi=120,
                       rewrite_gaia_stars_file=False, rewrite_light_curve_file=False):
    """
    Render the same scene as `plot_cuts` (log-flux heatmap + Gaia star
    markers + light curve with a moving cursor), but straight to a video
    file instead of an interactive window -- for sharing with people who
    don't have this package/your data cache set up.

    This is the Python equivalent of the Julia code's commented-out
        record(fig, "58.mp4", 400:500; framerate=10)

    Parameters
    ----------
    frame_range : (start, end) 1-based, inclusive, or None for every frame
        (can be a lot of frames -- e.g. a full TESS sector is often several
        thousand; consider a subrange, e.g. (400, 900), for a quick clip).
    fps : playback frames per second of the output video.
    output_path : where to save it. Defaults to
        plots/videos/{cut_width}x{cut_height}/{star}_sector_{sector}_frames_{start}-{end}.mp4
        (start/end are the actual 1-based frame numbers used, so repeated
        calls with different frame_range values don't overwrite each other).
        The extension picked (.mp4 vs .gif) also selects the writer, unless
        ffmpeg isn't available, in which case this automatically falls back
        to a .gif next to the requested path and prints a note explaining
        why (ffmpeg not found on PATH).
    show_background_pixels : draw the PRF-derived background-pixel mask
        (red x's) on every frame, same as ticking the checkbox in `plot_cuts`.

    Returns
    -------
    str : the path the video was actually written to.
    """
    if cut_height is None:
        cut_height = cut_width

    scene = _load_cutout_scene(star_name, sector, cut_width, cut_height, aperture_radius, d_mag_r, star_directory,
                                rewrite_gaia_stars_file, rewrite_light_curve_file)
    flux_cuts, n_cuts, mjds, phot_flux = scene["flux_cuts"], scene["n_cuts"], scene["mjds"], scene["phot_flux"]
    star_px, stars_x, stars_y, bkg_pixels = scene["star_px"], scene["stars_x"], scene["stars_y"], scene["bkg_pixels"]

    if frame_range is None:
        frames = list(range(1, n_cuts + 1))
    else:
        start, end = frame_range
        frames = list(range(max(1, start), min(n_cuts, end) + 1))

    if output_path is None:
        nospace = star_name.replace(" ", "_")
        out_dir = os.path.join("plots", "videos", f"{cut_width}x{cut_height}")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(
            out_dir, f"{nospace}_sector_{sector}_frames_{frames[0]}-{frames[-1]}.mp4")

    use_mp4 = output_path.lower().endswith(".mp4")
    if use_mp4 and not _mp4_available():
        print("Note: ffmpeg not found on PATH, can't write .mp4 -- falling back to .gif "
              "(install ffmpeg and make sure it's on PATH for mp4 output).")
        output_path = os.path.splitext(output_path)[0] + ".gif"
        use_mp4 = False

    def log_frame(i_cut_1based):
        cut = get_frame(flux_cuts, i_cut_1based)
        bkg_cut = fit_flat_background(cut, bkg_pixels)
        return np.log10(np.abs(cut - bkg_cut))

    fig = plt.figure(figsize=(9, 9))
    ax_cut = fig.add_axes([0.08, 0.42, 0.62, 0.5])
    ax_lc = fig.add_axes([0.08, 0.22, 0.85, 0.14])

    im = ax_cut.imshow(log_frame(frames[0]).T, origin="lower", cmap="viridis",
                        extent=(0.5, cut_width + 0.5, 0.5, cut_height + 0.5))
    fig.colorbar(im, ax=ax_cut, label="lg TESS flux")
    ax_cut.scatter(stars_x, stars_y, s=15, c="lightgray", edgecolors="k", linewidths=0.3)
    ax_cut.scatter(*star_px, marker="x", c="magenta", s=60, label=star_name)
    if show_background_pixels:
        bkg_x = [p[0] for p in bkg_pixels]
        bkg_y = [p[1] for p in bkg_pixels]
        ax_cut.scatter(bkg_x, bkg_y, marker="x", c="red", s=10)
    ax_cut.set_xlim(0.5, cut_width + 0.5)
    ax_cut.set_ylim(0.5, cut_height + 0.5)
    ax_cut.set_title(f"{star_name}, sector {sector}")
    ax_cut.legend(loc="upper right", fontsize=8)

    mags = calc_tess_magnitude(phot_flux)
    ax_lc.plot(mjds, mags, lw=0.8)
    ax_lc.invert_yaxis()
    ax_lc.set_xlabel("BTJD")
    ax_lc.set_ylabel("TESS magnitude")
    cursor_line = ax_lc.axvline(mjds[frames[0] - 1], color="red")
    cursor_point, = ax_lc.plot([mjds[frames[0] - 1]], [mags[frames[0] - 1]], "ro", ms=4)

    def update(i_cut_1based):
        im.set_data(log_frame(i_cut_1based).T)
        cursor_line.set_xdata([mjds[i_cut_1based - 1], mjds[i_cut_1based - 1]])
        cursor_point.set_data([mjds[i_cut_1based - 1]], [mags[i_cut_1based - 1]])
        return im, cursor_line, cursor_point

    anim = FuncAnimation(fig, update, frames=frames, blit=False)

    if use_mp4:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps, bitrate=1800)
    else:
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=fps)

    with tqdm(total=len(frames), desc=f"Rendering {os.path.basename(output_path)}", unit="frame") as bar:
        anim.save(output_path, writer=writer, dpi=dpi, progress_callback=lambda i, n: bar.update(1))

    plt.close(fig)
    return output_path
