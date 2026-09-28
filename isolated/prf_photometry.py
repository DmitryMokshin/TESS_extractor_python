"""
PRF-deblending photometry: fit every nearby Gaia star's PRF flux (+ a flat
background) simultaneously in a small window around a target, so a bright
neighbour's light is separated from the target's own instead of contaminating
a plain aperture sum. See ROADMAP.md, "Этап 3. PRF-фотометрия в пакете".

Ported from the `PRF_cleaning.py` prototype (`select_stars`, `prf_models`,
`design_matrix`, `deblend`) -- not a Julia port. Reuses the package's
existing PRF machinery (`psf.add_prf_cut`, the same per-star PRF-cut
convention `photometry.find_background_prf_gaia_mags`/`psf.create_gaia_prf_model`
already use) instead of a new PRF implementation. Deliberately does NOT
include the prototype's per-pixel amplitude maps / signal-attribution logic
-- that is ROADMAP's later Этап 6 ("Проверка происхождения сигнала") and
stays in `PRF_cleaning.py` for now.

Star selection here uses `geometry.calc_tmag_from_gaia` (Stassun et al. 2019,
Gaia G + BP-RP) rather than Gaia RP magnitude, deliberately scoped to just
this module -- the rest of the package (aperture photometry's background
mask, aperture-flux correction) keeps using RP as a stand-in for T, as it
always has, so numbers already in published results don't change.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .config import STAR_DIRECTORY
from .geometry import get_nospace_star_name, calc_tmag_from_gaia, calc_tess_magnitude, calc_tess_flux_from_mag
from .data_io import (load_tess_cutouts, load_gaia_stars_in_view_data, load_star_gaia_data, find_target_row,
                       read_light_curve_csv)
from .psf import get_tesscut_prf_supersampled, add_prf_cut
from .photometry import calc_prf_contamination_fraction
from .cleaning import quality_mask
from .provenance import write_csv_with_provenance


def select_prf_model_stars(gaia_stars_df, target_source_id, box_x0, box_y0, box_size,
                            dt_max=3.5, merge_px=1.0):
    """
    Field stars to include in the PRF model for a `box_size` x `box_size`
    window whose 1-based lower-left pixel is (box_x0, box_y0): stars inside
    the window (+2 px margin) and no fainter than the target by more than
    `dt_max` in T (Stassun et al. 2019, from Gaia G/BP-RP -- see module
    docstring for why not RP). Stars closer than `merge_px` to an
    already-kept, brighter star are dropped: their PRF footprint would be
    nearly degenerate with it (unfittable separately), so their light is
    left folded into the brighter star's fitted flux.

    Returns a DataFrame (subset of `gaia_stars_df`'s columns + "t_mag",
    sorted brightest-T-first); the target itself is always included (it
    trivially satisfies its own T < T + dt_max).
    """
    stars = gaia_stars_df.copy()
    stars["t_mag"] = calc_tmag_from_gaia(stars["phot_g_mean_mag"], stars["bp_rp"])
    target = stars[stars["source_id"] == target_source_id].iloc[0]

    in_window = ((stars["px_x"] > box_x0 - 2) & (stars["px_x"] < box_x0 + box_size + 2) &
                 (stars["px_y"] > box_y0 - 2) & (stars["px_y"] < box_y0 + box_size + 2) &
                 (stars["t_mag"] < target["t_mag"] + dt_max))
    stars = stars[in_window].sort_values("t_mag").reset_index(drop=True)

    keep = []
    for i, s in stars.iterrows():
        if any(np.hypot(s.px_x - stars.loc[j, "px_x"], s.px_y - stars.loc[j, "px_y"]) < merge_px for j in keep):
            continue
        keep.append(i)
    return stars.loc[keep].reset_index(drop=True)


def build_prf_star_cuts(supersampled_prf, stars_df, cut_width, cut_height, box_x0, box_y0, box_size):
    """
    Each star's own unit-flux PRF footprint (`psf.add_prf_cut`), cropped to
    the `box_size` x `box_size` window whose 1-based lower-left pixel is
    (box_x0, box_y0). Returns an (n_stars, box_size, box_size) array, same
    (x, y) axis convention as a cutout frame.
    """
    x0, y0 = box_x0 - 1, box_y0 - 1  # 1-based window origin -> 0-based slice start
    cuts = []
    for _, star in stars_df.iterrows():
        full_cut = np.zeros((cut_width, cut_height))
        add_prf_cut(full_cut, 1.0, supersampled_prf, cut_width, cut_height, star["px_x"], star["px_y"])
        cuts.append(full_cut[x0:x0 + box_size, y0:y0 + box_size])
    return np.array(cuts)


def _prf_design_matrix(star_cuts):
    n_stars, bx, by = star_cuts.shape
    xx, yy = np.meshgrid(np.arange(bx) - bx / 2, np.arange(by) - by / 2, indexing="ij")
    cols = [cut.ravel() for cut in star_cuts] + [np.ones(bx * by), xx.ravel(), yy.ravel()]
    return np.array(cols).T  # (n_pixels, n_stars + 3)


def deblend_prf_flux(cube_box, star_cuts):
    """
    Fit every star's flux plus a flat (tilted-plane) background
    simultaneously -- one linear least-squares solve across every frame at
    once (ROADMAP: "одна матрица модели на все кадры, решение одним lstsq"),
    rather than a nonlinear per-frame fit (see `isolated.fitting`, kept
    as-is for its own, different, non-PRF-deblending use).

    `cube_box`: (n_frames, box, box), same window as `star_cuts`
    (n_stars, box, box) from `build_prf_star_cuts`. A pixel that is
    non-finite in *any* frame is excluded from the fit for *every* frame
    (same behaviour as the prototype this was ported from).

    Returns (star_fluxes, background_coeffs): star_fluxes is
    (n_frames, n_stars); background_coeffs is (n_frames, 3), the
    (constant, x-slope, y-slope) plane each frame was fit with, in
    box-local coordinates centred on the box.
    """
    design = _prf_design_matrix(star_cuts)
    pixels = cube_box.reshape(cube_box.shape[0], -1).T  # (n_pixels, n_frames)
    finite_everywhere = np.all(np.isfinite(pixels), axis=1)
    coeffs, *_ = np.linalg.lstsq(design[finite_everywhere], pixels[finite_everywhere], rcond=None)
    n_stars = star_cuts.shape[0]
    return coeffs[:n_stars].T, coeffs[n_stars:].T


def load_prf_light_curve(star_name, sector, cut_width, cut_height=None, box=13, target_source_id=None,
                          dt_max=3.5, merge_px=1.0, aperture_radius=3, d_mag_r=6.0, quality_bitmask=175,
                          rewrite_file=False, rewrite_gaia_stars_file=False, star_directory=STAR_DIRECTORY):
    """
    PRF-deblended light curve for one sector: every field star's flux inside
    a `box` x `box` window around the target is fit simultaneously (see
    `deblend_prf_flux`), separating the target's own flux from any
    contaminating neighbour instead of summing both into one aperture.

    Unlike `data_io.load_light_curve`, frame *selection* here is just the
    TESS `QUALITY` bitmask (`quality_bitmask`, default 175 -- lightkurve's
    "default" mask; 0 disables it, None keeps only QUALITY==0). This one
    (unlike the manual-window cleaning below) isn't optional: a single
    non-finite pixel in one bad-quality frame would otherwise knock that
    pixel out of the fit for *every* frame (see `deblend_prf_flux`). It does
    NOT depend on a `_clean.csv` (that file may not exist yet for this star)
    -- the same manual/frame-window cleaning tools used for aperture
    photometry (`TESS_cleaning.py`, `pick_exclusion_windows`,
    `exclude_frame_windows`) apply just as well to this function's own CSV
    output, as a separate later step.

    `target_source_id`: the target's Gaia `source_id`; if None, resolved via
    `data_io.load_star_gaia_data`/`find_target_row`, same as `load_light_curve`.

    Writes/reads `{star_directory}/{star}/{cut_width}x{cut_height}/
    light_curve_sector_{sector}_prf.csv` (columns BTJD, FRAME, CADENCENO,
    QUALITY, FLUX, MAG -- no SN: PRF-deblending doesn't produce a separate
    background-only aperture sum the way aperture photometry does).
    `FRAME` is each surviving row's 1-based position in the *original*,
    not-yet-QUALITY-filtered sector (not a fresh 1..N of just the kept
    rows), so `exclude_frame_windows` still cuts the right cadences even
    though this function already drops some by `QUALITY` before writing.

    Prints a note if a plain `aperture_radius`-sized aperture here would
    have captured less than 80% of the target's own flux by the PRF model
    (`photometry.calc_prf_contamination_fraction`) -- for context on how
    crowded the field is; it does not indicate a problem with this
    function's own (already-deblended) output.

    Returns the target's DataFrame. For every fitted star's flux (e.g. to
    also save a bright neighbour's curve, or build per-pixel amplitude maps
    -- see `PRF_cleaning.py`), call `select_prf_model_stars` +
    `build_prf_star_cuts` + `deblend_prf_flux` directly instead: this
    function only persists the target's own curve, matching the rest of the
    pipeline's one-file-per-star convention.
    """
    if cut_height is None:
        cut_height = cut_width

    nospace = get_nospace_star_name(star_name)
    out_dir = os.path.join(star_directory, nospace, f"{cut_width}x{cut_height}")
    out_path = os.path.join(out_dir, f"light_curve_sector_{sector}_prf.csv")
    if os.path.isfile(out_path) and not rewrite_file:
        return read_light_curve_csv(out_path)

    cut_fits = load_tess_cutouts(star_name, cut_width, cut_height, star_directory, sector=sector)[sector]
    data = cut_fits[1].data
    t_all = np.asarray(data["TIME"], dtype=float)
    cadenceno_all = np.asarray(data["CADENCENO"], dtype=int)
    quality = np.asarray(data["QUALITY"], dtype=int)
    frame_no_all = np.arange(1, t_all.size + 1)
    cube = np.transpose(np.asarray(data["FLUX"], dtype=float), (0, 2, 1))  # (n_frames, x, y)
    prf = get_tesscut_prf_supersampled(cut_fits)

    gaia_data = load_star_gaia_data(star_name, star_directory)
    gaia_stars = load_gaia_stars_in_view_data(star_name, cut_fits, d_mag_r, rewrite_gaia_stars_file, star_directory)
    match = find_target_row(gaia_stars, gaia_data, sector)
    if target_source_id is None:
        target_source_id = int(match["source_id"])
    star_px_x, star_px_y = float(match["px_x"]), float(match["px_y"])

    box_x0 = int(round(star_px_x)) - box // 2
    box_y0 = int(round(star_px_y)) - box // 2
    stars = select_prf_model_stars(gaia_stars, target_source_id, box_x0, box_y0, box, dt_max, merge_px)
    target_idx = int(np.where(stars["source_id"] == target_source_id)[0][0])

    good = np.isfinite(t_all) & quality_mask(quality, quality_bitmask)

    x0, y0 = box_x0 - 1, box_y0 - 1
    cube_box = cube[good, x0:x0 + box, y0:y0 + box]
    t = t_all[good]
    frame_no = frame_no_all[good]
    cadenceno = cadenceno_all[good]
    quality_kept = quality[good]

    star_cuts = build_prf_star_cuts(prf, stars, cut_width, cut_height, box_x0, box_y0, box)
    fluxes, _background = deblend_prf_flux(cube_box, star_cuts)
    target_flux = fluxes[:, target_idx]

    contamination = calc_prf_contamination_fraction(
        star_px_x, star_px_y, stars["px_x"].to_numpy(), stars["px_y"].to_numpy(),
        calc_tess_flux_from_mag(stars["phot_rp_mean_mag"].to_numpy()), prf, cut_height, aperture_radius)
    if contamination < 0.8:
        print(f"Note: a plain r={aperture_radius}px aperture on {star_name} (sector {sector}) would have "
              f"modeled only {contamination:.0%} of its own flux -- crowded field, which is exactly what "
              f"this PRF-deblended curve corrects for.")

    os.makedirs(out_dir, exist_ok=True)
    lc_df = pd.DataFrame({
        "BTJD": t,
        "FRAME": frame_no,
        "CADENCENO": cadenceno,
        "QUALITY": quality_kept,
        "FLUX": target_flux,
        "MAG": calc_tess_magnitude(np.abs(target_flux)),
    })
    write_csv_with_provenance(lc_df, out_path, star_name=star_name, sector=sector, box=box,
                              dt_max=dt_max, merge_px=merge_px, quality_bitmask=quality_bitmask)
    return lc_df
