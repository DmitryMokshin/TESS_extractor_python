"""
Per-pixel amplitude maps + PRF-star attribution, and comparison-star signal
origin checks. Not a Julia port -- ported from the `PRF_cleaning.py`
prototype's `amplitude_maps`/`attribute`, generalized so any script can use
them, plus new comparison-star machinery. See ROADMAP.md, "Этап 6. Проверка
происхождения сигнала".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import calc_tmag_from_gaia
from .prf_photometry import select_prf_model_stars, build_prf_star_cuts, deblend_prf_flux
from .prewhitening import detrend_poly, amp_spectrum, red_noise_fit, local_noise


def amplitude_maps(t, cube_box, freqs):
    """Complex amplitude of each frequency at each pixel (e/s): A = a_cos + i a_sin."""
    tt = t - t.mean()
    cols = [np.ones_like(tt), tt, tt ** 2]
    for f in freqs:
        cols += [np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]
    X = np.array(cols).T
    D = np.nan_to_num(cube_box.reshape(cube_box.shape[0], -1))
    coef, *_ = np.linalg.lstsq(X, D, rcond=None)
    c = coef[3:].reshape(len(freqs), 2, *cube_box.shape[1:])
    return c[:, 0] + 1j * c[:, 1]  # (nfreq, box, box)


def attribute_amplitude(amap, models):
    """Decompose a complex amplitude map onto the PRF stars (+ constant).
    Returns |c_j| (e/s) -- the amplitude attributed to each star."""
    X = np.c_[models.reshape(models.shape[0], -1).T, np.ones(models[0].size)]
    y = amap.ravel()
    cr, *_ = np.linalg.lstsq(X, y.real, rcond=None)
    ci, *_ = np.linalg.lstsq(X, y.imag, rcond=None)
    return np.abs(cr[:-1] + 1j * ci[:-1])


def select_comparison_stars(gaia_stars_df, target_source_id, cut_width, cut_height, n=5, mag_tol=1.0, box=13):
    """
    Up to `n` field stars of comparable brightness to the target (T magnitude
    within `mag_tol`, Stassun et al. 2019 via `geometry.calc_tmag_from_gaia`
    -- same magnitude estimate `prf_photometry.select_prf_model_stars`
    already uses), excluding the target itself, closest-in-magnitude first.

    Also excludes stars too close to the cutout edge for a full `box` x
    `box` window (same 1-based box-corner math `deblend_star_curve` itself
    uses: `box_x0 = round(px_x) - box//2`) to fit -- the Gaia "stars in
    view" catalog is a cone search, not clipped to the square cutout, so it
    can include stars with a pixel position outside `[0, cut_width)` x
    `[0, cut_height)` entirely, or just inside it but too close to the edge
    for `deblend_star_curve`'s box slice to stay in bounds (a too-small
    `box_y0` there silently wraps around via numpy's negative indexing
    instead of raising, producing a garbage window rather than an error).

    Returns a DataFrame (subset of `gaia_stars_df`'s columns + "t_mag").
    """
    stars = gaia_stars_df.copy()
    stars["t_mag"] = calc_tmag_from_gaia(stars["phot_g_mean_mag"], stars["bp_rp"])
    target_t_mag = float(stars.loc[stars["source_id"] == target_source_id, "t_mag"].iloc[0])

    candidates = stars[stars["source_id"] != target_source_id].copy()
    candidates["mag_diff"] = (candidates["t_mag"] - target_t_mag).abs()
    candidates = candidates[candidates["mag_diff"] <= mag_tol]

    box_x0 = np.round(candidates["px_x"]).astype(int) - box // 2
    box_y0 = np.round(candidates["px_y"]).astype(int) - box // 2
    in_bounds = (box_x0 >= 1) & (box_x0 + box - 1 <= cut_width) & (box_y0 >= 1) & (box_y0 + box - 1 <= cut_height)
    candidates = candidates[in_bounds].sort_values("mag_diff")

    return candidates.head(n).drop(columns="mag_diff").reset_index(drop=True)


def deblend_star_curve(cube_good, prf, gaia_stars_df, cut_width, cut_height, source_id,
                        box=13, dt_max=3.5, merge_px=1.0):
    """
    PRF-deblended flux in a `box` x `box` window centred on an ARBITRARY
    field star (`source_id`), not necessarily the pipeline's official
    target. Unlike `prf_photometry.load_prf_light_curve` (whose window is
    always centred on its `star_name`; a `target_source_id` override there
    only picks which star's flux to return from THAT window, so it can't
    place the window on a star elsewhere in the cutout), this centres the
    window on `source_id`'s own position -- for comparison stars scattered
    across the field. Reuses the same deblending machinery as
    `prf_photometry` (`select_prf_model_stars`/`build_prf_star_cuts`/
    `deblend_prf_flux`) -- no new photometry, just a different window centre.

    `cube_good`: (n_frames, cut_width, cut_height), already restricted to
    the frames to use (e.g. by `cleaning.quality_mask`) -- this function
    does no frame selection of its own.

    Returns `(flux, stars_df, models, bkg)`: `flux` is `source_id`'s own
    deblended flux (n_frames,); `stars_df`/`models` are the star list/PRF
    cuts used for this window (for further attribution if needed); `bkg`
    is the per-frame fitted background plane (n_frames, 3).

    Raises `ValueError` if the box would fall outside the cutout -- a
    too-small box origin would otherwise silently wrap around via numpy's
    negative indexing instead of erroring (`select_comparison_stars` already
    filters candidates so this shouldn't trigger for its output) -- or if
    `source_id` itself has no independent PRF model in this window: it can
    be merged into a brighter neighbour within `merge_px`
    (`prf_photometry.select_prf_model_stars`'s own degeneracy rule), in
    which case its light is folded into that neighbour's fitted flux and
    there is no separate curve to return.
    """
    star_row = gaia_stars_df[gaia_stars_df["source_id"] == source_id].iloc[0]
    box_x0 = int(round(star_row["px_x"])) - box // 2
    box_y0 = int(round(star_row["px_y"])) - box // 2
    if box_x0 < 1 or box_x0 + box - 1 > cut_width or box_y0 < 1 or box_y0 + box - 1 > cut_height:
        raise ValueError(f"source_id={source_id}: box [{box_x0}, {box_x0 + box - 1}] x "
                         f"[{box_y0}, {box_y0 + box - 1}] falls outside the {cut_width}x{cut_height} cutout")
    stars = select_prf_model_stars(gaia_stars_df, source_id, box_x0, box_y0, box, dt_max, merge_px)
    if not (stars["source_id"] == source_id).any():
        raise ValueError(f"source_id={source_id}: merged into a brighter neighbour within "
                         f"merge_px={merge_px} -- no independent PRF model in this window")
    models = build_prf_star_cuts(prf, stars, cut_width, cut_height, box_x0, box_y0, box)

    x0, y0 = box_x0 - 1, box_y0 - 1
    cube_box = cube_good[:, x0:x0 + box, y0:y0 + box]
    fluxes, bkg = deblend_prf_flux(cube_box, models)
    idx = int(np.where(stars["source_id"] == source_id)[0][0])
    return fluxes[:, idx], stars, models, bkg


def signal_origin_table(star_curves, target_freqs, fmax=10.0, oversample=10, box=1.0, snr_threshold=4.0):
    """
    For every (star, target frequency) pair: amplitude and S/N AT THAT
    SPECIFIC frequency (not an independent peak search -- the question is
    "does this star show the target's frequency", not "what did we find on
    our own"). Builds the same kind of dense oversampled grid + red+white
    noise model `prewhitening.prewhiten` builds internally (same
    `detrend_poly`/`amp_spectrum`/`red_noise_fit`/`local_noise`, same S/N
    definitions -- `SNR_LOCAL` is `prewhiten`'s `snr_local` (Breger et al.
    1993, the pipeline's primary significance criterion -- see CLAUDE.md),
    `SNR_RED` its `snr_red` (Bowman et al. 2019) -- just evaluated at fixed
    target frequencies instead of at the grid's own peaks.

    `star_curves`: list of `(label, t, flux_ppt)`. `target_freqs`: the
    target's own significant frequencies (e.g. from `prewhitening.prewhiten`
    on the target's own curve). `fmax`/`oversample`/`box`: same meaning as
    the matching `prewhiten()` parameters; `fmax` is extended automatically
    if a target frequency lies above it.

    Returns a long-format DataFrame: LABEL, FREQUENCY, AMPLITUDE_PPT,
    SNR_LOCAL, SNR_RED, PRESENT (`SNR_LOCAL >= snr_threshold`) -- one row
    per (star, frequency).
    """
    target_freqs = np.asarray(target_freqs, dtype=float)
    if target_freqs.size:
        fmax = max(fmax, float(target_freqs.max()) * 1.05)

    rows = []
    for label, t, flux_ppt in star_curves:
        t = np.asarray(t, dtype=float)
        y = np.asarray(flux_ppt, dtype=float)
        if t.size < 10:  # too few points for a meaningful spectrum/noise estimate -- skip, don't crash
            continue
        T = t.max() - t.min()
        fmin = 0.5 / T
        df = 1.0 / (oversample * T)
        grid = np.arange(fmin, fmax, df)
        res = detrend_poly(t, y)
        A_res = amp_spectrum(t, res, grid)
        noise_model, _ = red_noise_fit(grid, A_res)
        amp_at_targets = amp_spectrum(t, res, target_freqs)
        for f, a in zip(target_freqs, amp_at_targets):
            snr_local = float(a / local_noise(grid, A_res, f, box))
            snr_red = float(a / noise_model(f))
            rows.append({"LABEL": label, "FREQUENCY": float(f), "AMPLITUDE_PPT": float(a),
                         "SNR_LOCAL": snr_local, "SNR_RED": snr_red,
                         "PRESENT": snr_local >= snr_threshold})
    return pd.DataFrame(rows)
