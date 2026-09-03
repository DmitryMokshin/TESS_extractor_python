"""
Background estimation and aperture photometry.

Ported from main.jl:
    get_n_min_mean_background, get_n_min_median_background,
    calc_aperture_photometry, calc_aperture_photometry_bkg_pixels,
    calc_aperture_photometry_bkg_pixels_with_sn_ratio,
    fit_flat_background, fit_flat_background_precise_indeces,
    find_background_prf_gaia_mags, find_background_prf,
    aperture_prf_correction, calc_aperture_prf_correction

Coordinate convention: x, y star/aperture positions are 1-based (Julia/FITS
style, pixel (1,1) = center of first pixel), matching the rest of this
package. photutils uses 0-based pixel centers, so we subtract 1 right before
calling into it.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from photutils.aperture import CircularAperture, aperture_photometry

from .psf import get_prf_cut, create_gaia_prf_model


def aperture_sum(data: np.ndarray, x: float, y: float, radius: float) -> float:
    """Sum of `data` inside a circular aperture, x/y 1-based like the rest of the package."""
    ap = CircularAperture((x - 1.0, y - 1.0), r=radius)
    table = aperture_photometry(data, ap, method="exact")
    return float(table["aperture_sum"][0])


def get_n_min_mean_background(cut_flux: np.ndarray, n_min: int) -> float:
    """Direct port of `get_n_min_mean_background`."""
    flat = np.sort(cut_flux.ravel())
    return float(np.sum(flat[:n_min]) / n_min)


def get_n_min_median_background(cut_flux: np.ndarray, n_min: int) -> float:
    """Direct port of `get_n_min_median_background`."""
    flat = np.sort(cut_flux.ravel())
    return float(np.median(flat[:n_min]))


def calc_aperture_photometry(cut: np.ndarray, star_px_x, star_px_y, aperture_radius) -> float:
    """Direct port of `calc_aperture_photometry`."""
    n_px = cut.shape[0] * cut.shape[1]
    bkg = get_n_min_median_background(cut, n_px // 4)
    return aperture_sum(cut - bkg, star_px_x, star_px_y, aperture_radius)


def calc_aperture_photometry_bkg_pixels(cut, bkg_positions, star_px_x, star_px_y, aperture_radius):
    """Direct port of `calc_aperture_photometry_bkg_pixels`."""
    if np.any(np.abs(cut) < 1e-8):
        return np.nan
    bkg_cut = fit_flat_background(cut, bkg_positions)
    return aperture_sum(cut - bkg_cut, star_px_x, star_px_y, aperture_radius)


def calc_aperture_photometry_bkg_pixels_with_sn_ratio(cut, bkg_positions, star_px_x, star_px_y, aperture_radius):
    """Direct port of `calc_aperture_photometry_bkg_pixels_with_sn_ratio`."""
    if np.any(np.abs(cut) < 1e-8):
        return np.nan, 0.0
    bkg_cut = fit_flat_background(cut, bkg_positions)
    phot = aperture_sum(cut - bkg_cut, star_px_x, star_px_y, aperture_radius)
    sn = phot / aperture_sum(bkg_cut, star_px_x, star_px_y, aperture_radius)
    return phot, sn


def _fit_plane(bkg_fluxes, bkg_xs, bkg_ys, cut_width, cut_height):
    """
    Shared plane-fit core of fit_flat_background / fit_flat_background_precise_indeces.
    Fits z(x, y) = (pars[3]*|pars[0:3]| - pars[0]*x - pars[1]*y) / pars[2]
    by Levenberg-Marquardt, exactly mirroring the Julia `to_optimize` closures.
    """
    bkg_xs = np.asarray(bkg_xs, dtype=float)
    bkg_ys = np.asarray(bkg_ys, dtype=float)
    bkg_fluxes = np.asarray(bkg_fluxes, dtype=float)

    def residual(pars):
        normal = np.sqrt(pars[0] ** 2 + pars[1] ** 2 + pars[2] ** 2)
        return bkg_fluxes - (pars[3] * normal - pars[0] * bkg_xs - pars[1] * bkg_ys) / pars[2]

    result = least_squares(residual, x0=[0.0, 0.0, 1.0, 100.0], method="lm")
    bkg_plane = result.x

    normal = np.sqrt(bkg_plane[0] ** 2 + bkg_plane[1] ** 2 + bkg_plane[2] ** 2)
    xs = np.arange(1, cut_width + 1)
    ys = np.arange(1, cut_height + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    bkg_cut = (bkg_plane[3] * normal - bkg_plane[0] * xx - bkg_plane[1] * yy) / bkg_plane[2]
    return bkg_cut


def fit_flat_background_precise_indeces(flux_cut, bkg_positions):
    """
    Direct port of `fit_flat_background_precise_indeces`. `bkg_positions` is a
    list of (x, y) 1-based pixel index tuples.
    """
    cut_width, cut_height = flux_cut.shape
    if len(bkg_positions) == 0:
        return np.full((cut_width, cut_height), np.nan)

    bkg_xs = [p[0] for p in bkg_positions]
    bkg_ys = [p[1] for p in bkg_positions]
    bkg_fluxes = [flux_cut[x - 1, y - 1] for x, y in bkg_positions]
    return _fit_plane(bkg_fluxes, bkg_xs, bkg_ys, cut_width, cut_height)


def fit_flat_background(flux_cut, bkg_positions):
    """
    Direct port of `fit_flat_background`: keeps only bkg_positions whose flux
    is (numerically) below the 70th percentile flux among bkg_positions
    before fitting the plane.
    """
    cut_width, cut_height = flux_cut.shape
    if len(bkg_positions) == 0:
        return np.full((cut_width, cut_height), np.nan)

    vals_at_bkg = np.sort([flux_cut[x - 1, y - 1] for x, y in bkg_positions])
    median_cut = vals_at_bkg[round(len(bkg_positions) * 0.7) - 1]  # Julia 1-based index

    final_positions = [p for p in bkg_positions if (flux_cut[p[0] - 1, p[1] - 1] - median_cut) < -1e-8]
    if len(final_positions) == 0:
        return np.full((cut_width, cut_height), np.nan)

    bkg_xs = [p[0] for p in final_positions]
    bkg_ys = [p[1] for p in final_positions]
    bkg_fluxes = [flux_cut[x - 1, y - 1] for x, y in final_positions]
    return _fit_plane(bkg_fluxes, bkg_xs, bkg_ys, cut_width, cut_height)


def find_background_prf_gaia_mags(flux_cut, supersampled_prf, stars_x, stars_y, stars_m_r):
    """
    Direct port of `find_background_prf_gaia_mags`: pick pixels where the
    *modeled* Gaia-PRF flux is below its own 70th percentile, i.e. pixels far
    from any bright star -> a background mask driven purely by star geometry.
    Returns a list of (x, y) 1-based pixel index tuples.
    """
    cut_width, cut_height = flux_cut.shape
    prf_cut = create_gaia_prf_model(supersampled_prf, cut_width, cut_height, stars_x, stars_y, stars_m_r, 0.0, 0.0)
    sorted_vals = np.sort(prf_cut.ravel())
    median_prf = sorted_vals[round(cut_width * cut_height * 0.7) - 1]

    positions = []
    for x in range(1, cut_width + 1):
        for y in range(1, cut_height + 1):
            if (median_prf - prf_cut[x - 1, y - 1]) > -1e-8:
                positions.append((x, y))
    return positions


def find_background_prf(flux_cut, supersampled_prf, stars_x, stars_y):
    """
    Direct port of `find_background_prf`. Requires `fit_stars_prf_flat_bkg`
    from `fitting.py` (imported lazily to avoid a circular import).
    """
    from .fitting import fit_stars_prf_flat_bkg

    n_stars = len(stars_x)
    cut_width, cut_height = flux_cut.shape
    start_pars = np.full(n_stars + 4, 100.0)
    res = fit_stars_prf_flat_bkg(supersampled_prf, flux_cut, stars_x, stars_y, start_pars)

    prfs = [abs(res[i]) * get_prf_cut(supersampled_prf, cut_width, cut_height, stars_x[i], stars_y[i])
            for i in range(n_stars)]
    empty_cut = sum(abs(res[i]) * np.full((cut_width, cut_height),
                                           get_n_min_median_background(prfs[i], cut_width * cut_height // 4))
                     for i in range(n_stars))
    prf_cut = sum(prfs)
    diff = prf_cut - empty_cut
    sorted_vals = np.sort(diff.ravel())
    median_prf = sorted_vals[round(cut_width * cut_height * 0.7) - 1]

    positions = []
    for x in range(1, cut_width + 1):
        for y in range(1, cut_height + 1):
            if (median_prf - diff[x - 1, y - 1]) > -1e-8:
                positions.append((x, y))
    return positions


def aperture_prf_correction(star_px_x, star_px_y, supersampled_prf, cut_size, aperture_radius):
    """Direct port of `aperture_prf_correction` / `calc_aperture_prf_correction`."""
    prf_cut = get_prf_cut(supersampled_prf, cut_size, cut_size, star_px_x, star_px_y)
    prf_flux = float(np.sum(prf_cut))
    ap_flux = aperture_sum(prf_cut, star_px_x, star_px_y, aperture_radius)
    return prf_flux / ap_flux


def calc_aperture_prf_correction(aperture_radius, star_px_x, star_px_y, supersampled_prf, cut_size,
                                  shift_x=0.0, shift_y=0.0):
    """Direct port of both `calc_aperture_prf_correction` methods."""
    return aperture_prf_correction(star_px_x + shift_x, star_px_y + shift_y, supersampled_prf, cut_size,
                                    aperture_radius)


def get_n_min_mean_background_from_cube(cube_slice, n_min):  # convenience alias
    return get_n_min_mean_background(cube_slice, n_min)
