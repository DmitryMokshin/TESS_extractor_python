"""
Point-spread / pixel-response function models.

Ported from main.jl:
    elliptic_powexp_PSF, elliptic_gaussian_PSF, gaussian_PSF,
    get_tesscut_prf_supersampled, get_prf_cut, add_prf_cut!,
    create_gaia_prf_model (+ _old), get_tesscut_ffi_coordinates

Pixel-coordinate convention: as in the Julia code / FITS/WCS, pixel (1,1)
is the *center* of the first pixel (1-based). All functions here keep that
convention for their arguments; array storage is the usual 0-based numpy,
so indices are offset by one internally where needed.
"""
from __future__ import annotations

import os
import numpy as np
from astropy.io import fits as pyfits

from .geometry import calc_tess_flux_from_mag
from .tess_point import tess_stars2px_sector

# in-process cache for get_tesscut_prf_supersampled, keyed by (sector, cam,
# ccd, ccd position, cache dir) -- ROADMAP.md Этап 9 ("кэшировать PRF на
# сектор и ПЗС"): avoids re-downloading/re-interpolating the same PRF within
# one process (e.g. a target + several comparison stars in the same cutout,
# isolated.localize's Этап 6 signal-origin check), without touching the
# on-disk PRF-file cache (already handled by `os.path.isfile` in `ensure()`).
_PRF_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# Analytic PSF models (not used by the main PRF-based pipeline, kept for
# completeness / fit_stars_gauss)
# --------------------------------------------------------------------------- #

def elliptic_powexp_psf(width, height, center_x, center_y, x_sigma, y_sigma, power, max_int, super_samp=10):
    """Direct port of `elliptic_powexp_PSF`."""
    model = np.zeros((width, height))
    d_ss = 1.0 / super_samp
    for x_px in range(1, width + 1):
        corner_x = x_px - 0.5
        for y_px in range(1, height + 1):
            corner_y = y_px - 0.5
            acc = 0.0
            for i_ss_x in range(1, super_samp + 1):
                x_ss = corner_x + i_ss_x * d_ss - d_ss / 2
                for i_ss_y in range(1, super_samp + 1):
                    y_ss = corner_y + i_ss_y * d_ss - d_ss / 2
                    acc += max_int * np.exp(-(abs((center_x - x_ss) / x_sigma) ** power +
                                               abs((center_y - y_ss) / y_sigma) ** power)) * d_ss ** 2
            model[x_px - 1, y_px - 1] = acc
    return model


def elliptic_gaussian_psf(width, height, center_x, center_y, x_sigma, y_sigma, max_int, super_samp=10):
    """Direct port of `elliptic_gaussian_PSF`."""
    model = np.zeros((width, height))
    d_ss = 1.0 / super_samp
    for x_px in range(1, width + 1):
        corner_x = x_px - 0.5
        for y_px in range(1, height + 1):
            corner_y = y_px - 0.5
            acc = 0.0
            for i_ss_x in range(1, super_samp + 1):
                x_ss = corner_x + i_ss_x * d_ss - d_ss / 2
                for i_ss_y in range(1, super_samp + 1):
                    y_ss = corner_y + i_ss_y * d_ss - d_ss / 2
                    acc += max_int * np.exp(-(((center_x - x_ss) / x_sigma) ** 2 +
                                               ((center_y - y_ss) / y_sigma) ** 2)) * d_ss ** 2
            model[x_px - 1, y_px - 1] = acc
    return model


def gaussian_psf(width, height, center_x, center_y, sigma, max_int, super_samp=10):
    """Direct port of `gaussian_PSF`."""
    model = np.zeros((width, height))
    d_ss = 1.0 / super_samp
    for x_px in range(1, width + 1):
        corner_x = x_px - 0.5
        for y_px in range(1, height + 1):
            corner_y = y_px - 0.5
            acc = 0.0
            for i_ss_x in range(1, super_samp + 1):
                x_ss = corner_x + i_ss_x * d_ss - d_ss / 2
                for i_ss_y in range(1, super_samp + 1):
                    y_ss = corner_y + i_ss_y * d_ss - d_ss / 2
                    acc += max_int * np.exp(-((center_x - x_ss) ** 2 +
                                               (center_y - y_ss) ** 2) / sigma ** 2) * d_ss ** 2
            model[x_px - 1, y_px - 1] = acc
    return model


# --------------------------------------------------------------------------- #
# TESS pixel-response function (PRF) handling
# --------------------------------------------------------------------------- #

def get_tesscut_ffi_coordinates(cut_fits):
    """Port of `get_tesscut_ffi_coordinates`."""
    from .fits_utils import get_reference_radec
    alpha_cut, delta_cut = get_reference_radec(cut_fits)
    sector = int(cut_fits[0].header["SECTOR"])
    return tess_stars2px_sector(alpha_cut, delta_cut, sector)


def get_tesscut_prf_supersampled(cut_fits, prf_cache_dir="prf", download=True):
    """
    Port of `get_tesscut_prf_supersampled`: locate/download the four nearest
    TESS PRF FITS files for this camera/CCD/sector and bilinearly interpolate
    the supersampled PRF at the cutout's CCD position.
    """
    import requests
    from .fits_utils import get_reference_radec

    alpha_cut, delta_cut = get_reference_radec(cut_fits)
    sector = int(cut_fits[0].header["SECTOR"])
    ccd_x, ccd_y = np.round(tess_stars2px_sector(alpha_cut, delta_cut, sector)).astype(int)
    cam = int(cut_fits[0].header["CAMERA"])
    ccd = int(cut_fits[0].header["CCD"])

    cache_key = (sector, cam, ccd, int(ccd_x), int(ccd_y), prf_cache_dir)
    if cache_key in _PRF_CACHE:
        return _PRF_CACHE[cache_key]

    prf_url = "https://archive.stsci.edu/missions/tess/models/prf_fitsfiles"
    prf_dir = f"start_s000{4 if sector > 3 else 1}/cam{cam}_ccd{ccd}"

    if sector < 4:
        prf_prefix = "tess2018243163601" if (cam > 2 or (cam == 2 and ccd == 4)) else "tess2018243163600"
    else:
        if cam > 3 or (cam == 3 and ccd >= 2):
            prf_prefix = "tess2019107181902"
        elif cam == 1 and ccd == 1:
            prf_prefix = "tess2019107181900"
        else:
            prf_prefix = "tess2019107181901"

    b_row = ((ccd_y - 1) // 512) * 512 + 1
    l_col = ((ccd_x - 45) // 512) * 512 + 45
    t_row = ((ccd_y - 1) // 512 + 1) * 512 + 1
    r_col = ((ccd_x - 45) // 512 + 1) * 512 + 45

    b_row -= 1 if b_row > 1500 else 0
    l_col -= 1 if l_col > 1500 else 0
    t_row -= 1 if t_row > 1500 else 0
    r_col -= 1 if r_col > 1500 else 0

    def fname(row, col):
        return f"{prf_prefix}-prf-{cam}-{ccd}-row{row:04d}-col{col:04d}.fits"

    bl_name, br_name = fname(b_row, l_col), fname(b_row, r_col)
    tl_name, tr_name = fname(t_row, l_col), fname(t_row, r_col)

    local_dir = os.path.join(prf_cache_dir, prf_dir)
    os.makedirs(local_dir, exist_ok=True)

    def ensure(name):
        path = os.path.join(local_dir, name)
        if not os.path.isfile(path) and download:
            r = requests.get(f"{prf_url}/{prf_dir}/{name}", timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
        return path

    with pyfits.open(ensure(br_name)) as f:
        br_prf = f[0].data.astype(float)
    with pyfits.open(ensure(bl_name)) as f:
        bl_prf = f[0].data.astype(float)
    with pyfits.open(ensure(tr_name)) as f:
        tr_prf = f[0].data.astype(float)
    with pyfits.open(ensure(tl_name)) as f:
        tl_prf = f[0].data.astype(float)

    interpolated_prf = ((ccd_y - b_row) * (ccd_x - l_col) * tr_prf
                         - (ccd_y - b_row) * (ccd_x - r_col) * tl_prf
                         - (ccd_y - t_row) * (ccd_x - l_col) * br_prf
                         + (ccd_y - t_row) * (ccd_x - r_col) * bl_prf) / (t_row - b_row) / (r_col - l_col)
    _PRF_CACHE[cache_key] = interpolated_prf
    return interpolated_prf


def _prf_axis_weights(n_px, supersampled_len, source, n_super=9, offset=59):
    """
    Vectorized form of `get_prf_cut`/`add_prf_cut`'s per-axis weight: for
    output pixel `x_px` (1-based) the corresponding supersampled-PRF centre
    is `x_prf_source = (x_px - source)*n_super + offset`, and the weight
    given to supersampled index `x_prf` (1-based) is a flat-top taper --
    `1` for `|x_prf - x_prf_source| <= 4`, linearly down to `0` over
    `(4, 5)`, `0` beyond -- identical to the original nested-loop formula
    (confirmed separable: the 2D weight there is exactly this function's
    outer product along x and y). Returns an `(n_px, supersampled_len)`
    weight matrix; `get_prf_cut`/`add_prf_cut` combine the x/y matrices with
    two matrix products instead of a Python double loop per output pixel.
    """
    px = np.arange(1, n_px + 1, dtype=float)
    prf_source = (px - source) * n_super + offset
    prf_idx = np.arange(1, supersampled_len + 1, dtype=float)
    d = np.abs(prf_idx[None, :] - prf_source[:, None])
    return np.where(d <= 4, 1.0, np.where(d < 5, 5.0 - d, 0.0))


def get_prf_cut(prf_supersampled, cut_width, cut_height, x_source, y_source):
    """
    Vectorized port of `get_prf_cut`: bin the 9x-supersampled PRF onto a
    cut_width x cut_height pixel grid centered on (x_source, y_source)
    (1-based pixel coordinates). Same formula as the original nested-loop
    reference implementation (see `_prf_axis_weights`), just computed as two
    matrix products instead of a double loop over every output pixel x every
    supersampled-PRF cell -- ROADMAP.md Этап 9 ("get_prf_cut/add_prf_cut:
    векторизовать").
    """
    supersampled_width, supersampled_height = prf_supersampled.shape
    wx = _prf_axis_weights(cut_width, supersampled_width, x_source)
    wy = _prf_axis_weights(cut_height, supersampled_height, y_source)
    val = (wx @ prf_supersampled @ wy.T) / 81
    return np.where(val >= 2e-4, val, 0.0)


def add_prf_cut(cut, flux, prf_supersampled, cut_width, cut_height, x_source, y_source):
    """
    Vectorized port of `add_prf_cut!`: adds `flux * PRF` into `cut` in-place
    and returns it. Same weight formula as `get_prf_cut` (see
    `_prf_axis_weights`), without `get_prf_cut`'s `>= 2e-4` threshold (the
    original windowed/optimized loop never applied one either).
    """
    supersampled_width, supersampled_height = prf_supersampled.shape
    wx = _prf_axis_weights(cut_width, supersampled_width, x_source)
    wy = _prf_axis_weights(cut_height, supersampled_height, y_source)
    cut += flux * (wx @ prf_supersampled @ wy.T) / 81
    return cut


def create_gaia_prf_model(supersampled_prf, cut_width, cut_height, stars_x, stars_y, stars_m_r,
                           shift_x=0.0, shift_y=0.0):
    """
    Direct port of `create_gaia_prf_model(supersampled_prf::AbstractMatrix, ...)`:
    sum of PRF cuts for every star, scaled by its TESS flux (from Gaia RP mag).
    """
    model_cut = np.zeros((cut_width, cut_height))
    for m_r, x, y in zip(stars_m_r, stars_x, stars_y):
        add_prf_cut(model_cut, calc_tess_flux_from_mag(m_r), supersampled_prf,
                    cut_width, cut_height, x + shift_x, y + shift_y)
    return model_cut


def create_gaia_prf_model_old(supersampled_prf, cut_width, cut_height, stars_x, stars_y, stars_m_r,
                               shift_x=0.0, shift_y=0.0):
    """Direct port of `create_gaia_prf_model_old` (uses the slow get_prf_cut)."""
    model_cut = np.zeros((cut_width, cut_height))
    for m_r, x, y in zip(stars_m_r, stars_x, stars_y):
        prf_cut = get_prf_cut(supersampled_prf, cut_width, cut_height, x + shift_x, y + shift_y)
        model_cut += prf_cut * calc_tess_flux_from_mag(m_r)
    return model_cut
