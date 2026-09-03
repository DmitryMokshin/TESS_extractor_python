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
    return interpolated_prf


def get_prf_cut(prf_supersampled, cut_width, cut_height, x_source, y_source):
    """
    Direct (unoptimized, matches the Julia reference implementation) port of
    `get_prf_cut`: bin the 9x-supersampled PRF onto a cut_width x cut_height
    pixel grid centered on (x_source, y_source) (1-based pixel coordinates).
    """
    prf = np.zeros((cut_width, cut_height))
    supersampled_width, supersampled_height = prf_supersampled.shape

    for x_px in range(1, cut_width + 1):
        for y_px in range(1, cut_height + 1):
            x_prf_source = (x_px - x_source) * 9 + 59
            y_prf_source = (y_px - y_source) * 9 + 59

            mask = np.ones((supersampled_width, supersampled_height))
            for x_prf in range(1, supersampled_width + 1):
                dx = abs(x_prf - x_prf_source)
                for y_prf in range(1, supersampled_height + 1):
                    dy = abs(y_prf - y_prf_source)
                    if dx >= 5 or dy >= 5:
                        mask[x_prf - 1, y_prf - 1] = 0.0
                        continue
                    if dx <= 4 and dy <= 4:
                        mask[x_prf - 1, y_prf - 1] = 1.0
                        continue
                    m = mask[x_prf - 1, y_prf - 1]
                    if dx > 4:
                        m *= 5 - dx
                    if dy > 4:
                        m *= 5 - dy
                    mask[x_prf - 1, y_prf - 1] = m

            val = np.sum(prf_supersampled * mask) / 81
            prf[x_px - 1, y_px - 1] = val if val >= 2e-4 else 0.0
    return prf


def add_prf_cut(cut, flux, prf_supersampled, cut_width, cut_height, x_source, y_source):
    """
    Direct port of `add_prf_cut!`: the windowed/optimized version of
    get_prf_cut, adding `flux * PRF` into `cut` in-place. Returns `cut`.
    """
    supersampled_width, supersampled_height = prf_supersampled.shape

    for x_px in range(1, cut_width + 1):
        for y_px in range(1, cut_height + 1):
            x_prf_source = (x_px - x_source) * 9 + 59
            y_prf_source = (y_px - y_source) * 9 + 59
            x_prf_source_int = round((x_px - x_source) * 9 + 59)
            y_prf_source_int = round((y_px - y_source) * 9 + 59)

            x_prf_start = max(1, x_prf_source_int - 6)
            x_prf_end = min(supersampled_width, x_prf_source_int + 6)
            y_prf_start = max(1, y_prf_source_int - 6)
            y_prf_end = min(supersampled_height, y_prf_source_int + 6)

            acc = 0.0
            for x_prf in range(x_prf_start, x_prf_end + 1):
                dx = abs(x_prf - x_prf_source)
                for y_prf in range(y_prf_start, y_prf_end + 1):
                    dy = abs(y_prf - y_prf_source)
                    if dx >= 5 or dy >= 5:
                        continue
                    mask = 1.0
                    if dx <= 4 and dy <= 4:
                        mask = 1.0
                    else:
                        if dx > 4:
                            mask *= 5 - dx
                        if dy > 4:
                            mask *= 5 - dy
                    acc += prf_supersampled[x_prf - 1, y_prf - 1] * mask / 81 * flux
            cut[x_px - 1, y_px - 1] += acc
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
