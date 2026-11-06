"""
Small helpers for reading the WCS-like keywords TESScut puts in extension 2
of its cutout FITS files, and computing the cutout's sky-coordinate corners.

Ported from main.jl:
    get_px_to_radec_matrix, get_reference_radec, get_reference_px,
    get_tesscut_corners
"""
from __future__ import annotations

import numpy as np

from .geometry import get_true_radec


def _wcs_keyword(cut_fits, pixels_key, aperture_key):
    """
    One cutout-WCS keyword. Newer TESScut files carry it in the PIXELS extension
    (e.g. `1CRPX4`, python `cut_fits[1]`; Julia `fits[2]`); older products (e.g.
    sector 18) don't have those, but always have the very same WCS (identical
    values wherever both exist) in the APERTURE extension (`CRPIX1`, ...,
    `cut_fits[2]`) -- used only when the PIXELS one is absent.
    """
    hdr = cut_fits[1].header  # Julia fits[2] == python cut_fits[1] (0-based HDU list)
    if pixels_key in hdr:
        return hdr[pixels_key]
    try:
        return cut_fits[2].header[aperture_key]
    except (IndexError, KeyError):
        raise KeyError(f"cutout has neither {pixels_key!r} in the PIXELS header nor {aperture_key!r} "
                       f"in the APERTURE header -- can't locate it on the sky") from None


def get_px_to_radec_matrix(cut_fits):
    """Direct port of `get_px_to_radec_matrix` (APERTURE-header fallback for older cutouts)."""
    return np.array([
        [_wcs_keyword(cut_fits, "11PC4", "PC1_1"), _wcs_keyword(cut_fits, "12PC4", "PC1_2")],
        [_wcs_keyword(cut_fits, "21PC4", "PC2_1"), _wcs_keyword(cut_fits, "22PC4", "PC2_2")],
    ])


def get_reference_radec(cut_fits):
    """Direct port of `get_reference_radec` (APERTURE-header fallback for older cutouts)."""
    return np.array([_wcs_keyword(cut_fits, "1CRVL4", "CRVAL1"), _wcs_keyword(cut_fits, "2CRVL4", "CRVAL2")])


def get_reference_px(cut_fits):
    """Direct port of `get_reference_px` (APERTURE-header fallback for older cutouts)."""
    return np.array([_wcs_keyword(cut_fits, "1CRPX4", "CRPIX1"), _wcs_keyword(cut_fits, "2CRPX4", "CRPIX2")])


def get_tesscut_corners(cut_fits):
    """
    Direct port of `get_tesscut_corners`: sky coordinates of the 4 corners of
    the cutout, in the order [bottom_left, top_left, top_right, bottom_right].
    """
    reference_px = get_reference_px(cut_fits)
    reference_radec = get_reference_radec(cut_fits)
    m = get_px_to_radec_matrix(cut_fits)

    flux = cut_fits[1].data["FLUX"]  # shape (n_cuts, height, width) as read by astropy
    # Julia `size(read(fits[2], "FLUX"))[1:2]` gives (width, height) in FITS/Julia
    # (column-major) convention; astropy gives the cube as (n_cuts, height, width).
    height, width = flux.shape[-2], flux.shape[-1]

    def corner(dxdy):
        rel = m @ (np.array(dxdy) - reference_px)
        return get_true_radec(reference_radec[0], reference_radec[1], rel[0], rel[1])

    bottom_left = corner([0.5, 0.5])
    bottom_right = corner([width + 0.5, 0.5])
    top_left = corner([0.5, height + 0.5])
    top_right = corner([width + 0.5, height + 0.5])

    return [bottom_left, top_left, top_right, bottom_right]
