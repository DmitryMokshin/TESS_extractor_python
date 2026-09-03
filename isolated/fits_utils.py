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


def get_px_to_radec_matrix(cut_fits):
    """Direct port of `get_px_to_radec_matrix` (extension index 2, 1-based -> 1 here)."""
    hdr = cut_fits[1].header  # Julia fits[2] == python cut_fits[1] (0-based HDU list)
    return np.array([
        [hdr["11PC4"], hdr["12PC4"]],
        [hdr["21PC4"], hdr["22PC4"]],
    ])


def get_reference_radec(cut_fits):
    """Direct port of `get_reference_radec`."""
    hdr = cut_fits[1].header
    return np.array([hdr["1CRVL4"], hdr["2CRVL4"]])


def get_reference_px(cut_fits):
    """Direct port of `get_reference_px`."""
    hdr = cut_fits[1].header
    return np.array([hdr["1CRPX4"], hdr["2CRPX4"]])


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
