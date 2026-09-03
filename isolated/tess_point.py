"""
Pure-Python replacement for the C shared library `tess_point_c` that main.jl
called via `@ccall`.

This is a faithful port of the "tess-point" focal-plane model
(Alan M. Levine / Christopher J. Burke, STScI/MIT), found in:
    tess_point_c/src/tess_stars2px.c
    tess_point_c/src/vec.c
    tess_point_c/src/mat_ra3.c

Only the pieces actually used by the Julia code are ported:
    tess_stars2px_sector(ra, dec, sector) -> (col_px, row_px)
        returns (-1.0, -1.0) if the star is not on any TESS CCD in that sector.

Julia usage this replaces:
    ptr = @ccall "./tess_point_c/libtess_stars2px.so.1.0.1".tess_stars2px_sector(...)
    coords = unsafe_wrap(Vector{Float64}, ptr, 2)

Equivalent, well-tested pixel coordinates are also available from the
official `tess-point` PyPI package (`pip install tess-point`), which
implements the same model. This module exists so the project has no
compiled/C dependency at all.
"""
from __future__ import annotations

import numpy as np

from ._tess_point_data import (
    _RAS, _DECS, _ROLLS,
    EULCAM, OPTCON, ASYMANG, ASYMFAC, CCDXY0, PIXSZ, CCDANG,
)

NSEC = 121
NCAM = 4


# --------------------------------------------------------------------------- #
# vec.c / mat_ra3.c helpers
# --------------------------------------------------------------------------- #

def _dsphcr(ra: float, dec: float) -> np.ndarray:
    """Spherical (ra, dec) in radians -> unit 3-vector. Port of dsphcr()."""
    return np.array([
        np.cos(ra) * np.cos(dec),
        np.sin(ra) * np.cos(dec),
        np.sin(dec),
    ])


def _dcrsph(vec: np.ndarray):
    """Cartesian 3-vector -> spherical (ra, dec) in radians. Port of dcrsph()."""
    length = np.linalg.norm(vec)
    ra = 0.0
    dec = 0.0
    if length > 0.0:
        dec = np.arcsin(vec[2] / length)
        if vec[0] != 0.0 or vec[1] != 0.0:
            ra = np.arctan2(vec[1], vec[0])
            if ra < 0.0:
                ra += 2.0 * np.pi
    return ra, dec


def _rotm1(nax: int, angle: float) -> np.ndarray:
    """Elementary rotation matrix about axis `nax` (0,1,2). Port of rotm1()."""
    n1 = nax
    n2 = (n1 + 1) % 3
    n3 = (n2 + 1) % 3
    s = np.sin(angle)
    c = np.cos(angle)
    mat = np.zeros((3, 3))
    mat[n1, n1] = 1.0
    mat[n3, n3] = c
    mat[n2, n2] = c
    mat[n2, n3] = s
    mat[n3, n2] = -s
    return mat


def _eulerm323(euler: np.ndarray) -> np.ndarray:
    """3-2-3 Euler-angle rotation matrix (radians). Port of eulerm323()."""
    mat1 = _rotm1(2, euler[0])
    mat2 = _rotm1(1, euler[1])
    mata = mat2 @ mat1
    mat1 = _rotm1(2, euler[2])
    return mat1 @ mata


# --------------------------------------------------------------------------- #
# tess_stars2px.c
# --------------------------------------------------------------------------- #

def _sky_to_sc_mat(ra_deg: float, dec_deg: float, roll_deg: float) -> np.ndarray:
    """Port of sky_to_sc_mat()."""
    xeul = np.array([
        np.radians(ra_deg),
        np.pi / 2.0 - np.radians(dec_deg),
        np.radians(roll_deg) + np.pi,
    ])
    return _eulerm323(xeul)


def _sc_to_cam_mat(euler_deg: np.ndarray) -> np.ndarray:
    """Port of sc_to_cam_mat()."""
    return _eulerm323(np.radians(euler_deg))


def _star_in_fov(lng_deg: float, lat_deg: float) -> bool:
    """Port of star_in_fov(): is the star within ~12.5 deg of the camera boresight?"""
    if lat_deg <= 70.0:
        return False
    lngr = np.radians(lng_deg)
    latr = np.radians(lat_deg)
    v = _dsphcr(lngr, latr)
    v = v / np.linalg.norm(v)
    return (abs(np.arctan(v[0] / v[2])) <= np.radians(12.5) and
            abs(np.arctan(v[1] / v[2])) <= np.radians(12.5))


def _xyrotate(angle_deg: float, xy: np.ndarray) -> np.ndarray:
    """Port of xyrotate()."""
    ca = np.cos(np.radians(angle_deg))
    sa = np.sin(np.radians(angle_deg))
    return np.array([ca * xy[0] + sa * xy[1], -sa * xy[0] + ca * xy[1]])


def _make_az_asym(icam: int, xy: np.ndarray) -> np.ndarray:
    """Port of make_az_asym(). No-op for TESS (asymfac=1, asymang=0 for all cams)."""
    xyp = _xyrotate(ASYMANG[icam], xy)
    xypa = np.array([ASYMFAC[icam] * xyp[0], xyp[1]])
    return _xyrotate(-ASYMANG[icam], xypa)


def _optics_fp(icam: int, lng_deg: float, lat_deg: float) -> np.ndarray:
    """Port of optics_fp(): sky angle -> focal-plane position in mm."""
    thetar = np.pi / 2.0 - np.radians(lat_deg)
    tanth = np.tan(thetar)
    cphi = np.cos(np.radians(lng_deg))
    sphi = np.sin(np.radians(lng_deg))
    rfp0 = OPTCON[icam, 0] * tanth
    rfp = 0.0
    for i in range(1, 6):  # NOPTCON = 6
        rfp += OPTCON[icam, i] * tanth ** (2 * (i - 1))
    xytmp = np.array([-cphi * rfp0 * rfp, -sphi * rfp0 * rfp])
    return _make_az_asym(icam, xytmp)


def _mm_to_pix(icam: int, xy: np.ndarray):
    """Port of mm_to_pix(): focal-plane mm -> CCD pixel coords + which CCD (0-3)."""
    if xy[0] >= 0.0:
        iccd = 0 if xy[1] >= 0.0 else 3
    else:
        iccd = 1 if xy[1] >= 0.0 else 2

    xyb = xy - CCDXY0[icam, iccd]
    xyccd = _xyrotate(CCDANG[icam, iccd], xyb)
    ccdpx = np.array([
        xyccd[0] / PIXSZ[icam, iccd, 0] - 0.5,
        xyccd[1] / PIXSZ[icam, iccd, 1] - 0.5,
    ])
    return ccdpx, iccd


def _ccdpx_to_outpx(ccdpx: np.ndarray):
    """Port of ccdpx_to_outpx(): CCD-local pixel -> full-frame output pixel."""
    x_use = ccdpx[0] + 45.0
    y_use = ccdpx[1] + 1.0
    x_min = 44.0
    y_max_coord = 2049.0
    x_max_coord = 2093.0

    if x_min < x_use < x_max_coord and 0.0 < y_use < y_max_coord:
        return True, np.array([x_use, y_use])
    return False, np.array([-1.0, -1.0])


def tess_stars2px_sector(ra_deg: float, dec_deg: float, sector: int) -> np.ndarray:
    """
    Direct port of `tess_stars2px_sector` from tess_stars2px.c.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Star coordinates in degrees (ICRS).
    sector : int
        TESS sector number, 1-based (as in the Julia code).

    Returns
    -------
    np.ndarray shape (2,)
        [col_px, row_px] on the CCD the star falls on, or [-1.0, -1.0] if the
        star was not observed by any camera in that sector.
    """
    isec = sector - 1
    if not (0 <= isec < NSEC):
        return np.array([-1.0, -1.0])

    ra_sc, dec_sc, roll_sc = _RAS[isec], _DECS[isec], _ROLLS[isec]
    rmat1 = _sky_to_sc_mat(ra_sc, dec_sc, roll_sc)

    vstar = _dsphcr(np.radians(ra_deg), np.radians(dec_deg))

    napx = np.array([-1.0, -1.0])
    found = False

    for icam in range(NCAM):
        rmat2 = _sc_to_cam_mat(EULCAM[icam])
        rmat4 = rmat2 @ rmat1

        vcam = rmat4 @ vstar
        lng, lat = _dcrsph(vcam)
        lng_deg, lat_deg = np.degrees(lng), np.degrees(lat)

        if _star_in_fov(lng_deg, lat_deg):
            xyfp = _optics_fp(icam, lng_deg, lat_deg)
            ccdpx, _iccd = _mm_to_pix(icam, xyfp)
            on_silicon, cand = _ccdpx_to_outpx(ccdpx)
            if on_silicon:
                found = True
                napx = cand

    if not found:
        napx = np.array([-1.0, -1.0])
    return napx


def is_in_sector(ra_deg: float, dec_deg: float, sector: int) -> bool:
    """Port of `is_in_sector` in main.jl."""
    px = tess_stars2px_sector(ra_deg, dec_deg, sector)
    return not (px[0] < 0.0 or px[1] < 0.0)


def find_tess_sectors(ra_deg: float, dec_deg: float, max_sector: int) -> list[int]:
    """Port of `find_tess_sectors(alpha, delta, max_sector)` in main.jl."""
    return [s for s in range(1, max_sector + 1) if is_in_sector(ra_deg, dec_deg, s)]
