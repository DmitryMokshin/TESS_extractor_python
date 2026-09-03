"""
Spherical astrometry helpers.

Ported 1:1 from the following functions in main.jl:
    get_nospace_star_name, get_true_radec, get_rel_radec, get_distance,
    calc_tess_magnitude, calc_tess_flux_from_mag, get_true_jd
"""
from __future__ import annotations

import numpy as np


def get_nospace_star_name(star_name: str) -> str:
    """Julia: replace(star_name, " " => "_")"""
    return star_name.replace(" ", "_")


def get_true_radec(alpha_0: float, delta_0: float, d_alpha: float, d_delta: float) -> np.ndarray:
    """
    Convert a small tangent-plane offset (d_alpha, d_delta), both in degrees,
    around a reference point (alpha_0, delta_0) (degrees) into true (ra, dec).

    Direct port of `get_true_radec` in main.jl.
    """
    a0 = np.radians(alpha_0)
    d0 = np.radians(delta_0)
    da = np.radians(d_alpha)
    dd = np.radians(d_delta)

    alpha_hat = np.array([-np.sin(a0), np.cos(a0), 0.0])
    delta_hat = np.array([-np.cos(a0) * np.sin(d0), -np.sin(a0) * np.sin(d0), np.cos(d0)])
    xyz0 = np.array([np.cos(a0) * np.cos(d0), np.sin(a0) * np.cos(d0), np.sin(d0)])

    xyz = xyz0 + da * alpha_hat + dd * delta_hat
    xyz = xyz / np.sqrt(np.sum(xyz ** 2))

    dec = np.degrees(np.arcsin(xyz[2]))
    ra = np.degrees(np.arctan2(xyz[1], xyz[0]))

    if ra < 0:
        ra += 360.0
    return np.array([ra, dec])


def get_rel_radec(alpha_0: float, delta_0: float, alpha: float, delta: float) -> np.ndarray:
    """
    Inverse of get_true_radec: tangent-plane offset (degrees) of (alpha, delta)
    relative to reference point (alpha_0, delta_0), both in degrees.

    Direct port of `get_rel_radec` in main.jl.
    """
    a0 = np.radians(alpha_0)
    d0 = np.radians(delta_0)
    a = np.radians(alpha)
    d = np.radians(delta)

    xyz0 = np.array([np.cos(a0) * np.cos(d0), np.sin(a0) * np.cos(d0), np.sin(d0)])
    xyz = np.array([np.cos(a) * np.cos(d), np.sin(a) * np.cos(d), np.sin(d)])
    xyz = xyz / np.dot(xyz, xyz0)
    d_xyz = xyz - xyz0

    alpha_hat = np.array([-np.sin(a0), np.cos(a0), 0.0])
    delta_hat = np.array([-np.cos(a0) * np.sin(d0), -np.sin(a0) * np.sin(d0), np.cos(d0)])

    d_alpha = np.dot(d_xyz, alpha_hat)
    d_delta = np.dot(d_xyz, delta_hat)

    return np.degrees(np.array([d_alpha, d_delta]))


def get_distance(alpha_1: float, delta_1: float, alpha_2: float, delta_2: float) -> float:
    """Great-circle distance in degrees. Direct port of `get_distance`."""
    a1, d1, a2, d2 = map(np.radians, (alpha_1, delta_1, alpha_2, delta_2))
    xyz1 = np.array([np.cos(a1) * np.cos(d1), np.sin(a1) * np.cos(d1), np.sin(d1)])
    xyz2 = np.array([np.cos(a2) * np.cos(d2), np.sin(a2) * np.cos(d2), np.sin(d2)])
    return float(np.degrees(np.arccos(np.clip(np.dot(xyz1, xyz2), -1.0, 1.0))))


def calc_tess_magnitude(flux):
    """Direct port of `calc_tess_magnitude`: -2.5*log10(abs(flux)) + 20.44"""
    return -2.5 * np.log10(np.abs(flux)) + 20.44


def calc_tess_flux_from_mag(mag):
    """Direct port of `calc_tess_flux_from_mag`: 10^(0.4*(20.44 - mag))"""
    return 10.0 ** (0.4 * (20.44 - mag))


def get_true_jd(mjd):
    """Direct port of `get_true_jd`: mjd + 2457000"""
    return mjd + 2457000
