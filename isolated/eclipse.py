"""
Analytic circular-disk eclipse model (both stars/disks assumed circular,
primary disk radius = 1, secondary disk radius = r).

Ported from eclipse-light-curve.jl:
    calc_intersect_area, calc_eclipse_magnitude, calc_transit_mag
"""
from __future__ import annotations

import numpy as np


def calc_intersect_area(h: float, x: float, r: float) -> float:
    """
    Direct port of `calc_intersect_area`: fractional overlap area (of the
    r-radius disk) with a unit-radius disk, disk centers separated by
    s = sqrt(h^2 + x^2), normalized by the r-disk's area (pi*r^2).
    """
    s = np.sqrt(h ** 2 + x ** 2)
    if s <= 1 - r:
        return 1.0
    if s >= r + 1:
        return 0.0

    cos_alpha = -(r ** 2 - 1 - s ** 2) / 2 / s
    cos_beta = -(1 - r ** 2 - s ** 2) / 2 / s / r
    if abs(cos_beta) > 1.0:
        cos_beta = cos_beta / abs(cos_beta)

    alpha = np.arccos(cos_alpha)
    beta = np.arccos(cos_beta)

    s_alpha = alpha - np.sin(alpha) * np.cos(alpha)
    s_beta = beta * r ** 2 - np.sin(beta) * np.cos(beta) * r ** 2

    return (s_alpha + s_beta) / np.pi / r ** 2


def calc_eclipse_magnitude(m_1: float, m_2: float, s: float) -> float:
    """Direct port of `calc_eclipse_magnitude`: combined magnitude with a fraction `s` of star 2 covered."""
    if 1 - s < 1e-6:
        m_2_ecl = 1000
    else:
        m_2_ecl = m_2 - 2.5 * np.log10(1 - s)
    return -2.5 * np.log10(10 ** (-0.4 * m_1) + 10 ** (-0.4 * m_2_ecl))


def calc_transit_mag(all_mag, primary_rel_flux, a, phi, phi_0, h, r):
    """Direct port of `calc_transit_mag`."""
    all_flux = 10 ** (0.4 * (10 - all_mag))
    primary_flux = all_flux / (1 + 1 / primary_rel_flux)
    secondary_flux = primary_flux / primary_rel_flux

    x = (phi - phi_0) * a
    s = calc_intersect_area(abs(h), x, abs(r))
    flux = all_flux - secondary_flux * s
    return 10 - 2.5 * np.log10(flux)
