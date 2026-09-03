"""
Levenberg-Marquardt fits of star positions/fluxes to cutout images.

Ported from main.jl:
    fit_stars_gauss, fit_stars_prf, fit_stars_prf_bkg, fit_stars_prf_flat_bkg

Julia's `LeastSquaresOptim.optimize(..., LevenbergMarquardt())` /
`Optim.optimize(..., LevenbergMarquardt())` map to
`scipy.optimize.least_squares(..., method="lm")`.

Note: the original Julia code supplies an analytic Jacobian for
`fit_stars_prf_flat_bkg` for speed; since this project explicitly does not
need that speed, this port lets scipy estimate the Jacobian numerically,
which gives the same fit result at (negligible, for this problem size) extra
cost.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .psf import elliptic_powexp_psf, get_prf_cut


def fit_stars_gauss(cut_no_bkg, stars_px_x, stars_px_y, super_samp=10):
    """
    Direct port of `fit_stars_gauss`: fit a shared elliptic power-law PSF
    (amplitude per star + shared x_sigma, y_sigma, power) to a background-
    subtracted cutout.

    Returns the fitted parameter vector
        [amp_1, ..., amp_n, power, x_sigma, y_sigma]
    (same order as the Julia `pars` vector).
    """
    width, height = cut_no_bkg.shape
    n_stars = len(stars_px_x)

    def residual(pars):
        max_ints = pars[:n_stars]
        power = pars[-3]
        x_sigma = pars[-2]
        y_sigma = pars[-1]
        model = cut_no_bkg.copy()
        for i in range(n_stars):
            model = model - elliptic_powexp_psf(width, height, stars_px_x[i], stars_px_y[i],
                                                 x_sigma, y_sigma, power, abs(max_ints[i]),
                                                 super_samp=super_samp)
        return model.ravel()

    start_pars = np.full(n_stars + 3, 100.0)
    start_pars[-2:] = 1.0
    start_pars[-3] = 2.0

    result = least_squares(residual, start_pars, method="lm")
    return result.x


def fit_stars_prf(supersampled_prf, cut_no_bkg, stars_px_x, stars_px_y, start_pars):
    """Direct port of `fit_stars_prf`: fit per-star PRF amplitudes, no background term."""
    width, height = cut_no_bkg.shape
    n_stars = len(stars_px_x)
    prf_cuts = [get_prf_cut(supersampled_prf, width, height, stars_px_x[i], stars_px_y[i]).ravel()
                for i in range(n_stars)]
    model_start = cut_no_bkg.ravel().copy()

    def residual(pars):
        model = model_start.copy()
        for i in range(n_stars):
            model = model - prf_cuts[i] * abs(pars[i])
        return model

    result = least_squares(residual, start_pars, method="lm")
    return result.x


def fit_stars_prf_bkg(supersampled_prf, cut, stars_px_x, stars_px_y):
    """Direct port of `fit_stars_prf_bkg`: per-star PRF amplitudes + one flat (constant) background."""
    width, height = cut.shape
    n_stars = len(stars_px_x)
    prf_cuts = [get_prf_cut(supersampled_prf, width, height, stars_px_x[i], stars_px_y[i])
                for i in range(n_stars)]

    def residual(pars):
        model = cut.copy()
        for i in range(n_stars):
            model = model - prf_cuts[i] * abs(pars[i])
        return (model - pars[-1]).ravel()

    start_pars = np.full(n_stars + 1, 100.0)
    result = least_squares(residual, start_pars, method="lm")
    return result.x


def fit_stars_prf_flat_bkg(supersampled_prf, cut, stars_px_x, stars_px_y, start_pars):
    """
    Direct port of `fit_stars_prf_flat_bkg`: per-star PRF amplitudes + a
    tilted (flat-plane) background, parametrized the same odd-but-equivalent
    way as `fit_flat_background` (normal-vector form, last 4 parameters).
    """
    width, height = cut.shape
    n_stars = len(stars_px_x)
    prf_cuts = [get_prf_cut(supersampled_prf, width, height, stars_px_x[i], stars_px_y[i]).ravel()
                for i in range(n_stars)]
    model_start = cut.ravel().copy()

    xs = np.arange(1, width + 1)
    ys = np.arange(1, height + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()

    def residual(pars):
        model = model_start.copy()
        for i in range(n_stars):
            model = model - prf_cuts[i] * abs(pars[i])
        n1, n2, n3, d = pars[-4], pars[-3], pars[-2], pars[-1]
        normal = np.sqrt(n1 ** 2 + n2 ** 2 + n3 ** 2)
        model = model - (d * normal - n1 * xx_flat - n2 * yy_flat) / n3
        return model

    result = least_squares(residual, start_pars, method="lm",
                            xtol=1e-10, ftol=1e-10, gtol=1e-10)
    return result.x
