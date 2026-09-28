"""
Synthetic test for ROADMAP.md Этап 9's pure-performance refactors (items
1-3 of 5; storage format and Gaia query pagination are out of scope for
this stage -- see plan): vectorized `isolated.psf.get_prf_cut`/`add_prf_cut`,
`isolated.psf.get_tesscut_prf_supersampled`'s in-process cache,
`isolated.photometry.aperture_sum`'s precomputed-mask path, and
`isolated.photometry._fit_plane`'s lstsq replacement.

These are all "same math, faster" refactors (not new behaviour), so the
tests check NUMERICAL EQUIVALENCE against a reference implementation (the
original nested-loop/Levenberg-Marquardt code, reproduced here as the
known-correct baseline) rather than injected-signal recovery. Plain script
with asserts -- matches the other tests in this directory (no pytest
dependency yet). Run directly:
    .venv/bin/python tests/test_performance_synthetic.py
"""
import os
import sys
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isolated.psf import get_prf_cut, add_prf_cut, get_tesscut_prf_supersampled, _PRF_CACHE
from isolated.photometry import aperture_sum, precompute_aperture_mask, _fit_plane

RNG = np.random.default_rng(5)


# --------------------------------------------------------------------------- #
# reference (pre-Этап-9) implementations, reproduced here as the known-correct
# baseline -- NOT imported from isolated/, so a future edit to the vectorized
# versions can't accidentally make this test compare a function to itself
# --------------------------------------------------------------------------- #

def _ref_get_prf_cut(prf_supersampled, cut_width, cut_height, x_source, y_source):
    prf = np.zeros((cut_width, cut_height))
    ssw, ssh = prf_supersampled.shape
    for x_px in range(1, cut_width + 1):
        for y_px in range(1, cut_height + 1):
            xps = (x_px - x_source) * 9 + 59
            yps = (y_px - y_source) * 9 + 59
            mask = np.ones((ssw, ssh))
            for x_prf in range(1, ssw + 1):
                dx = abs(x_prf - xps)
                for y_prf in range(1, ssh + 1):
                    dy = abs(y_prf - yps)
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


def _ref_add_prf_cut(cut, flux, prf_supersampled, cut_width, cut_height, x_source, y_source):
    ssw, ssh = prf_supersampled.shape
    for x_px in range(1, cut_width + 1):
        for y_px in range(1, cut_height + 1):
            xps = (x_px - x_source) * 9 + 59
            yps = (y_px - y_source) * 9 + 59
            xpsi = round(xps)
            ypsi = round(yps)
            xs, xe = max(1, xpsi - 6), min(ssw, xpsi + 6)
            ys, ye = max(1, ypsi - 6), min(ssh, ypsi + 6)
            acc = 0.0
            for x_prf in range(xs, xe + 1):
                dx = abs(x_prf - xps)
                for y_prf in range(ys, ye + 1):
                    dy = abs(y_prf - yps)
                    if dx >= 5 or dy >= 5:
                        continue
                    m = 1.0
                    if not (dx <= 4 and dy <= 4):
                        if dx > 4:
                            m *= 5 - dx
                        if dy > 4:
                            m *= 5 - dy
                    acc += prf_supersampled[x_prf - 1, y_prf - 1] * m / 81 * flux
            cut[x_px - 1, y_px - 1] += acc
    return cut


def _ref_fit_plane(bkg_fluxes, bkg_xs, bkg_ys, cut_width, cut_height):
    from scipy.optimize import least_squares
    bkg_xs = np.asarray(bkg_xs, dtype=float)
    bkg_ys = np.asarray(bkg_ys, dtype=float)
    bkg_fluxes = np.asarray(bkg_fluxes, dtype=float)

    def residual(pars):
        normal = np.sqrt(pars[0] ** 2 + pars[1] ** 2 + pars[2] ** 2)
        return bkg_fluxes - (pars[3] * normal - pars[0] * bkg_xs - pars[1] * bkg_ys) / pars[2]

    result = least_squares(residual, x0=[0.0, 0.0, 1.0, 100.0], method="lm")
    bp = result.x
    normal = np.sqrt(bp[0] ** 2 + bp[1] ** 2 + bp[2] ** 2)
    xs = np.arange(1, cut_width + 1)
    ys = np.arange(1, cut_height + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return (bp[3] * normal - bp[0] * xx - bp[1] * yy) / bp[2]


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

def test_get_prf_cut_and_add_prf_cut_match_reference():
    ssw = ssh = 39  # smaller than the real 117x117 PRF, plenty for equivalence
    prf_ss = RNG.normal(100, 10, (ssw, ssh)) ** 2
    for x_source, y_source in [(10.3, 10.7), (5.1, 5.9), (15.0, 15.0), (1.0, 20.0)]:
        ref = _ref_get_prf_cut(prf_ss, 9, 9, x_source, y_source)
        new = get_prf_cut(prf_ss, 9, 9, x_source, y_source)
        assert np.allclose(ref, new, atol=1e-9), f"get_prf_cut mismatch at ({x_source},{y_source})"

        cut_ref = np.zeros((9, 9))
        cut_new = np.zeros((9, 9))
        _ref_add_prf_cut(cut_ref, 3.5, prf_ss, 9, 9, x_source, y_source)
        add_prf_cut(cut_new, 3.5, prf_ss, 9, 9, x_source, y_source)
        assert np.allclose(cut_ref, cut_new, atol=1e-9), f"add_prf_cut mismatch at ({x_source},{y_source})"
    print("get_prf_cut / add_prf_cut: vectorized matches reference nested-loop implementation -- OK")


def test_add_prf_cut_matches_get_prf_cut_before_threshold():
    # get_prf_cut's docstring says add_prf_cut is "the windowed/optimized
    # version of get_prf_cut" -- same weights, minus get_prf_cut's own
    # >=2e-4 threshold. Check that relationship holds for the new code too.
    ssw = ssh = 39
    prf_ss = RNG.normal(50, 5, (ssw, ssh)) ** 2
    x_source, y_source = 7.3, 8.6
    cut = np.zeros((9, 9))
    add_prf_cut(cut, 1.0, prf_ss, 9, 9, x_source, y_source)
    thresholded = np.where(cut >= 2e-4, cut, 0.0)
    direct = get_prf_cut(prf_ss, 9, 9, x_source, y_source)
    assert np.allclose(thresholded, direct, atol=1e-9), "add_prf_cut(flux=1) should equal get_prf_cut after thresholding"
    print("add_prf_cut(flux=1) == get_prf_cut() (after its threshold) -- OK")


def test_get_tesscut_prf_supersampled_caches():
    # a synthetic cut_fits (no real WCS/network needed): fake out the two
    # calls get_tesscut_prf_supersampled makes BEFORE it even checks the
    # cache (get_reference_radec, tess_stars2px_sector) so a pre-seeded cache
    # entry short-circuits before any real download/interpolation happens --
    # if caching weren't working, this would fail trying to open PRF files
    # that don't exist.
    _PRF_CACHE.clear()

    class FakeHDU:
        def __init__(self, header):
            self.header = header

    cut_fits = [FakeHDU({"SECTOR": 80, "CAMERA": 2, "CCD": 3}), FakeHDU({})]
    fake_prf = RNG.normal(0, 1, (5, 5))
    cache_key = (80, 2, 3, 123, 456, "prf")
    _PRF_CACHE[cache_key] = fake_prf

    with mock.patch("isolated.fits_utils.get_reference_radec", lambda _cut_fits: (100.0, -10.0)), \
         mock.patch("isolated.psf.tess_stars2px_sector", lambda _ra, _dec, _sector: np.array([123.0, 456.0])):
        result1 = get_tesscut_prf_supersampled(cut_fits)
        result2 = get_tesscut_prf_supersampled(cut_fits)
    assert result1 is fake_prf and result2 is fake_prf, "expected the pre-seeded cache entry to be returned as-is"
    print("get_tesscut_prf_supersampled: cache hit returns the cached array without touching PRF files -- OK")
    _PRF_CACHE.clear()


def test_aperture_sum_mask_matches_default():
    data = RNG.normal(100, 10, (15, 15))
    for x, y, r in [(7.5, 7.5, 3.0), (1.2, 1.3, 2.5), (14.8, 14.9, 4.0), (0.5, 0.5, 3.0)]:
        old = aperture_sum(data, x, y, r)
        new = aperture_sum(data, x, y, r, mask=precompute_aperture_mask(x, y, r))
        assert abs(old - new) < 1e-9, f"aperture_sum mask mismatch at ({x},{y},{r}): {old} vs {new}"
    print("aperture_sum: precomputed-mask path matches the default photutils path -- OK")


def test_fit_plane_recovers_exact_plane_and_matches_reference():
    a, b, c = 120.0, 0.8, -1.3
    xs = RNG.uniform(1, 50, 40)
    ys = RNG.uniform(1, 50, 40)

    exact_fluxes = a + b * xs + c * ys
    new_exact = _fit_plane(exact_fluxes, xs, ys, 3, 3)
    xx, yy = np.meshgrid([1, 2, 3], [1, 2, 3], indexing="ij")
    expected = a + b * xx + c * yy
    assert np.allclose(new_exact, expected, atol=1e-8), "lstsq _fit_plane should exactly recover a noise-free plane"

    noisy_fluxes = exact_fluxes + RNG.normal(0, 2.0, xs.size)
    ref = _ref_fit_plane(noisy_fluxes, xs, ys, 50, 50)
    new = _fit_plane(noisy_fluxes, xs, ys, 50, 50)
    rel_diff = np.max(np.abs(ref - new)) / np.max(np.abs(ref))
    assert rel_diff < 1e-4, f"lstsq _fit_plane should closely match the converged LM reference, got {rel_diff:.2e}"
    print(f"_fit_plane: exact-plane recovery to 1e-8, noisy case matches LM reference (rel diff {rel_diff:.2e}) -- OK")


if __name__ == "__main__":
    test_get_prf_cut_and_add_prf_cut_match_reference()
    test_add_prf_cut_matches_get_prf_cut_before_threshold()
    test_get_tesscut_prf_supersampled_caches()
    test_aperture_sum_mask_matches_default()
    test_fit_plane_recovers_exact_plane_and_matches_reference()
    print("\nALL CHECKS PASSED")
