"""
Test for `isolated.photometry.calc_aperture_flux_error` (ROADMAP.md Этап 2:
propagate TESScut's own per-pixel FLUX_ERR through the aperture sum).

Monte Carlo validation, deliberately without any background fit (that would
mix in a different noise source and defeat the point of testing the
propagation formula in isolation): given a known per-pixel error `sigma`,
draw many independent Gaussian noise realizations and measure how much the
resulting aperture sum actually scatters -- that empirical scatter should
match what `calc_aperture_flux_error` predicts from `sigma` alone, since no
other signal is present.

Plain script with asserts, matching the other tests in this directory. Run:
    .venv/bin/python tests/test_flux_error_propagation.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isolated.photometry import calc_aperture_flux_error, aperture_sum

CUT_SIZE = 20
STAR_PX = (10.0, 10.0)
APERTURE_RADIUS = 3
SIGMA_PER_PIXEL = 20.0
N_TRIALS = 5000
# `calc_aperture_flux_error` uses ROADMAP.md's literal formula, sum(w_i *
# sigma_i^2) -- linear (not squared) aperture weights w_i, same convention
# lightkurve/SPOC use. That slightly *overestimates* the true error at the
# aperture boundary, where partial-pixel weights are 0 < w_i < 1 and the
# mathematically exact propagation would need w_i^2 there instead of w_i
# (interior pixels have w_i in {0, 1}, where w_i^2 == w_i, so they're exact
# either way). For r=3 (used here, and for the real SS 397 photometry) this
# is a real, deterministic ~6-7% overestimate, not sampling noise -- checked
# empirically that it shrinks with a bigger aperture (r=8 -> ~1.5%), exactly
# as the boundary-pixel explanation predicts. Tolerance is set above that
# known bias, to still catch a real bug (wrong exponent, missing term, off
# by a large factor) rather than this expected small effect.
RELATIVE_TOLERANCE = 0.10


def main():
    rng = np.random.default_rng(0)

    flux_err_cut = np.full((CUT_SIZE, CUT_SIZE), SIGMA_PER_PIXEL)
    predicted_sigma = calc_aperture_flux_error(flux_err_cut, *STAR_PX, APERTURE_RADIUS)

    noise_realizations = rng.normal(0.0, SIGMA_PER_PIXEL, size=(N_TRIALS, CUT_SIZE, CUT_SIZE))
    measured = np.array([aperture_sum(noise, *STAR_PX, APERTURE_RADIUS) for noise in noise_realizations])
    empirical_sigma = measured.std()

    rel_diff = abs(predicted_sigma - empirical_sigma) / predicted_sigma
    print(f"predicted (propagated) sigma: {predicted_sigma:.3f}")
    print(f"empirical (Monte Carlo, n={N_TRIALS}) sigma: {empirical_sigma:.3f}")
    print(f"relative difference: {rel_diff:.1%}")

    assert rel_diff < RELATIVE_TOLERANCE, (
        f"predicted vs Monte-Carlo empirical sigma differ by {rel_diff:.1%}, "
        f"expected < {RELATIVE_TOLERANCE:.0%}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
