"""
Synthetic test for `isolated/photometry.py`'s aperture-contamination
functions: `calc_prf_contamination_fraction` (existing, previously only
exercised against real cached data in `test_ss397_regression.py`) and
`find_dominant_contaminant` (new -- auto-picks the single biggest
contaminating neighbour instead of a hand-picked Gaia `source_id`, see
`PRF_cleaning.py`). Both share their computation (`_prf_aperture_fluxes`),
so these checks also guard that refactor against the two functions
diverging. Real PRF from `prf/`, synthetic Gaia-like star positions/fluxes,
no network.

Checks:
  - two stars (target + one neighbour): `find_dominant_contaminant`'s
    fraction is exactly `1 - calc_prf_contamination_fraction(...)` -- with
    only one other star in the field, all of "not the target" is that
    neighbour.
  - three stars (target + close bright neighbour + far faint star):
    `find_dominant_contaminant` picks the close bright one, not the far
    faint one.
  - one star (target only): `find_dominant_contaminant` returns (None, 0.0).
  - `rank_contaminants`: several neighbours ordered by aperture share,
    `n` cap, `min_fraction` cut, first entry == `find_dominant_contaminant`.

Plain script with asserts -- matches the other tests in this directory.
Run directly:
    .venv/bin/python tests/test_contamination_synthetic.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isolated.photometry import calc_prf_contamination_fraction, find_dominant_contaminant, rank_contaminants

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CUT_SIZE = 25
APERTURE_RADIUS = 3
TARGET_PX = (12.0, 12.0)


def load_real_prf():
    from astropy.io import fits as pyfits
    candidates = sorted(glob.glob(os.path.join(REPO_ROOT, "prf", "**", "*.fits"), recursive=True))
    if not candidates:
        raise FileNotFoundError("No cached PRF FITS files under prf/ -- can't run the synthetic test.")
    return pyfits.getdata(candidates[0]).astype(float)


def test_two_stars_neighbour_fraction_complements_target(prf):
    stars_x = np.array([TARGET_PX[0], TARGET_PX[0] + 3.2])
    stars_y = np.array([TARGET_PX[1], TARGET_PX[1] + 0.5])
    stars_flux = np.array([12000.0, 18000.0])  # brighter neighbour, like the real SS 397 case

    target_frac = calc_prf_contamination_fraction(
        TARGET_PX[0], TARGET_PX[1], stars_x, stars_y, stars_flux, prf, CUT_SIZE, APERTURE_RADIUS)
    idx, neighbour_frac = find_dominant_contaminant(
        TARGET_PX[0], TARGET_PX[1], stars_x, stars_y, stars_flux, prf, CUT_SIZE, APERTURE_RADIUS)

    assert idx == 1, f"expected the only other star (index 1) as dominant contaminant, got {idx}"
    assert abs(neighbour_frac - (1.0 - target_frac)) < 1e-9, (
        f"with exactly one other star, its fraction ({neighbour_frac}) must equal "
        f"1 - target's own fraction ({1.0 - target_frac})")
    print(f"two stars: target={target_frac:.3f}, neighbour={neighbour_frac:.3f} "
          f"(sum={target_frac + neighbour_frac:.6f}) -- OK")


def test_three_stars_picks_close_bright_not_far_faint(prf):
    stars_x = np.array([TARGET_PX[0], TARGET_PX[0] + 3.2, TARGET_PX[0] + 20.0])
    stars_y = np.array([TARGET_PX[1], TARGET_PX[1] + 0.5, TARGET_PX[1]])
    stars_flux = np.array([12000.0, 18000.0, 50000.0])  # far star is much brighter but negligible in the aperture

    idx, frac = find_dominant_contaminant(
        TARGET_PX[0], TARGET_PX[1], stars_x, stars_y, stars_flux, prf, CUT_SIZE, APERTURE_RADIUS)
    assert idx == 1, f"expected the close neighbour (index 1), not the far bright star (index 2), got {idx}"
    assert frac > 0.1, f"expected a non-trivial contaminating fraction, got {frac}"
    print(f"three stars: dominant contaminant index={idx}, fraction={frac:.3f} -- OK")


def test_single_star_no_neighbour(prf):
    stars_x = np.array([TARGET_PX[0]])
    stars_y = np.array([TARGET_PX[1]])
    stars_flux = np.array([12000.0])

    idx, frac = find_dominant_contaminant(
        TARGET_PX[0], TARGET_PX[1], stars_x, stars_y, stars_flux, prf, CUT_SIZE, APERTURE_RADIUS)
    assert idx is None and frac == 0.0, f"expected (None, 0.0) for an isolated target, got ({idx}, {frac})"

    target_frac = calc_prf_contamination_fraction(
        TARGET_PX[0], TARGET_PX[1], stars_x, stars_y, stars_flux, prf, CUT_SIZE, APERTURE_RADIUS)
    assert abs(target_frac - 1.0) < 1e-9, f"an isolated target should get ~100% of its own aperture, got {target_frac}"
    print("single star: find_dominant_contaminant -> (None, 0.0), contamination fraction -> 1.0 -- OK")


def test_rank_contaminants(prf):
    stars_x = np.array([TARGET_PX[0], TARGET_PX[0] + 3.2, TARGET_PX[0] - 4.5, TARGET_PX[0] + 20.0])
    stars_y = np.array([TARGET_PX[1], TARGET_PX[1] + 0.5, TARGET_PX[1] + 1.0, TARGET_PX[1]])
    stars_flux = np.array([12000.0, 18000.0, 9000.0, 50000.0])  # [target, close bright, farther fainter, far bright]
    args = (TARGET_PX[0], TARGET_PX[1], stars_x, stars_y, stars_flux, prf, CUT_SIZE, APERTURE_RADIUS)

    ranked = rank_contaminants(*args, n=5, min_fraction=0.0)
    assert [i for i, _ in ranked][:2] == [1, 2], f"expected close bright then farther fainter first, got {ranked}"
    fracs = [f for _, f in ranked]
    assert fracs == sorted(fracs, reverse=True), f"fractions must be descending, got {fracs}"
    assert 0 not in [i for i, _ in ranked], "the target itself must never appear as its own contaminant"

    assert len(rank_contaminants(*args, n=1, min_fraction=0.0)) == 1
    cut = rank_contaminants(*args, n=5, min_fraction=fracs[1] * 1.01)
    assert [i for i, _ in cut] == [1], f"min_fraction must drop weaker stars, got {cut}"

    top_idx, top_frac = find_dominant_contaminant(*args)
    assert ranked[0][0] == top_idx and abs(ranked[0][1] - top_frac) < 1e-12
    assert rank_contaminants(TARGET_PX[0], TARGET_PX[1], stars_x[:1], stars_y[:1], stars_flux[:1],
                             prf, CUT_SIZE, APERTURE_RADIUS) == []
    print(f"rank_contaminants: order {[i for i, _ in ranked]}, fractions {[round(f, 3) for f in fracs]} -- OK")


if __name__ == "__main__":
    prf = load_real_prf()
    test_two_stars_neighbour_fraction_complements_target(prf)
    test_three_stars_picks_close_bright_not_far_faint(prf)
    test_single_star_no_neighbour(prf)
    test_rank_contaminants(prf)

    print("\nALL CHECKS PASSED")
