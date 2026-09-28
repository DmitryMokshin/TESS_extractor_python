"""
Synthetic test for `isolated/localize.py` (ROADMAP.md Этап 6: "Проверка
происхождения сигнала"). Same harness as `test_prf_photometry_synthetic.py`:
real PRF from `prf/`, a small synthetic Gaia field, injected sinusoids of
known amplitude, Gaussian noise.

Checks:
  - `select_comparison_stars`: only stars within `mag_tol` of the target are
    picked, closest-in-magnitude first, capped at `n`.
  - `deblend_star_curve` + `signal_origin_table` end-to-end on three stars in
    one cutout: a target, a "contaminated" comparison star sharing the
    target's exact frequency (simulating a field-wide systematic) and a
    "clean" comparison star with no signal at that frequency -- the
    contaminated star must be flagged PRESENT with amplitude recovered
    within the project's existing +/-5% tolerance, the clean one must not.

Plain script with asserts -- matches the other tests in this directory
(no pytest dependency yet). Run directly:
    .venv/bin/python tests/test_localize_synthetic.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits as pyfits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isolated.localize import select_comparison_stars, deblend_star_curve, signal_origin_table


def test_deblend_star_curve_merged_away():
    # id=10 sits within merge_px of a much brighter id=11 -- select_prf_model_stars'
    # own degeneracy rule folds id=10 into id=11 instead of fitting it separately,
    # so deblend_star_curve must raise a clear error, not IndexError/garbage
    gaia = pd.DataFrame({
        "source_id": [10, 11],
        "px_x": [12.0, 12.3],
        "px_y": [12.0, 12.2],
        "phot_g_mean_mag": [13.0, 11.0],
        "bp_rp": [1.0, 1.0],
    })
    try:
        deblend_star_curve(None, None, gaia, cut_width=25, cut_height=25, source_id=10, box=13, merge_px=1.0)
        assert False, "expected a ValueError for a star merged into a brighter neighbour"
    except ValueError as exc:
        assert "merge" in str(exc).lower(), f"expected a merge-related message, got: {exc}"
    print("deblend_star_curve: a star merged into a brighter neighbour raises a clear ValueError -- OK")

RNG = np.random.default_rng(1)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CUT_SIZE = 45
BOX = 13
TARGET_PX = (10.0, 10.0)
CONTAMINATED_PX = (35.0, 35.0)   # far from the target -- own, non-overlapping PRF window
CLEAN_PX = (10.0, 35.0)          # far from both

N_FRAMES = 2500
T_SPAN_DAYS = 20.0
MEAN_FLUX = 20000.0
SHARED_FREQ = 1.7        # 1/d -- the "systematic" the target and the contaminated star share
CLEAN_STAR_FREQ = 3.4    # unrelated, only on the clean star -- must NOT show up as the target's own freq
AMP_PPT = 15.0
NOISE_SIGMA = 12.0        # e-/s per pixel per frame

AMPLITUDE_TOLERANCE = 0.05
SNR_THRESHOLD = 4.0


def load_real_prf():
    candidates = sorted(glob.glob(os.path.join(REPO_ROOT, "prf", "**", "*.fits"), recursive=True))
    if not candidates:
        raise FileNotFoundError("No cached PRF FITS files under prf/ -- can't run the synthetic test.")
    return pyfits.getdata(candidates[0]).astype(float)


def test_select_comparison_stars():
    gaia = pd.DataFrame({
        "source_id": [1, 2, 3, 4, 5, 6],
        "px_x": [10.0, 11.0, 12.0, 13.0, 14.0, 12.0],
        "px_y": [10.0, 11.0, 12.0, 13.0, 14.0, 2.0],  # id=6: near-comparable mag, but too close to the edge
        # target (id=1) T~12.0 (g=12, bp_rp=1 -> matches select_prf_model_stars' own formula indirectly);
        # id=2 close in mag, id=3 within tol, id=4 just outside tol, id=5 far outside, id=6 within tol
        "phot_g_mean_mag": [12.0, 12.3, 12.9, 13.3, 15.0, 12.4],
        "bp_rp": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })
    cut_width = cut_height = 25
    box = 13  # box//2 == 6 -> a star needs px_y in [7, cut_height-6] == [7, 19]; id=6's px_y=2.0 fails that

    # bp_rp is identical for every row, so T-mag differences equal the G-mag
    # differences exactly: target(id=1) is the reference; id=2 is 0.3 away
    # (in), id=3 is 0.9 away (in), id=4 is 1.3 away (out), id=5 is 3.0 away (out),
    # id=6 is 0.4 away (in on magnitude, but excluded for being too close to the edge)
    picked = select_comparison_stars(gaia, target_source_id=1, cut_width=cut_width, cut_height=cut_height,
                                      n=5, mag_tol=1.0, box=box)
    assert picked["source_id"].tolist() == [2, 3], (
        f"expected [2, 3] (within mag_tol and box bounds, closest-first), got {picked['source_id'].tolist()}")

    capped = select_comparison_stars(gaia, target_source_id=1, cut_width=cut_width, cut_height=cut_height,
                                      n=1, mag_tol=5.0, box=box)
    assert len(capped) == 1
    print("select_comparison_stars: mag_tol filter, edge-of-cutout exclusion, closest-first ordering, n cap -- OK")


def main():
    test_select_comparison_stars()
    test_deblend_star_curve_merged_away()

    prf = load_real_prf()

    gaia_stars = pd.DataFrame({
        "source_id": [1, 2, 3],
        "px_x": [TARGET_PX[0], CONTAMINATED_PX[0], CLEAN_PX[0]],
        "px_y": [TARGET_PX[1], CONTAMINATED_PX[1], CLEAN_PX[1]],
        "phot_g_mean_mag": [12.0, 12.2, 12.1],
        "bp_rp": [1.0, 1.0, 1.0],
    })

    t = np.sort(RNG.uniform(0, T_SPAN_DAYS, N_FRAMES))

    def make_cube_box(center_px, freq, amp_ppt):
        box_x0 = int(round(center_px[0])) - BOX // 2
        box_y0 = int(round(center_px[1])) - BOX // 2
        star_row = gaia_stars[(gaia_stars.px_x == center_px[0]) & (gaia_stars.px_y == center_px[1])]
        sid = int(star_row["source_id"].iloc[0])
        from isolated.prf_photometry import select_prf_model_stars, build_prf_star_cuts
        stars = select_prf_model_stars(gaia_stars, sid, box_x0, box_y0, BOX, dt_max=5.0, merge_px=1.0)
        assert len(stars) == 1, "stars are far apart -- each window should contain only its own star"
        star_cuts = build_prf_star_cuts(prf, stars, CUT_SIZE, CUT_SIZE, box_x0, box_y0, BOX)
        flux_true = MEAN_FLUX * (1 + amp_ppt * 1e-3 * np.sin(2 * np.pi * freq * t))
        cube_box = (flux_true[:, None, None] * star_cuts[0] +
                    RNG.normal(0.0, NOISE_SIGMA, size=(N_FRAMES, BOX, BOX)))
        x0, y0 = box_x0 - 1, box_y0 - 1
        return cube_box, x0, y0

    # build the full (N_FRAMES, CUT_SIZE, CUT_SIZE) cube by placing each star's own
    # noisy PRF cutout at its own window (windows are far apart -> no overlap)
    cube = RNG.normal(0.0, NOISE_SIGMA, size=(N_FRAMES, CUT_SIZE, CUT_SIZE))
    for center_px, freq, amp in [(TARGET_PX, SHARED_FREQ, AMP_PPT),
                                  (CONTAMINATED_PX, SHARED_FREQ, AMP_PPT),
                                  (CLEAN_PX, CLEAN_STAR_FREQ, AMP_PPT)]:
        cube_box, x0, y0 = make_cube_box(center_px, freq, amp)
        cube[:, x0:x0 + BOX, y0:y0 + BOX] = cube_box

    target_flux, _s, _m, _b = deblend_star_curve(cube, prf, gaia_stars, CUT_SIZE, CUT_SIZE, source_id=1, box=BOX)
    contaminated_flux, _s, _m, _b = deblend_star_curve(cube, prf, gaia_stars, CUT_SIZE, CUT_SIZE, source_id=2, box=BOX)
    clean_flux, _s, _m, _b = deblend_star_curve(cube, prf, gaia_stars, CUT_SIZE, CUT_SIZE, source_id=3, box=BOX)

    def ppt(flux):
        return (flux / np.median(flux) - 1.0) * 1e3

    star_curves = [("target", t, ppt(target_flux)), ("contaminated", t, ppt(contaminated_flux)),
                   ("clean", t, ppt(clean_flux))]
    table = signal_origin_table(star_curves, target_freqs=[SHARED_FREQ], fmax=10.0, snr_threshold=SNR_THRESHOLD)
    print(table.to_string())

    target_row = table[table.LABEL == "target"].iloc[0]
    contaminated_row = table[table.LABEL == "contaminated"].iloc[0]
    clean_row = table[table.LABEL == "clean"].iloc[0]

    assert target_row["PRESENT"], f"target must show its own injected frequency, got {target_row.to_dict()}"
    assert abs(target_row["AMPLITUDE_PPT"] - AMP_PPT) / AMP_PPT <= AMPLITUDE_TOLERANCE, (
        f"target amplitude recovery {target_row['AMPLITUDE_PPT']:.3f} ppt vs injected {AMP_PPT} ppt "
        f"exceeds +/-{AMPLITUDE_TOLERANCE:.0%}")

    assert contaminated_row["PRESENT"], (
        f"contaminated comparison star (shares the target's frequency) must be flagged PRESENT, "
        f"got {contaminated_row.to_dict()}")
    assert abs(contaminated_row["AMPLITUDE_PPT"] - AMP_PPT) / AMP_PPT <= AMPLITUDE_TOLERANCE, (
        f"contaminated star amplitude recovery {contaminated_row['AMPLITUDE_PPT']:.3f} ppt vs injected "
        f"{AMP_PPT} ppt exceeds +/-{AMPLITUDE_TOLERANCE:.0%}")

    assert not clean_row["PRESENT"], (
        f"clean comparison star (no signal at the target's frequency) must NOT be flagged PRESENT, "
        f"got {clean_row.to_dict()}")

    print(f"target: recovered {target_row['AMPLITUDE_PPT']:.3f} ppt @ {SHARED_FREQ} 1/d "
          f"(S/N_local={target_row['SNR_LOCAL']:.1f})")
    print(f"contaminated comparison star: recovered {contaminated_row['AMPLITUDE_PPT']:.3f} ppt "
          f"(S/N_local={contaminated_row['SNR_LOCAL']:.1f}, PRESENT={contaminated_row['PRESENT']})")
    print(f"clean comparison star: recovered {clean_row['AMPLITUDE_PPT']:.3f} ppt "
          f"(S/N_local={clean_row['SNR_LOCAL']:.1f}, PRESENT={clean_row['PRESENT']})")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
