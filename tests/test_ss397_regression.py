"""
Regression test against ROADMAP.md's SS 397 sector 80 benchmark (ROADMAP.md
Этап 10: "регрессию на эталон SS 397 (для этого теста нужны реальные
данные)"). The ONE test in this project that needs real, locally-cached
data instead of synthetic input -- `stars_python/` is gitignored (see
CLAUDE.md: per-star cache, large, regenerated on demand, never committed),
so a fresh clone won't have it. If the needed cache files aren't present,
each check below SKIPS with a clear message (prints and returns) instead of
failing or reaching out to the network. Regenerate the cache by running
`TESS_cleaning.py`/`PRF_cleaning.py` for SS 397 sector 80 first.

Does NOT include the amplitude-map benchmark item ("g1 sits on SS 397,
share >= 0.8 vs the neighbour") -- that needs a full PRF deblending of
target + neighbour across the whole ~11000-frame sector (~10 minutes with
ROADMAP.md Этап 8's bootstrap/CLEAN diagnostics also enabled), far too slow
for a routine test; it was verified by hand during Этап 6 instead.

Run directly:
    .venv/bin/python tests/test_ss397_regression.py
"""
import matplotlib
matplotlib.use("Agg")

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAR_NAME = "SS 397"
STAR_NOSPACE = "SS_397"
SECTOR = 80
CUT = 50
STAR_DIR = f"stars_python/{STAR_NOSPACE}/{CUT}x{CUT}"
LC_PRF_CLEAN = f"{STAR_DIR}/light_curve_sector_{SECTOR}_prf_clean.csv"
GAIA_STARS_FILE = f"{STAR_DIR}/gaia_stars_in_view_sector_{SECTOR}.csv"

# ROADMAP.md benchmark (SS 397, сектор 80, вырезка 50x50); tolerances as stated there
FREQ_TOL = 0.005       # 1/d
AMP_TOL_REL = 0.10     # +/-10%
G1_FREQS = [1.5188, 1.5836, 1.6243, 1.6675, 1.7380]
G1_AMPS_MMAG = [7.45, 13.54, 8.29, 5.21, 5.50]
G2_FREQS = [3.259, 3.301]
G3_FREQS = [4.888]
CONTAMINATION_EXPECTED = 0.43
CONTAMINATION_TOL = 0.05  # absolute -- ROADMAP gives "≈0.43", no explicit tolerance for this one


def _closest(freqs, target):
    arr = np.asarray(freqs)
    idx = int(np.argmin(np.abs(arr - target)))
    return idx, float(arr[idx])


def test_frequencies_and_amplitudes():
    if not os.path.isfile(LC_PRF_CLEAN):
        print(f"[SKIP] {LC_PRF_CLEAN} not found locally -- run PRF_cleaning.py for SS 397 sector {SECTOR} first")
        return

    from Peridogram_compute import periodogram_compute_analise
    peaks = periodogram_compute_analise(STAR_NOSPACE, SECTOR, CUT, CUT, lc_suffix="_prf_clean", make_ls_plot=False)
    freqs = np.array([p["frequency"] for p in peaks])
    amps = np.array([p["amplitude"] for p in peaks])

    for f_expected, a_expected in zip(G1_FREQS, G1_AMPS_MMAG):
        idx, f_found = _closest(freqs, f_expected)
        assert abs(f_found - f_expected) < FREQ_TOL, (
            f"g1 frequency {f_expected} not recovered within +/-{FREQ_TOL} 1/d (closest found: {f_found:.4f})")
        a_found = amps[idx]
        rel_diff = abs(a_found - a_expected) / a_expected
        assert rel_diff < AMP_TOL_REL, (
            f"g1 @ {f_expected}: amplitude {a_found:.2f} vs benchmark {a_expected} mmag "
            f"exceeds +/-{AMP_TOL_REL:.0%} (got {rel_diff:.1%})")
        print(f"g1 {f_expected:.4f} 1/d: found {f_found:.4f}, amplitude {a_found:.2f} vs {a_expected} mmag -- OK")

    for f_expected in G2_FREQS + G3_FREQS:
        _idx, f_found = _closest(freqs, f_expected)
        assert abs(f_found - f_expected) < FREQ_TOL, (
            f"frequency {f_expected} not recovered within +/-{FREQ_TOL} 1/d (closest found: {f_found:.4f})")
        print(f"{f_expected:.4f} 1/d: found {f_found:.4f} -- OK")

    print("SS 397 S80: g1/g2/g3 frequencies and g1 amplitudes match the ROADMAP.md benchmark -- OK")


def test_contamination_fraction():
    if not (os.path.isfile(GAIA_STARS_FILE) and os.path.isdir("prf")):
        print(f"[SKIP] {GAIA_STARS_FILE} or prf/ not found locally -- run PRF_cleaning.py for "
              f"SS 397 sector {SECTOR} first")
        return

    from isolated.data_io import load_tess_cutouts, load_star_gaia_data, find_target_row, _read_gaia_csv
    from isolated.psf import get_tesscut_prf_supersampled
    from isolated.photometry import calc_prf_contamination_fraction
    from isolated.geometry import calc_tess_flux_from_mag

    cut_fits = load_tess_cutouts(STAR_NAME, CUT, CUT, sector=SECTOR)[SECTOR]
    prf = get_tesscut_prf_supersampled(cut_fits)
    gaia_stars = _read_gaia_csv(GAIA_STARS_FILE)
    gaia_data = load_star_gaia_data(STAR_NAME)
    match = find_target_row(gaia_stars, gaia_data, SECTOR)
    star_px_x, star_px_y = float(match["px_x"]), float(match["px_y"])

    stars_x = gaia_stars["px_x"].to_numpy()
    stars_y = gaia_stars["px_y"].to_numpy()
    stars_flux = calc_tess_flux_from_mag(gaia_stars["phot_rp_mean_mag"].to_numpy())

    contamination = calc_prf_contamination_fraction(star_px_x, star_px_y, stars_x, stars_y, stars_flux,
                                                     prf, CUT, aperture_radius=3)
    print(f"SS 397 S80: contamination fraction at r=3px = {contamination:.4f} "
          f"(benchmark ~{CONTAMINATION_EXPECTED})")
    assert abs(contamination - CONTAMINATION_EXPECTED) < CONTAMINATION_TOL, (
        f"contamination fraction {contamination:.4f} vs benchmark {CONTAMINATION_EXPECTED} "
        f"exceeds tolerance {CONTAMINATION_TOL}")
    print("SS 397 S80: contamination fraction matches the ROADMAP.md benchmark -- OK")


if __name__ == "__main__":
    test_frequencies_and_amplitudes()
    test_contamination_fraction()

    print("\nALL CHECKS PASSED (or cleanly skipped -- see [SKIP] lines above, if any)")
