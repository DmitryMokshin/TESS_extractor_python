"""
Synthetic test for `isolated/prf_photometry.py`, per ROADMAP.md's Этап 3
acceptance criteria: real PRF from `prf/`, two Gaia-like stars in one
deblending window (a target and a bright, close neighbour -- like the real
SS 397 case), sinusoids of known amplitude injected on each (15 ppt on the
target, 10 ppt on the neighbour, per the roadmap's own wording), plus a
tilted-plane background and Gaussian noise.

Checks:
  - deblend_prf_flux recovers both amplitudes within +/-5%;
  - cross-talk (the OTHER star's frequency leaking into a star's own
    recovered flux) is < 1 ppt.

Plain script with asserts -- this project has no pytest dependency/test
harness yet (see ROADMAP.md Этап 10). Run directly:
    .venv/bin/python tests/test_prf_photometry_synthetic.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits as pyfits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isolated.prf_photometry import select_prf_model_stars, build_prf_star_cuts, deblend_prf_flux
from isolated.prewhitening import fit_all

RNG = np.random.default_rng(0)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# window/star geometry, chosen to resemble the real SS 397 case (~3.2 px separation)
CUT_SIZE = 21
BOX = 13
TARGET_PX = (11.0, 11.0)
NEIGHBOUR_PX = (14.0, 11.5)  # 3.04 px from the target

N_FRAMES = 2000
T_SPAN_DAYS = 20.0
TARGET_MEAN_FLUX = 20000.0
NEIGHBOUR_MEAN_FLUX = 15000.0
TARGET_FREQ, NEIGHBOUR_FREQ = 1.6, 2.3  # 1/d -- different so cross-talk is separable
TARGET_AMP_PPT, NEIGHBOUR_AMP_PPT = 15.0, 10.0
NOISE_SIGMA = 15.0  # e-/s per pixel per frame

AMPLITUDE_TOLERANCE = 0.05
CROSSTALK_LIMIT_PPT = 1.0


def load_real_prf():
    candidates = sorted(glob.glob(os.path.join(REPO_ROOT, "prf", "**", "*.fits"), recursive=True))
    if not candidates:
        raise FileNotFoundError("No cached PRF FITS files under prf/ -- can't run the synthetic test.")
    return pyfits.getdata(candidates[0]).astype(float)


def measure_amplitude_ppt(t, flux, freq):
    """Amplitude (ppt) of a sinusoid at a known, fixed frequency, via isolated.prewhitening.fit_all
    (anchored tightly on `freq` so it measures that frequency specifically, not whatever peak is
    strongest) -- reuses the package's own multi-sine fit instead of a one-off DFT."""
    y_ppt = (flux / np.median(flux) - 1.0) * 1e3
    _, amps, _, _ = fit_all(t, y_ppt, freqs=[freq], amps=[np.std(y_ppt)], phases=[0.0],
                            anchors=[freq], halfwidth=1e-6)
    return float(amps[0])


def main():
    prf = load_real_prf()

    gaia_stars = pd.DataFrame({
        "source_id": [1, 2],
        "px_x": [TARGET_PX[0], NEIGHBOUR_PX[0]],
        "px_y": [TARGET_PX[1], NEIGHBOUR_PX[1]],
        "phot_g_mean_mag": [12.0, 11.0],
        "bp_rp": [1.0, 1.5],
    })

    box_x0 = int(round(TARGET_PX[0])) - BOX // 2
    box_y0 = int(round(TARGET_PX[1])) - BOX // 2
    stars = select_prf_model_stars(gaia_stars, target_source_id=1, box_x0=box_x0, box_y0=box_y0,
                                    box_size=BOX, dt_max=5.0, merge_px=1.0)
    assert len(stars) == 2, f"expected both synthetic stars to be selected into the model, got {len(stars)}"
    target_idx = int(np.where(stars["source_id"] == 1)[0][0])
    neighbour_idx = int(np.where(stars["source_id"] == 2)[0][0])

    star_cuts = build_prf_star_cuts(prf, stars, CUT_SIZE, CUT_SIZE, box_x0, box_y0, BOX)

    t = np.sort(RNG.uniform(0, T_SPAN_DAYS, N_FRAMES))
    target_flux_true = TARGET_MEAN_FLUX * (1 + TARGET_AMP_PPT * 1e-3 * np.sin(2 * np.pi * TARGET_FREQ * t))
    neighbour_flux_true = NEIGHBOUR_MEAN_FLUX * (
        1 + NEIGHBOUR_AMP_PPT * 1e-3 * np.sin(2 * np.pi * NEIGHBOUR_FREQ * t))

    xx, yy = np.meshgrid(np.arange(BOX) - BOX / 2, np.arange(BOX) - BOX / 2, indexing="ij")
    background_plane = 50.0 + 0.5 * xx + 0.3 * yy

    cube_box = (target_flux_true[:, None, None] * star_cuts[target_idx] +
                neighbour_flux_true[:, None, None] * star_cuts[neighbour_idx] +
                background_plane[None, :, :] +
                RNG.normal(0.0, NOISE_SIGMA, size=(N_FRAMES, BOX, BOX)))

    fluxes, _background_fit = deblend_prf_flux(cube_box, star_cuts)
    target_flux_fit = fluxes[:, target_idx]
    neighbour_flux_fit = fluxes[:, neighbour_idx]

    target_amp = measure_amplitude_ppt(t, target_flux_fit, TARGET_FREQ)
    neighbour_amp = measure_amplitude_ppt(t, neighbour_flux_fit, NEIGHBOUR_FREQ)
    target_crosstalk = measure_amplitude_ppt(t, target_flux_fit, NEIGHBOUR_FREQ)
    neighbour_crosstalk = measure_amplitude_ppt(t, neighbour_flux_fit, TARGET_FREQ)

    print(f"target:    injected {TARGET_AMP_PPT:.2f} ppt @ {TARGET_FREQ} 1/d -> recovered {target_amp:.3f} ppt")
    print(f"neighbour: injected {NEIGHBOUR_AMP_PPT:.2f} ppt @ {NEIGHBOUR_FREQ} 1/d -> recovered {neighbour_amp:.3f} ppt")
    print(f"cross-talk: neighbour's freq in target's curve = {target_crosstalk:.3f} ppt, "
          f"target's freq in neighbour's curve = {neighbour_crosstalk:.3f} ppt")

    assert abs(target_amp - TARGET_AMP_PPT) / TARGET_AMP_PPT <= AMPLITUDE_TOLERANCE, (
        f"target amplitude recovery {target_amp:.3f} ppt vs injected {TARGET_AMP_PPT} ppt "
        f"exceeds +/-{AMPLITUDE_TOLERANCE:.0%}")
    assert abs(neighbour_amp - NEIGHBOUR_AMP_PPT) / NEIGHBOUR_AMP_PPT <= AMPLITUDE_TOLERANCE, (
        f"neighbour amplitude recovery {neighbour_amp:.3f} ppt vs injected {NEIGHBOUR_AMP_PPT} ppt "
        f"exceeds +/-{AMPLITUDE_TOLERANCE:.0%}")
    assert target_crosstalk < CROSSTALK_LIMIT_PPT, (
        f"neighbour's signal leaks {target_crosstalk:.3f} ppt into the target's curve "
        f"(limit {CROSSTALK_LIMIT_PPT} ppt)")
    assert neighbour_crosstalk < CROSSTALK_LIMIT_PPT, (
        f"target's signal leaks {neighbour_crosstalk:.3f} ppt into the neighbour's curve "
        f"(limit {CROSSTALK_LIMIT_PPT} ppt)")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
