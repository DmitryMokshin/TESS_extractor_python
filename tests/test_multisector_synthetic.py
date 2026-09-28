"""
Synthetic test for `isolated/multisector.py` (ROADMAP.md Этап 7: "Несколько
секторов"). No real multi-sector data exists yet for any star in this
project's cache (checked for SS 397: only sector 80 has been observed;
tess-point's geometric model finds a second pass only in sector 118, a
~2027 pointing -- in the future relative to today, nothing to download) --
so, as agreed, this stage is verified synthetically only.

Two synthetic "sectors": a ~25-day baseline each, separated by a ~365-day
gap (TESS's yearly revisit cadence), with a DIFFERENT zero-point offset and
a DIFFERENT slow polynomial trend in each -- exactly what
`stitch_light_curves` is supposed to remove before concatenating. Two close
injected frequencies (Δν = 0.02 1/d) simulate a Be-star-like group: closer
than 1/T of one 25-day sector (~0.04 1/d) but much wider than 1/T of the
full ~390-day combined baseline (~0.0026 1/d).

Checks:
  - stitching removes the offset/trend artifact: spurious low-frequency
    power (from the unremoved step between sectors) is much smaller after
    stitching than a naive concatenation.
  - resolution ~1/T (ROADMAP's literal criterion, "Be-star groups, Δν ~ 0.04,
    separate only on a long baseline"): one sector's 1/T is coarser than
    Δν (genuinely unresolved there); the stitched baseline's 1/T is far
    finer than Δν (clearly resolved) -- tested directly on the time arrays,
    not via automatic peak search, since a real two-block gap like this
    creates heavy spectral-window aliasing (exactly why ROADMAP separately
    insists the spectral window must be shown -- see
    `Peridogram_compute.py`'s amplitude-spectrum zoom panel).
  - the injected frequencies are recovered on the stitched series within
    the project's existing +/-5% amplitude tolerance, via a tightly
    anchored fit at the known frequencies (same technique
    `test_prf_photometry_synthetic.py` uses) rather than the automatic
    search, for the same aliasing reason.

Plain script with asserts -- matches the other tests in this directory
(no pytest dependency yet). Run directly:
    .venv/bin/python tests/test_multisector_synthetic.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isolated.multisector import stitch_light_curves, stitch_sector_csvs
from isolated.prewhitening import amp_spectrum, fit_all

RNG = np.random.default_rng(42)

T_SECTOR = 25.0          # days, per sector
GAP = 365.0              # days between sectors (yearly revisit)
N_PER_SECTOR = 1500
F1, F2 = 1.60, 1.62      # 1/d, Delta_nu = 0.02 -- unresolved in one sector, resolved combined
AMP1, AMP2 = 10.0, 8.0   # mmag
NOISE_SIGMA = 2.0        # mmag
OFFSET_A, OFFSET_B = 0.0, 30.0   # different zero-points between sectors
AMPLITUDE_TOLERANCE = 0.05


def _make_sector(t0, trend_coeffs, offset):
    t = np.sort(RNG.uniform(t0, t0 + T_SECTOR, N_PER_SECTOR))
    signal = AMP1 * np.sin(2 * np.pi * F1 * t) + AMP2 * np.sin(2 * np.pi * F2 * t)
    trend = np.polyval(trend_coeffs, t - t.mean())
    noise = RNG.normal(0, NOISE_SIGMA, t.size)
    y = offset + trend + signal + noise
    return t, y


def test_stitching_removes_offset_trend_artifact():
    t_a, y_a = _make_sector(0.0, [0.002, 0.05, 0.0], OFFSET_A)
    t_b, y_b = _make_sector(T_SECTOR + GAP, [-0.0015, -0.3, 0.0], OFFSET_B)

    t_naive = np.concatenate([t_a, t_b])
    y_naive = np.concatenate([y_a, y_b])
    order = np.argsort(t_naive)
    t_naive, y_naive = t_naive[order], y_naive[order] - np.median(y_naive)

    t_stitch, y_stitch, sector_tags = stitch_light_curves([t_a, t_b], [y_a, y_b], detrend_deg=2)
    assert t_stitch.size == t_a.size + t_b.size
    assert np.array_equal(sector_tags[t_stitch < T_SECTOR + GAP / 2], np.full(np.sum(t_stitch < T_SECTOR + GAP / 2), 1))

    low_freq = np.linspace(0.002, 0.05, 300)  # well below F1/F2 (~1.6), where the step artifact shows up
    A_naive = amp_spectrum(t_naive, y_naive, low_freq)
    A_stitch = amp_spectrum(t_stitch, y_stitch, low_freq)
    print(f"low-frequency spurious power: naive max={A_naive.max():.2f} mmag, stitched max={A_stitch.max():.2f} mmag")
    assert A_stitch.max() < 0.3 * A_naive.max(), (
        f"stitching should strongly suppress the low-frequency offset/trend artifact: "
        f"stitched={A_stitch.max():.2f} vs naive={A_naive.max():.2f} mmag")
    print("stitch_light_curves: removes the sector offset/trend artifact -- OK")
    return t_a, y_a, t_b, y_b, t_stitch, y_stitch


def test_amplitude_recovery_and_resolution(t_a, y_a, t_b, y_b, t_stitch, y_stitch):
    # ROADMAP's literal criterion is frequency RESOLUTION ~1/T -- a property of the
    # time baseline itself. Testing it directly (rather than via prewhiten's automatic
    # peak search) sidesteps the heavy spectral-window aliasing a real two-block gap
    # like this creates -- exactly why ROADMAP separately insists the spectral window
    # must be shown (see Peridogram_compute.py's amplitude-spectrum zoom panel).
    T_a = t_a.max() - t_a.min()
    T_stitch = t_stitch.max() - t_stitch.min()
    delta_nu = F2 - F1
    print(f"single sector: T={T_a:.2f} d, 1/T={1 / T_a:.4f} 1/d (Delta_nu={delta_nu})")
    print(f"stitched: T={T_stitch:.2f} d, 1/T={1 / T_stitch:.5f} 1/d (Delta_nu={delta_nu})")
    assert 1.0 / T_a > delta_nu, (
        f"test setup check: expected one sector's resolution (1/T={1 / T_a:.4f}) to be "
        f"coarser than Delta_nu ({delta_nu}) so the pair is genuinely unresolved there")
    assert 1.0 / T_stitch < delta_nu / 5, (
        f"expected the stitched baseline's resolution (1/T={1 / T_stitch:.5f}) to be far "
        f"finer than Delta_nu ({delta_nu}); got only {delta_nu * T_stitch:.1f}x margin")
    print("resolution ~1/T: unresolved on one sector, clearly resolved on the stitched baseline -- OK")

    # Amplitude recovery on the stitched series, anchored tightly at the known
    # frequencies (same technique as test_prf_photometry_synthetic.py's
    # measure_amplitude_ppt -- a fixed-frequency fit, not the automatic search,
    # so aliasing/sidelobes from the gap can't pull the fit away from F1/F2).
    y0 = y_stitch - np.median(y_stitch)
    fr, am, ph, res = fit_all(t_stitch, y0, freqs=[F1, F2], amps=[AMP1, AMP2], phases=[0.0, 0.0],
                              anchors=[F1, F2], halfwidth=1e-6)
    print(f"stitched, anchored fit: F1={F1} -> {am[0]:.2f} mmag (injected {AMP1}), "
          f"F2={F2} -> {am[1]:.2f} mmag (injected {AMP2})")
    assert abs(am[0] - AMP1) / AMP1 <= AMPLITUDE_TOLERANCE, (
        f"F1 amplitude {am[0]:.2f} vs injected {AMP1} exceeds +/-{AMPLITUDE_TOLERANCE:.0%}")
    assert abs(am[1] - AMP2) / AMP2 <= AMPLITUDE_TOLERANCE, (
        f"F2 amplitude {am[1]:.2f} vs injected {AMP2} exceeds +/-{AMPLITUDE_TOLERANCE:.0%}")
    print("amplitude recovery on stitched series (anchored fit at F1, F2) -- OK")


def test_stitch_sector_csvs_smoke(tmp_dir):
    # minimal wiring check: two tiny CSVs (pipeline column convention) -> same
    # stitching as stitch_light_curves, via the file-reading entry point
    t_a = np.linspace(0.0, 10.0, 200)
    t_b = np.linspace(400.0, 410.0, 200)
    mag_a = 12.0 + 0.01 * np.sin(2 * np.pi * 1.0 * t_a)
    mag_b = 12.3 + 0.01 * np.sin(2 * np.pi * 1.0 * t_b)  # different zero-point (12.3 vs 12.0)

    path_a = os.path.join(tmp_dir, "light_curve_sector_1_clean.csv")
    path_b = os.path.join(tmp_dir, "light_curve_sector_2_clean.csv")
    pd.DataFrame({"BTJD": t_a, "FRAME": np.arange(1, t_a.size + 1), "MAG": mag_a}).to_csv(path_a, index=False)
    pd.DataFrame({"BTJD": t_b, "FRAME": np.arange(1, t_b.size + 1), "MAG": mag_b}).to_csv(path_b, index=False)

    t, y, sector_tags = stitch_sector_csvs([path_a, path_b], sector_labels=[1, 2], value_col="MAG", detrend_deg=2)
    # value_col="MAG" runs the same cleaned_jds_mags sigma-clip Peridogram_compute.py
    # already applies to a single file, so a handful of points may legitimately be
    # dropped -- not an exact count match
    assert t.size > 0.9 * (t_a.size + t_b.size), f"unexpectedly many points dropped: {t.size}"
    assert set(np.unique(sector_tags).tolist()) == {1, 2}
    assert np.all(np.diff(t) >= 0)
    print("stitch_sector_csvs: reads + stitches per-sector CSVs -- OK")


if __name__ == "__main__":
    import tempfile

    t_a, y_a, t_b, y_b, t_stitch, y_stitch = test_stitching_removes_offset_trend_artifact()
    test_amplitude_recovery_and_resolution(t_a, y_a, t_b, y_b, t_stitch, y_stitch)
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_stitch_sector_csvs_smoke(tmp_dir)

    print("\nALL CHECKS PASSED")
