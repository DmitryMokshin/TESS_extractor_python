"""
Synthetic test for the pure helpers of `Comparison_check.py` (environment
check of neighbours/standards): injected sinusoids of known amplitude plus
trend and noise, no cutouts or network.

Checks:
  - `fixed_freq_fit` recovers injected amplitudes on fixed frequencies within
    +/-5 % and gives high S/N there but not at an unrelated frequency;
  - `auto_dyn_band`: band hugs the target's frequencies inside `frange`
    (+/- margin, clipped), falls back to the whole `frange` when none lie in it;
  - `variability_stats`: RMS grows with the injected signal; the binned RMS
    sees a slow signal but suppresses white noise;
  - `dyn_summary`: tracks the injected frequency inside the band and reports
    NaN (no crash) when the band is outside the computed range.

Plain script with asserts. Run directly:
    .venv/bin/python tests/test_comparison_check_synthetic.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")

import Comparison_check as CC

RNG = np.random.default_rng(3)
T_SPAN = 26.0
N = 5000
t = np.sort(RNG.uniform(0.0, T_SPAN, N)) + 3480.0
FREQS = [1.52, 1.74, 3.30]
AMPS = [10.0, 5.0, 3.0]    # ppt
NOISE = 2.0                # ppt


def make_series(freqs, amps, noise=NOISE, trend=True):
    y = noise * RNG.normal(size=t.size)
    for f, a in zip(freqs, amps):
        y += a * np.sin(2 * np.pi * f * t + RNG.uniform(0, 6.28))
    if trend:
        x = t - t.mean()
        y += 0.3 * x + 0.01 * x ** 2
    return y


def test_fixed_freq_fit():
    y = make_series(FREQS, AMPS)
    amps, sa, snr, noise, d = CC.fixed_freq_fit(t, y, FREQS + [2.4])   # 2.4: nothing injected there
    for f, a_true, a in zip(FREQS, AMPS, amps[:3]):
        assert abs(a - a_true) / a_true < 0.05, f"{f}: recovered {a:.2f} vs injected {a_true} ppt"
    assert all(snr[:3] > CC.SNR_STOP), f"injected frequencies must be significant, got S/N {snr[:3]}"
    assert snr[3] < CC.SNR_STOP, f"no signal at 2.4 -- must not be significant, got S/N {snr[3]:.1f}"
    print(f"fixed_freq_fit: amplitudes {np.round(amps[:3], 2)} vs {AMPS}, S/N {np.round(snr, 1)} -- OK")


def test_auto_dyn_band():
    assert CC.auto_dyn_band([0.1, 1.52, 1.74, 3.3], (0.8, 2.5), 0.15) == (1.37, 1.89)
    lo, hi = CC.auto_dyn_band([0.85, 2.45], (0.8, 2.5), 0.15)
    assert (lo, hi) == (0.8, 2.5), "band must be clipped to frange"
    assert CC.auto_dyn_band([0.1, 3.3], (0.8, 2.5), 0.15) == (0.8, 2.5), "no freq inside -> whole frange"
    assert CC.auto_dyn_band([], (0.8, 2.5), 0.15) == (0.8, 2.5)
    print("auto_dyn_band: hugs target frequencies, clipped to frange, falls back to frange -- OK")


def test_variability_stats():
    quiet = make_series([], [], noise=NOISE)
    slow = make_series([0.8], [8.0], noise=NOISE)
    rms_q, rb_q = CC.variability_stats(t, quiet)
    rms_s, rb_s = CC.variability_stats(t, slow)
    assert abs(rms_q - NOISE) / NOISE < 0.1, f"white-noise RMS {rms_q:.2f} vs {NOISE}"
    assert rb_q < 0.5 * rms_q, f"binning must suppress white noise: RMS {rms_q:.2f} -> binned {rb_q:.2f}"
    assert rms_s > 2 * rms_q and rb_s > 3 * rb_q, "slow signal must raise both RMS and binned RMS"
    assert rb_s > 0.8 * 8.0 / np.sqrt(2) * 0.8, f"binned RMS {rb_s:.2f} must keep a slow 8 ppt signal"
    print(f"variability_stats: quiet {rms_q:.2f}/{rb_q:.2f}, slow signal {rms_s:.2f}/{rb_s:.2f} ppt -- OK")


def test_dyn_summary():
    y = make_series([1.60], [12.0], noise=1.5, trend=False)
    ds = CC.dyn_summary(t, y, band=(1.4, 1.8))
    assert ds["n_win"] > 10
    assert abs(ds["f_med"] - 1.60) < 0.03, f"tracked frequency {ds['f_med']:.3f} vs injected 1.60"
    assert abs(ds["a_med"] - 12.0) / 12.0 < 0.15, f"tracked amplitude {ds['a_med']:.2f} vs injected 12"
    off = CC.dyn_summary(t, y, band=(5.0, 6.0))
    assert np.isnan(off["f_med"]), "band outside the computed range must give NaN, not crash"
    print(f"dyn_summary: tracked {ds['f_med']:.3f} 1/d, {ds['a_med']:.1f} ppt (injected 1.60, 12) -- OK")


if __name__ == "__main__":
    test_fixed_freq_fit()
    test_auto_dyn_band()
    test_variability_stats()
    test_dyn_summary()
    print("\nALL CHECKS PASSED")
