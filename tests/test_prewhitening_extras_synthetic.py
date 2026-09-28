"""
Synthetic test for `isolated/prewhitening.py`'s ROADMAP.md Этап 8 additions
("частотный анализ, дальше"): bootstrap/split-half frequency errors, sliding-
window amplitude/phase stability, CLEAN, and error-tolerant combination
frequencies. Generic time-series processing, not photometry -- a plain
synthetic (t, y) series is enough (same approach as `test_frame_windows.py`/
`test_auto_cleaning.py`), no real PRF/Gaia data needed.

Plain script with asserts -- matches the other tests in this directory
(no pytest dependency yet). Run directly:
    .venv/bin/python tests/test_prewhitening_extras_synthetic.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isolated.prewhitening import (prewhiten, bootstrap_frequency_errors, split_half_frequency_errors,
                                   frequency_stability, clean_periodogram, find_combination_frequencies)

RNG = np.random.default_rng(11)

T_SPAN = 25.0
N_POINTS = 2000
FREQ = 1.6          # 1/d
AMP = 12.0           # mmag
NOISE_SIGMA = 2.0    # mmag
AMPLITUDE_TOLERANCE = 0.05


def _make_series():
    t = np.sort(RNG.uniform(0, T_SPAN, N_POINTS))
    y = AMP * np.sin(2 * np.pi * FREQ * t) + RNG.normal(0, NOISE_SIGMA, t.size)
    return t, y


def test_bootstrap_and_split_half_errors():
    t, y = _make_series()
    peaks, extra = prewhiten(t, y, fmax=3.0, nmax=5, snr_stop=4.0)
    assert len(peaks) == 1, f"expected exactly one significant peak, got {len(peaks)}: {peaks}"
    formula_sigma_f = peaks[0]["frequency_err"]

    boot = bootstrap_frequency_errors(t, y, peaks, extra, n_boot=150, rng=np.random.default_rng(1))
    boot_sigma_f = boot["frequency"][0]
    print(f"sigma_f: formula={formula_sigma_f:.6f}, bootstrap={boot_sigma_f:.6f}")
    # two independent estimates of the same thing -- not an exact match, but
    # should agree to within an order of magnitude (neither wildly small nor huge)
    assert 0.1 * formula_sigma_f < boot_sigma_f < 10 * formula_sigma_f, (
        f"bootstrap sigma_f ({boot_sigma_f:.6f}) too far from the formula's "
        f"({formula_sigma_f:.6f}) -- more than an order of magnitude off")

    first, second = split_half_frequency_errors(t, y, peaks)
    half_diff = abs(first["frequency"][0] - second["frequency"][0])
    print(f"split-half frequency difference: {half_diff:.6f} (formula sigma_f={formula_sigma_f:.6f})")
    assert half_diff < 10 * formula_sigma_f, (
        f"split-half difference ({half_diff:.6f}) implausibly large vs formula sigma_f "
        f"({formula_sigma_f:.6f})")
    print("bootstrap_frequency_errors / split_half_frequency_errors: same order of magnitude as "
          "errors_mo99's formula -- OK")


def test_frequency_stability_detects_amplitude_ramp():
    t = np.sort(RNG.uniform(0, T_SPAN, N_POINTS))
    amp_t = 5.0 + 1.0 * t  # amplitude ramps from 5 to ~30 mmag over the run
    y = amp_t * np.sin(2 * np.pi * FREQ * t) + RNG.normal(0, NOISE_SIGMA, t.size)

    centers, amplitudes, phases = frequency_stability(t, y, [FREQ], window=6.0, step=1.0)
    valid = np.isfinite(amplitudes[:, 0])
    assert valid.sum() > 5, "too few valid stability windows to test a trend"
    c, a = centers[valid], amplitudes[valid, 0]
    slope = np.polyfit(c, a, 1)[0]
    print(f"frequency_stability: fitted amplitude-vs-time slope = {slope:.3f} mmag/d (injected ~1.0 mmag/d)")
    assert slope > 0.5, f"expected a clearly rising amplitude trend (slope > 0.5 mmag/d), got {slope:.3f}"
    assert a[0] < a[-1], "expected the tracked amplitude to increase from start to end of the run"
    print("frequency_stability: detects an injected amplitude ramp -- OK")


def test_clean_periodogram_recovers_injected_peak():
    t, y = _make_series()
    freqs = np.arange(0.1, 3.0, 1.0 / (3 * T_SPAN))
    components, residual, dirty = clean_periodogram(t, y, freqs, gain=0.2, n_iter=100, stop_fraction=0.1)
    assert components, "CLEAN found no components at all"

    near_freq = [c for c in components if abs(c["frequency"] - FREQ) < 1.0 / T_SPAN]
    assert near_freq, f"no CLEAN component near the injected frequency {FREQ}, got {components[:5]}"
    total_amp_near = sum(c["amplitude"] for c in near_freq)
    print(f"CLEAN: {len(components)} components, {len(near_freq)} near F={FREQ}, "
          f"summed amplitude={total_amp_near:.2f} mmag (injected {AMP})")
    assert abs(total_amp_near - AMP) / AMP <= AMPLITUDE_TOLERANCE, (
        f"CLEAN recovered amplitude {total_amp_near:.2f} vs injected {AMP} exceeds "
        f"+/-{AMPLITUDE_TOLERANCE:.0%}")
    print("clean_periodogram: recovers the injected frequency/amplitude -- OK")


def test_find_combination_frequencies_uses_error_tolerance():
    # three "parent-ish" peaks plus one candidate "child" that sits within the
    # ERROR-based tolerance of parent1+parent2, and a second candidate that the
    # OLD flat 0.5/T tolerance would have flagged but the error-based one must not
    T = 25.0
    flat_tol = 0.5 / T  # the tolerance Peridogram_diagnostics.py used before this stage
    peaks = [
        {"frequency": 1.20, "frequency_err": 0.0005, "amplitude": 20.0},  # parent1
        {"frequency": 0.35, "frequency_err": 0.0005, "amplitude": 15.0},  # parent2
        # true combination: 1.20 + 0.35 = 1.55, well within a tiny error tolerance
        {"frequency": 1.55, "frequency_err": 0.0005, "amplitude": 5.0},   # real child
        # false positive under the OLD flat 0.5/T tolerance (0.5/25=0.02): sits
        # ~0.015 off the true sum, inside 0.5/T but WAY outside the tiny formal errors
        {"frequency": 1.20 + 0.35 + 0.015, "frequency_err": 0.0005, "amplitude": 4.0},
    ]
    assert abs((1.20 + 0.35 + 0.015) - 1.55) < flat_tol, "test setup check: false positive must be inside 0.5/T"

    found = find_combination_frequencies(peaks, n_top=2, n_sigma=3.0)
    children = {c["child"] for c in found}
    print(f"find_combination_frequencies: flagged children {children} (real=2, false-positive-under-old-rule=3)")
    assert 2 in children, "expected the real combination (index 2) to be found"
    assert 3 not in children, (
        "the error-based tolerance should reject the false positive the old flat 0.5/T "
        "tolerance would have accepted -- got it flagged anyway")
    print("find_combination_frequencies: error-based tolerance -- finds the real combination, "
          "rejects the old rule's false positive -- OK")


if __name__ == "__main__":
    test_bootstrap_and_split_half_errors()
    test_frequency_stability_detects_amplitude_ramp()
    test_clean_periodogram_recovers_injected_peak()
    test_find_combination_frequencies_uses_error_tolerance()
    print("\nALL CHECKS PASSED")
