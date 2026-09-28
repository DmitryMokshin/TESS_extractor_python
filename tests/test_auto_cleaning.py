"""
Test for `isolated.cleaning` (ROADMAP.md Этап 5: automatic, ordered
cleaning rules -- FLUX validity, QUALITY, STAR_BKG_RATIO, then local
point-to-point noise -- plus the combined log). Generic frame-selection
logic, not photometry, so a plain synthetic DataFrame is enough (same
approach as `test_frame_windows.py`); no real PRF/Gaia data needed.

Plain script with asserts -- matches the other tests in this directory
(no pytest dependency yet). Run directly:
    .venv/bin/python tests/test_auto_cleaning.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isolated.cleaning import quality_mask, auto_clean_light_curve, summarize_cleaning_log
from isolated.lightcurve_tools import windows_to_log


def test_quality_mask():
    quality = np.array([0, 128, 2, 0, 4096, 128 + 2])
    # default bitmask 175 (includes bits 2 and 128, not 4096)
    mask = quality_mask(quality, 175)
    assert mask.tolist() == [True, False, False, True, True, False], mask.tolist()
    # None -> strictly QUALITY == 0
    mask_strict = quality_mask(quality, None)
    assert mask_strict.tolist() == [True, False, False, True, False, False]
    # 0 -> disabled, keep everything
    mask_off = quality_mask(quality, 0)
    assert mask_off.all()
    print("quality_mask: bitmask / None (strict) / 0 (disabled) -- OK")


def _make_synthetic_lc():
    n = 1000
    step = 200.0 / 86400.0  # TESS cadence, days
    frame = np.arange(1, n + 1)
    btjd = frame * step

    quality = np.zeros(n, dtype=int)
    quality[[199, 200, 201]] = 128  # frames 200,201,202 (1-based) -- bad QUALITY (bit 128, in mask 175)

    star_bkg_ratio = np.full(n, 1.5)
    star_bkg_ratio[[399, 400]] = 0.05  # frames 400,401 -- low ratio
    # frame 200 is bad for BOTH reasons -- checks rule ordering (QUALITY wins, no double count)
    star_bkg_ratio[199] = 0.05

    # Gaussian jitter, not a perfectly periodic wiggle: an exact period-2
    # alternation degenerates the MAD-based sigma estimator (median cancels
    # it out on an odd-length window) -- real point-to-point noise doesn't
    # have that pathology, so a fixed-seed random draw is the honest test.
    rng = np.random.default_rng(12345)
    flux = 1000.0 + rng.normal(0, 0.3, n)
    burst = slice(699, 729)  # frames 700-729 (1-based) -- a 30-frame noisy stretch
    flux[burst] = 1000.0 + rng.normal(0, 20, burst.stop - burst.start)

    df = pd.DataFrame({"BTJD": btjd, "FRAME": frame, "QUALITY": quality,
                        "STAR_BKG_RATIO": star_bkg_ratio, "FLUX": flux})
    return df


def test_auto_clean_light_curve_default():
    df = _make_synthetic_lc()
    # local_noise_window=0.03d ~13 cadences: >10 (function's minimum for a
    # real estimate) and comfortably smaller than the 30-frame burst, so
    # windows centered well inside the burst are all-burst
    df_clean, log = auto_clean_light_curve(df, quality_bitmask=175, star_bkg_ratio_min=None,
                                            local_noise_kappa=2.5, local_noise_window=0.03)

    reasons = dict(zip(log["FRAME"], log["REASON"]))
    assert reasons[200] == "QUALITY" and reasons[201] == "QUALITY" and reasons[202] == "QUALITY"
    # star_bkg_ratio_min=None -> rule 2 disabled: frame 401 (only bad by ratio) must survive
    assert 401 not in reasons
    # the core of the noisy burst must be flagged as LOCAL_NOISE
    core = range(710, 720)
    for f in core:
        assert reasons.get(f) == "LOCAL_NOISE", f"frame {f}: expected LOCAL_NOISE, got {reasons.get(f)}"
    # a quiet frame far from QUALITY/burst issues must survive untouched
    assert 500 not in reasons
    assert len(df_clean) + len(log) == len(df)
    print(f"auto_clean_light_curve (star_bkg_ratio_min=None): dropped {len(log)}/{len(df)}, "
          f"reasons: {log['REASON'].value_counts().to_dict()} -- OK")


def test_auto_clean_light_curve_with_ratio_rule():
    df = _make_synthetic_lc()
    df_clean, log = auto_clean_light_curve(df, quality_bitmask=175, star_bkg_ratio_min=0.1,
                                            local_noise_kappa=2.5, local_noise_window=0.03)
    reasons = dict(zip(log["FRAME"], log["REASON"]))
    # frame 200: bad in BOTH QUALITY and STAR_BKG_RATIO -- rule order means QUALITY wins, no double count
    assert reasons[200] == "QUALITY"
    # frame 401: bad ONLY in STAR_BKG_RATIO
    assert reasons[401] == "STAR_BKG_RATIO"
    assert (log["FRAME"] == 401).sum() == 1  # not logged twice
    print("auto_clean_light_curve (star_bkg_ratio_min=0.1): rule ordering, no double-counting -- OK")


def test_auto_clean_light_curve_invalid_flux():
    # a flat run of FLUX=0.0 (the real SS 397 S80 defect this rule exists
    # for) has ZERO internal point-to-point scatter -- rule 3 alone would
    # wave it through as "quiet", not noisy. Confirm the new INVALID_FLUX
    # rule (run BEFORE rule 3) catches it instead.
    df = _make_synthetic_lc()
    flux = df["FLUX"].to_numpy().copy()
    dropout = slice(499, 519)  # frames 500-519 (1-based) -- a flat FLUX=0.0 stretch
    flux[dropout] = 0.0
    flux[42] = np.nan  # a lone non-finite point elsewhere, must also be caught
    df["FLUX"] = flux

    df_clean, log = auto_clean_light_curve(df, quality_bitmask=175, star_bkg_ratio_min=None,
                                            local_noise_kappa=2.5, local_noise_window=0.03)
    reasons = dict(zip(log["FRAME"], log["REASON"]))
    for f in range(500, 520):
        assert reasons.get(f) == "INVALID_FLUX", f"frame {f}: expected INVALID_FLUX, got {reasons.get(f)}"
    assert reasons.get(43) == "INVALID_FLUX"  # frame 43 (1-based) == index 42
    assert 0.0 not in df_clean["FLUX"].to_numpy()
    assert not np.isnan(df_clean["FLUX"].to_numpy()).any()
    print("auto_clean_light_curve: flat FLUX=0.0 dropout + lone NaN caught as INVALID_FLUX -- OK")


def test_auto_clean_light_curve_quality_disabled_no_column():
    # quality_bitmask=0 (disabled) must work on a DataFrame that has no QUALITY
    # column at all -- the pattern `PRF_cleaning.py` uses for a per-star curve
    # built from an already QUALITY-filtered cube (rule 1 applied upstream,
    # this call is only for rule 3)
    df = _make_synthetic_lc().drop(columns=["QUALITY", "STAR_BKG_RATIO"])
    df_clean, log = auto_clean_light_curve(df, quality_bitmask=0, local_noise_kappa=2.5,
                                            local_noise_window=0.03)
    assert (log["REASON"] == "QUALITY").sum() == 0
    assert (log["REASON"] == "LOCAL_NOISE").sum() > 0  # the noisy burst is still caught
    print("auto_clean_light_curve (quality_bitmask=0, no QUALITY column): no crash, rule 3 still runs -- OK")


def test_windows_to_log():
    df = pd.DataFrame({"BTJD": np.arange(10) * 0.1, "FRAME": np.arange(1, 11), "FLUX": np.arange(10, dtype=float)})
    log = windows_to_log(df, [[3, 5]], "manual:test")
    assert sorted(log["FRAME"].tolist()) == [3, 4, 5]
    assert (log["REASON"] == "manual:test").all()
    assert log.loc[log["FRAME"] == 4, "BTJD"].iloc[0] == df.loc[df["FRAME"] == 4, "BTJD"].iloc[0]
    print("windows_to_log: FRAME/BTJD/REASON match the excluded window -- OK")


def test_summarize_cleaning_log():
    log = pd.DataFrame({"FRAME": [10, 11, 20], "BTJD": [1.0, 1.1, 2.0], "REASON": ["QUALITY", "QUALITY", "LOCAL_NOISE"]})
    summary = summarize_cleaning_log(log, total_frames=100)
    assert "dropped: 3" in summary and "kept: 97" in summary
    assert "QUALITY: 2" in summary and "LOCAL_NOISE: 1" in summary
    # frames 10-11 are contiguous -> one interval; frame 20 is separate -> another
    assert summary.count("frames ") == 2, summary
    print("summarize_cleaning_log: counts and intervals -- OK")


if __name__ == "__main__":
    test_quality_mask()
    test_auto_clean_light_curve_default()
    test_auto_clean_light_curve_with_ratio_rule()
    test_auto_clean_light_curve_invalid_flux()
    test_auto_clean_light_curve_quality_disabled_no_column()
    test_windows_to_log()
    test_summarize_cleaning_log()
    print("\nALL CHECKS PASSED")
