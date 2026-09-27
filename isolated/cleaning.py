"""
Automatic, ordered light-curve cleaning rules, with a full log of why each
frame was dropped. Not a Julia port. See ROADMAP.md, "Этап 5. Автоматическая
чистка с логом".
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .config import STAR_DIRECTORY
from .geometry import get_nospace_star_name
from .lightcurve_tools import local_point_to_point_sigma


def quality_mask(quality, quality_bitmask=175):
    """
    Which frames pass the TESS `QUALITY` flag check. `quality_bitmask` is
    lightkurve's convention: a frame is kept if none of the bitmask's bits
    are set in its `QUALITY` (default 175 -- lightkurve's "default" mask: 1
    AttitudeTweak, 2 SafeMode, 4 CoarsePoint, 8 EarthPoint, 32 Desat, 128
    ManualExclude). `None` keeps only `QUALITY == 0` (strict); `0`/falsy
    disables the check (keeps everything).

    Previously duplicated inline in `ss397_localize.py` and
    `prf_photometry.load_prf_light_curve` -- same formula, now one place.
    """
    quality = np.asarray(quality)
    if quality_bitmask is None:
        return quality == 0
    if not quality_bitmask:
        return np.ones(quality.shape, dtype=bool)
    return (quality & quality_bitmask) == 0


def auto_clean_light_curve(df_lc, quality_bitmask=175, star_bkg_ratio_min=None,
                            local_noise_kappa=2.5, local_noise_window=0.25):
    """
    Apply ROADMAP.md Этап 5's three rules, in order, each computed on the
    survivors of the previous one:
      1. `QUALITY` bitmask (`quality_mask`).
      2. `STAR_BKG_RATIO` < `star_bkg_ratio_min` -- aperture-mode only;
         skipped if the column is absent, or if `star_bkg_ratio_min` is None
         (the default: ROADMAP.md gives no threshold, and this rule directly
         changes the published "_clean" curve, so it only applies when a
         caller consciously sets a value, e.g. via `run_config.RunConfig`).
      3. local point-to-point noise (`lightcurve_tools.local_point_to_point_sigma`
         on normalized `FLUX`) > `local_noise_kappa` * median.
    A frame dropped by an earlier rule is never re-evaluated by a later one.

    Returns `(df_clean, log_df)`: `df_clean` is `df_lc` restricted to the
    surviving rows (index reset); `log_df` has one row per DROPPED row,
    columns `FRAME, BTJD, REASON` ("QUALITY"/"STAR_BKG_RATIO"/"LOCAL_NOISE"),
    sorted by `FRAME` -- same format as `lightcurve_tools.windows_to_log`, so
    a manual-window log can be concatenated with this one into a single
    combined cleaning log.
    """
    time_col = "BTJD" if "BTJD" in df_lc.columns else "MJD"
    n = len(df_lc)
    frame_no = df_lc["FRAME"].to_numpy() if "FRAME" in df_lc.columns else np.arange(1, n + 1)
    btjd = df_lc[time_col].to_numpy()

    keep = np.ones(n, dtype=bool)
    reason = np.full(n, "", dtype=object)

    q_ok = quality_mask(df_lc["QUALITY"].to_numpy(), quality_bitmask)
    reason[keep & ~q_ok] = "QUALITY"
    keep &= q_ok

    if star_bkg_ratio_min is not None and "STAR_BKG_RATIO" in df_lc.columns:
        ratio_ok = df_lc["STAR_BKG_RATIO"].to_numpy() >= star_bkg_ratio_min
        drop_now = keep & ~ratio_ok
        reason[drop_now] = "STAR_BKG_RATIO"
        keep &= ratio_ok

    if keep.sum() > 2:
        idx = np.where(keep)[0]
        flux = df_lc["FLUX"].to_numpy()[idx]
        norm_flux = flux / np.median(flux)
        sigma = local_point_to_point_sigma(btjd[idx], norm_flux, window=local_noise_window)
        # `>=`, mirroring `ss397_localize.py`'s original inline `sigma < kappa * median`
        # keep-condition exactly (a kept frame satisfies strict "<"; this is its negation)
        noisy = sigma >= local_noise_kappa * np.median(sigma)
        drop_idx = idx[noisy]
        reason[drop_idx] = "LOCAL_NOISE"
        keep[drop_idx] = False

    log_df = pd.DataFrame({
        "FRAME": frame_no[~keep],
        "BTJD": btjd[~keep],
        "REASON": reason[~keep],
    }).sort_values("FRAME").reset_index(drop=True)
    return df_lc.loc[keep].reset_index(drop=True), log_df


def summarize_cleaning_log(log_df, total_frames):
    """
    Human-readable summary of a combined cleaning log (automatic rules +
    manual windows, `FRAME, BTJD, REASON` format): counts per reason, and the
    dropped BTJD intervals (consecutive dropped frame numbers merged into
    one interval, regardless of reason) -- the "abzac Методы" ROADMAP.md
    Этап 5 asks for.
    """
    lines = [f"Frames total: {total_frames}, dropped: {len(log_df)}, "
             f"kept: {total_frames - len(log_df)}"]
    for reason, count in log_df["REASON"].value_counts().items():
        lines.append(f"  {reason}: {count}")

    if len(log_df):
        ordered = log_df.sort_values("FRAME")
        frames = ordered["FRAME"].to_numpy()
        btjd = ordered["BTJD"].to_numpy()
        breaks = np.where(np.diff(frames) > 1)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, frames.size - 1]
        lines.append("Dropped intervals (BTJD):")
        for s, e in zip(starts, ends):
            lines.append(f"  frames {frames[s]}-{frames[e]}: BTJD {btjd[s]:.3f}-{btjd[e]:.3f}")
    return "\n".join(lines)


def save_cleaning_log(log_df, star_name, sector, cut_width, cut_height=None,
                       star_directory=STAR_DIRECTORY, suffix="_clean"):
    """
    Write a combined cleaning log next to `light_curve_sector_{sector}{suffix}.csv`,
    same naming convention as `lightcurve_tools.save_clean_light_curve`:
        light_curve_sector_{sector}{suffix}_log.csv

    Returns the path written to.
    """
    if cut_height is None:
        cut_height = cut_width
    out_dir = os.path.join(star_directory, get_nospace_star_name(star_name), f"{cut_width}x{cut_height}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"light_curve_sector_{sector}{suffix}_log.csv")
    log_df.to_csv(out_path, index=False)
    return out_path
