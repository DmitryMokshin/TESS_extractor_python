"""
Stitching several sectors' light curves into one series for frequency
analysis: each sector is normalized (its own median subtracted) and
detrended (its own polynomial trend removed) independently before
concatenation, so a differing zero-point or slow drift between sectors --
and the year-scale gaps TESS's yearly-revisit cadence puts between them --
don't leak into the lowest frequencies once combined. See ROADMAP.md,
"Этап 7. Несколько секторов". Not a Julia port.

Downloading/photometering/cleaning stays entirely per-sector and unchanged
(`data_io.load_tess_cutouts`'s `sector=` already supports fetching one
sector at a time into the same cache; `TESS_cleaning.py`/`PRF_cleaning.py`
already produce one `_clean`/`_prf_clean` CSV per sector) -- this module
only combines the already-cleaned per-sector outputs.
"""
from __future__ import annotations

import numpy as np

from .prewhitening import detrend_poly
from .stats import cleaned_jds_mags
from .data_io import read_light_curve_csv


def stitch_light_curves(t_list, y_list, sector_labels=None, detrend_deg=2):
    """
    Combine several per-sector series into one: each sector independently
    has its own median subtracted (normalization) and its own polynomial
    trend of degree `detrend_deg` removed (`prewhitening.detrend_poly` --
    same detrending already used elsewhere in the pipeline, not a new
    formula), then all sectors are concatenated and sorted by time.

    `sector_labels`: optional label per input series (e.g. sector numbers);
    defaults to the 1-based index into `t_list`.

    Returns `(t, y, sector)`, sorted by `t`; `sector` tags each point with
    its origin, for diagnostics (e.g. colouring a plot by sector, or
    checking how much of the baseline gap between sectors is really gap).
    """
    t_parts, y_parts, tag_parts = [], [], []
    for i, (t_i, y_i) in enumerate(zip(t_list, y_list)):
        t_i = np.asarray(t_i, dtype=float)
        y_i = np.asarray(y_i, dtype=float) - np.median(y_i)
        y_i = detrend_poly(t_i, y_i, deg=detrend_deg)
        label = sector_labels[i] if sector_labels is not None else i + 1
        t_parts.append(t_i)
        y_parts.append(y_i)
        tag_parts.append(np.full(t_i.shape, label))

    t = np.concatenate(t_parts)
    y = np.concatenate(y_parts)
    sector = np.concatenate(tag_parts)
    order = np.argsort(t)
    return t[order], y[order], sector[order]


def stitch_sector_csvs(paths, sector_labels=None, value_col="MAG", detrend_deg=2,
                        jd_box=0.1, sigma_tol=10, n_out=20):
    """
    Read and stitch one light-curve CSV per sector (`data_io.read_light_curve_csv`
    -- tolerates old `MJD`/`SN` column names same as everywhere else).

    `value_col="MAG"` (the default, matching `Peridogram_compute.py`'s own
    convention): each file is cleaned the same way `compute_ls_periodogram`/
    the prewhitening step already clean a single file
    (`stats.cleaned_jds_mags` -- NaN-drop + box-smooth sigma-clip), so
    behaviour on a single sector is unchanged from today; time comes back as
    full BJD (`geometry.get_true_jd`) and is converted to the pipeline's
    usual `BJD-2457000` (BTJD) convention after stitching.

    `value_col="FLUX"` (for a PRF-mode curve, or any future caller that
    wants flux instead of magnitude): just NaN-dropped and sorted by time,
    no sigma-clipping -- the `_clean`/`_prf_clean` files are already cleaned
    by `isolated.cleaning.auto_clean_light_curve` (ROADMAP.md Этап 5), so
    reapplying `cleaned_jds_mags`'s own box-smooth clip (tuned for MAG) isn't
    appropriate here.

    `sector_labels`: passed through to `stitch_light_curves`; defaults to
    the 1-based index into `paths` if not given.

    Returns `(t, y, sector)` -- see `stitch_light_curves`. `t` is BTJD
    (`BJD-2457000`).
    """
    t_list, y_list = [], []
    for path in paths:
        df = read_light_curve_csv(path)
        if value_col == "MAG":
            jds, mags = cleaned_jds_mags(df, jd_box, sigma_tol, n_out)
            t_list.append(jds - 2457000.0)
            y_list.append(mags)
        else:
            time_col = "BTJD" if "BTJD" in df.columns else "MJD"
            t_i = df[time_col].to_numpy(dtype=float)
            y_i = df[value_col].to_numpy(dtype=float)
            finite = np.isfinite(t_i) & np.isfinite(y_i)
            order = np.argsort(t_i[finite])
            t_list.append(t_i[finite][order])
            y_list.append(y_i[finite][order])

    return stitch_light_curves(t_list, y_list, sector_labels=sector_labels, detrend_deg=detrend_deg)
