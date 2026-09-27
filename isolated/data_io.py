"""
On-disk caching / loading layer, mirroring the `$star_directory/...` file
layout used throughout main.jl.

Ported from main.jl:
    extract_tess_cutouts, get_tess_sectors (file-based method),
    get_star_tesscut_fits, load_star_gaia_data, load_tess_cutouts,
    load_gaia_stars_in_view_data, create_gaia_datafiles, load_light_curve
"""
from __future__ import annotations

import io
import os
import re
import zipfile

import numpy as np
import pandas as pd
from astropy.io import fits as pyfits
from tqdm.auto import tqdm

from .config import STAR_DIRECTORY, TESS_MAX_SECTORS
from .geometry import get_nospace_star_name, get_rel_radec, get_distance, calc_tess_magnitude
from .fits_utils import get_tesscut_corners
from .databases import get_star_gaia_data
from .tess_queries import get_tess_cutouts as _download_tess_cutouts
from .databases import _run_query, _gaia as _gaia_tap


def _read_gaia_csv(path: str) -> pd.DataFrame:
    """
    pd.read_csv + lowercase columns. Gaia/Simbad-derived CSV caches on disk
    may have been written before column names were normalized to lowercase
    (see isolated.databases._run_query) -- reading through this helper
    instead of bare pd.read_csv makes old caches self-heal instead of
    raising KeyError on columns like 'source_id'.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]
    return df


def read_light_curve_csv(path: str) -> pd.DataFrame:
    """
    Read a light-curve CSV (`load_light_curve`/`prf_photometry.
    load_prf_light_curve` output, or anything with the same column
    convention), tolerating old cached files written before ROADMAP.md's
    Этап 1/2: they call the time column "MJD" (it was always really BTJD --
    see CLAUDE.md), have no `FRAME`/`CADENCENO`/`QUALITY`/`FLUX_ERR`/...
    columns at all, and call the star/background aperture ratio "SN" (it
    was never a signal-to-noise ratio -- see CLAUDE.md). New files already
    say "BTJD"/"STAR_BKG_RATIO"; old ones get renamed on the fly so every
    caller can just use the new names either way. Not a Julia port -- same
    self-healing idea as `_read_gaia_csv`, for a different file family.
    """
    df = pd.read_csv(path)
    renames = {}
    if "BTJD" not in df.columns and "MJD" in df.columns:
        renames["MJD"] = "BTJD"
    if "STAR_BKG_RATIO" not in df.columns and "SN" in df.columns:
        renames["SN"] = "STAR_BKG_RATIO"
    if renames:
        df = df.rename(columns=renames)
    return df


def find_target_row(gaia_stars_data: pd.DataFrame, gaia_data: pd.Series, sector, max_sep_arcsec: float = 2.0) -> pd.Series:
    """
    Find the target star's row inside a Gaia-stars-in-view catalog.

    Tries an exact `source_id` match first (cheap, unambiguous). Some TAP
    responses hand back `source_id` (a 19-digit integer) as float64, which
    only has ~15-16 significant digits of precision -- so on rare occasions
    the exact match silently fails even though the star *is* in the table.
    As a fallback, match by nearest sky position instead (ra/dec agree to
    much better than an arcsecond for the same Gaia source across queries).
    """
    exact = gaia_stars_data[gaia_stars_data["source_id"] == gaia_data["source_id"]]
    if len(exact) > 0:
        return exact.iloc[0]

    sep_deg = np.sqrt((np.cos(np.radians(gaia_data["dec"])) * (gaia_stars_data["ra"] - gaia_data["ra"])) ** 2 +
                       (gaia_stars_data["dec"] - gaia_data["dec"]) ** 2)
    i_nearest = sep_deg.idxmin()
    if sep_deg[i_nearest] * 3600.0 <= max_sep_arcsec:
        return gaia_stars_data.loc[i_nearest]

    raise RuntimeError(
        f"Target star (Gaia source_id={gaia_data['source_id']}, "
        f"phot_rp_mean_mag={gaia_data.get('phot_rp_mean_mag')}) was not found in the "
        f"Gaia-stars-in-view catalog for sector {sector}, either by source_id or by sky "
        f"position (nearest match was {sep_deg[i_nearest] * 3600.0:.2f}\" away). This usually "
        f"means the star is fainter than the query's magnitude cutoff, or a stale/partial "
        f"gaia_stars_in_view_sector_{sector}.csv is cached. Try calling load_light_curve(...) "
        f"with rewrite_gaia_stars_file=True to refetch it."
    )


class ErrorNoTESScut(Exception):
    def __init__(self, star_name):
        super().__init__(f"No tesscut file for {star_name} is found. Download it using get_tess_cutouts(...).")


class ErrorTESSWrongSector(Exception):
    def __init__(self, sector_int):
        super().__init__(f"Wrong sector: {sector_int}")


def _star_zip_path(star_name, star_directory=STAR_DIRECTORY, cut_width=None, cut_height=None):
    nospace = get_nospace_star_name(star_name)
    if cut_width is None:
        return os.path.join(star_directory, nospace, f"{nospace}.zip")
    return os.path.join(star_directory, nospace, f"{cut_width}x{cut_height}", f"{nospace}.zip")


def extract_tess_cutouts(star_name="star", star_directory=STAR_DIRECTORY):
    """Direct port of `extract_tess_cutouts`."""
    zip_path = _star_zip_path(star_name, star_directory)
    if not os.path.isfile(zip_path):
        raise ErrorNoTESScut(star_name)
    nospace = get_nospace_star_name(star_name)
    out_dir = os.path.join(star_directory, nospace)
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            with archive.open(name) as src, open(os.path.join(out_dir, os.path.basename(name)), "wb") as dst:
                dst.write(src.read())


def get_tess_sectors_from_file(star_name, star_directory=STAR_DIRECTORY):
    """Direct port of the file-based `get_tess_sectors(star_name; query=false)` method."""
    zip_path = _star_zip_path(star_name, star_directory)
    if not os.path.isfile(zip_path):
        raise ErrorNoTESScut(star_name)
    sectors = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            with archive.open(name) as f:
                with pyfits.open(io.BytesIO(f.read())) as hdul:
                    sectors.append(int(hdul[0].header["SECTOR"]))
    return sectors


def get_star_tesscut_fits(star_name, sector, star_directory=STAR_DIRECTORY):
    """Direct port of `get_star_tesscut_fits`."""
    zip_path = _star_zip_path(star_name, star_directory)
    if not os.path.isfile(zip_path):
        raise ErrorNoTESScut(star_name)
    sectors = get_tess_sectors_from_file(star_name, star_directory)
    if sector not in sectors:
        raise ErrorTESSWrongSector(sector)
    idx = sectors.index(sector)
    with zipfile.ZipFile(zip_path) as archive:
        name = archive.namelist()[idx]
        with archive.open(name) as f:
            return pyfits.open(io.BytesIO(f.read()))


def load_star_gaia_data(star_name, star_directory=STAR_DIRECTORY) -> pd.Series:
    """Direct port of `load_star_gaia_data`."""
    out_dir = os.path.join(star_directory, get_nospace_star_name(star_name))
    os.makedirs(out_dir, exist_ok=True)
    gaia_data_file = os.path.join(out_dir, "gaia_target.csv")

    if not os.path.isfile(gaia_data_file):
        data = get_star_gaia_data(star_name)
        pd.DataFrame([data]).to_csv(gaia_data_file, index=False)
        return data
    return _read_gaia_csv(gaia_data_file).iloc[0]


def load_tess_cutouts(star_name, cut_width, cut_height=None, star_directory=STAR_DIRECTORY, sector=None):
    """
    Direct port of `load_tess_cutouts`. Returns a dict {sector_int: fits.HDUList}.

    `sector`: if given and not already present in the cached zip, downloads
    just that one sector instead of every sector the star was ever observed
    in (see `tess_queries.get_tess_cutouts`'s `sector` parameter) -- much
    faster, and avoids astroquery's "Timeout limit of 600 exceeded" on a
    large cutout or many sectors (see tesscut_timeout_notes.dat). Default
    (`sector=None`) is the original behaviour, unchanged: on a fresh cache,
    fetch every sector at once.
    """
    if cut_height is None:
        cut_height = cut_width

    nospace = get_nospace_star_name(star_name)
    out_dir = os.path.join(star_directory, nospace)
    os.makedirs(out_dir, exist_ok=True)

    gaia_data = load_star_gaia_data(star_name, star_directory)
    ra, dec = float(gaia_data["ra"]), float(gaia_data["dec"])

    cutouts_dir = os.path.join(out_dir, f"{cut_width}x{cut_height}")
    cutouts_file = os.path.join(cutouts_dir, f"{nospace}.zip")

    need_download = not os.path.isfile(cutouts_file)
    if not need_download and sector is not None:
        with zipfile.ZipFile(cutouts_file) as archive:
            cached_sectors = set()
            for name in archive.namelist():
                match = re.match(r"tess-sector(\d+)-cutout\.fits$", name)
                if match:
                    cached_sectors.add(int(match.group(1)))
        need_download = sector not in cached_sectors

    if need_download:
        os.makedirs(cutouts_dir, exist_ok=True)
        _download_tess_cutouts(ra, dec, cut_width, cut_height, star_name=star_name,
                                star_directory=star_directory, sector=sector)

    result = {}
    with zipfile.ZipFile(cutouts_file) as archive:
        for name in archive.namelist():
            with archive.open(name) as f:
                hdul = pyfits.open(io.BytesIO(f.read()))
                sector = int(hdul[0].header["SECTOR"])
                result[sector] = hdul
    return result


def load_gaia_stars_in_view_data(star_name, cut_fits, d_mag_r=5.0, rewrite_file=False,
                                  star_directory=STAR_DIRECTORY) -> pd.DataFrame:
    """Direct port of `load_gaia_stars_in_view_data`."""
    gaia_data = load_star_gaia_data(star_name, star_directory)
    corners = get_tesscut_corners(cut_fits)

    header3 = cut_fits[2].header
    cut_width, cut_height = int(header3["NAXIS1"]), int(header3["NAXIS2"])
    sector = int(cut_fits[0].header["SECTOR"])

    out_dir = os.path.join(star_directory, get_nospace_star_name(star_name), f"{cut_width}x{cut_height}")
    gaia_stars_file = os.path.join(out_dir, f"gaia_stars_in_view_sector_{sector}.csv")

    center = sum(corners) / 4.0
    distance = get_distance(*corners[0], *corners[2]) / 2.0

    # The magnitude cutoff should only ever need to reach `d_mag_r` fainter
    # than the target itself (that's the whole point of `d_mag_r` -- how far
    # below the target's brightness a neighbour still matters for the PRF/
    # background model). A previous version of this function floored the
    # cutoff at 20 unconditionally; for a *bright* target (e.g. RP~10) in a
    # crowded/dense field that pulls in tens of thousands of irrelevant faint
    # stars down to RP=20, which silently hits the Gaia TAP sync job's ~2000
    # row cap -- and since the query isn't sorted, the 2000 rows that do come
    # back are an essentially arbitrary, spatially/magnitude-biased subset
    # (this is what caused both the earlier "target star missing" error and
    # the "stars don't line up with the image" issue). Only fall back to 20
    # if the target's own magnitude isn't known.
    target_rp_mag = gaia_data.get("phot_rp_mean_mag", np.nan)
    if pd.isna(target_rp_mag):
        mag_cutoff = 20.0
    else:
        mag_cutoff = float(target_rp_mag) + d_mag_r + 0.5

    if not os.path.isfile(gaia_stars_file) or rewrite_file:
        os.makedirs(out_dir, exist_ok=True)
        adql = (f"select * from gaiadr3.gaia_source where DISTANCE(POINT('ICRS', {center[0]}, {center[1]}), "
                f"POINT('ICRS', ra, dec)) < {distance * 1.2} and phot_rp_mean_mag < {mag_cutoff} "
                f"order by phot_rp_mean_mag asc")
        data = _run_query(_gaia_tap(), adql, desc=f"Querying Gaia for stars near {star_name} (sector {sector})")
        if len(data) >= 2000:
            import warnings
            warnings.warn(
                f"Gaia query for '{star_name}' sector {sector} returned {len(data)} rows -- this "
                f"looks like it hit the TAP service's sync-job row cap (results are then an "
                f"incomplete, order-dependent subset of the true field). Consider lowering d_mag_r."
            )

        # The cone-search query above has, in practice, occasionally come back
        # without the target star itself even though it's well within the
        # search radius and magnitude cutoff (root cause unclear -- seen in
        # crowded/heavily-extincted fields; possibly a TAP-service-side quirk).
        # Since we already have the target's own Gaia row (`gaia_data`), just
        # guarantee its presence here instead of depending on the query.
        target_present = len(data) > 0 and (data["source_id"] == gaia_data["source_id"]).any()
        if not target_present:
            target_row = gaia_data.reindex(data.columns) if len(data.columns) > 0 else gaia_data
            data = pd.concat([data, target_row.to_frame().T], ignore_index=True)
            for numeric_col in ("ra", "dec", "phot_rp_mean_mag", "phot_g_mean_mag", "source_id"):
                if numeric_col in data.columns:
                    data[numeric_col] = pd.to_numeric(data[numeric_col], errors="coerce")

        header2 = cut_fits[1].header
        reference_px = np.array([header2["1CRPX4"], header2["2CRPX4"]])
        reference_radec = np.array([header2["1CRVL4"], header2["2CRVL4"]])
        conv_px_to_radec = np.array([[header2["11PC4"], header2["12PC4"]],
                                      [header2["21PC4"], header2["22PC4"]]])
        conv_radec_to_px = np.linalg.inv(conv_px_to_radec)

        stars_x = np.zeros(len(data))
        stars_y = np.zeros(len(data))
        for i in tqdm(range(len(data)), desc=f"{star_name} sector {sector}: star pixel coords",
                      unit="star", leave=False):
            star_radec = (data["ra"].iloc[i], data["dec"].iloc[i])
            delta_radec = get_rel_radec(reference_radec[0], reference_radec[1], *star_radec)
            xy = reference_px + conv_radec_to_px @ delta_radec
            stars_x[i], stars_y[i] = xy

        data = data.assign(px_x=stars_x, px_y=stars_y)
        data.to_csv(gaia_stars_file, index=False)
        gaia_stars_df = data
    else:
        gaia_stars_df = _read_gaia_csv(gaia_stars_file)

    return gaia_stars_df[gaia_stars_df["phot_rp_mean_mag"] < (gaia_data["phot_rp_mean_mag"] + d_mag_r)]


def create_gaia_datafiles(star_name, rewrite=False, star_directory=STAR_DIRECTORY):
    """Direct port of `create_gaia_datafiles`."""
    nospace = get_nospace_star_name(star_name)
    out_dir = os.path.join(star_directory, nospace)
    gaia_data_file = os.path.join(out_dir, "gaia_target.csv")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(gaia_data_file) or rewrite:
        gaia_data = get_star_gaia_data(star_name)
        pd.DataFrame([gaia_data]).to_csv(gaia_data_file, index=False)
    else:
        gaia_data = _read_gaia_csv(gaia_data_file).iloc[0]

    zip_path = _star_zip_path(star_name, star_directory)
    if os.path.isfile(zip_path):
        sectors = get_tess_sectors_from_file(star_name, star_directory)
        cut_fits = get_star_tesscut_fits(star_name, sectors[0], star_directory)
        corners = get_tesscut_corners(cut_fits)

        gaia_stars_file = os.path.join(out_dir, "gaia_stars_in_view.csv")
        center = sum(corners) / 4.0
        distance = get_distance(*corners[0], *corners[2]) / 2.0

        if not os.path.isfile(gaia_stars_file) or rewrite:
            target_rp_mag = gaia_data.get("phot_rp_mean_mag", np.nan)
            mag_cutoff = 20.0 if pd.isna(target_rp_mag) else float(target_rp_mag) + 5.5
            adql = (f"select * from gaiadr2.gaia_source where DISTANCE(POINT('ICRS', {center[0]}, {center[1]}), "
                    f"POINT('ICRS', ra, dec)) < {distance * 1.2} and phot_rp_mean_mag < {mag_cutoff} "
                    f"order by phot_rp_mean_mag asc")
            data = _run_query(_gaia_tap(), adql, desc=f"Querying Gaia for stars near {star_name}")
            data.to_csv(gaia_stars_file, index=False)


def load_light_curve(star_name, sector, cut_width, cut_height=None, d_mag_r=5.0, rewrite_file=False,
                      rewrite_gaia_stars_file=False, aperture_radius=3, star_directory=STAR_DIRECTORY) -> pd.DataFrame:
    """
    Direct port of `load_light_curve`: aperture photometry (with PRF-derived
    background mask and PRF aperture correction) light curve for one sector.
    """
    from .psf import get_tesscut_prf_supersampled
    from .photometry import (find_background_prf_gaia_mags, calc_aperture_prf_correction,
                              calc_aperture_photometry_with_diagnostics, calc_aperture_flux_error,
                              calc_prf_contamination_fraction)
    from .geometry import calc_tess_flux_from_mag

    if cut_height is None:
        cut_height = cut_width

    cut_fits = load_tess_cutouts(star_name, cut_width, cut_height, star_directory, sector=sector)[sector]
    flux_cuts = cut_fits[1].data["FLUX"]  # shape (n_cuts, height, width)
    flux_bkg_cuts = cut_fits[1].data["FLUX_BKG"]
    flux_err_cuts = cut_fits[1].data["FLUX_ERR"]
    n_cuts = flux_cuts.shape[0]
    mjds = cut_fits[1].data["TIME"]
    cadenceno = cut_fits[1].data["CADENCENO"]
    quality = cut_fits[1].data["QUALITY"]
    pos_corr1 = cut_fits[1].data["POS_CORR1"]
    pos_corr2 = cut_fits[1].data["POS_CORR2"]

    gaia_stars_data = load_gaia_stars_in_view_data(star_name, cut_fits, d_mag_r, rewrite_gaia_stars_file,
                                                    star_directory)
    gaia_data = load_star_gaia_data(star_name, star_directory)

    match = find_target_row(gaia_stars_data, gaia_data, sector)
    star_px = (float(match["px_x"]), float(match["px_y"]))
    star_mag = float(match["phot_rp_mean_mag"])

    mask = gaia_stars_data["phot_rp_mean_mag"] < star_mag + d_mag_r
    stars_x = gaia_stars_data.loc[mask, "px_x"].to_numpy()
    stars_y = gaia_stars_data.loc[mask, "px_y"].to_numpy()
    stars_mag = gaia_stars_data.loc[mask, "phot_rp_mean_mag"].to_numpy()

    out_dir = os.path.join(star_directory, get_nospace_star_name(star_name), f"{cut_width}x{cut_height}")
    light_curve_file = os.path.join(out_dir, f"light_curve_sector_{sector}.csv")

    if not os.path.isfile(light_curve_file) or rewrite_file:
        os.makedirs(out_dir, exist_ok=True)
        prf = get_tesscut_prf_supersampled(cut_fits)

        # note: flux_cuts is (n_cuts, height, width) in numpy/astropy order;
        # transpose the per-frame cut to (width, height) to match the rest
        # of this package's (x, y) convention (as in Julia's column-major FLUX array)
        def frame(i):
            return flux_cuts[i].T

        def frame_bkg(i):
            return flux_bkg_cuts[i].T

        def frame_err(i):
            return flux_err_cuts[i].T

        bkg_pixels = find_background_prf_gaia_mags(frame(n_cuts // 4), prf, stars_x, stars_y, stars_mag)

        aperture_correction = calc_aperture_prf_correction(aperture_radius, star_px[0], star_px[1], prf, cut_height)

        contamination = calc_prf_contamination_fraction(
            star_px[0], star_px[1], stars_x, stars_y, calc_tess_flux_from_mag(stars_mag), prf, cut_height,
            aperture_radius)
        if contamination < 0.8:
            print(f"Warning: only {contamination:.0%} of the aperture flux is modeled to be {star_name}'s own "
                  f"(sector {sector}, r={aperture_radius}px) -- a crowded field; consider PRF-deblending "
                  f"photometry (isolated.prf_photometry) instead of aperture photometry for this star.")

        phot_flux = np.zeros(n_cuts)
        sn = np.zeros(n_cuts)
        flux_bkg = np.zeros(n_cuts)
        centroid_x = np.zeros(n_cuts)
        centroid_y = np.zeros(n_cuts)
        flux_err = np.zeros(n_cuts)
        for i in tqdm(range(n_cuts), desc=f"{star_name} sector {sector}: photometry", unit="frame"):
            phot_flux[i], sn[i], flux_bkg[i], centroid_x[i], centroid_y[i] = calc_aperture_photometry_with_diagnostics(
                frame(i), frame_bkg(i), bkg_pixels, star_px[0], star_px[1], aperture_radius)
            flux_err[i] = calc_aperture_flux_error(frame_err(i), star_px[0], star_px[1], aperture_radius)
        phot_flux *= aperture_correction
        flux_err *= aperture_correction

        lc_df = pd.DataFrame({
            "BTJD": mjds,
            "FRAME": np.arange(1, n_cuts + 1),
            "CADENCENO": cadenceno,
            "QUALITY": quality,
            "FLUX": phot_flux,
            "FLUX_ERR": flux_err,
            "MAG": calc_tess_magnitude(np.abs(phot_flux)),
            "STAR_BKG_RATIO": sn,
            "FLUX_BKG": flux_bkg,
            "POS_CORR1": pos_corr1,
            "POS_CORR2": pos_corr2,
            "CENTROID_X": centroid_x,
            "CENTROID_Y": centroid_y,
        })
        lc_df.to_csv(light_curve_file, index=False)
        return lc_df

    return read_light_curve_csv(light_curve_file)
