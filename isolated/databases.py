"""
Simbad / Gaia TAP (ADQL) queries.

Ported from databases.jl. The Julia code used `VirtualObservatory.jl`'s
`TAPService(:simbad)` / `TAPService(:gaia)` + `execute(...)`; here we use
`astroquery`'s generic TAP client (`astroquery.utils.tap.core.TapPlus`)
against the same underlying services, and keep the ADQL queries themselves
essentially verbatim.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from astroquery.utils.tap.core import TapPlus

from .progress import heartbeat

SIMBAD_TAP_URL = "https://simbad.cds.unistra.fr/simbad/sim-tap"
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"

_simbad_tap = None
_gaia_tap = None


def _simbad() -> TapPlus:
    global _simbad_tap
    if _simbad_tap is None:
        _simbad_tap = TapPlus(url=SIMBAD_TAP_URL)
    return _simbad_tap


def _gaia() -> TapPlus:
    global _gaia_tap
    if _gaia_tap is None:
        _gaia_tap = TapPlus(url=GAIA_TAP_URL)
    return _gaia_tap


def _run_query(tap: TapPlus, adql: str, desc: str = "Querying TAP service", async_query: bool = False) -> pd.DataFrame:
    """
    Direct analogue of `convert_vo_to_df(execute(TAPService(...), adql))`.

    The Simbad/Gaia TAP services don't return consistent column-name casing
    (e.g. Gaia returns `SOURCE_ID`/`DESIGNATION` in uppercase but most other
    columns in lowercase) -- everywhere else in this package assumes
    lowercase column names (matching the Julia code, whose TAP client
    normalized this for you), so we normalize here, once, for every query.

    `async_query` (ROADMAP.md Этап 9): the Gaia TAP service caps a SYNC job
    (`launch_job`, the default here) at exactly 2000 rows for a query with no
    explicit `TOP` -- verified live (a real cone-search query that should
    return ~36000 rows came back truncated to 2000 over sync, complete over
    async). An ASYNC job (`launch_job_async`) has no such cap, at the cost of
    being slower (polls for completion instead of a direct response). Default
    `False` keeps every existing caller's behaviour/speed unchanged -- none
    of the small lookup queries (Simbad identification, single-source_id
    Gaia rows) ever approached the cap; only `data_io.load_gaia_stars_in_view_data`'s
    field-star query does.
    """
    with heartbeat(desc):
        job = tap.launch_job_async(adql) if async_query else tap.launch_job(adql)
        table = job.get_results()
    df = table.to_pandas()
    df.columns = [str(c).lower() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)

    # Gaia `source_id` is a 19-digit integer, well beyond float64's ~15-16
    # significant digits of precision. If the TAP/VOTable round-trip handed
    # it back as float64 (this happens inconsistently depending on the exact
    # query/service), naive equality comparisons against it will silently
    # fail on a different-but-close value. Force it to a (nullable) integer
    # dtype here so every query returns bit-exact ids.
    if "source_id" in df.columns and pd.api.types.is_float_dtype(df["source_id"]):
        df["source_id"] = df["source_id"].round().astype("Int64")

    return df


def get_star_gaia_data(star_name: str, gaia: str = "dr3") -> pd.Series:
    """Direct port of `get_star_gaia_data`."""
    df_simbad = _run_query(_simbad(), f"""select top 5 oid, main_id, ids
    FROM ident
    JOIN basic ON ident.oidref = basic.oid
    JOIN ids on ident.oidref = ids.oidref
    where id = '{star_name}'""")

    gaia_regex = re.compile(rf"gaia {gaia} (?P<id>[0-9]+)", re.IGNORECASE)
    match = gaia_regex.search(df_simbad["ids"].iloc[0])
    if match is None:
        raise ValueError(f"No Gaia {gaia} identifier found for '{star_name}' in Simbad")
    gaia_id = int(match.group("id"))

    df_gaia = _run_query(_gaia(), f"select top 1 * from gaia{gaia}.gaia_source where source_id = {gaia_id}")
    return df_gaia.iloc[0]


def get_star_coords(star_name: str):
    """Direct port of `get_star_coords`."""
    df_simbad = _run_query(_simbad(),
                            f"select top 5 * FROM ident JOIN basic ON ident.oidref = basic.oid where id = '{star_name}'")
    return float(df_simbad["ra"].iloc[0]), float(df_simbad["dec"].iloc[0])


def get_star_coords_gaia(star_name: str, gaia: str = "dr3"):
    """Direct port of `get_star_coords_gaia`."""
    df_simbad = _run_query(_simbad(), f"""select top 5 oid, main_id, ids
    FROM ident
    JOIN basic ON ident.oidref = basic.oid
    JOIN ids on ident.oidref = ids.oidref
    where id = '{star_name}'""")

    gaia_regex = re.compile(rf"gaia {gaia} (?P<id>[0-9]+)", re.IGNORECASE)
    match = gaia_regex.search(df_simbad["ids"].iloc[0])
    gaia_id = int(match.group("id"))

    df_gaia = _run_query(_gaia(),
                          f"select top 1 source_id, ra, dec from gaia{gaia}.gaia_source where source_id = {gaia_id}")
    return float(df_gaia["ra"].iloc[0]), float(df_gaia["dec"].iloc[0])


def brightest_near_relative_mag(star_ra: float, star_dec: float, box_width: float, box_height: float,
                                 gaia: str = "dr3") -> float:
    """Direct port of `brightest_near_relative_mag`."""
    df_gaia = _run_query(_gaia(), f"""select * from gaia{gaia}.gaia_source where CONTAINS(POINT('ICRS', ra, dec),
    BOX('ICRS', {star_ra}, {star_dec}, {box_width}, {box_height})) = 1""")

    dist = np.sqrt((np.cos(np.radians(star_dec)) * (df_gaia["ra"] - star_ra)) ** 2 +
                    (df_gaia["dec"] - star_dec) ** 2)
    df_gaia = df_gaia.assign(dist=dist).sort_values("dist").reset_index(drop=True)

    near_g_mag_sorted = df_gaia.loc[1:, "phot_g_mean_mag"].dropna()

    try:
        if pd.isna(df_gaia.loc[0, "phot_g_mean_mag"]):
            raise ValueError("missing phot_g_mean_mag for the star")
        star_g_mag = df_gaia.loc[0, "phot_g_mean_mag"]
    except Exception as e:
        warnings.warn(str(e))
        star_g_mag = 0.0

    try:
        brightest_near_g_mag = float(near_g_mag_sorted.min())
        if np.isnan(brightest_near_g_mag):
            raise ValueError("no neighbouring stars with phot_g_mean_mag")
    except Exception as e:
        warnings.warn(str(e))
        brightest_near_g_mag = star_g_mag

    return brightest_near_g_mag - star_g_mag


def check_star_isolation_by_name(star_name: str, box_width: float, box_height: float, mag_threshold: float,
                                  gaia: str = "dr3") -> bool:
    """Explicit, unambiguous version of the Julia `check_star_for_isolation(star_name, ...)` method."""
    star_ra, star_dec = get_star_coords(star_name)
    return check_star_isolation_by_radec(star_ra, star_dec, box_width, box_height, mag_threshold, gaia=gaia)


def check_star_isolation_by_radec(star_ra: float, star_dec: float, box_width: float, box_height: float,
                                   mag_threshold: float, gaia: str = "dr3") -> bool:
    """Explicit, unambiguous version of the Julia `check_star_for_isolation(ra, dec, ...)` method."""
    df_gaia = _run_query(_gaia(), f"""select * from gaia{gaia}.gaia_source where CONTAINS(POINT('ICRS', ra, dec),
    BOX('ICRS', {star_ra}, {star_dec}, {box_width}, {box_height})) = 1""")

    dist = np.sqrt((np.cos(np.radians(star_dec)) * (df_gaia["ra"] - star_ra)) ** 2 +
                    (df_gaia["dec"] - star_dec) ** 2)
    df_gaia = df_gaia.assign(dist=dist).sort_values("dist").reset_index(drop=True)

    star_rp_mag = df_gaia["phot_rp_mean_mag"].iloc[0]
    bright_stars_mag = df_gaia.loc[df_gaia["phot_rp_mean_mag"] < star_rp_mag + mag_threshold, "phot_rp_mean_mag"]

    return (len(bright_stars_mag) - 1) == 0


def get_simbad_young_stars(mag_min: float, mag_max: float) -> pd.DataFrame:
    """Direct port of `get_simbad_young_stars`."""
    df_simbad_sp = _run_query(_simbad(), f"""select oid, otype, main_id, ra, dec, sptype, V, R, "year",  ids
from basic
join allfluxes on oid = allfluxes.oidref
join ids on oid = ids.oidref
join (select oidref, sptype, "year"
      from mesSpT join REF on mesSpT.bibcode = REF.bibcode order by oidref,"year" desc) as sptyear on oid = sptyear.oidref
where (otype = 'TT*' or otype = 'Ae*' or otype = 'Or*') and V < {mag_min} and V > {mag_max}""")

    rows = []
    old_main_id = ""
    for _, row in df_simbad_sp.iterrows():
        if row["main_id"] != old_main_id:
            old_main_id = row["main_id"]
            rows.append(row)
    df_simbad = pd.DataFrame(rows).reset_index(drop=True)

    df_simbad["star_name"] = df_simbad["main_id"]
    for i in range(len(df_simbad)):
        star_names = df_simbad.loc[i, "ids"].split("|")
        main_name = df_simbad.loc[i, "star_name"]
        for name in star_names:
            if name[:2] == "V*":
                main_name = name[3:]
        first_word = main_name.split()[0]
        if first_word.endswith("*"):
            main_name = " ".join(main_name.split()[1:])
        main_name = " ".join(main_name.split())
        df_simbad.loc[i, "star_name"] = main_name

    return df_simbad_sp.assign(star_name=df_simbad["star_name"])


def check_for_isolation(df_stars: pd.DataFrame, box_size: float, mag_thres: float, gaia: str = "dr2") -> pd.DataFrame:
    """Direct port of `check_for_isolation`."""
    isolated_rows = []
    for _, row in df_stars.iterrows():
        star_name, star_ra, star_dec = row["star_name"], row["ra"], row["dec"]
        is_isolated = check_star_isolation_by_radec(star_ra, star_dec, box_size / 60, box_size / 60, mag_thres,
                                                     gaia=gaia)
        if is_isolated:
            print(f"{star_name} is isolated")
            isolated_rows.append(row)
        else:
            print(f"{star_name} is not isolated")
    return pd.DataFrame(isolated_rows).reset_index(drop=True)
