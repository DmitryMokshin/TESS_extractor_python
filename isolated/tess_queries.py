"""
MAST TESScut API access.

Ported from tess-queries.jl:
    get_tess_sectors(ra, dec), get_tess_cutouts(ra, dec, width, height; ...)

The raw `mast.stsci.edu/tesscut/...` HTTP endpoint (what the Julia code hits
with `HTTP.download`, and what a naive `requests.get` port of it hits too) is
known to be flaky from some networks: it 30x-redirects to a separate
cloud/S3-backed host, and that handshake is where corporate proxies /
antivirus TLS-inspection / some ISPs most often break the connection with
errors like `SSLEOFError`. `astroquery.mast.Tesscut` talks to the same
service but with STScI's own battle-tested retry/redirect handling, so it is
used here as the primary path; the raw-HTTP function is kept as a fallback
and for reference.

`astroquery.exceptions.TimeoutError: Timeout limit of 600 exceeded` is a
*different* failure mode from the network issues above -- it means
`Tesscut.get_cutouts` was asked for every sector the star was ever observed
in (no `sector=` given) and/or a large cutout, the request legitimately took
longer than astroquery's internal 600s budget to prepare and download, and
astroquery discarded the (successfully received) response rather than
returning it late. See `tesscut_timeout_notes.dat` in the repo root for the
full investigation. `get_tess_cutouts`'s `sector` parameter is the fix.
"""
from __future__ import annotations

import io
import os
import zipfile

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm.auto import tqdm

from .config import STAR_DIRECTORY
from .geometry import get_nospace_star_name
from .progress import heartbeat

_BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; isolated-python/1.0)"}


def _session_with_retries() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update(_BROWSER_HEADERS)
    return session


def get_tess_sectors_for_radec(star_ra: float, star_dec: float, product: str = "SPOC") -> list[dict]:
    """Direct port of `get_tess_sectors(star_ra, star_dec; product)` (the network-query method)."""
    session = _session_with_retries()
    resp = session.get("https://mast.stsci.edu/tesscut/api/v0.1/sector",
                        params={"ra": star_ra, "dec": star_dec, "radius": "1m", "product": product},
                        timeout=60)
    resp.raise_for_status()
    return resp.json()["results"]


def get_tess_cutouts(star_ra: float, star_dec: float, width: int, height: int,
                      star_name: str = "star", product: str = "SPOC",
                      star_directory: str = STAR_DIRECTORY, sector: int | None = None) -> str:
    """
    Download a TESScut zip archive for the given position/size and save it
    to the same directory layout the rest of the package expects (one zip
    per star/cutout-size, containing one FITS file per sector). Returns the
    path to the (created or already-cached) zip file.

    `sector`: if given, ask TESScut for just this one sector instead of
    every sector the star was ever observed in -- much faster, and avoids
    the "Timeout limit of 600 exceeded" failure mode documented in
    `tesscut_timeout_notes.dat` (request size grows with cutout area x
    frames-per-sector x number-of-sectors). Repeated calls with different
    `sector` values for the same star/size append to the same cached zip
    instead of overwriting it, so the file ends up the same as a single
    `sector=None` call would have produced. Default (`sector=None`) is the
    original "fetch every sector at once" behaviour -- unchanged.

    Tries `astroquery.mast.Tesscut` first (recommended, robust); falls back
    to a raw HTTP request with retries if astroquery isn't installed or the
    query fails outright.
    """
    nospace_star_name = get_nospace_star_name(star_name)
    out_dir = os.path.join(star_directory, nospace_star_name, f"{width}x{height}")
    os.makedirs(out_dir, exist_ok=True)
    cutouts_file = os.path.join(out_dir, f"{nospace_star_name}.zip")

    try:
        _get_tess_cutouts_astroquery(star_ra, star_dec, width, height, cutouts_file, sector=sector)
    except ImportError:
        _get_tess_cutouts_raw(star_ra, star_dec, width, height, cutouts_file)
    except Exception as e:
        from astroquery.exceptions import TimeoutError as AstroqueryTimeoutError
        if isinstance(e, AstroqueryTimeoutError):
            raise ConnectionError(
                "TESScut/astroquery timed out waiting for the cutout to be prepared "
                f"(sector={sector!r}, size={width}x{height}).\n"
                f"Original error: {e!r}\n\n"
                "This is almost always NOT a network/TLS problem -- astroquery only checks "
                "its timeout *after* the response is fully received, so this means the "
                "request itself was just too big to prepare/transfer within its budget. "
                "See tesscut_timeout_notes.dat for the full writeup. In order:\n"
                "  1. Pass sector=<int> (this function already does, if you got here through "
                "load_light_curve/load_tess_cutouts with a sector) so TESScut only cuts the "
                "one sector you need instead of every sector the star was ever observed in.\n"
                "  2. A very large cutout or a very long sector can still be slow to prepare; "
                "astroquery's own request timeout is already raised to 3600s here "
                "(Tesscut._service_api_connection.TIMEOUT), but you can raise it further.\n"
                "  3. Retry -- MAST's cutout service does occasionally have transient slow spells."
            ) from e
        raise ConnectionError(
            "Failed to download TESS cutouts from MAST via astroquery.\n"
            f"Original error: {e!r}\n\n"
            "This is almost always a network/TLS issue on your machine, not a bug in the "
            "query itself -- try, in order:\n"
            "  1. pip install -U astroquery requests urllib3 certifi\n"
            "  2. Check whether https://mast.stsci.edu is reachable from a normal browser "
            "on this machine/network (corporate proxies and some antivirus TLS-inspection "
            "software are the most common cause of SSLEOFError here).\n"
            "  3. If you're behind a proxy, make sure HTTPS_PROXY/HTTP_PROXY are set for "
            "this process.\n"
            "  4. Retry -- MAST's cutout service does occasionally have transient outages."
        ) from e

    return cutouts_file


def _get_tess_cutouts_astroquery(star_ra, star_dec, width, height, cutouts_file, sector=None):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.mast import Tesscut

    # astroquery's own request timeout is only checked *after* the response is
    # fully received (see tesscut_timeout_notes.dat), so 600s (the astroquery
    # default) is too tight for a large cutout or an all-sectors request even
    # when the request itself is perfectly healthy.
    Tesscut._service_api_connection.TIMEOUT = 3600

    coord = SkyCoord(ra=star_ra * u.deg, dec=star_dec * u.deg)
    sector_desc = f"sector {sector}" if sector is not None else "all sectors"
    with heartbeat(f"Downloading TESS cutouts (ra={star_ra:.4f}, dec={star_dec:.4f}, "
                   f"{width}x{height}px, {sector_desc})"):
        hdulists = Tesscut.get_cutouts(coordinates=coord, size=[height, width], sector=sector)

    if not hdulists:
        raise RuntimeError(f"MAST TESScut returned no cutouts for ra={star_ra}, dec={star_dec}, {sector_desc}")

    zip_mode = "a" if os.path.isfile(cutouts_file) else "w"
    with zipfile.ZipFile(cutouts_file, zip_mode) as archive:
        already_cached = set(archive.namelist()) if zip_mode == "a" else set()
        for i, hdul in enumerate(hdulists):
            cutout_sector = hdul[0].header.get("SECTOR", i)
            entry_name = f"tess-sector{cutout_sector:04d}-cutout.fits"
            if entry_name in already_cached:
                continue
            buf = io.BytesIO()
            hdul.writeto(buf)
            archive.writestr(entry_name, buf.getvalue())


def _get_tess_cutouts_raw(star_ra, star_dec, width, height, cutouts_file):
    """Fallback used only if astroquery is not installed."""
    session = _session_with_retries()
    url = "https://mast.stsci.edu/tesscut/api/v0.1/astrocut"
    params = {"ra": star_ra, "dec": star_dec, "y": height, "x": width}

    resp = session.get(url, params=params, allow_redirects=True, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(cutouts_file, "wb") as f, tqdm(total=total or None, unit="B", unit_scale=True,
                                              desc="Downloading TESS cutouts") as bar:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            bar.update(len(chunk))
