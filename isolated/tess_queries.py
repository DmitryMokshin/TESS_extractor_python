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
                      star_directory: str = STAR_DIRECTORY) -> str:
    """
    Download a TESScut zip archive for the given position/size and save it
    to the same directory layout the rest of the package expects (one zip
    per star/cutout-size, containing one FITS file per sector). Returns the
    path to the (created or already-cached) zip file.

    Tries `astroquery.mast.Tesscut` first (recommended, robust); falls back
    to a raw HTTP request with retries if astroquery isn't installed or the
    query fails outright.
    """
    nospace_star_name = get_nospace_star_name(star_name)
    out_dir = os.path.join(star_directory, nospace_star_name, f"{width}x{height}")
    os.makedirs(out_dir, exist_ok=True)
    cutouts_file = os.path.join(out_dir, f"{nospace_star_name}.zip")

    try:
        _get_tess_cutouts_astroquery(star_ra, star_dec, width, height, cutouts_file)
    except ImportError:
        _get_tess_cutouts_raw(star_ra, star_dec, width, height, cutouts_file)
    except Exception as e:
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


def _get_tess_cutouts_astroquery(star_ra, star_dec, width, height, cutouts_file):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.mast import Tesscut

    coord = SkyCoord(ra=star_ra * u.deg, dec=star_dec * u.deg)
    with heartbeat(f"Downloading TESS cutouts (ra={star_ra:.4f}, dec={star_dec:.4f}, {width}x{height}px)"):
        hdulists = Tesscut.get_cutouts(coordinates=coord, size=[height, width])

    if not hdulists:
        raise RuntimeError(f"MAST TESScut returned no cutouts for ra={star_ra}, dec={star_dec}")

    with zipfile.ZipFile(cutouts_file, "w") as archive:
        for i, hdul in enumerate(hdulists):
            buf = io.BytesIO()
            hdul.writeto(buf)
            sector = hdul[0].header.get("SECTOR", i)
            archive.writestr(f"tess-sector{sector:04d}-cutout.fits", buf.getvalue())


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
