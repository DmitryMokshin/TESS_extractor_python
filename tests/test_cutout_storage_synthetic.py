"""
Test for ROADMAP.md Этап 9's remaining items (4-5 of 5): per-sector FITS
storage (`isolated.data_io.load_tess_cutouts`, replacing the previous
whole-zip-in-memory cache) and the async Gaia TAP query
(`isolated.databases._run_query`'s `async_query=True`, replacing the
2000-row sync cap). Entirely offline -- the actual live-network behaviour
(sync truncates at 2000 rows, async doesn't; 2000 vs 35812 rows on the same
real query) was verified by hand while planning this stage, not repeated
here as an automated test (this project's test suite has no live-network
tests anywhere else either).

Plain script with asserts -- matches the other tests in this directory
(no pytest dependency yet). Run directly:
    .venv/bin/python tests/test_cutout_storage_synthetic.py
"""
import os
import shutil
import sys
import tempfile
import zipfile
from unittest import mock

import numpy as np
from astropy.io import fits as pyfits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import isolated.data_io as data_io
from isolated.databases import _run_query


def _make_cutout_hdul(sector):
    primary = pyfits.PrimaryHDU()
    primary.header["SECTOR"] = sector
    data = np.arange(9, dtype=float).reshape(3, 3)
    image = pyfits.ImageHDU(data=data, name="PIXELS")
    return pyfits.HDUList([primary, image])


def test_load_tess_cutouts_writes_and_reads_loose_fits(tmp_dir):
    star_directory = tmp_dir
    star_name = "Test Star"
    nospace = "Test_Star"
    cut_width = cut_height = 21
    sector = 80

    gaia_row = {"ra": 10.0, "dec": -5.0, "source_id": 123, "phot_rp_mean_mag": 12.0}
    cutouts_dir = os.path.join(star_directory, nospace, f"{cut_width}x{cut_height}")

    def fake_download(ra, dec, width, height, star_name=None, star_directory=None, sector=None):
        os.makedirs(cutouts_dir, exist_ok=True)
        path = data_io._sector_fits_path(cutouts_dir, nospace, sector)
        _make_cutout_hdul(sector).writeto(path, overwrite=True)

    with mock.patch.object(data_io, "load_star_gaia_data", return_value=gaia_row), \
         mock.patch.object(data_io, "_download_tess_cutouts", side_effect=fake_download) as mock_download:
        result = data_io.load_tess_cutouts(star_name, cut_width, cut_height, star_directory, sector=sector)

    assert mock_download.call_count == 1, "expected exactly one download on a cold cache"
    assert set(result.keys()) == {sector}
    assert int(result[sector][0].header["SECTOR"]) == sector
    expected_path = data_io._sector_fits_path(cutouts_dir, nospace, sector)
    assert os.path.isfile(expected_path), "expected a loose per-sector FITS file, not a zip"
    print("load_tess_cutouts: writes/reads a loose per-sector FITS file, downloads once -- OK")

    # second call: already cached -- must NOT download again
    with mock.patch.object(data_io, "load_star_gaia_data", return_value=gaia_row), \
         mock.patch.object(data_io, "_download_tess_cutouts", side_effect=fake_download) as mock_download2:
        result2 = data_io.load_tess_cutouts(star_name, cut_width, cut_height, star_directory, sector=sector)
    assert mock_download2.call_count == 0, "expected no download on a warm cache"
    assert int(result2[sector][0].header["SECTOR"]) == sector
    print("load_tess_cutouts: warm cache reuses the loose file, no redownload -- OK")


def test_load_tess_cutouts_migrates_from_legacy_zip(tmp_dir):
    star_directory = tmp_dir
    star_name = "Legacy Star"
    nospace = "Legacy_Star"
    cut_width = cut_height = 21
    sector = 42

    gaia_row = {"ra": 20.0, "dec": 15.0, "source_id": 456, "phot_rp_mean_mag": 11.0}
    cutouts_dir = os.path.join(star_directory, nospace, f"{cut_width}x{cut_height}")
    os.makedirs(cutouts_dir, exist_ok=True)
    legacy_zip = os.path.join(cutouts_dir, f"{nospace}.zip")

    hdul = _make_cutout_hdul(sector)
    buf_path = os.path.join(cutouts_dir, "_tmp.fits")
    hdul.writeto(buf_path, overwrite=True)
    with zipfile.ZipFile(legacy_zip, "w") as archive:
        archive.write(buf_path, arcname=f"tess-sector{sector:04d}-cutout.fits")
    os.remove(buf_path)
    zip_mtime_before = os.path.getmtime(legacy_zip)
    zip_size_before = os.path.getsize(legacy_zip)

    with mock.patch.object(data_io, "load_star_gaia_data", return_value=gaia_row), \
         mock.patch.object(data_io, "_download_tess_cutouts") as mock_download:
        result = data_io.load_tess_cutouts(star_name, cut_width, cut_height, star_directory, sector=sector)

    assert mock_download.call_count == 0, "expected NO network download -- the sector was in the legacy zip"
    assert int(result[sector][0].header["SECTOR"]) == sector
    new_path = data_io._sector_fits_path(cutouts_dir, nospace, sector)
    assert os.path.isfile(new_path), "expected the sector to be extracted into the new loose-file layout"
    assert os.path.isfile(legacy_zip), "the legacy zip must not be deleted"
    assert os.path.getmtime(legacy_zip) == zip_mtime_before and os.path.getsize(legacy_zip) == zip_size_before, (
        "the legacy zip must not be modified")
    print("load_tess_cutouts: migrates a needed sector out of a legacy zip with no network, zip left untouched -- OK")


def test_run_query_async_dispatches_to_launch_job_async():
    class FakeColumn:
        def __init__(self, values):
            self.values = values

    class FakeTable:
        def to_pandas(self):
            import pandas as pd
            # source_id as float64 (the real "TAP/VOTable round-trip lost precision"
            # case _run_query's Int64 normalization exists for -- see its docstring)
            return pd.DataFrame({"SOURCE_ID": [123456789012345678.0], "ra": [10.0], "dec": [-5.0]})

    class FakeJob:
        def get_results(self):
            return FakeTable()

    fake_tap = mock.Mock()
    fake_tap.launch_job = mock.Mock(side_effect=AssertionError("sync launch_job should NOT be called"))
    fake_tap.launch_job_async = mock.Mock(return_value=FakeJob())

    df = _run_query(fake_tap, "select top 1 * from gaiadr3.gaia_source", async_query=True)
    assert fake_tap.launch_job_async.called and not fake_tap.launch_job.called
    assert list(df.columns) == ["source_id", "ra", "dec"], "column lowercasing should still happen"
    assert str(df["source_id"].dtype) == "Int64", "source_id should still be normalized to Int64"
    print("_run_query(async_query=True): dispatches to launch_job_async, same post-processing -- OK")

    fake_tap2 = mock.Mock()
    fake_tap2.launch_job = mock.Mock(return_value=FakeJob())
    fake_tap2.launch_job_async = mock.Mock(side_effect=AssertionError("async launch_job_async should NOT be called"))
    _run_query(fake_tap2, "select top 1 * from gaiadr3.gaia_source")
    assert fake_tap2.launch_job.called and not fake_tap2.launch_job_async.called
    print("_run_query (default): still dispatches to sync launch_job -- OK")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir_1:
        test_load_tess_cutouts_writes_and_reads_loose_fits(tmp_dir_1)
    with tempfile.TemporaryDirectory() as tmp_dir_2:
        test_load_tess_cutouts_migrates_from_legacy_zip(tmp_dir_2)
    test_run_query_async_dispatches_to_launch_job_async()

    print("\nALL CHECKS PASSED")
