"""
Test for `isolated/provenance.py` (ROADMAP.md Этап 10: "В заголовок каждого
выходного CSV и .dat: git-хеш, дата, параметры, маски, число точек").

Checks:
  - `git_commit_hash` never raises, even outside a git repo.
  - `provenance_header` includes git hash, timestamp, point count, and every
    passed field, in order.
  - `write_csv_with_provenance` writes a `#`-prefixed header block before the
    data, and the result round-trips through `data_io.read_light_curve_csv`/
    `_read_gaia_csv` (both updated to `comment="#"`) with the DATA unchanged.
  - a PRE-Этап-10 file (no header at all) still reads correctly through
    those same functions -- the point of `comment="#"` being purely additive.

Plain script with asserts -- matches the other tests in this directory
(no pytest dependency yet). Run directly:
    .venv/bin/python tests/test_provenance_synthetic.py
"""
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isolated.provenance import git_commit_hash, provenance_header, write_csv_with_provenance
from isolated.data_io import read_light_curve_csv, _read_gaia_csv


def test_git_commit_hash_never_raises(tmp_dir):
    # inside this repo: a real short hash
    h = git_commit_hash()
    assert h != "" and h != "unknown", f"expected a real hash inside the repo, got {h!r}"

    # outside any git repo: must degrade to "unknown", not raise
    empty_dir = os.path.join(tmp_dir, "not_a_repo")
    os.makedirs(empty_dir, exist_ok=True)
    old_cwd = os.getcwd()
    try:
        os.chdir(empty_dir)
        h_outside = git_commit_hash()
    finally:
        os.chdir(old_cwd)
    assert h_outside == "unknown", f"expected 'unknown' outside a git repo, got {h_outside!r}"
    print("git_commit_hash: real hash inside the repo, 'unknown' (no raise) outside -- OK")


def test_provenance_header_contents():
    lines = provenance_header(n_points=42, star_name="SS 397", sector=80)
    assert any(line.startswith("git commit: ") for line in lines)
    assert any(line.startswith("generated (UTC): ") for line in lines)
    assert "N points: 42" in lines
    assert "star_name: SS 397" in lines
    assert "sector: 80" in lines
    # order: git hash, timestamp, N points, then fields in the order given
    assert lines.index("N points: 42") < lines.index("star_name: SS 397") < lines.index("sector: 80")
    print("provenance_header: git hash, timestamp, N points, fields in order -- OK")


def test_write_csv_with_provenance_roundtrips(tmp_dir):
    df = pd.DataFrame({"BTJD": [1.0, 2.0, 3.0], "FRAME": [1, 2, 3], "FLUX": [100.0, 101.0, 102.0]})
    path = os.path.join(tmp_dir, "light_curve_sector_80_clean.csv")
    write_csv_with_provenance(df, path, star_name="SS 397", sector=80, aperture_radius=3)

    with open(path) as fo:
        raw_lines = fo.readlines()
    header_lines = [l for l in raw_lines if l.startswith("#")]
    assert len(header_lines) >= 4, f"expected at least git/date/N/field header lines, got {header_lines}"
    assert any("aperture_radius: 3" in l for l in header_lines)

    read_back = read_light_curve_csv(path)
    assert list(read_back.columns) == ["BTJD", "FRAME", "FLUX"]
    assert read_back["FLUX"].tolist() == [100.0, 101.0, 102.0]
    print("write_csv_with_provenance + read_light_curve_csv: header written, data round-trips unchanged -- OK")


def test_gaia_csv_roundtrips_with_provenance(tmp_dir):
    df = pd.DataFrame({"SOURCE_ID": [123, 456], "ra": [10.0, 11.0], "dec": [-5.0, -4.0]})
    path = os.path.join(tmp_dir, "gaia_stars_in_view_sector_80.csv")
    write_csv_with_provenance(df, path, star_name="SS 397", sector=80)
    read_back = _read_gaia_csv(path)
    assert list(read_back.columns) == ["source_id", "ra", "dec"], "column lowercasing must still happen"
    assert read_back["source_id"].tolist() == [123, 456]
    print("write_csv_with_provenance + _read_gaia_csv: header written, lowercasing still works -- OK")


def test_pre_etap10_file_without_header_still_reads(tmp_dir):
    # a plain CSV with no '#' header at all -- simulates a cache file
    # written before this stage; comment="#" must be purely additive
    path = os.path.join(tmp_dir, "light_curve_sector_80_old.csv")
    pd.DataFrame({"BTJD": [5.0, 6.0], "FRAME": [1, 2], "FLUX": [50.0, 51.0]}).to_csv(path, index=False)
    read_back = read_light_curve_csv(path)
    assert read_back["FLUX"].tolist() == [50.0, 51.0]
    print("read_light_curve_csv: a header-less pre-Этап-10 file still reads correctly -- OK")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_git_commit_hash_never_raises(tmp_dir)
        test_provenance_header_contents()
        test_write_csv_with_provenance_roundtrips(tmp_dir)
        test_gaia_csv_roundtrips_with_provenance(tmp_dir)
        test_pre_etap10_file_without_header_still_reads(tmp_dir)

    print("\nALL CHECKS PASSED")
