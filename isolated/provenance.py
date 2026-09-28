"""
Provenance headers for output files: git commit hash, generation timestamp,
parameters/masks, point count (ROADMAP.md Этап 10: "В заголовок каждого
выходного CSV и .dat: git-хеш, дата, параметры, маски, число точек"). Not a
Julia port.

Only two functions in the whole package read a CSV back (`data_io.py`'s
`_read_gaia_csv`/`read_light_curve_csv` -- everything else goes through
them), and nothing reads a `.dat` file back at all (those are for a human or
another program, e.g. CLEAN/LS comparisons) -- so a `#`-prefixed comment
header ahead of the data is safe everywhere: `np.savetxt`'s own `header=`
already adds the `#` prefix; `write_csv_with_provenance` below does the same
for a `pandas` CSV, and the two central readers pass `comment="#"` to skip
it transparently (also skips right past an older cache file that has no
header at all -- nothing to skip there, same result as before).
"""
from __future__ import annotations

import datetime
import subprocess


def git_commit_hash(short: bool = True) -> str:
    """
    Current git commit hash, or `"unknown"` if this isn't a git checkout, git
    isn't installed, or anything else goes wrong -- never raises (a missing
    git hash shouldn't ever be a reason the actual output can't be written).
    """
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def provenance_header(n_points: int | None = None, **fields) -> list[str]:
    """
    Standard provenance lines for an output file: git commit hash, UTC
    generation timestamp, point count (if given), then one "key: value" line
    per `fields` entry, in the order given -- whatever parameters/masks
    produced this particular file (e.g. `aperture_radius=3,
    quality_bitmask=175`). Returns plain lines, NOT yet `#`-prefixed -- pass
    to `np.savetxt`'s `header=` (which adds the prefix itself) or to
    `write_csv_header`/`write_csv_with_provenance` below.
    """
    lines = [f"git commit: {git_commit_hash()}",
             f"generated (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
    if n_points is not None:
        lines.append(f"N points: {n_points}")
    lines.extend(f"{key}: {value}" for key, value in fields.items())
    return lines


def write_csv_header(path: str, lines: list[str]) -> None:
    """Write `lines` as `#`-prefixed comment lines at the start of `path` (mode "w")."""
    with open(path, "w") as fo:
        for line in lines:
            fo.write(f"# {line}\n")


def write_csv_with_provenance(df, path: str, index: bool = False, **fields) -> str:
    """
    Write `df` to `path` as CSV with a provenance header (`provenance_header`,
    `n_points=len(df)`) before the data. Returns `path`. Readers must use
    `comment="#"` (already the case for `data_io.read_light_curve_csv`/
    `_read_gaia_csv`) to skip the header transparently.
    """
    write_csv_header(path, provenance_header(n_points=len(df), **fields))
    df.to_csv(path, mode="a", index=index)
    return path
