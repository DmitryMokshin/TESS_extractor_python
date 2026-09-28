"""
Test for `isolated.lightcurve_tools.exclude_frame_windows`'s frame-number
fix (ROADMAP.md Этап 1's acceptance criterion: "вырезать окна после
удаления части строк, и удаляются именно нужные кадры").

Plain script with asserts -- matches the other tests in this directory
(no pytest dependency yet). Run directly:
    .venv/bin/python tests/test_frame_windows.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isolated.lightcurve_tools import exclude_frame_windows


def main():
    # --- new-format curve: FRAME already has a gap, as if frames 6-9 were
    # excised by an earlier cleaning pass. Row position no longer equals
    # frame number here -- exactly the bug this column exists to fix.
    frame = np.array([1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15])
    df_new = pd.DataFrame({"FRAME": frame, "BTJD": frame * 0.1, "FLUX": np.arange(len(frame), dtype=float)})

    # Cut frames 11-13 (by FRAME value): rows at positions 6,7,8 (0-based),
    # i.e. frame values 11,12,13 -- NOT rows at positions 11,12,13, which
    # don't even exist in this 11-row frame.
    clean, trash = exclude_frame_windows(df_new, [[11, 13]])
    assert sorted(trash["FRAME"].tolist()) == [11, 12, 13], (
        f"expected trash frames [11,12,13], got {sorted(trash['FRAME'].tolist())}")
    assert sorted(clean["FRAME"].tolist()) == [1, 2, 3, 4, 5, 10, 14, 15], (
        f"expected clean frames [1,2,3,4,5,10,14,15], got {sorted(clean['FRAME'].tolist())}")
    print("new-format (FRAME column, gap already present): cut the correct frames -- OK")

    # A naive position-based cut (the old bug) would instead have cut
    # positions 11-13, which don't exist in an 11-row frame -- confirm that
    # is NOT what happened (nothing left over 15, no IndexError, no
    # accidental no-op).
    assert len(clean) + len(trash) == len(df_new)
    assert 15 in clean["FRAME"].to_numpy() and 14 in clean["FRAME"].to_numpy()

    # --- old-format curve: no FRAME column at all (pre-Этап-1 cache file).
    # Falls back to the original position-based convention: row i (0-based)
    # is frame i+1, exactly like before this change.
    df_old = pd.DataFrame({"MJD": np.arange(10) * 0.1, "FLUX": np.arange(10, dtype=float)})
    clean_old, trash_old = exclude_frame_windows(df_old, [[3, 5]])
    assert trash_old["FLUX"].tolist() == [2.0, 3.0, 4.0], (
        f"expected positions 3-5 (0-based values 2,3,4) removed, got {trash_old['FLUX'].tolist()}")
    assert len(clean_old) == 7
    print("old-format (no FRAME column): still cuts by row position -- OK")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
