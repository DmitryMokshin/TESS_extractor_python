"""
Test for the network-robustness helpers of `isolated/databases.py`, with fake
TAP clients (no network):
  - `_simbad_name_candidates`: "EM AS 14" also tries Simbad's "EM* AS 14" spelling
    (and V*/* prefixes); an already-starred name or a one-word name is left alone;
  - `_simbad_ident_df`: rows of the spelling closest to the typed name come first;
    a star Simbad doesn't know gives a clear ValueError (not an IndexError later);
  - `_simbad`: skips an unreachable mirror and uses the next one; if none is
    reachable -- ConnectionError with a helpful message, state reset for the next try;
  - `_run_simbad`: a connection error mid-query moves on to the next mirror;
  - `_launch_job`: a "busy" 503 is retried (as ASYNC), other HTTP errors are not.

Plain script with asserts. Run directly:
    .venv/bin/python tests/test_databases_synthetic.py
"""
import os
import sys

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isolated import databases as db


class FakeTap:
    def __init__(self, url):
        self.url = url


def reset_simbad_state():
    db._simbad_tap = None
    db._simbad_url = None
    db._simbad_bad_urls.clear()


def test_name_candidates():
    assert db._simbad_name_candidates("EM AS 14") == ["EM AS 14", "EM* AS 14", "V* EM AS 14", "* EM AS 14"]
    assert db._simbad_name_candidates("  SS   397 ")[0] == "SS 397"
    assert db._simbad_name_candidates("EM* AS 14") == ["EM* AS 14"]
    assert db._simbad_name_candidates("TIC12345") == ["TIC12345"]
    print("_simbad_name_candidates: starred spellings tried, as-typed first -- OK")


def test_ident_df_order_and_unknown():
    seen = {}

    def fake_run(adql, desc="", async_query=False):
        seen["adql"] = adql
        return pd.DataFrame({"ids": ["b|Gaia DR3 2", "a|Gaia DR3 1"], "matched_id": ["EM* AS   14", "EM AS 14"]})

    orig = db._run_simbad
    db._run_simbad = fake_run
    try:
        df = db._simbad_ident_df("oid, ids", "JOIN basic ON ident.oidref = basic.oid", "EM AS 14")
        assert df["matched_id"].tolist() == ["EM AS 14", "EM* AS   14"], "as-typed spelling must come first"
        assert "'EM* AS 14'" in seen["adql"] and "'EM AS 14'" in seen["adql"]
        db._simbad_ident_df("oid", "", "O'Neil 1")
        assert "'O''Neil 1'" in seen["adql"], "single quotes must be doubled in ADQL"

        db._run_simbad = lambda adql, desc="", async_query=False: pd.DataFrame({"ids": [], "matched_id": []})
        try:
            db._simbad_ident_df("oid", "", "Zzz 99")
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "Zzz 99" in str(exc) and "Simbad" in str(exc)
    finally:
        db._run_simbad = orig
    print("_simbad_ident_df: as-typed spelling first, quotes escaped, unknown star -> clear ValueError -- OK")


def test_mirror_choice_and_failover():
    orig_reach, orig_tap, orig_query = db._reachable, db.TapPlus, db._run_query
    db.TapPlus = lambda url: FakeTap(url)
    try:
        reset_simbad_state()
        db._reachable = lambda url, timeout=0: url == db.SIMBAD_TAP_URLS[1]
        tap = db._simbad()
        assert tap.url == db.SIMBAD_TAP_URLS[1], "unreachable first mirror must be skipped"
        assert db._simbad() is tap, "the chosen mirror is cached"

        reset_simbad_state()
        db._reachable = lambda url, timeout=0: False
        try:
            db._simbad()
            assert False, "expected ConnectionError"
        except ConnectionError as exc:
            assert "Simbad" in str(exc)
        assert not db._simbad_bad_urls, "state must reset after total failure so a later retry can succeed"

        # mid-session drop on the first mirror -> second mirror answers
        reset_simbad_state()
        db._reachable = lambda url, timeout=0: True
        calls = []

        def fake_query(tap, adql, desc="", async_query=False):
            calls.append(tap.url)
            if tap.url == db.SIMBAD_TAP_URLS[0]:
                raise TimeoutError("Connection timed out")
            return pd.DataFrame({"x": [1]})

        db._run_query = fake_query
        df = db._run_simbad("select 1")
        assert calls == db.SIMBAD_TAP_URLS and df["x"].tolist() == [1], f"failover order wrong: {calls}"
    finally:
        db._reachable, db.TapPlus, db._run_query = orig_reach, orig_tap, orig_query
        reset_simbad_state()
    print("_simbad/_run_simbad: unreachable mirror skipped, mid-query drop fails over -- OK")


def test_busy_retry():
    class Tap:
        def __init__(self, sync_error):
            self.sync_error, self.log = sync_error, []

        def launch_job(self, adql):
            self.log.append("sync")
            raise requests.exceptions.HTTPError(self.sync_error)

        def launch_job_async(self, adql):
            self.log.append("async")
            return "JOB"

    orig_sleep, orig_delays = db.time.sleep, db.RETRY_DELAYS
    db.time.sleep = lambda s: None
    try:
        tap = Tap("Error 503:\nServer overloaded: synchronous job rejected")
        assert db._launch_job(tap, "q", False) == "JOB"
        assert tap.log == ["sync", "async"], f"expected sync then async retry, got {tap.log}"

        tap = Tap("Error 400:\nbad ADQL")
        try:
            db._launch_job(tap, "q", False)
            assert False, "a non-busy HTTP error must propagate"
        except requests.exceptions.HTTPError:
            assert tap.log == ["sync"], "non-busy errors must not be retried"
    finally:
        db.time.sleep, db.RETRY_DELAYS = orig_sleep, orig_delays
    print("_launch_job: busy 503 retried as async, other errors raised at once -- OK")


if __name__ == "__main__":
    test_name_candidates()
    test_ident_df_order_and_unknown()
    test_mirror_choice_and_failover()
    test_busy_retry()
    print("\nALL CHECKS PASSED")
