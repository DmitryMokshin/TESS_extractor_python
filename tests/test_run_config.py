"""
Test for `run_config.RunConfig` (ROADMAP.md Этап 4: unified pipeline config).

Plain script with asserts, matching the other tests in this directory. Run:
    .venv/bin/python tests/test_run_config.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_config import RunConfig


def main():
    prf_cfg = RunConfig(photometry_mode="prf")
    assert prf_cfg.lc_suffix == "_prf_clean", prf_cfg.lc_suffix

    aperture_cfg = RunConfig(photometry_mode="aperture")
    assert aperture_cfg.lc_suffix == "_clean", aperture_cfg.lc_suffix
    print("lc_suffix resolves correctly for both photometry modes -- OK")

    try:
        RunConfig(photometry_mode="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an invalid photometry_mode")
    print("invalid photometry_mode rejected -- OK")

    cfg = RunConfig(star_name="EM* AS 14", cut_width=15, cut_height=15)
    assert cfg.star_dir == "stars_python/EM*_AS_14/15x15", cfg.star_dir
    print("star_dir builds the expected path for a star name with a space -- OK")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
