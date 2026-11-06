"""
Test for `isolated/fits_utils.py`'s cutout-WCS readers with synthetic in-memory
cutouts (no files, no network):
  - NEW-format cutout (WCS in the PIXELS header as `1CRPX4`, `1CRVL4`, `11PC4`, ...):
    read from there, as before;
  - OLD-format cutout (e.g. sector 18: no `*4` keywords in PIXELS, the same WCS only in
    the APERTURE header as `CRPIX1`, `CRVAL1`, `PC1_1`, ...): same numbers, same corners;
  - both present: the PIXELS ones win (existing behaviour unchanged);
  - neither present: a clear KeyError naming both places.

Plain script with asserts. Run directly:
    .venv/bin/python tests/test_fits_utils_synthetic.py
"""
import os
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isolated.fits_utils import (get_reference_px, get_reference_radec, get_px_to_radec_matrix,
                                 get_tesscut_corners)

REF_PX = (25.3, 25.5)
REF_RADEC = (19.0, 59.16)
PC = ((-0.0045885, 0.0034903), (-0.0035442, -0.0046339))


def make_cutout(pixels_wcs=True, aperture_wcs=True, pixels_wcs_scale=1.0):
    flux = np.zeros((4, 50, 50), dtype=np.float32)
    table = fits.BinTableHDU.from_columns([fits.Column(name="FLUX", format="2500E", dim="(50,50)", array=flux)])
    if pixels_wcs:
        h = table.header
        h["1CRPX4"], h["2CRPX4"] = (v * pixels_wcs_scale for v in REF_PX)
        h["1CRVL4"], h["2CRVL4"] = REF_RADEC
        h["11PC4"], h["12PC4"], h["21PC4"], h["22PC4"] = PC[0][0], PC[0][1], PC[1][0], PC[1][1]
    aperture = fits.ImageHDU(np.zeros((50, 50), dtype=np.int32))
    if aperture_wcs:
        h = aperture.header
        h["CRPIX1"], h["CRPIX2"] = REF_PX
        h["CRVAL1"], h["CRVAL2"] = REF_RADEC
        h["PC1_1"], h["PC1_2"], h["PC2_1"], h["PC2_2"] = PC[0][0], PC[0][1], PC[1][0], PC[1][1]
    return fits.HDUList([fits.PrimaryHDU(), table, aperture])


def test_new_and_old_format_agree():
    new = make_cutout(pixels_wcs=True, aperture_wcs=True)
    old = make_cutout(pixels_wcs=False, aperture_wcs=True)
    for f in (get_reference_px, get_reference_radec, get_px_to_radec_matrix):
        assert np.array_equal(f(new), f(old)), f"{f.__name__}: old-format fallback differs from new-format values"
    assert np.allclose(get_reference_px(old), REF_PX) and np.allclose(get_reference_radec(old), REF_RADEC)
    assert np.allclose(get_px_to_radec_matrix(old), PC)
    assert np.allclose(get_tesscut_corners(new), get_tesscut_corners(old))
    print("fits_utils: old-format (APERTURE-header) cutout gives the same WCS and corners as new-format -- OK")


def test_pixels_header_wins():
    both = make_cutout(pixels_wcs=True, aperture_wcs=True, pixels_wcs_scale=1.1)
    assert np.allclose(get_reference_px(both), np.array(REF_PX) * 1.1), "PIXELS-header value must take precedence"
    print("fits_utils: when both exist the PIXELS-header value is used (behaviour unchanged) -- OK")


def test_missing_everywhere():
    none = make_cutout(pixels_wcs=False, aperture_wcs=False)
    try:
        get_reference_px(none)
        assert False, "expected KeyError"
    except KeyError as exc:
        assert "1CRPX4" in str(exc) and "CRPIX1" in str(exc), f"message should name both places: {exc}"
    print("fits_utils: no WCS anywhere -> KeyError naming both headers -- OK")


if __name__ == "__main__":
    test_new_and_old_format_agree()
    test_pixels_header_wins()
    test_missing_everywhere()
    print("\nALL CHECKS PASSED")
