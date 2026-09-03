"""
isolated: Python port of the dmitrievdv/Isolated Julia project
(TESS light curves of isolated young stars via Gaia-PRF-model photometry).

See README.md for the file-by-file mapping to the original .jl files.
"""

from isolated.config import STAR_DIRECTORY, TESS_MAX_SECTORS
from isolated.geometry import (
    get_nospace_star_name, get_true_radec, get_rel_radec, get_distance,
    calc_tess_magnitude, calc_tess_flux_from_mag, get_true_jd,
)
from isolated.tess_point import tess_stars2px_sector, is_in_sector, find_tess_sectors
from isolated.data_io import (
    load_star_gaia_data, load_tess_cutouts, load_gaia_stars_in_view_data,
    load_light_curve, create_gaia_datafiles, get_star_tesscut_fits,
    extract_tess_cutouts, get_tess_sectors_from_file,
)
from isolated.lightcurve_tools import (
    delete_nans, box_smooth, clean_flux_sigma, clean_flux, find_sampling,
    find_acf, save_lc_figure, get_all_data,
)
from isolated.stats import calc_noise_rms, find_period, calc_periodicity, calc_asymmetry
from isolated.eclipse import calc_intersect_area, calc_eclipse_magnitude, calc_transit_mag

__all__ = [
    "STAR_DIRECTORY", "TESS_MAX_SECTORS",
    "get_nospace_star_name", "get_true_radec", "get_rel_radec", "get_distance",
    "calc_tess_magnitude", "calc_tess_flux_from_mag", "get_true_jd",
    "tess_stars2px_sector", "is_in_sector", "find_tess_sectors",
    "load_star_gaia_data", "load_tess_cutouts", "load_gaia_stars_in_view_data",
    "load_light_curve", "create_gaia_datafiles", "get_star_tesscut_fits",
    "extract_tess_cutouts", "get_tess_sectors_from_file",
    "delete_nans", "box_smooth", "clean_flux_sigma", "clean_flux", "find_sampling",
    "find_acf", "save_lc_figure", "get_all_data",
    "calc_noise_rms", "find_period", "calc_periodicity", "calc_asymmetry",
    "calc_intersect_area", "calc_eclipse_magnitude", "calc_transit_mag",
]
