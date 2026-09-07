"""
isolated: Python port of the dmitrievdv/Isolated Julia project
(TESS light curves of isolated young stars via Gaia-PRF-model photometry).

See README.md for the file-by-file mapping to the original .jl files.
"""

from .config import STAR_DIRECTORY, TESS_MAX_SECTORS
from .geometry import (
    get_nospace_star_name, get_true_radec, get_rel_radec, get_distance,
    calc_tess_magnitude, calc_tess_flux_from_mag, get_true_jd,
)
from .tess_point import tess_stars2px_sector, is_in_sector, find_tess_sectors
from .data_io import (
    load_star_gaia_data, load_tess_cutouts, load_gaia_stars_in_view_data,
    load_light_curve, create_gaia_datafiles, get_star_tesscut_fits,
    extract_tess_cutouts, get_tess_sectors_from_file,
)
from .lightcurve_tools import (
    delete_nans, box_smooth, clean_flux_sigma, clean_flux, find_sampling,
    find_acf, save_lc_figure, get_all_data,
    exclude_frame_windows, pick_exclusion_windows, save_clean_light_curve, save_trash_light_curve,
)
from .stats import calc_noise_rms, find_period, calc_periodicity, calc_asymmetry, cleaned_jds_mags
from .eclipse import calc_intersect_area, calc_eclipse_magnitude, calc_transit_mag
from .viewer import plot_cuts, save_cutout_video
from .periodogram import (
    compute_ls_periodogram, find_periodogram_peaks, save_periodogram, plot_periodogram,
    fit_multi_sine, fit_periodogram_approximation, plot_periodogram_approximation,
    suggest_period_range,
)
from .stray_light import (
    pick_stripe_pixels, stray_light_monitor_curve, plot_stray_light_diagnostics, correct_stray_light,
)

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
    "exclude_frame_windows", "pick_exclusion_windows", "save_clean_light_curve", "save_trash_light_curve",
    "calc_noise_rms", "find_period", "calc_periodicity", "calc_asymmetry", "cleaned_jds_mags",
    "calc_intersect_area", "calc_eclipse_magnitude", "calc_transit_mag",
    "plot_cuts", "save_cutout_video",
    "compute_ls_periodogram", "find_periodogram_peaks", "save_periodogram", "plot_periodogram",
    "fit_multi_sine", "fit_periodogram_approximation", "plot_periodogram_approximation",
    "suggest_period_range",
    "pick_stripe_pixels", "stray_light_monitor_curve", "plot_stray_light_diagnostics", "correct_stray_light",
]
