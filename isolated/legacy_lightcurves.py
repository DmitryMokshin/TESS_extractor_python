"""
Legacy simple-aperture light-curve extraction.

Ported from lightcurves.jl. NOTE: this module was already superseded in the
Julia project by the Gaia-PRF-model pipeline in main.jl/data_io.py -- the
`include("lightcurves.jl")` line in main.jl is commented out. It's ported
here only for completeness; prefer `isolated.data_io.load_light_curve` for
new work.

Ported functions:
    findbkg, findstar, getlc, cleanlc, uniformlc, tessmag, mergelc, get_lcs
"""
from __future__ import annotations

import os
import zipfile
import io

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from astropy.time import Time

from .config import STAR_DIRECTORY
from .geometry import get_nospace_star_name


def _negative_to_nan(x):
    x = np.asarray(x, dtype=float)
    return np.where(x <= 0, np.nan, x)


def _nan_to_big(x):
    return np.where(np.isnan(x), 1e100, x)


def findbkg(cut, aperture_size):
    """Direct port of `findbkg`: darkest aperture_size-radius square box in the cut."""
    width, height = cut.shape
    aperture_flux = np.zeros((width - 2 * aperture_size, height - 2 * aperture_size))
    for i in range(aperture_size, width - aperture_size):
        for j in range(aperture_size, height - aperture_size):
            box = cut[i - aperture_size:i + aperture_size + 1, j - aperture_size:j + aperture_size + 1]
            aperture_flux[i - aperture_size, j - aperture_size] = np.sum(_nan_to_big(box))
    idx = np.unravel_index(np.argmin(aperture_flux), aperture_flux.shape)
    ix, iy = idx[0] + aperture_size, idx[1] + aperture_size
    return (slice(ix - aperture_size, ix + aperture_size + 1),
            slice(iy - aperture_size, iy + aperture_size + 1))


def findstar(cut, aperture_size, search_box):
    """Direct port of `findstar`: brightest aperture_size-radius square box near the cut center."""
    width, height = cut.shape
    center_x = (width + 1) // 2
    center_y = (height + 1) // 2
    aperture_flux = np.zeros((2 * search_box + 1, 2 * search_box + 1))
    no_search_width = max(center_x - 1 - search_box, aperture_size)

    for i in range(1, 2 * search_box + 2):
        for j in range(1, 2 * search_box + 2):
            x0 = i + no_search_width - aperture_size - 1
            x1 = i + no_search_width + aperture_size
            y0 = j + no_search_width - aperture_size - 1
            y1 = j + no_search_width + aperture_size
            aperture_flux[i - 1, j - 1] = np.sum(cut[x0:x1, y0:y1])

    idx = np.unravel_index(np.argmax(aperture_flux), aperture_flux.shape)
    ix = idx[0] + 1 + no_search_width
    iy = idx[1] + 1 + no_search_width
    return (slice(ix - aperture_size - 1, ix + aperture_size),
            slice(iy - aperture_size - 1, iy + aperture_size))


def tessmag(flux):
    """Direct port of `tessmag`."""
    return -2.5 * np.log10(flux) + 20.44


def getlc(star, cut_fits, sector=0, star_directory=STAR_DIRECTORY):
    """Direct port of `getlc`."""
    cuts = _negative_to_nan(cut_fits[1].data["FLUX"])  # (n_cuts, height, width)
    n_cuts, height, width = cuts.shape
    n_bkg = n_cuts // 4

    jd = cut_fits[1].data["TIME"] + 2457000

    aperture_size = 2
    raw_flux = np.zeros(n_cuts)
    bkg_flux = np.zeros(n_cuts)
    flux = np.zeros(n_cuts)

    def frame(i):
        return cuts[i].T  # -> (width, height)

    for i in range(n_cuts):
        c = frame(i)
        ap = findstar(c, aperture_size, 2)
        bkg_ap = findbkg(c, aperture_size)
        raw_flux[i] = np.sum(c[ap])
        bkg_flux[i] = np.sum(c[bkg_ap])
        flux[i] = _negative_to_nan(np.array([raw_flux[i] - bkg_flux[i]]))[0]

    out_dir = os.path.join(star_directory, star)
    os.makedirs(out_dir, exist_ok=True)

    ap = findstar(frame(n_bkg), aperture_size, 2)
    bkg_ap = findbkg(frame(n_bkg), aperture_size)

    fig, ax = plt.subplots()
    ax.imshow(np.log10(frame(n_bkg)).T, origin="lower")
    for sl, color, label in [(bkg_ap, "tab:orange", "bkg"), (ap, "tab:blue", "star")]:
        x0, x1 = sl[0].start + 0.5, sl[0].stop - 0.5
        y0, y1 = sl[1].start + 0.5, sl[1].stop - 0.5
        ax.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0], lw=2, color=color, label=label)
    ax.legend()
    fig.savefig(os.path.join(out_dir, f"{sector}-map.pdf"))
    plt.close(fig)

    with open(os.path.join(out_dir, f"{sector}-rawlc.dat"), "w") as f:
        f.write("#jd flux raw_flux bkg_flux\n")
        for n in range(n_cuts):
            f.write(f"{jd[n]:15.6f} {flux[n]:12.3e} {raw_flux[n]:12.3e} {bkg_flux[n]:12.3e}\n")

    return jd.astype(float), flux.astype(float)


def cleanlc(jd, flux):
    """Direct port of `cleanlc`: drop isolated (< 10-point) runs of good data around NaN gaps."""
    jd = np.asarray(jd, dtype=float)
    flux = np.asarray(flux, dtype=float)
    n = len(jd)
    non_nan_index = -1
    nan_index = -1
    clean_jd, clean_flux = [], []

    for i in range(n):
        if not np.isnan(flux[i]) and nan_index == non_nan_index:
            non_nan_index = i
        elif np.isnan(flux[i]):
            nan_index = i
            if non_nan_index != -1 and (nan_index - non_nan_index) > 10:
                clean_jd.extend(jd[non_nan_index:nan_index + 1])
                clean_flux.extend(flux[non_nan_index:nan_index + 1])
            non_nan_index = i

    if non_nan_index != -1 and (n - 1 - non_nan_index) > 10:
        clean_jd.extend(jd[non_nan_index:n])
        clean_flux.extend(flux[non_nan_index:n])

    clean_jd = np.array(clean_jd)
    clean_flux = np.array(clean_flux)
    if len(clean_jd) > 0 and np.isnan(clean_jd[-1]):
        return clean_jd[:-1], clean_flux[:-1]
    return clean_jd, clean_flux


def uniformlc(jd, flux, eps_jd=1.5, eps_flux=0.1):
    """Direct port of `uniformlc`."""
    jd = np.asarray(jd, dtype=float)
    flux = np.asarray(flux, dtype=float)
    n = len(jd)
    uniform_jd, uniform_flux = [], []
    jd_step = jd[1] - jd[0]
    flux_step = flux[1] - flux[0]

    for i in range(1, n):
        new_jd_step = jd[i] - jd[i - 1]
        new_flux_step = flux[i] - flux[i - 1]

        if (new_jd_step / jd_step > eps_jd) or (jd_step / new_jd_step > eps_jd):
            uniform_jd.append(np.nan)
            uniform_flux.append(np.nan)
        elif abs(new_flux_step) / flux[i - 1] > eps_flux:
            handled = False
            if i < n - 1:
                if abs(flux[i + 1] - flux[i - 1]) / flux[i - 1] < eps_flux:
                    uniform_jd.append(jd[i])
                    uniform_flux.append((flux[i + 1] + flux[i - 1]) / 2)
                    handled = True
            if not handled:
                uniform_jd.append(np.nan)
                uniform_flux.append(np.nan)
        else:
            uniform_jd.append(jd[i])
            uniform_flux.append(flux[i])
            if np.isnan(uniform_flux[-1]):
                uniform_jd[-1] = np.nan

        jd_step = new_jd_step
        flux_step = new_flux_step

    return np.array(uniform_jd), np.array(uniform_flux)


def mergelc(jd1, flux1, jd2, flux2):
    """Direct port of `mergelc`."""
    jd1, flux1 = np.asarray(jd1, dtype=float), np.asarray(flux1, dtype=float)
    jd2, flux2 = np.asarray(jd2, dtype=float), np.asarray(flux2, dtype=float)
    if len(jd1) == 0:
        return jd2, flux2
    if len(jd2) == 0:
        return jd1, flux1

    if jd1[-1] < jd2[0]:
        first_jd, first_flux, second_jd, second_flux = jd1, flux1, jd2, flux2
    else:
        first_jd, first_flux, second_jd, second_flux = jd2, flux2, jd1, flux1

    jd = np.concatenate([first_jd, [second_jd[0]], second_jd])
    flux = np.concatenate([first_flux, [np.nan], second_flux])
    return jd, flux


def get_lcs(star_name, star_directory=STAR_DIRECTORY):
    """Direct port of `get_lcs`."""
    nospace = get_nospace_star_name(star_name)
    zip_path = os.path.join(star_directory, nospace, f"{nospace}.zip")

    jd_all = np.array([], dtype=float)
    flux_all = np.array([], dtype=float)

    with zipfile.ZipFile(zip_path) as archive:
        for entry_name in archive.namelist():
            with archive.open(entry_name) as f:
                cut_fits = pyfits.open(io.BytesIO(f.read()))
            sector = int(entry_name[6:10])  # Julia entry_name[7:10], 1-based -> 0-based slice[6:10]

            jd, flux = getlc(nospace, cut_fits, sector=sector, star_directory=star_directory)
            jd, flux = uniformlc(jd, flux)
            jd, flux = cleanlc(jd, flux)
            mag = -2.5 * np.log10(flux) + 20.44

            out_dir = os.path.join(star_directory, nospace)
            with open(os.path.join(out_dir, f"{sector}-lc.dat"), "w") as fout:
                fout.write("#jd flux\n")
                for n in range(len(jd)):
                    fout.write(f"{jd[n]:15.6f} {flux[n]:15.6e} {mag[n]:15.6f}\n")

            if len(jd) > 0:
                fig, ax = plt.subplots()
                ax.plot(jd, tessmag(flux))
                ax.invert_yaxis()
                ax.set_ylabel("TESS magnitude")
                int_start = int(np.ceil(jd[0] / 5) * 5)
                int_end = int(np.floor(jd[-1] / 5) * 5)
                ticks = np.arange(int_start, int_end + 1, 5)
                ax.set_xticks(ticks)
                ax.set_xticklabels([Time(t, format="jd").datetime.strftime("%-d %b %Y") for t in ticks],
                                    rotation=45, ha="right")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"{sector}-lc.pdf"))
                fig.savefig(os.path.join(out_dir, f"{sector}-lc.png"))
                plt.close(fig)

                jd_all, flux_all = mergelc(jd_all, flux_all, jd, flux)

    return jd_all, flux_all
