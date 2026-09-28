"""
Поиск частот по очищенной кривой блеска -- для одного сектора или для
нескольких склеенных секторов сразу (ROADMAP.md Этап 7: "Несколько
секторов"; см. `isolated/multisector.py`).

Что делается (по порядку):
 1. Периодограмма Ломба-Скаргла, как раньше, но с вычитанием медленного тренда
    (иначе тренд за сектор уходит в самые низкие частоты: пик на 1/T или 2/T —
    артефакт, а не период). Это быстрый обзор; FAP здесь "naive", т.е. для белого
    шума, и для статьи не годится.
 2. Выбеливание (isolated/prewhitening.py): частоты извлекаются по одной, после каждой
    все частоты, амплитуды, фазы и тренд переподгоняются вместе. Значимость —
    S/N >= 4 относительно реального уровня шума (Breger et al. 1993; модель
    красного шума Bowman et al. 2019). Ошибки — Montgomery & O'Donoghue (1999)
    с поправкой на корреляцию остатков. Амплитуды в mmag.
 3. Рисунки: амплитудный спектр (с уровнем 4×шум и спектральным окном --
    для нескольких секторов особенно важно: годовой разрыв дает сильные
    алиасы, окно показывает их явно), кривая блеска с моделью, динамический
    спектр.

Все файлы кладутся рядом с кривой блеска:
    periodogram_sector_{N}{суффикс}.dat / .{fmt}          — LS (как раньше)
    periodogram_sector_{N}{суффикс}_frequencies.dat       — таблица частот с ошибками и S/N
    periodogram_sector_{N}{суффикс}_amplitude.{fmt}       — амплитудный спектр
    periodogram_sector_{N}{суффикс}_fit.{fmt}             — кривая + модель
    periodogram_sector_{N}{суффикс}_dynamic.{fmt}         — динамический спектр
Для нескольких секторов -- то же самое, но `periodogram_sectors_{N1-N2-...}{суффикс}*`.
"""
import numpy as np

from isolated.data_io import read_light_curve_csv
from isolated.periodogram import compute_ls_periodogram_from_series, find_periodogram_peaks
from isolated.periodogram import save_periodogram, plot_periodogram
from isolated.periodogram import suggest_period_range
from isolated.stats import cleaned_jds_mags
from isolated.multisector import stitch_sector_csvs
from isolated.prewhitening import (prewhiten, format_frequency_table, save_frequency_table,
                                   plot_amplitude_spectrum, plot_prewhitening_fit,
                                   dynamic_spectrum, plot_dynamic_spectrum)
from isolated.provenance import provenance_header


def _analyse_prepared_series(t, y_raw, base, title, name_file_data, plot_star_name, plot_sector,
                             max_peaks_analyse=10, image_format="png",
                             nyquist_factor=2.0, max_period_fraction=0.5, detrend_deg=2,
                             prewhiten_fmax=10.0, snr_stop=4.0, n_max_freq=15,
                             dyn_window=10.0, dyn_step=0.25, dyn_frange=(0.8, 2.5),
                             make_ls_plot=True, y_ls=None, save_official_ls=False, cadr_width=None,
                             cadr_height=None):
    """
    Shared "given an already time-sorted (t, y_raw) series -- y_raw a mag
    difference from its own median, NOT yet scaled to mmag -- run the full
    LS-quick-look + prewhitening + 3-plot pipeline and save everything next
    to `base`" body, identical for one sector or several stitched together
    (`periodogram_compute_analise`/`periodogram_compute_analise_multisector`
    below): they only differ in how `(t, y_raw)`/`base`/`title` are prepared.

    `y_ls`: the series to use for the LS quick-look panel specifically, if it
    should differ from `y_raw` (single-sector: `compute_ls_periodogram` used
    to additionally detrend by `detrend_deg` for JUST the LS panel, while
    prewhitening relies on its own internal degree-2 detrend -- preserved
    here via this override so the refactor doesn't change either function's
    numbers). Defaults to `y_raw` (multi-sector: no separate LS-only detrend
    -- the per-sector detrend already applied before stitching covers it).

    `save_official_ls`: if True (single-sector, `lc_suffix == "_clean"` --
    same condition as before this was refactored out), write the LS `.dat`
    through `periodogram.save_periodogram` (the single-sector "official"
    pipeline file, needs `cadr_width`/`cadr_height`) instead of a plain file
    next to `base`. `plot_star_name`/`plot_sector` are always used for the
    LS plot's title (and `save_periodogram`'s own file-naming, when used).
    """
    if y_ls is None:
        y_ls = y_raw

    # ------------------------------------------------------------------ 1. LS
    period_range = suggest_period_range(t, nyquist_factor=nyquist_factor, max_period_fraction=max_period_fraction)
    print(f"period_range = {period_range[0]:.5f} .. {period_range[1]:.3f} d")

    freq, power, ls = compute_ls_periodogram_from_series(t, y_ls, period_range=period_range)
    peaks_ls = find_periodogram_peaks(freq, power, ls, fap_levels=(0.1, 0.01, 0.001), max_peaks=max_peaks_analyse)
    print("\nLS (быстрый обзор; FAP 'naive' предполагает белый шум — в статью не брать):")
    for p in peaks_ls[:max_peaks_analyse]:
        print(f"   {p['frequency']:.4f} 1/d   {p['period']:.4f} d   power={p['power']:.4f}")
    if make_ls_plot:
        if save_official_ls:
            dat_path = save_periodogram(plot_star_name, plot_sector, freq, power, ls,
                                        cut_width=cadr_width, cut_height=cadr_height, peaks=peaks_ls)
        else:                                        # своя кривая — свой файл, основной не трогаем
            dat_path = f"{base}.dat"
            header_lines = provenance_header(n_points=freq.size, star_name=plot_star_name, sector=plot_sector,
                                             detrend_deg=detrend_deg) + [
                f"Lomb-Scargle periodogram, {name_file_data}", "frequency[1/d] power"]
            np.savetxt(dat_path, np.column_stack([freq, power]), fmt="%.6f %.6e", header="\n".join(header_lines))
        plot_periodogram(plot_star_name, plot_sector, freq, power, ls, peaks=peaks_ls,
                         out_path=dat_path.replace(".dat", f".{image_format}"),
                         max_peaks_for_plotting=max_peaks_analyse)

    # ------------------------------------------------------ 2. выбеливание
    y = y_raw * 1e3                                        # mmag
    peaks, extra = prewhiten(t, y, fmax=prewhiten_fmax, nmax=n_max_freq, snr_stop=snr_stop)
    T = extra["T"]
    table = format_frequency_table(peaks, units="mmag")
    print(f"\nВыбеливание: N = {t.size}, T = {T:.2f} d, 1/T = {1 / T:.4f}, 1.5/T = {1.5 / T:.4f} 1/d; "
          f"sigma(остатков) = {extra['sigma_res']:.2f} mmag, D = {extra['D']:.1f}")
    print(table)
    save_frequency_table(f"{base}_frequencies.dat", peaks, header_lines=[
        *provenance_header(star_name=plot_star_name, sector=plot_sector, detrend_deg=detrend_deg),
        f"{title}, file {name_file_data}",
        f"N = {t.size}, T = {T:.3f} d, 1/T = {1 / T:.4f} 1/d, 1.5/T = {1.5 / T:.4f} 1/d",
        f"prewhitening: trend deg {detrend_deg}, S/N stop {snr_stop}, fmax {prewhiten_fmax} 1/d",
        f"errors: Montgomery & O'Donoghue (1999) x sqrt(D), D = {extra['D']:.1f}; "
        f"sigma_res = {extra['sigma_res']:.3f} mmag",
        "S/N_loc: mean residual amplitude within +/-0.5 1/d (Breger+1993); S/N_red: red-noise model (Bowman+2019)",
        "for members of close groups (spacing < 1.5/T) the realistic frequency error is ~0.005-0.01 1/d",
    ])

    # ---------------------------------------------------------- 3. рисунки
    real = [p for p in peaks if p["frequency"] > 2.0 / T]           # без тренда (< 2/T)
    zoom = None
    if real:
        f_main = max(real, key=lambda p: p["amplitude"])["frequency"]
        zoom = (max(0.0, f_main - 0.35), f_main + 0.35)
    plot_amplitude_spectrum(t, y, peaks, extra, fmax=min(prewhiten_fmax, 6.0), units="mmag", zoom=zoom,
                            title=title, out_path=f"{base}_amplitude.{image_format}")
    plot_prewhitening_fit(t, y, peaks, units="mmag", title=title, out_path=f"{base}_fit.{image_format}")
    centers, fd, dyn = dynamic_spectrum(t, y, window=dyn_window, step=dyn_step, frange=dyn_frange)
    plot_dynamic_spectrum(centers, fd, dyn, peaks=peaks, units="mmag", window=dyn_window,
                          title=f"{title}: dynamic spectrum", out_path=f"{base}_dynamic.{image_format}")
    return peaks


def periodogram_compute_analise(star_name_process, sector_number, cadr_width_for_analyse, cadr_height_for_analyse,
                                max_peaks_analyse=10, image_format="png",
                                nyquist_factor=2.0, max_period_fraction=0.5,
                                lc_suffix="_clean", detrend_deg=2,
                                prewhiten_fmax=10.0, snr_stop=4.0, n_max_freq=15,
                                dyn_window=10.0, dyn_step=0.25, dyn_frange=(0.8, 2.5),
                                make_ls_plot=True):
    star_dir = f"stars_python/{star_name_process}/{cadr_width_for_analyse}x{cadr_height_for_analyse}"
    name_file_data = f"{star_dir}/light_curve_sector_{sector_number}{lc_suffix}.csv"
    base = f"{star_dir}/periodogram_sector_{sector_number}{lc_suffix.replace('_clean', '')}"

    df_clean = read_light_curve_csv(name_file_data)
    print(f"Кривая: {name_file_data}, {len(df_clean)} точек")

    jds, mags = cleaned_jds_mags(df_clean)
    t = jds - 2457000.0                                   # BJD - 2457000
    y_raw = mags - np.median(mags)
    y_ls = y_raw
    if detrend_deg > 0:                                   # only the LS quick-look panel used this extra detrend
        x = t - t.mean()
        y_ls = y_raw - np.polyval(np.polyfit(x, y_raw, detrend_deg), x)

    return _analyse_prepared_series(
        t, y_raw, base, f"{star_name_process}, sector {sector_number}", name_file_data,
        star_name_process, sector_number,
        max_peaks_analyse=max_peaks_analyse, image_format=image_format,
        nyquist_factor=nyquist_factor, max_period_fraction=max_period_fraction, detrend_deg=detrend_deg,
        prewhiten_fmax=prewhiten_fmax, snr_stop=snr_stop, n_max_freq=n_max_freq,
        dyn_window=dyn_window, dyn_step=dyn_step, dyn_frange=dyn_frange,
        make_ls_plot=make_ls_plot, y_ls=y_ls, save_official_ls=(lc_suffix == "_clean"),
        cadr_width=cadr_width_for_analyse, cadr_height=cadr_height_for_analyse)


def periodogram_compute_analise_multisector(star_name_process, sector_numbers, cadr_width_for_analyse,
                                            cadr_height_for_analyse, max_peaks_analyse=10, image_format="png",
                                            nyquist_factor=2.0, max_period_fraction=0.5,
                                            lc_suffix="_clean", detrend_deg=2,
                                            prewhiten_fmax=10.0, snr_stop=4.0, n_max_freq=15,
                                            dyn_window=10.0, dyn_step=0.25, dyn_frange=(0.8, 2.5),
                                            make_ls_plot=True):
    """
    Same analysis as `periodogram_compute_analise`, but on several sectors'
    already-cleaned curves stitched into one series (ROADMAP.md Этап 7):
    each sector independently normalized (own median subtracted) and
    detrended (own polynomial, degree `detrend_deg`) before concatenation --
    see `isolated.multisector.stitch_sector_csvs` -- so a differing
    zero-point/drift between sectors and the (often year-scale) gap between
    them don't leak into the lowest frequencies. Each sector's own
    `light_curve_sector_{N}{lc_suffix}.csv` must already exist (run
    `TESS_cleaning.py`/`PRF_cleaning.py` once per sector beforehand, same
    as for a single sector -- unchanged).
    """
    star_dir = f"stars_python/{star_name_process}/{cadr_width_for_analyse}x{cadr_height_for_analyse}"
    paths = [f"{star_dir}/light_curve_sector_{s}{lc_suffix}.csv" for s in sector_numbers]
    sectors_tag = "-".join(str(s) for s in sector_numbers)
    base = f"{star_dir}/periodogram_sectors_{sectors_tag}{lc_suffix.replace('_clean', '')}"
    title = f"{star_name_process}, sectors {sectors_tag}"

    t, y_raw, sector_of = stitch_sector_csvs(paths, sector_labels=sector_numbers,
                                             value_col="MAG", detrend_deg=detrend_deg)
    print(f"Кривая: {', '.join(paths)}, {t.size} точек ({len(sector_numbers)} секторов, "
          f"склейка/тренд по каждому отдельно)")
    for s in sector_numbers:
        n_s = int(np.sum(sector_of == s))
        print(f"   сектор {s}: {n_s} точек")

    return _analyse_prepared_series(
        t, y_raw, base, title, ", ".join(paths), star_name_process, sectors_tag,
        max_peaks_analyse=max_peaks_analyse, image_format=image_format,
        nyquist_factor=nyquist_factor, max_period_fraction=max_period_fraction, detrend_deg=detrend_deg,
        prewhiten_fmax=prewhiten_fmax, snr_stop=snr_stop, n_max_freq=n_max_freq,
        dyn_window=dyn_window, dyn_step=dyn_step, dyn_frange=dyn_frange,
        make_ls_plot=make_ls_plot)


if __name__ == "__main__":
    # Звезда / сектор(ы) / размер кадра / режим фотометрии -- из run_config.py (Этап 4),
    # тот же конфиг, что и run.py/TESS_cleaning.py/PRF_cleaning.py/Peridogram_diagnostics.py.
    # CONFIG.sectors (Этап 7) -- если задан (больше одного сектора), анализ идет по
    # объединенному ряду; иначе (по умолчанию) -- как раньше, один CONFIG.sector.
    from run_config import CONFIG

    if CONFIG.sectors:
        periodogram_compute_analise_multisector(
            CONFIG.star_name.replace(' ', '_'), CONFIG.sectors, CONFIG.cut_width, CONFIG.cut_height,
            max_peaks_analyse=CONFIG.max_peaks_analyse, image_format=CONFIG.image_format,
            nyquist_factor=CONFIG.nyquist_factor, max_period_fraction=CONFIG.max_period_fraction,
            lc_suffix=CONFIG.lc_suffix, detrend_deg=CONFIG.detrend_deg,
            prewhiten_fmax=CONFIG.prewhiten_fmax, snr_stop=CONFIG.snr_stop, n_max_freq=CONFIG.n_max_freq,
            dyn_window=CONFIG.dyn_window, dyn_step=CONFIG.dyn_step, dyn_frange=CONFIG.dyn_frange)
    else:
        periodogram_compute_analise(
            CONFIG.star_name.replace(' ', '_'), CONFIG.sector, CONFIG.cut_width, CONFIG.cut_height,
            max_peaks_analyse=CONFIG.max_peaks_analyse, image_format=CONFIG.image_format,
            nyquist_factor=CONFIG.nyquist_factor, max_period_fraction=CONFIG.max_period_fraction,
            lc_suffix=CONFIG.lc_suffix, detrend_deg=CONFIG.detrend_deg,
            prewhiten_fmax=CONFIG.prewhiten_fmax, snr_stop=CONFIG.snr_stop, n_max_freq=CONFIG.n_max_freq,
            dyn_window=CONFIG.dyn_window, dyn_step=CONFIG.dyn_step, dyn_frange=CONFIG.dyn_frange)
