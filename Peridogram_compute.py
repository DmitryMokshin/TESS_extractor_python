"""
Поиск частот по очищенной кривой блеска.

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
 3. Рисунки: амплитудный спектр (с уровнем 4×шум и спектральным окном),
    кривая блеска с моделью, динамический спектр.

Все файлы кладутся рядом с кривой блеска:
    periodogram_sector_{N}{суффикс}.dat / .{fmt}          — LS (как раньше)
    periodogram_sector_{N}{суффикс}_frequencies.dat       — таблица частот с ошибками и S/N
    periodogram_sector_{N}{суффикс}_amplitude.{fmt}       — амплитудный спектр
    periodogram_sector_{N}{суффикс}_fit.{fmt}             — кривая + модель
    periodogram_sector_{N}{суффикс}_dynamic.{fmt}         — динамический спектр
"""
import numpy as np
import pandas as pd

from isolated.periodogram import compute_ls_periodogram, find_periodogram_peaks
from isolated.periodogram import save_periodogram, plot_periodogram
from isolated.periodogram import suggest_period_range
from isolated.stats import cleaned_jds_mags
from isolated.prewhitening import (prewhiten, format_frequency_table, save_frequency_table,
                                   plot_amplitude_spectrum, plot_prewhitening_fit,
                                   dynamic_spectrum, plot_dynamic_spectrum)


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

    df_clean = pd.read_csv(name_file_data)
    print(f"Кривая: {name_file_data}, {len(df_clean)} точек")

    # ------------------------------------------------------------------ 1. LS
    # max_period_fraction=0.5: период должен уложиться в ряд хотя бы дважды
    period_range = suggest_period_range(df_clean['MJD'], nyquist_factor=nyquist_factor,
                                        max_period_fraction=max_period_fraction)
    print(f"period_range = {period_range[0]:.5f} .. {period_range[1]:.3f} d")

    freq, power, ls = compute_ls_periodogram(df_clean, period_range=period_range, detrend_deg=detrend_deg)
    peaks_ls = find_periodogram_peaks(freq, power, ls, fap_levels=(0.1, 0.01, 0.001), max_peaks=max_peaks_analyse)
    print("\nLS (быстрый обзор; FAP 'naive' предполагает белый шум — в статью не брать):")
    for p in peaks_ls[:max_peaks_analyse]:
        print(f"   {p['frequency']:.4f} 1/d   {p['period']:.4f} d   power={p['power']:.4f}")
    if make_ls_plot:
        if lc_suffix == "_clean":
            dat_path = save_periodogram(star_name_process, sector_number, freq, power, ls,
                                        cut_width=cadr_width_for_analyse, cut_height=cadr_height_for_analyse,
                                        peaks=peaks_ls)
        else:                                        # своя кривая — свой файл, основной не трогаем
            dat_path = f"{base}.dat"
            np.savetxt(dat_path, np.column_stack([freq, power]), fmt="%.6f %.6e",
                       header=f"Lomb-Scargle periodogram (detrend deg {detrend_deg}), {name_file_data}\n"
                              "frequency[1/d] power")
        plot_periodogram(star_name_process, sector_number, freq, power, ls, peaks=peaks_ls,
                         out_path=dat_path.replace(".dat", f".{image_format}"),
                         max_peaks_for_plotting=max_peaks_analyse)

    # ------------------------------------------------------ 2. выбеливание
    jds, mags = cleaned_jds_mags(df_clean)
    t = jds - 2457000.0                                   # BJD - 2457000
    y = (mags - np.median(mags)) * 1e3                    # mmag
    peaks, extra = prewhiten(t, y, fmax=prewhiten_fmax, nmax=n_max_freq, snr_stop=snr_stop)
    T = extra["T"]
    table = format_frequency_table(peaks, units="mmag")
    print(f"\nВыбеливание: N = {t.size}, T = {T:.2f} d, 1/T = {1 / T:.4f}, 1.5/T = {1.5 / T:.4f} 1/d; "
          f"sigma(остатков) = {extra['sigma_res']:.2f} mmag, D = {extra['D']:.1f}")
    print(table)
    save_frequency_table(f"{base}_frequencies.dat", peaks, header_lines=[
        f"{star_name_process}, sector {sector_number}, file {name_file_data}",
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
                            title=f"{star_name_process}, sector {sector_number}",
                            out_path=f"{base}_amplitude.{image_format}")
    plot_prewhitening_fit(t, y, peaks, units="mmag", title=f"{star_name_process}, sector {sector_number}",
                          out_path=f"{base}_fit.{image_format}")
    centers, fd, dyn = dynamic_spectrum(t, y, window=dyn_window, step=dyn_step, frange=dyn_frange)
    plot_dynamic_spectrum(centers, fd, dyn, peaks=peaks, units="mmag", window=dyn_window,
                          title=f"{star_name_process}, sector {sector_number}: dynamic spectrum",
                          out_path=f"{base}_dynamic.{image_format}")
    return peaks


if __name__ == "__main__":
    # Звезда / сектор / размер кадра — те же, что в run.py и TESS_cleaning.py (см. README).
    star_name = "SS 397"
    sector = 80
    cadr_width = 50
    cadr_height = 50
    # "_clean"      — кривая после TESS_cleaning.py (апертурная фотометрия);
    # "_prf_clean"  — PRF-кривая из ss397_localize.py (без света соседней звезды).
    lc_suffix = "_prf_clean"

    periodogram_compute_analise(star_name.replace(' ', '_'), sector, cadr_width, cadr_height,
                                max_peaks_analyse=10, image_format="png", lc_suffix=lc_suffix)
