import pandas as pd
from isolated.periodogram import compute_ls_periodogram
from isolated.periodogram import find_periodogram_peaks
from isolated.periodogram import save_periodogram, plot_periodogram, plot_periodogram_approximation
from isolated.periodogram import suggest_period_range


def periodogram_compute_analise(star_name_process, sector_number, cadr_width_for_analyse, cadr_height_for_analyse,
                                max_peaks_analyse=10, n_frequencies_fit=5, image_format="png",
                                nyquist_factor=2.0, max_period_fraction=1.0):
    name_file_data = f"stars_python/{star_name_process}/{cadr_width_for_analyse}x{cadr_height_for_analyse}/light_curve_sector_{sector_number}_clean.csv"

    df_clean = pd.read_csv(name_file_data)

    # Мин. период -- 2x медианный шаг по времени (предел Найквиста)
    period_range = suggest_period_range(df_clean['MJD'], nyquist_factor=nyquist_factor,
                                        max_period_fraction=max_period_fraction)
    print(f"period_range = {period_range[0]:.5f} .. {period_range[1]:.3f} d")

    freq, power, ls = compute_ls_periodogram(df_clean, period_range=period_range)

    peaks = find_periodogram_peaks(freq, power, ls, fap_levels=(0.1, 0.01, 0.001), max_peaks=max_peaks_analyse)
    for p in peaks[:max_peaks_analyse + 1]:
        print(f"{p['frequency']:.3f} 1/d   {p['period']:.3f} d   power={p['power']:.3f}   FAP={p['fap']:.2e}")

    dat_path = save_periodogram(star_name_process, sector_number, freq, power, ls, cut_width=cadr_width_for_analyse,
                                peaks=peaks)

    plot_periodogram(star_name_process, sector_number, freq, power, ls, peaks=peaks,
                     out_path=dat_path.replace(".dat", f".{image_format}"), max_peaks_for_plotting=max_peaks_analyse)

    # overlay the strongest n_frequencies_fit peaks back onto the light curve, as a sanity check on the periodogram
    plot_periodogram_approximation(star_name_process, sector_number, df_clean, peaks,
                                   n_frequencies=n_frequencies_fit,
                                   out_path=dat_path.replace(".dat", f"_fit.{image_format}"))


if __name__ == "__main__":
    star_name = "EM* AS 14"
    sector = 18
    cadr_width = 15
    cadr_height = 15

    periodogram_compute_analise(star_name.replace(' ', '_'), sector, cadr_width, cadr_height, 5, 5, 'eps')
