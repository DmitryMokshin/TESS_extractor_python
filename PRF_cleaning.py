"""
PRF-режим шага 2 пайплайна: чей сигнал? Локализация частот TESS по пикселям
+ PRF-разделение кривых блеска (звезда/сектор/вырезка/режим — из
`run_config.py`, работает для любой звезды в конфиге).

Зачем. В тесных полях простая апертура собирает чужой свет: звезда Gaia
рядом с целью может давать заметную долю потока в апертуре r=3 пикс — код
пайплайна без PRF-деблендинга вычитает только плоский фон, и свет соседей
остаётся в кривой (например, для SS 397, сектор 80: на цель приходится лишь
~43% потока в апертуре, на ближайшего соседа ~46% -- обе звезды сравнимой
яркости).

Что делает скрипт:
 1. Берёт вырезку сектора и выбирает кадры: QUALITY_BITMASK (стандартная маска
    lightkurve, 175). Дальнейшая (окно-по-окну) чистка -- отдельный шаг через
    уже существующие TESS_cleaning.py/pick_exclusion_windows/exclude_frame_windows
    поверх сохранённого здесь light_curve_sector_{SECTOR}_prf.csv, как и для
    апертурного режима -- этот скрипт не требует заранее посчитанного
    light_curve_sector_{SECTOR}_clean.csv.
 2. PRF-фотометрия (isolated.prf_photometry): в окне BOX×BOX вокруг звезды каждый
    кадр описывается суммой PRF звёзд Gaia (положения фиксированы) + плоский фон.
    Потоки звёзд линейны, решение точное и быстрое (один lstsq на все кадры сразу).
    Цель находится автоматически (isolated.data_io.find_target_row, тот же
    механизм, что isolated.prf_photometry.load_prf_light_curve), доминирующий
    сосед (наибольшая доля PRF-потока в апертуре цели) -- тоже автоматически
    (isolated.photometry.find_dominant_contaminant), без ручных Gaia source_id.
    На выходе отдельные кривые цели и соседа (CSV в формате пайплайна).
 3. Проверка происхождения сигнала (isolated.localize, ROADMAP.md Этап 6): свежим
    выбеливанием находятся значимые частоты цели, автоматически выбираются до
    N_COMPARISON звёзд сопоставимого блеска в вырезке (не только в окне цели),
    им строятся кривые тем же PRF-методом -- и для каждой проверяется, есть ли
    у неё частоты цели (S/N на заданной частоте). Если да у нескольких звёзд
    сразу -- вероятна общая систематика (рассеянный свет/инструмент), а не
    сигнал именно цели.
 4. Карты амплитуд (isolated.localize): в каждом пикселе подгоняются синусоиды
    на свежих значимых частотах цели (или на AMP_MAP_FREQS_OVERRIDE, если
    задан вручную) + квадратичный тренд. Комплексная карта амплитуды
    раскладывается по PRF звёзд: видно, какая звезда несёт каждую частоту.

Как читать вывод (интерпретация для тесных полей) -- см. README.md, разделы
"Правила интерпретации при тесных полях" и "Как использовать isolated/localize.py
напрямую".

Выход: out_localize/
    amp_maps.pdf              — карты амплитуд с положениями звёзд
    attribution.txt           — доли амплитуды по звёздам для каждой частоты
    comparison_stars.csv      — таблица происхождения сигнала (звезда x частота
                                 цели: амплитуда, S/N, есть ли частота)
    comparison_stars.txt      — то же, сводной таблицей для чтения глазами
    lc_deblended_<id>.csv     — PRF-кривые (BTJD, FRAME, FLUX) для цели и соседа
    lc_deblended.pdf          — сравнение кривых
и в CONFIG.star_dir (stars_python/{звезда}/{вырезка}/):
    light_curve_sector_{N}_prf.csv        — канонический PRF-выход пайплайна
                                             (QUALITY-фильтр, без доп. чистки;
                                             то же самое, что дал бы
                                             isolated.prf_photometry.load_prf_light_curve)
    light_curve_sector_{N}_prf_clean.csv  — тот же, плюс отбраковка шумных участков
                                             (см. п.1 выше), для быстрого CLEAN/LS
Если RUN_DIAGNOSTICS_AFTER = True, сразу после этого разделённые кривые прогоняются через
Peridogram_diagnostics.py (частоты, ошибки, динамический спектр) — результаты в out_tess/.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from isolated.data_io import load_tess_cutouts, load_gaia_stars_in_view_data, load_star_gaia_data, find_target_row
from isolated.psf import get_tesscut_prf_supersampled
from isolated.geometry import get_nospace_star_name, calc_tess_magnitude, calc_tess_flux_from_mag
from isolated.lightcurve_tools import local_point_to_point_sigma
from isolated.cleaning import quality_mask, auto_clean_light_curve, save_cleaning_log, summarize_cleaning_log
from isolated.prf_photometry import select_prf_model_stars, build_prf_star_cuts, deblend_prf_flux
from isolated.photometry import find_dominant_contaminant
from isolated.localize import amplitude_maps, attribute_amplitude, select_comparison_stars, \
    deblend_star_curve, signal_origin_table
from isolated.prewhitening import prewhiten
from isolated.provenance import provenance_header, write_csv_with_provenance
from run_config import CONFIG

# =============================================================================
# НАСТРОЙКИ — звезда/сектор/вырезка/чистка теперь в run_config.py (Этап 4);
# то, что осталось здесь, специфично для этого шага пайплайна (не для звезды).
# =============================================================================
STAR_NAME = CONFIG.star_name
SECTOR = CONFIG.sector
CUT = CONFIG.cut_width
BOX = CONFIG.prf_box            # размер окна для PRF-фотометрии (нечетный), пикс
DT_MAX = CONFIG.prf_dt_max       # включать звезды Gaia не слабее T_target + DT_MAX
MERGE_PX = CONFIG.prf_merge_px   # звезды ближе MERGE_PX к более яркой не фитуются отдельно (вырождение)
# частоты для карт амплитуд: None -- свежие значимые частоты цели (target_freqs,
# см. ниже); задать вручную (список 1/сут), если нужны конкретные/кандидатные
# частоты вместо автовыбора (например, чтобы воспроизвести конкретную фигуру)
AMP_MAP_FREQS_OVERRIDE = None
OUTDIR = "out_localize"
# звезды сравнения (ROADMAP.md Этап 6): сколько брать и насколько близко по T-величине
N_COMPARISON = 5
COMPARISON_MAG_TOL = 1.0
# Какие флаги QUALITY выбрасывают кадр. 175 = стандартная маска lightkurve "default":
# 1 AttitudeTweak, 2 SafeMode, 4 CoarsePoint, 8 EarthPoint, 32 Desat, 128 ManualExclude.
# Флаги рассеянного света (2048, 4096 и т.п.) НЕ выбрасываем: такие кадры часто
# нормальные, плохие участки потом отсекаются по локальному шуму.
# 0 — не использовать флаги вообще; None — брать только QUALITY == 0 (слишком строго для S80).
QUALITY_BITMASK = CONFIG.quality_bitmask
LOCAL_NOISE_KAPPA = CONFIG.local_noise_kappa  # отбраковка "_prf_clean": локальный шум > KAPPA × медиана
RUN_DIAGNOSTICS_AFTER = True     # сразу прогнать Peridogram_diagnostics.analyse по разделенным кривым
SHOW_PLOTS = True         # открыть окна с рисунками в конце (PDF сохраняются всегда)


def export_clean_lc(t, flux, dat_path, star, sector, csv_path=None, note="", frame_no=None):
    """
    Очищенная кривая блеска для коллег (ASCII, для CLEAN/LS в любых программах):
        BJD-2457000   dF/F[ppt]   sigma[ppt]   dmag[mag]   FLUX[e/s]
    и, если задан csv_path, копия в формате пайплайна (BTJD, FRAME, FLUX,
    MAG) — её можно подать в Peridogram_compute.py.
    """
    med = np.median(flux)
    ppt = (flux / med - 1.0) * 1e3
    err = local_point_to_point_sigma(t, ppt)
    dmag = -2.5 * np.log10(flux / med)
    header_lines = provenance_header(n_points=t.size, star_name=star, sector=sector,
                                     quality_bitmask=QUALITY_BITMASK, local_noise_kappa=LOCAL_NOISE_KAPPA) + [
        f"{star}, TESS sector {sector}: PRF-photometry light curve (deblended from Gaia neighbours)",
        f"frames: TESScut FFI, cadence 200 s; QUALITY bitmask {QUALITY_BITMASK} removed; points with local",
        f"point-to-point noise > {LOCAL_NOISE_KAPPA} x median removed. {note}",
        f"median flux = {med:.1f} e/s, median sigma = {np.median(err):.2f} ppt",
        "columns: BJD-2457000  dF/F[ppt]  sigma[ppt]  dmag[mag]  FLUX[e/s]",
    ]
    np.savetxt(dat_path, np.c_[t, ppt, err, dmag, flux], fmt="%.6f %9.3f %7.3f %9.5f %11.3f",
              header="\n".join(header_lines))
    if csv_path:
        write_csv_with_provenance(pd.DataFrame({"BTJD": t, "FRAME": frame_no, "FLUX": flux,
                                                "MAG": calc_tess_magnitude(flux)}),
                                  csv_path, star_name=star, sector=sector)
    return dat_path


# =============================================================================
# запуск на реальных данных
# =============================================================================
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # пути пакета isolated относительные
    os.makedirs(OUTDIR, exist_ok=True)
    star_dir = os.path.join("stars_python", get_nospace_star_name(STAR_NAME), f"{CUT}x{CUT}")

    cut_fits = load_tess_cutouts(STAR_NAME, CUT, CUT, sector=SECTOR)[SECTOR]
    data = cut_fits[1].data
    t_all = np.asarray(data["TIME"], float)
    q_all = np.asarray(data["QUALITY"], int)
    frame_no_all = np.arange(1, t_all.size + 1)  # 1-based позиция в исходном, ещё не отфильтрованном секторе
    cube = np.transpose(np.asarray(data["FLUX"], float), (0, 2, 1))   # (nt, x, y), как frame(i)
    prf = get_tesscut_prf_supersampled(cut_fits)

    gaia_data = load_star_gaia_data(STAR_NAME)
    gaia = load_gaia_stars_in_view_data(STAR_NAME, cut_fits, d_mag_r=6.0)
    tgt = find_target_row(gaia, gaia_data, SECTOR)
    TARGET_ID = int(tgt["source_id"])
    box_x0 = int(round(tgt["px_x"])) - BOX // 2       # 1-based начало окна
    box_y0 = int(round(tgt["px_y"])) - BOX // 2
    stars = select_prf_model_stars(gaia, TARGET_ID, box_x0, box_y0, BOX, DT_MAX, MERGE_PX)
    print(f"Звезд в модели окна: {len(stars)}")
    print(stars[["source_id", "t_mag", "px_x", "px_y"]].to_string())

    # --- выбор кадров: только QUALITY (см. докстринг модуля; правило 1, ROADMAP.md Этап 5) ---
    bits = [b for b in range(20) if np.any(q_all & (1 << b))]
    print("Флаги QUALITY в секторе (бит: число кадров): " +
          ", ".join(f"{1 << b}: {np.count_nonzero(q_all & (1 << b))}" for b in bits))
    finite = np.isfinite(t_all)
    q_ok = quality_mask(q_all, QUALITY_BITMASK)
    good = finite & q_ok
    print(f"Кадров: всего {t_all.size}, используем после фильтра QUALITY: {good.sum()}")
    log_quality = pd.DataFrame({"FRAME": frame_no_all[finite & ~q_ok], "BTJD": t_all[finite & ~q_ok],
                                "REASON": "QUALITY"})

    x0, y0 = box_x0 - 1, box_y0 - 1  # 0-based для среза куба
    box = cube[good, x0:x0 + BOX, y0:y0 + BOX]
    t = t_all[good]
    frame_no = frame_no_all[good]
    models = build_prf_star_cuts(prf, stars, CUT, CUT, box_x0, box_y0, BOX)
    fluxes, bkg = deblend_prf_flux(box, models)

    it = int(np.where(stars["source_id"] == TARGET_ID)[0][0])

    # доминирующий сосед -- звезда поля с наибольшей долей PRF-потока в апертуре
    # цели, а не заранее известный Gaia source_id (isolated.photometry, новое
    # в этой версии скрипта)
    iN, neighbour_frac = find_dominant_contaminant(
        tgt["px_x"], tgt["px_y"], stars["px_x"].to_numpy(), stars["px_y"].to_numpy(),
        calc_tess_flux_from_mag(stars["phot_rp_mean_mag"].to_numpy()), prf, CUT, CONFIG.aperture_radius)
    if iN is not None:
        NEIGHBOUR_ID = int(stars.iloc[iN]["source_id"])
        print(f"\nДоминирующий сосед в апертуре r={CONFIG.aperture_radius}px: source_id={NEIGHBOUR_ID} "
              f"(T={stars.iloc[iN]['t_mag']:.2f}), доля потока цели+соседа ~{neighbour_frac:.0%}.")
        if neighbour_frac > 0.2:
            print("  -> существенное загрязнение (>20%): доверять можно только PRF-кривой цели, "
                  "апертурная фотометрия здесь искажена соседом. Прежде чем приписывать найденную "
                  "частоту цели, сверьтесь с comparison_stars.csv (PRESENT) и долей `share` в "
                  "attribution.txt/amp_maps.pdf -- см. README.md, \"Правила интерпретации при "
                  "тесных полях\".")
        elif neighbour_frac > 0.05:
            print("  -> заметное, но не доминирующее загрязнение (5-20%) -- иметь в виду при "
                  "сравнении с апертурной фотометрией.")
    else:
        NEIGHBOUR_ID = None
        print("\nСоседей в окне модели не найдено -- поле изолированное, PRF- и апертурный "
              "режимы должны давать близкие результаты.")

    # канонический PRF-выход пайплайна (без доп. чистки) -- то же самое, что дал бы
    # isolated.prf_photometry.load_prf_light_curve; сохраняем сами, чтобы не гонять
    # деблендинг по 13000 кадров дважды
    os.makedirs(star_dir, exist_ok=True)
    write_csv_with_provenance(
        pd.DataFrame({"BTJD": t, "FRAME": frame_no, "FLUX": fluxes[:, it],
                     "MAG": calc_tess_magnitude(np.abs(fluxes[:, it]))}),
        os.path.join(star_dir, f"light_curve_sector_{SECTOR}_prf.csv"),
        star_name=STAR_NAME, sector=SECTOR, box=BOX, dt_max=DT_MAX, merge_px=MERGE_PX,
        quality_bitmask=QUALITY_BITMASK)

    # отбраковка по шуму PRF-кривой цели (правило 3, ROADMAP.md Этап 5) -- через общую
    # auto_clean_light_curve (bitmask=0: правило 1 уже применено выше), для _prf_clean.csv
    # и карт амплитуд ниже
    df_target = pd.DataFrame({"BTJD": t, "FRAME": frame_no, "FLUX": fluxes[:, it]})
    _df_target_clean, log_noise = auto_clean_light_curve(df_target, quality_bitmask=0,
                                                          local_noise_kappa=LOCAL_NOISE_KAPPA)
    m = ~df_target["FRAME"].isin(log_noise["FRAME"]).to_numpy()
    print(f"После отбраковки шумных участков: {m.sum()} кадров")
    t, box, fluxes, bkg, frame_no = t[m], box[m], fluxes[m], bkg[m], frame_no[m]

    # общий лог (QUALITY + локальный шум) -- по нему можно дословно восстановить,
    # почему выброшен каждый кадр (ROADMAP.md Этап 5, критерий готовности)
    log_combined = pd.concat([log_quality, log_noise], ignore_index=True).sort_values("FRAME").reset_index(drop=True)
    log_path = save_cleaning_log(log_combined, STAR_NAME, SECTOR, CUT, suffix="_prf_clean")
    print(f"\nЛог чистки: {log_path}\n{summarize_cleaning_log(log_combined, total_frames=t_all.size)}")

    for j, tag in [(it, get_nospace_star_name(STAR_NAME)), (iN, "neighbour")]:
        if j is None:
            continue
        write_csv_with_provenance(pd.DataFrame({"BTJD": t, "FRAME": frame_no, "FLUX": fluxes[:, j]}),
                                  os.path.join(OUTDIR, f"lc_deblended_{tag}.csv"),
                                  star_name=STAR_NAME, sector=SECTOR, tag=tag, box=BOX)
        print(f"{tag}: медианный поток {np.median(fluxes[:, j]):.0f} e/s "
              f"(ожидание по Gaia {10 ** (0.4 * (20.44 - stars.loc[j, 't_mag'])):.0f})")

    # очищенная кривая цели + копия в формате пайплайна
    dat = export_clean_lc(t, fluxes[:, it], os.path.join(OUTDIR, f"{get_nospace_star_name(STAR_NAME)}_TESS_S{SECTOR}_PRF.dat"),
                          star=STAR_NAME, sector=SECTOR,
                          csv_path=os.path.join(star_dir, f"light_curve_sector_{SECTOR}_prf_clean.csv"),
                          frame_no=frame_no)
    print(f"Кривая блеска: {dat}")

    # =========================================================================
    # звезды сравнения: сигнал свой у цели или общий для поля/инструмента?
    # (ROADMAP.md Этап 6, "Проверка происхождения сигнала")
    # =========================================================================
    # свежие значимые частоты цели -- считаются один раз и переиспользуются
    # дальше и для таблицы происхождения сигнала, и (по умолчанию) для карт
    # амплитуд (AMP_MAP_FREQS_OVERRIDE) ниже
    t0 = t  # already BTJD = BJD-2457000 (see export_clean_lc's own header above), not full BJD
    y_target_ppt = (fluxes[:, it] / np.median(fluxes[:, it]) - 1.0) * 1e3
    target_peaks, target_extra = prewhiten(t0, y_target_ppt, fmax=CONFIG.prewhiten_fmax,
                                            nmax=CONFIG.n_max_freq, snr_stop=CONFIG.snr_stop)
    T_target = target_extra["T"]
    target_freqs = [p["frequency"] for p in target_peaks if p["frequency"] > 2.0 / T_target]
    print(f"\nЗначимые частоты цели (S/N >= {CONFIG.snr_stop}, не тренд): "
          + ", ".join(f"{f:.4f}" for f in target_freqs))

    comparison_stars = select_comparison_stars(gaia, TARGET_ID, CUT, CUT, n=N_COMPARISON,
                                                mag_tol=COMPARISON_MAG_TOL, box=BOX)
    print(f"\nЗвезды сравнения ({len(comparison_stars)}, |dT| <= {COMPARISON_MAG_TOL}):")
    print(comparison_stars[["source_id", "t_mag", "px_x", "px_y"]].to_string())

    # тот же good (правило 1, QUALITY), что и у цели -- "тем же методом"
    cube_good = cube[good]
    t_quality = t_all[good]
    frame_quality = frame_no_all[good]

    star_curves = [("target", t0, y_target_ppt)]
    if target_freqs and len(comparison_stars):
        for _, cstar in comparison_stars.iterrows():
            label = f"src{int(cstar['source_id']) % 100000}(T={cstar['t_mag']:.1f})"
            try:
                c_flux, _c_stars, _c_models, _c_bkg = deblend_star_curve(
                    cube_good, prf, gaia, CUT, CUT, int(cstar["source_id"]), box=BOX, dt_max=DT_MAX,
                    merge_px=MERGE_PX)
            except ValueError as exc:
                print(f"Пропускаю {label}: {exc}")
                continue
            df_c = pd.DataFrame({"BTJD": t_quality, "FRAME": frame_quality, "FLUX": c_flux})
            # правило 3 автоочистки (локальный шум) -- своя чистка на кадр для каждой звезды,
            # правило 1 (QUALITY) уже применено выше через cube_good/t_quality
            df_c_clean, _log_c = auto_clean_light_curve(df_c, quality_bitmask=0, local_noise_kappa=LOCAL_NOISE_KAPPA)
            if len(df_c_clean) < 50:  # too few surviving points for a meaningful noise estimate
                print(f"Пропускаю {label}: после очистки осталось только {len(df_c_clean)} кадров "
                      f"(вероятно, плохо смоделированная/слабая звезда в этом окне)")
                continue
            t_c = df_c_clean["BTJD"].to_numpy()  # already BTJD, not full BJD (see t0 above)
            flux_c = df_c_clean["FLUX"].to_numpy()
            y_c_ppt = (flux_c / np.median(flux_c) - 1.0) * 1e3
            star_curves.append((label, t_c, y_c_ppt))

        origin_table = signal_origin_table(star_curves, target_freqs, fmax=CONFIG.prewhiten_fmax,
                                            snr_threshold=CONFIG.snr_stop)
        write_csv_with_provenance(origin_table, os.path.join(OUTDIR, "comparison_stars.csv"),
                                  star_name=STAR_NAME, sector=SECTOR, n_comparison=N_COMPARISON,
                                  comparison_mag_tol=COMPARISON_MAG_TOL,
                                  target_freqs=[round(f, 4) for f in target_freqs])
        pivot = origin_table.pivot(index="LABEL", columns="FREQUENCY", values="SNR_LOCAL").round(1)
        summary = (f"S/N_local по частотам цели (порог значимости S/N >= {CONFIG.snr_stop}; "
                   f"да/нет -- колонка PRESENT в comparison_stars.csv):\n{pivot.to_string()}")
        with open(os.path.join(OUTDIR, "comparison_stars.txt"), "w") as fo:
            fo.write(summary + "\n")
        print(f"\nПроверка происхождения сигнала:\n{summary}")
        print(f"Таблица звезд сравнения: {os.path.join(OUTDIR, 'comparison_stars.csv')}")
    else:
        print("\nНет значимых частот цели или звезд сравнения -- таблица происхождения сигнала пропущена")

    # --- карты амплитуд и разложение по звездам ---
    freqs_for_maps = AMP_MAP_FREQS_OVERRIDE or target_freqs
    if not freqs_for_maps:
        print("\nНет значимых частот цели и AMP_MAP_FREQS_OVERRIDE не задан -- карты амплитуд "
              "пропущены (задайте AMP_MAP_FREQS_OVERRIDE вручную, если нужно посмотреть карту для "
              "конкретной кандидатной частоты).")
    else:
        # вычитаем подогнанный плоский фон (иначе медленные вариации рассеянного света
        # ложатся на все пиксели и мешают на низких частотах). bkg уже посчитан выше
        # (один lstsq для всех кадров) и просто отфильтрован маской m вместе с box/t/fluxes --
        # пересчитывать deblend_prf_flux заново для этого не нужно, оно не зависит от
        # того, какое подмножество кадров решается: результат для каждого кадра
        # определяется только его собственными пикселями.
        bx = by = BOX
        xx, yy = np.meshgrid(np.arange(bx) - bx / 2, np.arange(by) - by / 2, indexing="ij")
        box_nobkg = box - (bkg[:, 0, None, None] + bkg[:, 1, None, None] * xx + bkg[:, 2, None, None] * yy)
        amaps = amplitude_maps(t, box_nobkg, freqs_for_maps)
        lines = ["(доли ориентировочные: часть амплитуды расходится по слабым звездам модели;",
                 " главное — какая звезда несет наибольшую долю)",
                 "ν, 1/сут   " + "  ".join(f"{int(s) % 100000:>12d}" for s in stars["source_id"]) +
                 "   | цель/(цель+сосед)    | A_цель / F_цель, ppt"]
        medF = np.median(fluxes, axis=0)
        for k, f in enumerate(freqs_for_maps):
            c = attribute_amplitude(amaps[k], models)
            share = c[it] / (c[it] + c[iN]) if iN is not None else c[it] / c.sum()
            lines.append(f"{f:8.4f}   " + "  ".join(f"{v:12.2f}" for v in c) +
                         f"   | {share:18.2f}  | {1e3 * c[it] / medF[it]:8.2f}")
        txt = "\n".join(lines)
        print("\nАмплитуда (e/s), приписанная каждой звезде (последние 5 цифр source_id):\n" + txt)
        with open(os.path.join(OUTDIR, "attribution.txt"), "w") as fo:
            fo.write(stars[["source_id", "t_mag", "px_x", "px_y"]].to_string() + "\n\n" + txt + "\n")

        # --- рисунки ---
        nf = len(freqs_for_maps)
        ncol = 4
        fig, axes = plt.subplots(int(np.ceil(nf / ncol)), ncol, figsize=(3.2 * ncol, 3.2 * np.ceil(nf / ncol)))
        axes_flat = np.atleast_1d(axes).flat
        for ax, k in zip(axes_flat, range(nf)):
            im = ax.imshow(np.abs(amaps[k]).T, origin="lower", cmap="viridis")
            for j, s in stars.iterrows():
                col = "r" if j == it else ("w" if j == iN else "0.7")
                ax.plot(s["px_x"] - 1 - x0, s["px_y"] - 1 - y0, "+", color=col, ms=10 if j in (it, iN) else 5)
            circ = plt.Circle((tgt["px_x"] - 1 - x0, tgt["px_y"] - 1 - y0), CONFIG.aperture_radius,
                              fill=False, color="r", ls="--")
            ax.add_patch(circ)
            ax.set_title(f"ν = {freqs_for_maps[k]:.4f}", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)
        for ax in list(axes_flat)[nf:]:
            ax.axis("off")
        fig.suptitle(f"|A| по пикселям (e/s); красный + {STAR_NAME} и апертура "
                     f"r={CONFIG.aperture_radius}, белый + доминирующий сосед")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, "amp_maps.pdf"))

    fig, ax = plt.subplots(figsize=(12, 4))
    for j, tag, col in [(it, f"{STAR_NAME} (PRF)", "k"), (iN, "сосед (PRF)", "C3")]:
        if j is None:
            continue
        ax.plot(t, 1e3 * (fluxes[:, j] / np.median(fluxes[:, j]) - 1), ".", ms=1, color=col, label=tag)
    ax.set_xlabel("BTJD"); ax.set_ylabel("ΔF / F, ppt"); ax.legend(markerscale=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "lc_deblended.pdf"))
    print(f"\nГотово: {OUTDIR}/")

    if RUN_DIAGNOSTICS_AFTER:
        import Peridogram_diagnostics
        Peridogram_diagnostics.run_all(runs=[(os.path.join(star_dir, f"light_curve_sector_{SECTOR}_prf_clean.csv"), "prf"),
                                 (os.path.join(OUTDIR, "lc_deblended_neighbour.csv"), "neighbour")],
                           show=False)
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
