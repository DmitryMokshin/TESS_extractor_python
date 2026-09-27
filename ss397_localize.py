"""
Чей сигнал? Локализация частот TESS по пикселям + PRF-разделение кривых блеска.

Зачем. В апертуре r = 3 пикс вокруг SS 397 (T ≈ 10.8) на расстоянии 3.2 пикс (67″)
сидит звезда Gaia DR3 4154999332652688000 (T ≈ 9.7, в 2.8 раза ярче; красный гигант).
По модели PRF на SS 397 приходится лишь ~43 % потока в апертуре, на соседа ~46 %.
Код пайплайна вычитает только плоский фон, поэтому свет соседей остается в кривой.

Что делает скрипт:
 1. Берет вырезку сектора и выбирает кадры: QUALITY_BITMASK (стандартная маска
    lightkurve, 175). Дальнейшая (окно-по-окну) чистка -- отдельный шаг через
    уже существующие TESS_cleaning.py/pick_exclusion_windows/exclude_frame_windows
    поверх сохраненного здесь light_curve_sector_{SECTOR}_prf.csv, как и для
    апертурного режима -- этот скрипт больше не требует заранее посчитанного
    light_curve_sector_{SECTOR}_clean.csv.
 2. PRF-фотометрия (isolated.prf_photometry): в окне BOX×BOX вокруг звезды каждый
    кадр описывается суммой PRF звезд Gaia (положения фиксированы) + плоский фон.
    Потоки звезд линейны, решение точное и быстрое (один lstsq на все кадры сразу).
    На выходе отдельные кривые SS 397 и соседа (CSV в формате пайплайна).
 3. Карты амплитуд: в каждом пикселе подгоняются синусоиды на частотах FREQS
    (+ квадратичный тренд). Комплексная карта амплитуды раскладывается по PRF звезд:
    видно, какая звезда несет каждую частоту. (Это все еще живет здесь, а не в
    пакете -- обобщенная версия для произвольной звезды/звезд сравнения --
    отдельный, более поздний этап ROADMAP.md, "Этап 6. Проверка происхождения
    сигнала".)

Выход: out_localize/
    amp_maps.pdf              — карты амплитуд с положениями звезд
    attribution.txt           — доли амплитуды по звездам для каждой частоты
    lc_deblended_<id>.csv     — PRF-кривые (BTJD, FRAME, FLUX) для SS 397 и соседа
    lc_deblended.pdf          — сравнение кривых
и в stars_python/SS_397/50x50/:
    light_curve_sector_80_prf.csv        — канонический PRF-выход пайплайна
                                            (QUALITY-фильтр, без доп. чистки;
                                            то же самое, что дал бы
                                            isolated.prf_photometry.load_prf_light_curve)
    light_curve_sector_80_prf_clean.csv  — тот же, плюс отбраковка шумных участков
                                            (см. п.1 выше), для быстрого CLEAN/LS
Если RUN_TESS_AFTER = True, сразу после этого разделенные кривые прогоняются через
ss397_tess.py (частоты, ошибки, динамический спектр) — результаты в out_tess/.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from isolated.data_io import load_tess_cutouts, load_gaia_stars_in_view_data
from isolated.psf import get_tesscut_prf_supersampled
from isolated.geometry import get_nospace_star_name, calc_tess_magnitude
from isolated.lightcurve_tools import local_point_to_point_sigma
from isolated.cleaning import quality_mask, auto_clean_light_curve, save_cleaning_log, summarize_cleaning_log
from isolated.prf_photometry import select_prf_model_stars, build_prf_star_cuts, deblend_prf_flux
from run_config import CONFIG

# =============================================================================
# НАСТРОЙКИ — звезда/сектор/вырезка/чистка теперь в run_config.py (Этап 4);
# то, что осталось здесь, специфично для этого конкретного исследования.
# =============================================================================
STAR_NAME = CONFIG.star_name
SECTOR = CONFIG.sector
CUT = CONFIG.cut_width
BOX = CONFIG.prf_box            # размер окна для PRF-фотометрии (нечетный), пикс
DT_MAX = CONFIG.prf_dt_max       # включать звезды Gaia не слабее T_target + DT_MAX
MERGE_PX = CONFIG.prf_merge_px   # звезды ближе MERGE_PX к более яркой не фитуются отдельно (вырождение)
NEIGHBOUR_ID = 4154999332652688000
TARGET_ID = 4155000844481174656
# частоты для карт (из анализа clean-кривой), 1/сут
FREQS = [1.5810, 1.6333, 1.5182, 1.6593, 1.7412, 0.1295, 0.0895, 0.1972]
OUTDIR = "out_localize"
# Какие флаги QUALITY выбрасывают кадр. 175 = стандартная маска lightkurve "default":
# 1 AttitudeTweak, 2 SafeMode, 4 CoarsePoint, 8 EarthPoint, 32 Desat, 128 ManualExclude.
# Флаги рассеянного света (2048, 4096 и т.п.) НЕ выбрасываем: такие кадры часто
# нормальные, плохие участки потом отсекаются по локальному шуму.
# 0 — не использовать флаги вообще; None — брать только QUALITY == 0 (слишком строго для S80).
QUALITY_BITMASK = CONFIG.quality_bitmask
LOCAL_NOISE_KAPPA = CONFIG.local_noise_kappa  # отбраковка "_prf_clean": локальный шум > KAPPA × медиана
RUN_TESS_AFTER = True     # сразу прогнать ss397_tess.analyse по разделенным кривым
SHOW_PLOTS = True         # открыть окна с рисунками в конце (PDF сохраняются всегда)


# =============================================================================
# карты амплитуд / разложение по звездам (Этап 6 ROADMAP.md -- пока только здесь)
# =============================================================================
def amplitude_maps(t, cube_box, freqs):
    """Комплексная амплитуда каждой частоты в каждом пикселе (e/s): A = a_cos + i a_sin."""
    tt = t - t.mean()
    cols = [np.ones_like(tt), tt, tt ** 2]
    for f in freqs:
        cols += [np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]
    X = np.array(cols).T
    D = np.nan_to_num(cube_box.reshape(cube_box.shape[0], -1))
    coef, *_ = np.linalg.lstsq(X, D, rcond=None)
    c = coef[3:].reshape(len(freqs), 2, *cube_box.shape[1:])
    return c[:, 0] + 1j * c[:, 1]                 # (nfreq, box, box)


def attribute(amap, models):
    """Разложение комплексной карты амплитуды по PRF звезд (+ константа).
    Возвращает |c_j| (e/s) — амплитуду, которую несет каждая звезда."""
    X = np.c_[models.reshape(models.shape[0], -1).T, np.ones(models[0].size)]
    y = amap.ravel()
    cr, *_ = np.linalg.lstsq(X, y.real, rcond=None)
    ci, *_ = np.linalg.lstsq(X, y.imag, rcond=None)
    return np.abs(cr[:-1] + 1j * ci[:-1])


def export_clean_lc(t, flux, dat_path, csv_path=None, star="SS 397", sector=80, note="", frame_no=None):
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
    hdr = (f"{star}, TESS sector {sector}: PRF-photometry light curve (deblended from Gaia neighbours)\n"
           f"frames: TESScut FFI, cadence 200 s; QUALITY bitmask {QUALITY_BITMASK} removed; points with local\n"
           f"point-to-point noise > {LOCAL_NOISE_KAPPA} x median removed. {note}\n"
           f"N = {t.size}, median flux = {med:.1f} e/s, median sigma = {np.median(err):.2f} ppt\n"
           f"columns: BJD-2457000  dF/F[ppt]  sigma[ppt]  dmag[mag]  FLUX[e/s]")
    np.savetxt(dat_path, np.c_[t, ppt, err, dmag, flux], fmt="%.6f %9.3f %7.3f %9.5f %11.3f", header=hdr)
    if csv_path:
        pd.DataFrame({"BTJD": t, "FRAME": frame_no, "FLUX": flux, "MAG": calc_tess_magnitude(flux)}).to_csv(
            csv_path, index=False)
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

    gaia = load_gaia_stars_in_view_data(STAR_NAME, cut_fits, d_mag_r=6.0)
    tgt = gaia[gaia["source_id"] == TARGET_ID].iloc[0]
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
    inb = np.where(stars["source_id"] == NEIGHBOUR_ID)[0]
    iN = int(inb[0]) if inb.size else None

    # канонический PRF-выход пайплайна (без доп. чистки) -- то же самое, что дал бы
    # isolated.prf_photometry.load_prf_light_curve; сохраняем сами, чтобы не гонять
    # деблендинг по 13000 кадров дважды
    os.makedirs(star_dir, exist_ok=True)
    pd.DataFrame({"BTJD": t, "FRAME": frame_no, "FLUX": fluxes[:, it],
                  "MAG": calc_tess_magnitude(np.abs(fluxes[:, it]))}).to_csv(
        os.path.join(star_dir, f"light_curve_sector_{SECTOR}_prf.csv"), index=False)

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

    for j, tag in [(it, "SS397"), (iN, "neighbour")]:
        if j is None:
            continue
        pd.DataFrame({"BTJD": t, "FRAME": frame_no, "FLUX": fluxes[:, j]}).to_csv(
            os.path.join(OUTDIR, f"lc_deblended_{tag}.csv"), index=False)
        print(f"{tag}: медианный поток {np.median(fluxes[:, j]):.0f} e/s "
              f"(ожидание по Gaia {10 ** (0.4 * (20.44 - stars.loc[j, 't_mag'])):.0f})")

    # очищенная кривая SS 397 для коллег + копия в формате пайплайна
    dat = export_clean_lc(t, fluxes[:, it], os.path.join(OUTDIR, f"SS397_TESS_S{SECTOR}_PRF.dat"),
                          csv_path=os.path.join(star_dir, f"light_curve_sector_{SECTOR}_prf_clean.csv"),
                          star=STAR_NAME, sector=SECTOR, frame_no=frame_no)
    print(f"Кривая для коллег: {dat}")

    # --- карты амплитуд и разложение по звездам ---
    # вычитаем подогнанный плоский фон (иначе медленные вариации рассеянного света
    # ложатся на все пиксели и мешают на низких частотах). bkg уже посчитан выше
    # (один lstsq для всех кадров) и просто отфильтрован маской m вместе с box/t/fluxes --
    # пересчитывать deblend_prf_flux заново для этого не нужно, оно не зависит от
    # того, какое подмножество кадров решается: результат для каждого кадра
    # определяется только его собственными пикселями.
    bx = by = BOX
    xx, yy = np.meshgrid(np.arange(bx) - bx / 2, np.arange(by) - by / 2, indexing="ij")
    box_nobkg = box - (bkg[:, 0, None, None] + bkg[:, 1, None, None] * xx + bkg[:, 2, None, None] * yy)
    amaps = amplitude_maps(t, box_nobkg, FREQS)
    lines = ["(доли ориентировочные: часть амплитуды расходится по слабым звездам модели;",
             " главное — какая звезда несет наибольшую долю)",
             "ν, 1/сут   " + "  ".join(f"{int(s) % 100000:>12d}" for s in stars["source_id"]) +
             "   | SS397/(SS397+сосед) | A_SS397 / F_SS397, ppt"]
    medF = np.median(fluxes, axis=0)
    for k, f in enumerate(FREQS):
        c = attribute(amaps[k], models)
        share = c[it] / (c[it] + c[iN]) if iN is not None else c[it] / c.sum()
        lines.append(f"{f:8.4f}   " + "  ".join(f"{v:12.2f}" for v in c) +
                     f"   | {share:18.2f}  | {1e3 * c[it] / medF[it]:8.2f}")
    txt = "\n".join(lines)
    print("\nАмплитуда (e/s), приписанная каждой звезде (последние 5 цифр source_id):\n" + txt)
    with open(os.path.join(OUTDIR, "attribution.txt"), "w") as fo:
        fo.write(stars[["source_id", "t_mag", "px_x", "px_y"]].to_string() + "\n\n" + txt + "\n")

    # --- рисунки ---
    nf = len(FREQS)
    ncol = 4
    fig, axes = plt.subplots(int(np.ceil(nf / ncol)), ncol, figsize=(3.2 * ncol, 3.2 * np.ceil(nf / ncol)))
    for ax, k in zip(axes.flat, range(nf)):
        im = ax.imshow(np.abs(amaps[k]).T, origin="lower", cmap="viridis")
        for j, s in stars.iterrows():
            col = "r" if j == it else ("w" if j == iN else "0.7")
            ax.plot(s["px_x"] - 1 - x0, s["px_y"] - 1 - y0, "+", color=col, ms=10 if j in (it, iN) else 5)
        circ = plt.Circle((tgt["px_x"] - 1 - x0, tgt["px_y"] - 1 - y0), 3, fill=False, color="r", ls="--")
        ax.add_patch(circ)
        ax.set_title(f"ν = {FREQS[k]:.4f}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
    for ax in list(axes.flat)[nf:]:
        ax.axis("off")
    fig.suptitle("|A| по пикселям (e/s); красный + SS 397 и апертура r=3, белый + яркий сосед")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "amp_maps.pdf"))

    fig, ax = plt.subplots(figsize=(12, 4))
    for j, tag, col in [(it, "SS 397 (PRF)", "k"), (iN, "сосед (PRF)", "C3")]:
        if j is None:
            continue
        ax.plot(t, 1e3 * (fluxes[:, j] / np.median(fluxes[:, j]) - 1), ".", ms=1, color=col, label=tag)
    ax.set_xlabel("BTJD"); ax.set_ylabel("ΔF / F, ppt"); ax.legend(markerscale=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "lc_deblended.pdf"))
    print(f"\nГотово: {OUTDIR}/")

    if RUN_TESS_AFTER:
        import ss397_tess
        ss397_tess.run_all(runs=[(os.path.join(star_dir, f"light_curve_sector_{SECTOR}_prf_clean.csv"), "ss397_prf"),
                                 (os.path.join(OUTDIR, "lc_deblended_neighbour.csv"), "neighbour")],
                           show=False)
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
