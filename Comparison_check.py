"""
Проверка звезд сравнения для SS 397 (ответ рецензенту / коллегам: "не общий ли
это сигнал поля или инструмента?").

Что делает:
  1. Берет ту же вырезку, те же кадры (маска QUALITY) и тот же PRF-деблендинг,
     что PRF_cleaning.py, и строит PRF-кривые для звезд сравнения: звезды поля
     сопоставимого блеска (|T - T_цели| <= MAG_TOL), окно BOX x BOX целиком в
     вырезке. Звезды, у которых ближе MIN_SEP_PX пикселя есть звезда ярче или
     сравнимая (T < T_звезды + 1), помечаются как "тесные": их кривая
     вырождена с соседом, в выводах на них не опираемся (но считаем и
     показываем).
  2. Для каждой кривой (цель, яркий сосед, звезды сравнения):
       - выбеливание (isolated.prewhitening.prewhiten, S/N >= 4) -- собственные
         значимые частоты звезды;
       - подгонка на ФИКСИРОВАННЫХ частотах цели (все сразу + тренд 2-й степени)
         -> амплитуда каждой частоты цели в этой звезде, ошибка по
         Монтгомери-О'Донохью x sqrt(D) и S/N относительно шума спектра
         ОСТАТКОВ (в отличие от comparison_stars.csv из PRF_cleaning.py, где
         шум брался из спектра до выбеливания и сама цель не проходила порог);
       - амплитуда в долях потока SS 397 (A_звезды * F_звезды / F_цели): какой
         вклад звезда могла бы дать, если бы ее свет попал в кривую цели;
       - динамический спектр (окно DYN_WINDOW, шаг DYN_STEP) и трекинг
         сильнейшего пика в DYN_BAND: есть ли устойчивая частота.
  3. Для каждой звезды сравнения дополнительно -- полный стандартный прогон
     Peridogram_diagnostics.analyse (отчет, CLEAN, стабильность амплитуд/фаз),
     чтобы все звезды были обработаны тем же методом, что цель.

Выход (OUTDIR):
  lc_comp_<id>.csv             -- PRF-кривые звезд сравнения (BTJD, FRAME, FLUX)
  comparison_summary.txt       -- сводка (ее и присылать)
  comparison_fixed_freqs.csv   -- амплитуды/S/N на частотах цели по всем звездам
  comparison_dynamic.pdf       -- амплитудные и динамические спектры всех звезд
  comp_<id>_report.txt и пр.   -- стандартные отчеты Peridogram_diagnostics

Запуск из PyCharm: поправь блок НАСТРОЙКИ и нажми Run. Файл лежит в корне
проекта (рядом с PRF_cleaning.py). Нужна вырезка 50x50 (как для PRF_cleaning.py).
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from isolated.data_io import load_tess_cutouts, load_gaia_stars_in_view_data, load_star_gaia_data, find_target_row
from isolated.psf import get_tesscut_prf_supersampled
from isolated.geometry import get_nospace_star_name, calc_tmag_from_gaia
from isolated.cleaning import quality_mask, auto_clean_light_curve
from isolated.localize import deblend_star_curve
from isolated.prewhitening import (prewhiten, detrend_poly, amp_spectrum, local_noise, errors_mo99,
                                   dynamic_spectrum)
from isolated.provenance import write_csv_with_provenance
from isolated.plotting import finish_figure, set_show_plots, print_saved_summary
from run_config import CONFIG

# =============================================================================
# НАСТРОЙКИ
# =============================================================================
STAR_NAME = CONFIG.star_name
SECTOR = CONFIG.sector
CUT = CONFIG.cut_width
BOX = CONFIG.prf_box
DT_MAX = CONFIG.prf_dt_max
MERGE_PX = CONFIG.prf_merge_px
QUALITY_BITMASK = CONFIG.quality_bitmask
KAPPA = CONFIG.local_noise_kappa

MAG_TOL = 1.5          # |T - T_цели| для звезд сравнения (в PRF_cleaning.py было 1.0 -> всего 3 звезды)
N_COMP_MAX = 10        # максимум звезд сравнения
MIN_SEP_PX = 2.0       # ближе -- "тесная" звезда (вырождение PRF с соседом)
NEIGHBOUR_ID = 4154999332652688000   # яркий сосед SS 397 (его кривая уже есть в out_localize/)

# частоты цели для проверки (по умолчанию -- свежий прогон выбеливания по кривой цели);
# можно задать вручную списком, например частоты из табл. 2 статьи
TARGET_FREQS_OVERRIDE = None
FMAX = CONFIG.prewhiten_fmax
SNR_STOP = CONFIG.snr_stop
NOISE_BOX = 1.0        # ширина окна для локального шума, 1/сут (как в prewhiten: +-0.5)

DYN_WINDOW = CONFIG.dyn_window   # 10 сут
DYN_STEP = CONFIG.dyn_step       # 0.25 сут
DYN_FRANGE = CONFIG.dyn_frange   # (0.8, 2.5)
DYN_BAND = (1.3, 1.9)            # где ищем устойчивый пик (группа g1)

RUN_FULL_DIAGNOSTICS = True      # стандартный Peridogram_diagnostics.analyse для каждой звезды сравнения
OUTDIR = "out_compare"
SHOW_PLOTS = CONFIG.show_plots   # False: рисунки только сохраняются, пути -- в консоли (run_config.py)


# =============================================================================
def ppt(flux):
    return (flux / np.median(flux) - 1.0) * 1e3


def fixed_freq_fit(t, y, freqs):
    """Линейная подгонка синусоид на фиксированных частотах + тренд 2-й степени.
    Возвращает амплитуды (ppt), ошибки (MO99 x sqrt(D)), S/N к шуму спектра остатков, D."""
    t = np.asarray(t, float)
    tt = t - t.mean()
    cols = [np.ones_like(tt), tt, tt ** 2]
    for f in freqs:
        cols += [np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]
    X = np.array(cols).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    c = coef[3:].reshape(-1, 2)
    amps = np.hypot(c[:, 0], c[:, 1])
    _sf, sa, _sp, d, _sn = errors_mo99(t, res, np.maximum(amps, 1e-9))
    T = t.max() - t.min()
    grid = np.arange(0.5 / T, FMAX, 1.0 / (10 * T))
    A_res = amp_spectrum(t, res, grid)
    noise = np.array([local_noise(grid, A_res, f, NOISE_BOX) for f in freqs])
    return amps, sa, amps / noise, noise, d


def dyn_summary(t, y):
    centers, freqs, dyn = dynamic_spectrum(t, y, window=DYN_WINDOW, step=DYN_STEP, frange=DYN_FRANGE)
    band = (freqs >= DYN_BAND[0]) & (freqs <= DYN_BAND[1])
    ok = np.isfinite(dyn).all(axis=1)
    j = np.argmax(dyn[ok][:, band], axis=1)
    fpk = freqs[band][j]
    apk = dyn[ok][:, band][np.arange(j.size), j]
    return dict(centers=centers, freqs=freqs, dyn=dyn, f_lo=fpk.min(), f_hi=fpk.max(), f_med=np.median(fpk),
                f_iqr=np.subtract(*np.percentile(fpk, [75, 25])), a_lo=apk.min(), a_hi=apk.max(),
                a_med=np.median(apk), n_win=int(ok.sum()))


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(OUTDIR, exist_ok=True)
    set_show_plots(SHOW_PLOTS)
    star_dir = os.path.join("stars_python", get_nospace_star_name(STAR_NAME), f"{CUT}x{CUT}")

    # --- те же данные и кадры, что в PRF_cleaning.py ---
    cut_fits = load_tess_cutouts(STAR_NAME, CUT, CUT, sector=SECTOR)[SECTOR]
    data = cut_fits[1].data
    t_all = np.asarray(data["TIME"], float)
    q_all = np.asarray(data["QUALITY"], int)
    frame_no_all = np.arange(1, t_all.size + 1)
    cube = np.transpose(np.asarray(data["FLUX"], float), (0, 2, 1))
    prf = get_tesscut_prf_supersampled(cut_fits)
    gaia_data = load_star_gaia_data(STAR_NAME)
    gaia = load_gaia_stars_in_view_data(STAR_NAME, cut_fits, d_mag_r=6.0)
    tgt = find_target_row(gaia, gaia_data, SECTOR)
    TARGET_ID = int(tgt["source_id"])
    good = np.isfinite(t_all) & quality_mask(q_all, QUALITY_BITMASK)
    cube_good, t_q, fr_q = cube[good], t_all[good], frame_no_all[good]

    # --- кривые: цель и сосед тем же деблендингом в окне цели (как в PRF_cleaning.py) ---
    curves = []   # (label, source_id, T, t, flux, note)
    gaia = gaia.copy()
    gaia["t_mag"] = calc_tmag_from_gaia(gaia["phot_g_mean_mag"], gaia["bp_rp"])
    T_tgt = float(gaia.loc[gaia["source_id"] == TARGET_ID, "t_mag"].iloc[0])

    def clean_and_store(label, sid, flux, note):
        df = pd.DataFrame({"BTJD": t_q, "FRAME": fr_q, "FLUX": flux})
        dfc, _log = auto_clean_light_curve(df, quality_bitmask=0, local_noise_kappa=KAPPA)
        if len(dfc) < 1000:
            print(f"Пропускаю {label}: после очистки {len(dfc)} кадров")
            return None
        T_s = float(gaia.loc[gaia["source_id"] == sid, "t_mag"].iloc[0])
        curves.append((label, sid, T_s, dfc["BTJD"].to_numpy(), dfc["FLUX"].to_numpy(), note))
        return dfc

    for sid, label, note in [(TARGET_ID, "SS 397", "цель"), (NEIGHBOUR_ID, "сосед", "яркий сосед, 3.2 пикс")]:
        try:
            flux, *_ = deblend_star_curve(cube_good, prf, gaia, CUT, CUT, sid, box=BOX, dt_max=DT_MAX,
                                          merge_px=MERGE_PX)
        except ValueError as exc:
            print(f"{label}: {exc}")
            continue
        clean_and_store(label, sid, flux, note)

    # --- звезды сравнения ---
    cand = gaia[(gaia["source_id"] != TARGET_ID) & (gaia["source_id"] != NEIGHBOUR_ID)].copy()
    cand = cand[(cand["t_mag"] - T_tgt).abs() <= MAG_TOL]
    bx0 = np.round(cand["px_x"]).astype(int) - BOX // 2
    by0 = np.round(cand["px_y"]).astype(int) - BOX // 2
    cand = cand[(bx0 >= 1) & (bx0 + BOX - 1 <= CUT) & (by0 >= 1) & (by0 + BOX - 1 <= CUT)]

    def nearest_rival(r):
        d = np.hypot(gaia["px_x"] - r["px_x"], gaia["px_y"] - r["px_y"])
        m = (gaia["t_mag"] < r["t_mag"] + 1.0) & (d > 0.01)
        return float(d[m].min()) if m.any() else np.inf

    cand["sep_rival"] = cand.apply(nearest_rival, axis=1)
    cand["dist_tgt"] = np.hypot(cand["px_x"] - tgt["px_x"], cand["px_y"] - tgt["px_y"])
    cand = cand.sort_values("t_mag").head(N_COMP_MAX)
    print("\nКандидаты в звезды сравнения:")
    print(cand[["source_id", "t_mag", "px_x", "px_y", "sep_rival", "dist_tgt"]].to_string())

    for _, r in cand.iterrows():
        sid = int(r["source_id"])
        label = f"...{sid % 100000:05d}"
        close = r["sep_rival"] < MIN_SEP_PX
        note = (f"T={r['t_mag']:.2f}, {r['dist_tgt']:.0f} пикс от цели, ближайшая сравнимая звезда "
                f"{r['sep_rival']:.1f} пикс" + (" -- ТЕСНАЯ, кривая ненадежна" if close else ""))
        try:
            flux, *_ = deblend_star_curve(cube_good, prf, gaia, CUT, CUT, sid, box=BOX, dt_max=DT_MAX,
                                          merge_px=MERGE_PX)
        except ValueError as exc:
            print(f"Пропускаю {label}: {exc}")
            continue
        dfc = clean_and_store(label, sid, flux, note)
        if dfc is not None:
            write_csv_with_provenance(dfc, os.path.join(OUTDIR, f"lc_comp_{sid}.csv"), star_name=STAR_NAME,
                                      sector=SECTOR, source_id=sid, box=BOX, note=note)

    analyse_curves(curves)


def analyse_curves(curves):
    """curves: список (label, source_id, T_mag, t, flux, note); первая -- цель, вторая -- сосед."""
    # --- частоты цели ---
    _, _, _, t0, f0, _ = curves[0]
    y0 = ppt(f0)
    if TARGET_FREQS_OVERRIDE:
        target_freqs = list(TARGET_FREQS_OVERRIDE)
    else:
        pk, ex = prewhiten(t0, y0, fmax=FMAX, nmax=CONFIG.n_max_freq, snr_stop=SNR_STOP)
        target_freqs = [p["frequency"] for p in pk if p["frequency"] > 2.0 / ex["T"]]
    target_freqs = np.array(sorted(target_freqs))
    F_tgt = np.median(f0)

    # --- анализ всех кривых ---
    rows, rep, dyns = [], [], []
    P = rep.append
    P(f"# Проверка звезд сравнения: {STAR_NAME}, сектор {SECTOR}, вырезка {CUT}x{CUT}, PRF-окно {BOX}x{BOX}")
    P(f"# Частоты цели: " + ", ".join(f"{f:.4f}" for f in target_freqs))
    P(f"# Звезды сравнения: |T - T_цели| <= {MAG_TOL}; 'тесная' -- сравнимая звезда ближе {MIN_SEP_PX} пикс")
    P("")
    for label, sid, T_s, t, flux, note in curves:
        y = ppt(flux)
        T = t.max() - t.min()
        p2p = 1e3 * np.median(np.abs(np.diff(flux))) / np.median(flux) / np.sqrt(2) * 1.4826
        own, own_ex = prewhiten(t, y, fmax=FMAX, nmax=CONFIG.n_max_freq, snr_stop=SNR_STOP)
        own_f = [(p["frequency"], p["amplitude"], p["snr_local"]) for p in own if p["frequency"] > 2.0 / T]
        amps, sa, snr, noise, d = fixed_freq_fit(t, y, target_freqs)
        scale = np.median(flux) / F_tgt
        ds = dyn_summary(t, y)
        dyns.append((label, t, y, ds, note))
        for f, a, e, s, n in zip(target_freqs, amps, sa, snr, noise):
            rows.append(dict(LABEL=label, SOURCE_ID=sid, T_MAG=round(T_s, 2), FREQUENCY=round(f, 4),
                             AMP_PPT=a, AMP_ERR_PPT=e, NOISE_PPT=n, SNR=s, AMP_IN_SS397_PPT=a * scale,
                             CLOSE="ТЕСНАЯ" in note))
        P(f"## {label} (Gaia DR3 {sid}): {note}")
        P(f"   N = {t.size}, шум точка-к-точке {p2p:.2f} ppt, поток/поток SS 397 = {scale:.2f}, D = {d:.1f}")
        P("   Собственные значимые частоты (выбеливание, S/N>=4, nu > 2/T):")
        if own_f:
            for f, a, s in own_f:
                ft = target_freqs[np.argmin(np.abs(target_freqs - f))]
                mark = label != "SS 397" and abs(ft - f) < 1.5 / T
                P(f"      {f:8.4f} 1/сут  A = {a:6.2f} ppt  S/N = {s:5.1f}"
                  + (f"   <-- в пределах 1.5/T от частоты цели {ft:.4f}" if mark else ""))
        else:
            P("      нет")
        P("   На частотах цели (подгонка на фиксированных частотах):")
        P("      nu, 1/сут     A, ppt         S/N    A в долях потока SS 397, ppt")
        for f, a, e, s in zip(target_freqs, amps, sa, snr):
            P(f"      {f:8.4f}   {a:6.2f} ± {e:4.2f}   {s:5.1f}   {a * scale:7.2f}")
        P(f"   Динамический спектр (окно {DYN_WINDOW} сут, шаг {DYN_STEP} сут, {ds['n_win']} окон): сильнейший пик "
          f"в {DYN_BAND[0]}–{DYN_BAND[1]} 1/сут -- частота {ds['f_lo']:.3f}–{ds['f_hi']:.3f} "
          f"(медиана {ds['f_med']:.3f}, IQR {ds['f_iqr']:.3f}), амплитуда {ds['a_lo']:.2f}–{ds['a_hi']:.2f} ppt "
          f"(медиана {ds['a_med']:.2f})")
        P("")

    tab = pd.DataFrame(rows)
    write_csv_with_provenance(tab, os.path.join(OUTDIR, "comparison_fixed_freqs.csv"), star_name=STAR_NAME,
                              sector=SECTOR, mag_tol=MAG_TOL, target_freqs=[round(f, 4) for f in target_freqs])

    # сводная таблица S/N на частотах цели
    piv = tab.pivot_table(index="LABEL", columns="FREQUENCY", values="SNR", sort=False).round(1)
    P("Сводка: S/N на частотах цели (шум -- спектр остатков после подгонки этих частот, окно ±0.5 1/сут)")
    P(piv.to_string())
    comp = tab[(tab["LABEL"] != "SS 397") & (tab["LABEL"] != "сосед") & (~tab["CLOSE"])]
    if len(comp):
        g1 = comp[(comp["FREQUENCY"] > 1.4) & (comp["FREQUENCY"] < 1.9)]
        P(f"\nНетесные звезды сравнения: {comp['LABEL'].nunique()}; максимальное S/N на частотах цели "
          f"{comp['SNR'].max():.1f}; на частотах группы g1 -- амплитуда до {g1['AMP_PPT'].max():.2f} ppt "
          f"собственного потока (до {g1['AMP_IN_SS397_PPT'].max():.2f} ppt в долях потока SS 397).")
    txt = "\n".join(rep)
    print("\n" + txt)
    with open(os.path.join(OUTDIR, "comparison_summary.txt"), "w") as fo:
        fo.write(txt + "\n")

    # --- рисунок: амплитудный спектр + динамический спектр для каждой звезды ---
    n = len(dyns)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.2 * n), gridspec_kw=dict(width_ratios=[1.3, 1]), squeeze=False)
    for k, (label, t, y, ds, note) in enumerate(dyns):
        T = t.max() - t.min()
        fg = np.arange(0.5 / T, 6.0, 1 / (10 * T))
        A = amp_spectrum(t, detrend_poly(t, y), fg)
        ax = axes[k, 0]
        ax.plot(fg, A, lw=0.6, color="k")
        for f in target_freqs:
            ax.axvline(f, color="C3", lw=0.5, ls=":")
        ax.set_xlim(0, 6); ax.set_ylabel("A, ppt")
        ax.text(0.99, 0.9, f"{label}  {note.split(',')[0]}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax2 = axes[k, 1]
        im = ax2.pcolormesh(ds["freqs"], ds["centers"], ds["dyn"], shading="auto", cmap="viridis")
        for f in target_freqs[(target_freqs > DYN_FRANGE[0]) & (target_freqs < DYN_FRANGE[1])]:
            ax2.axvline(f, color="w", lw=0.4, ls=":")
        plt.colorbar(im, ax=ax2, label="A, ppt")
        ax2.set_ylabel("BTJD")
    axes[-1, 0].set_xlabel("Частота, 1/сут"); axes[-1, 1].set_xlabel("Частота, 1/сут")
    fig.tight_layout()
    finish_figure(fig, os.path.join(OUTDIR, "comparison_dynamic.pdf"), dpi=None)

    if RUN_FULL_DIAGNOSTICS:
        import Peridogram_diagnostics as PD
        for label, sid, *_ in curves[2:]:
            path = os.path.join(OUTDIR, f"lc_comp_{sid}.csv")
            PD.analyse(path, f"comp_{sid}", OUTDIR, PD.FMAX, nmax=PD.N_MAX, dyn_win=PD.DYN_WINDOW,
                       dyn_frange=PD.DYN_FRANGE, dyn_freq_step=PD.DYN_FREQ_STEP, kappa=PD.KAPPA,
                       n_sigma_combination=PD.N_SIGMA_COMBINATION, do_bootstrap=False,
                       do_stability=True, stability_window=PD.STABILITY_WINDOW,
                       stability_step=PD.STABILITY_STEP, do_clean=True, clean_gain=PD.CLEAN_GAIN,
                       clean_niter=PD.CLEAN_NITER)
    print_saved_summary("Comparison_check: что где лежит", extra_files=[
        os.path.join(OUTDIR, n) for n in ["comparison_summary.txt", "comparison_fixed_freqs.csv"] +
        [f"lc_comp_{sid}.csv" for _l, sid, *_ in curves[2:]] +
        [f"comp_{sid}_{k}" for _l, sid, *_ in curves[2:] for k in ("report.txt", "freqs.tex")]])
    print(f"Главное -- {os.path.abspath(os.path.join(OUTDIR, 'comparison_summary.txt'))}")
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
