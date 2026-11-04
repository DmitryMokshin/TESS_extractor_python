"""
Проверка окружения цели: соседи и звезды сравнения (стандарты), их
переменность и динамический спектр. Отвечает на вопрос "не общий ли это
сигнал поля или инструмента?" для любой звезды из run_config.py.

Что делает:
  1. Берет ту же вырезку, те же кадры (маска QUALITY) и тот же PRF-деблендинг,
     что PRF_cleaning.py, и строит PRF-кривые трех ролей:
       - target    -- сама цель;
       - neighbour -- до N_NEIGHBOURS соседей, чьи PRF дают наибольшую долю
                      потока в апертуре цели (isolated.photometry.rank_contaminants,
                      доля >= NEIGHBOUR_MIN_FRACTION; можно задать вручную --
                      NEIGHBOUR_IDS_OVERRIDE);
       - standard  -- до N_STANDARDS звезд поля сопоставимого блеска
                      (|T - T_цели| <= MAG_TOL), окно BOX x BOX целиком в
                      вырезке. Звезды, у которых ближе MIN_SEP_PX пикселя есть
                      звезда ярче или сравнимая (T < T_звезды + 1), помечаются
                      как "тесные": их кривая вырождена с соседом, в выводах на
                      них не опираемся (но считаем и показываем).
  2. Для каждой кривой:
       - переменность: разброс кривой после тренда (RMS_PPT) и разброс
         средних по бинам 0.25 сут (RMS_BINNED_PPT -- отделяет реальную
         медленную переменность от шума отдельных точек);
       - выбеливание (isolated.prewhitening.prewhiten, S/N >= 4) -- собственные
         значимые частоты звезды;
       - подгонка на ФИКСИРОВАННЫХ частотах цели (все сразу + тренд 2-й степени)
         -> амплитуда каждой частоты цели в этой звезде, ошибка по
         Монтгомери-О'Донохью x sqrt(D) и S/N относительно шума спектра
         ОСТАТКОВ;
       - амплитуда в долях потока цели (A_звезды * F_звезды / F_цели): какой
         вклад звезда могла бы дать, если бы ее свет попал в кривую цели;
       - динамический спектр (окно DYN_WINDOW, шаг DYN_STEP) и трекинг
         сильнейшего пика в DYN_BAND: есть ли устойчивая частота.
  3. Для соседей и стандартов дополнительно -- полный стандартный прогон
     Peridogram_diagnostics.analyse (отчет, CLEAN, стабильность амплитуд/фаз),
     чтобы все звезды были обработаны тем же методом, что цель.

Выход (OUTDIR):
  lc_nbr_<id>.csv, lc_comp_<id>.csv -- PRF-кривые соседей / стандартов (BTJD, FRAME, FLUX)
  comparison_summary.txt            -- сводка (ее и присылать)
  comparison_fixed_freqs.csv        -- амплитуды/S/N на частотах цели по всем звездам
  comparison_variability.csv        -- переменность и динамический спектр: одна строка на звезду
  comparison_dynamic_p<k>.pdf       -- кривая, амплитудный и динамический спектры всех звезд
  nbr_<id>_report.txt, comp_<id>_report.txt и пр. -- стандартные отчеты Peridogram_diagnostics

Запуск из PyCharm: поправь блок НАСТРОЙКИ и нажми Run. Файл лежит в корне
проекта (рядом с PRF_cleaning.py). Нужна та же вырезка, что для PRF_cleaning.py.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from isolated.data_io import load_tess_cutouts, load_gaia_stars_in_view_data, load_star_gaia_data, find_target_row
from isolated.psf import get_tesscut_prf_supersampled
from isolated.geometry import get_nospace_star_name, calc_tmag_from_gaia, calc_tess_flux_from_mag
from isolated.cleaning import quality_mask, auto_clean_light_curve
from isolated.localize import deblend_star_curve
from isolated.photometry import rank_contaminants
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

# соседи: звезды с наибольшей долей PRF-потока в апертуре цели
N_NEIGHBOURS = 3
NEIGHBOUR_MIN_FRACTION = 0.02          # соседи с меньшей долей потока в апертуре не берем
NEIGHBOUR_IDS_OVERRIDE = None          # вручную: список Gaia source_id вместо автопоиска
# стандарты: звезды поля сопоставимого блеска
N_STANDARDS = 8
MAG_TOL = 1.5          # |T - T_цели| для стандартов
MIN_SEP_PX = 2.0       # ближе -- "тесная" звезда (вырождение PRF с соседом)

# частоты цели для проверки (по умолчанию -- свежий прогон выбеливания по кривой цели);
# можно задать вручную списком, например частоты из другого анализа
TARGET_FREQS_OVERRIDE = None
FMAX = CONFIG.prewhiten_fmax
SNR_STOP = CONFIG.snr_stop
NOISE_BOX = 1.0        # ширина окна для локального шума, 1/сут (как в prewhiten: +-0.5)

DYN_WINDOW = CONFIG.dyn_window   # окно динамического спектра, сут
DYN_STEP = CONFIG.dyn_step       # шаг окна, сут
DYN_FRANGE = CONFIG.dyn_frange   # диапазон частот динамического спектра, 1/сут
# где ищем устойчивый пик: None -- автоматически вокруг значимых частот цели внутри
# DYN_FRANGE (± DYN_BAND_MARGIN); или кортеж (f1, f2) вручную
DYN_BAND = None
DYN_BAND_MARGIN = 0.15
BIN_DAYS = 0.25        # бин для RMS_BINNED_PPT, сут

PLOT_FMAX = 6.0        # верх оси частот на амплитудных спектрах
ROWS_PER_FIGURE = 5    # звезд на одной странице рисунка
RUN_FULL_DIAGNOSTICS = True      # стандартный Peridogram_diagnostics.analyse для соседей и стандартов
OUTDIR = "out_compare"
SHOW_PLOTS = CONFIG.show_plots   # False: рисунки только сохраняются, пути -- в консоли (run_config.py)


# =============================================================================
def ppt(flux):
    return (flux / np.median(flux) - 1.0) * 1e3


def fixed_freq_fit(t, y, freqs):
    """Линейная подгонка синусоид на фиксированных частотах + тренд 2-й степени.
    Возвращает амплитуды (ppt), ошибки (MO99 x sqrt(D)), S/N к шуму спектра остатков, шум, D."""
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


def auto_dyn_band(target_freqs, frange, margin=0.15):
    """Диапазон поиска устойчивого пика: вокруг значимых частот цели внутри `frange`
    (± margin, обрезано по frange); если таких частот нет -- весь `frange`."""
    inside = [float(f) for f in target_freqs if frange[0] <= f <= frange[1]]
    if not inside:
        return (float(frange[0]), float(frange[1]))
    return (max(frange[0], min(inside) - margin), min(frange[1], max(inside) + margin))


def variability_stats(t, y, bin_days=0.25, min_pts=5):
    """(RMS кривой после тренда 2-й степени, RMS средних по бинам `bin_days`), ppt."""
    t = np.asarray(t, float)
    res = detrend_poly(t, np.asarray(y, float))
    idx = np.floor((t - t.min()) / bin_days).astype(int)
    sums = np.bincount(idx, weights=res)
    cnt = np.bincount(idx)
    ok = cnt >= min_pts
    means = sums[ok] / cnt[ok]
    return float(np.std(res)), (float(np.std(means)) if means.size > 2 else float("nan"))


def dyn_summary(t, y, band):
    """Динамический спектр в DYN_FRANGE + разброс частоты/амплитуды сильнейшего пика в `band`."""
    centers, freqs, dyn = dynamic_spectrum(t, y, window=DYN_WINDOW, step=DYN_STEP, frange=DYN_FRANGE)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    ok = np.isfinite(dyn).all(axis=1)
    out = dict(centers=centers, freqs=freqs, dyn=dyn, n_win=int(ok.sum()))
    if not ok.any() or not in_band.any():
        nan = float("nan")
        out.update(f_lo=nan, f_hi=nan, f_med=nan, f_iqr=nan, a_lo=nan, a_hi=nan, a_med=nan)
        return out
    j = np.argmax(dyn[ok][:, in_band], axis=1)
    fpk = freqs[in_band][j]
    apk = dyn[ok][:, in_band][np.arange(j.size), j]
    out.update(f_lo=fpk.min(), f_hi=fpk.max(), f_med=np.median(fpk),
               f_iqr=np.subtract(*np.percentile(fpk, [75, 25])), a_lo=apk.min(), a_hi=apk.max(),
               a_med=np.median(apk))
    return out


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(OUTDIR, exist_ok=True)
    set_show_plots(SHOW_PLOTS)

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

    gaia = gaia.copy()
    gaia["t_mag"] = calc_tmag_from_gaia(gaia["phot_g_mean_mag"], gaia["bp_rp"])
    gaia["dist_tgt"] = np.hypot(gaia["px_x"] - tgt["px_x"], gaia["px_y"] - tgt["px_y"])
    T_tgt = float(gaia.loc[gaia["source_id"] == TARGET_ID, "t_mag"].iloc[0])

    # --- соседи: кто дает больше всего потока в апертуре цели ---
    if NEIGHBOUR_IDS_OVERRIDE:
        nbr_frac = {int(s): float("nan") for s in NEIGHBOUR_IDS_OVERRIDE}
    else:
        with_flux = gaia[np.isfinite(gaia["phot_rp_mean_mag"])].reset_index(drop=True)
        ranked = rank_contaminants(
            tgt["px_x"], tgt["px_y"], with_flux["px_x"].to_numpy(), with_flux["px_y"].to_numpy(),
            calc_tess_flux_from_mag(with_flux["phot_rp_mean_mag"].to_numpy()), prf, CUT, CONFIG.aperture_radius,
            n=N_NEIGHBOURS, min_fraction=NEIGHBOUR_MIN_FRACTION)
        nbr_frac = {int(with_flux.loc[i, "source_id"]): frac for i, frac in ranked}
    print(f"\nСоседи (доля потока в апертуре r={CONFIG.aperture_radius}px): "
          + (", ".join(f"{sid} ({fr:.0%})" for sid, fr in nbr_frac.items()) if nbr_frac else "нет"))

    # --- кривые: цель, соседи, стандарты тем же деблендингом (окно на самой звезде) ---
    curves = []   # dict(label, role, sid, T, t, flux, note, dist, close, csv, tag)

    def add_curve(role, sid, label, note, close=False):
        try:
            flux, *_ = deblend_star_curve(cube_good, prf, gaia, CUT, CUT, sid, box=BOX, dt_max=DT_MAX,
                                          merge_px=MERGE_PX)
        except ValueError as exc:
            print(f"Пропускаю {label}: {exc}")
            return
        df = pd.DataFrame({"BTJD": t_q, "FRAME": fr_q, "FLUX": flux})
        dfc, _log = auto_clean_light_curve(df, quality_bitmask=0, local_noise_kappa=KAPPA)
        if len(dfc) < 1000:
            print(f"Пропускаю {label}: после очистки {len(dfc)} кадров")
            return
        row = gaia[gaia["source_id"] == sid].iloc[0]
        prefix = {"target": None, "neighbour": "nbr", "standard": "comp"}[role]
        csv = None
        if prefix:
            csv = os.path.join(OUTDIR, f"lc_{prefix}_{sid}.csv")
            write_csv_with_provenance(dfc, csv, star_name=STAR_NAME, sector=SECTOR, source_id=sid, box=BOX,
                                      role=role, note=note)
        curves.append(dict(label=label, role=role, sid=sid, T=float(row["t_mag"]), t=dfc["BTJD"].to_numpy(),
                           flux=dfc["FLUX"].to_numpy(), note=note, dist=float(row["dist_tgt"]), close=close,
                           csv=csv, tag=f"{prefix}_{sid}" if prefix else None))

    add_curve("target", TARGET_ID, STAR_NAME, "цель")
    for sid, frac in nbr_frac.items():
        row = gaia[gaia["source_id"] == sid].iloc[0]
        frac_txt = f", {frac:.0%} потока в апертуре цели" if np.isfinite(frac) else ""
        add_curve("neighbour", sid, f"nbr_{sid % 100000:05d}",
                  f"сосед, T={row['t_mag']:.2f}, {row['dist_tgt']:.1f} пикс от цели{frac_txt}")
    if not curves or curves[0]["role"] != "target":
        raise SystemExit("Не удалось построить кривую цели -- проверять нечего")

    # --- стандарты ---
    cand = gaia[(gaia["source_id"] != TARGET_ID) & (~gaia["source_id"].isin(list(nbr_frac)))].copy()
    cand = cand[(cand["t_mag"] - T_tgt).abs() <= MAG_TOL]
    bx0 = np.round(cand["px_x"]).astype(int) - BOX // 2
    by0 = np.round(cand["px_y"]).astype(int) - BOX // 2
    cand = cand[(bx0 >= 1) & (bx0 + BOX - 1 <= CUT) & (by0 >= 1) & (by0 + BOX - 1 <= CUT)]

    def nearest_rival(r):
        d = np.hypot(gaia["px_x"] - r["px_x"], gaia["px_y"] - r["px_y"])
        m = (gaia["t_mag"] < r["t_mag"] + 1.0) & (d > 0.01)
        return float(d[m].min()) if m.any() else np.inf

    cand["sep_rival"] = cand.apply(nearest_rival, axis=1)
    cand = cand.sort_values("t_mag").head(N_STANDARDS)
    print("\nКандидаты в стандарты:")
    print(cand[["source_id", "t_mag", "px_x", "px_y", "sep_rival", "dist_tgt"]].to_string())

    for _, r in cand.iterrows():
        sid = int(r["source_id"])
        close = r["sep_rival"] < MIN_SEP_PX
        note = (f"стандарт, T={r['t_mag']:.2f}, {r['dist_tgt']:.0f} пикс от цели, ближайшая сравнимая звезда "
                f"{r['sep_rival']:.1f} пикс" + (" -- ТЕСНАЯ, кривая ненадежна" if close else ""))
        add_curve("standard", sid, f"std_{sid % 100000:05d}", note, close=close)

    analyse_curves(curves)


def analyse_curves(curves):
    """curves: список dict (см. main); первая -- цель."""
    target = curves[0]
    y0 = ppt(target["flux"])
    if TARGET_FREQS_OVERRIDE:
        target_freqs = list(TARGET_FREQS_OVERRIDE)
    else:
        pk, ex = prewhiten(target["t"], y0, fmax=FMAX, nmax=CONFIG.n_max_freq, snr_stop=SNR_STOP)
        target_freqs = [p["frequency"] for p in pk if p["frequency"] > 2.0 / ex["T"]]
    target_freqs = np.array(sorted(target_freqs))
    F_tgt = np.median(target["flux"])
    band = tuple(DYN_BAND) if DYN_BAND else auto_dyn_band(target_freqs, DYN_FRANGE, DYN_BAND_MARGIN)

    rows, var_rows, rep, dyns = [], [], [], []
    P = rep.append
    n_nbr = sum(c["role"] == "neighbour" for c in curves)
    n_std = sum(c["role"] == "standard" for c in curves)
    P(f"# Проверка окружения: {STAR_NAME}, сектор {SECTOR}, вырезка {CUT}x{CUT}, PRF-окно {BOX}x{BOX}")
    P(f"# Частоты цели: " + (", ".join(f"{f:.4f}" for f in target_freqs) or "нет значимых"))
    P(f"# Соседей: {n_nbr}, стандартов: {n_std} (|T - T_цели| <= {MAG_TOL}); "
      f"'тесная' -- сравнимая звезда ближе {MIN_SEP_PX} пикс")
    P(f"# Динамический спектр: диапазон {DYN_FRANGE[0]:.2f}-{DYN_FRANGE[1]:.2f}, "
      f"трекинг пика в {band[0]:.2f}-{band[1]:.2f} 1/сут")
    P("")
    for c in curves:
        label, role, sid, T_s, t, flux, note = (c[k] for k in ("label", "role", "sid", "T", "t", "flux", "note"))
        y = ppt(flux)
        T = t.max() - t.min()
        p2p = 1e3 * np.median(np.abs(np.diff(flux))) / np.median(flux) / np.sqrt(2) * 1.4826
        rms, rms_b = variability_stats(t, y, BIN_DAYS)
        own, own_ex = prewhiten(t, y, fmax=FMAX, nmax=CONFIG.n_max_freq, snr_stop=SNR_STOP)
        own_f = [(p["frequency"], p["amplitude"], p["snr_local"]) for p in own if p["frequency"] > 2.0 / T]
        scale = np.median(flux) / F_tgt
        ds = dyn_summary(t, y, band)
        dyns.append((c, y, ds))
        d = float("nan")
        if len(target_freqs):
            amps, sa, snr, noise, d = fixed_freq_fit(t, y, target_freqs)
            for f, a, e, s, n in zip(target_freqs, amps, sa, snr, noise):
                rows.append(dict(LABEL=label, ROLE=role, SOURCE_ID=sid, T_MAG=round(T_s, 2), FREQUENCY=round(f, 4),
                                 AMP_PPT=a, AMP_ERR_PPT=e, NOISE_PPT=n, SNR=s, AMP_IN_TARGET_PPT=a * scale,
                                 CLOSE=c["close"]))
        top = max(own_f, key=lambda x: x[1]) if own_f else (np.nan, np.nan, np.nan)
        var_rows.append(dict(LABEL=label, ROLE=role, SOURCE_ID=sid, T_MAG=round(T_s, 2), DIST_PX=round(c["dist"], 1),
                             FLUX_RATIO=round(scale, 3), CLOSE=c["close"], N=t.size, NOISE_P2P_PPT=round(p2p, 3),
                             RMS_PPT=round(rms, 3), RMS_BINNED_PPT=round(rms_b, 3), N_OWN_FREQS=len(own_f),
                             TOP_FREQ=round(top[0], 4), TOP_AMP_PPT=round(top[1], 3),
                             DYN_F_MED=round(ds["f_med"], 4), DYN_F_IQR=round(ds["f_iqr"], 4),
                             DYN_A_MED_PPT=round(ds["a_med"], 3)))
        P(f"## {label} [{role}] (Gaia DR3 {sid}): {note}")
        P(f"   N = {t.size}, шум точка-к-точке {p2p:.2f} ppt, RMS после тренда {rms:.2f} ppt "
          f"(по бинам {BIN_DAYS} сут: {rms_b:.2f} ppt), поток/поток цели = {scale:.2f}, D = {d:.1f}")
        P("   Собственные значимые частоты (выбеливание, S/N>=4, nu > 2/T):")
        if own_f:
            for f, a, s in own_f:
                ft = target_freqs[np.argmin(np.abs(target_freqs - f))] if len(target_freqs) else np.nan
                mark = role != "target" and abs(ft - f) < 1.5 / T
                P(f"      {f:8.4f} 1/сут  A = {a:6.2f} ppt  S/N = {s:5.1f}"
                  + (f"   <-- в пределах 1.5/T от частоты цели {ft:.4f}" if mark else ""))
        else:
            P("      нет")
        if len(target_freqs):
            P("   На частотах цели (подгонка на фиксированных частотах):")
            P("      nu, 1/сут     A, ppt         S/N    A в долях потока цели, ppt")
            for f, a, e, s in zip(target_freqs, amps, sa, snr):
                P(f"      {f:8.4f}   {a:6.2f} ± {e:4.2f}   {s:5.1f}   {a * scale:7.2f}")
        if ds["n_win"]:
            P(f"   Динамический спектр (окно {DYN_WINDOW} сут, шаг {DYN_STEP} сут, {ds['n_win']} окон): сильнейший пик "
              f"в {band[0]:.2f}–{band[1]:.2f} 1/сут -- частота {ds['f_lo']:.3f}–{ds['f_hi']:.3f} "
              f"(медиана {ds['f_med']:.3f}, IQR {ds['f_iqr']:.3f}), амплитуда {ds['a_lo']:.2f}–{ds['a_hi']:.2f} ppt "
              f"(медиана {ds['a_med']:.2f})")
        else:
            P("   Динамический спектр: нет окон с достаточным заполнением")
        P("")

    var_tab = pd.DataFrame(var_rows)
    write_csv_with_provenance(var_tab, os.path.join(OUTDIR, "comparison_variability.csv"), star_name=STAR_NAME,
                              sector=SECTOR, bin_days=BIN_DAYS, dyn_band=[round(b, 3) for b in band])
    P("Сводка переменности (ppt; RMS после тренда 2-й степени):")
    P(var_tab[["LABEL", "ROLE", "T_MAG", "DIST_PX", "FLUX_RATIO", "NOISE_P2P_PPT", "RMS_PPT", "RMS_BINNED_PPT",
               "N_OWN_FREQS", "TOP_FREQ", "TOP_AMP_PPT", "CLOSE"]].to_string(index=False))
    P("")

    out_files = ["comparison_summary.txt", "comparison_variability.csv"]
    if rows:
        tab = pd.DataFrame(rows)
        write_csv_with_provenance(tab, os.path.join(OUTDIR, "comparison_fixed_freqs.csv"), star_name=STAR_NAME,
                                  sector=SECTOR, mag_tol=MAG_TOL, target_freqs=[round(f, 4) for f in target_freqs])
        out_files.append("comparison_fixed_freqs.csv")

        piv = tab.pivot_table(index="LABEL", columns="FREQUENCY", values="SNR", sort=False).round(1)
        P("Сводка: S/N на частотах цели (шум -- спектр остатков после подгонки этих частот, окно ±0.5 1/сут)")
        P(piv.to_string())
        comp = tab[(tab["ROLE"] == "standard") & (~tab["CLOSE"])]
        if len(comp):
            in_band = comp[(comp["FREQUENCY"] >= band[0]) & (comp["FREQUENCY"] <= band[1])]
            P(f"\nНетесные стандарты: {comp['LABEL'].nunique()}; максимальное S/N на частотах цели "
              f"{comp['SNR'].max():.1f}" + (
                  f"; на частотах цели в {band[0]:.2f}–{band[1]:.2f} -- амплитуда до {in_band['AMP_PPT'].max():.2f} ppt "
                  f"собственного потока (до {in_band['AMP_IN_TARGET_PPT'].max():.2f} ppt в долях потока цели)."
                  if len(in_band) else "."))
            hit = comp[comp["SNR"] >= SNR_STOP].groupby("FREQUENCY")["LABEL"].agg(list)
            if len(hit):
                P(f"Частоты цели, значимые (S/N >= {SNR_STOP}) у нетесных стандартов:")
                for f, labels in hit.items():
                    P(f"   {f:.4f}: у {len(labels)} ({', '.join(labels)})"
                      + ("  <-- у >=2 звезд: вероятна общая систематика поля/инструмента" if len(labels) >= 2 else ""))
            else:
                P(f"Ни одна частота цели не значима (S/N >= {SNR_STOP}) ни у одного нетесного стандарта.")
    else:
        P("Нет значимых частот цели -- подгонка на частотах цели пропущена.")

    txt = "\n".join(rep)
    print("\n" + txt)
    with open(os.path.join(OUTDIR, "comparison_summary.txt"), "w") as fo:
        fo.write(txt + "\n")

    # --- рисунок: кривая + амплитудный спектр + динамический спектр, по ROWS_PER_FIGURE звезд на страницу ---
    pages = [dyns[i:i + ROWS_PER_FIGURE] for i in range(0, len(dyns), ROWS_PER_FIGURE)]
    for k, page in enumerate(pages, start=1):
        n = len(page)
        fig, axes = plt.subplots(n, 3, figsize=(15, 2.2 * n + 0.4), squeeze=False,
                                 gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
        for r, (c, y, ds) in enumerate(page):
            t = c["t"]
            T = t.max() - t.min()
            ax = axes[r, 0]
            ax.plot(t, y, ".", ms=1, color="k")
            ax.set_ylabel("ΔF, ppt")
            ax.text(0.99, 0.92, f"{c['label']} [{c['role']}]  {c['note'].split(',')[0] if c['role'] != 'target' else ''}"
                    f"{'  ТЕСНАЯ' if c['close'] else ''}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
            fg = np.arange(0.5 / T, PLOT_FMAX, 1 / (10 * T))
            A = amp_spectrum(t, detrend_poly(t, y), fg)
            ax = axes[r, 1]
            ax.plot(fg, A, lw=0.6, color="k")
            for f in target_freqs:
                ax.axvline(f, color="C3", lw=0.5, ls=":")
            ax.set_xlim(0, PLOT_FMAX); ax.set_ylabel("A, ppt")
            ax2 = axes[r, 2]
            im = ax2.pcolormesh(ds["freqs"], ds["centers"], ds["dyn"], shading="auto", cmap="viridis")
            for f in target_freqs[(target_freqs > DYN_FRANGE[0]) & (target_freqs < DYN_FRANGE[1])]:
                ax2.axvline(f, color="w", lw=0.4, ls=":")
            plt.colorbar(im, ax=ax2, label="A, ppt")
            ax2.set_ylabel("BTJD")
        axes[-1, 0].set_xlabel("BTJD"); axes[-1, 1].set_xlabel("Частота, 1/сут"); axes[-1, 2].set_xlabel("Частота, 1/сут")
        fig.tight_layout()
        finish_figure(fig, os.path.join(OUTDIR, f"comparison_dynamic_p{k}.pdf"), dpi=None)

    followers = [c for c in curves if c["role"] != "target"]
    if RUN_FULL_DIAGNOSTICS:
        import Peridogram_diagnostics as PD
        for c in followers:
            PD.analyse(c["csv"], c["tag"], OUTDIR, PD.FMAX, nmax=PD.N_MAX, dyn_win=PD.DYN_WINDOW,
                       dyn_frange=PD.DYN_FRANGE, dyn_freq_step=PD.DYN_FREQ_STEP,
                       comparison_freqs=list(target_freqs), kappa=PD.KAPPA,
                       n_sigma_combination=PD.N_SIGMA_COMBINATION, do_bootstrap=False,
                       do_stability=True, stability_window=PD.STABILITY_WINDOW,
                       stability_step=PD.STABILITY_STEP, do_clean=True, clean_gain=PD.CLEAN_GAIN,
                       clean_niter=PD.CLEAN_NITER)
    print_saved_summary("Comparison_check: что где лежит", extra_files=[
        os.path.join(OUTDIR, n) for n in out_files] +
        [c["csv"] for c in followers] +
        ([os.path.join(OUTDIR, f"{c['tag']}_{k}") for c in followers for k in ("report.txt", "freqs.tex")]
         if RUN_FULL_DIAGNOSTICS else []))
    print(f"Главное -- {os.path.abspath(os.path.join(OUTDIR, 'comparison_summary.txt'))}")
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
