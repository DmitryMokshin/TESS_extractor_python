"""
Чей сигнал? Локализация частот TESS по пикселям + PRF-разделение кривых блеска.

Зачем. В апертуре r = 3 пикс вокруг SS 397 (T ≈ 10.8) на расстоянии 3.2 пикс (67″)
сидит звезда Gaia DR3 4154999332652688000 (T ≈ 9.7, в 2.8 раза ярче; красный гигант).
По модели PRF на SS 397 приходится лишь ~43 % потока в апертуре, на соседа ~46 %.
Код пайплайна вычитает только плоский фон, поэтому свет соседей остается в кривой.

Что делает скрипт:
 1. Берет вырезку сектора и выбирает кадры: QUALITY == 0, из clean.csv, без участков
    с повышенным шумом.
 2. PRF-фотометрия: в окне BOX×BOX вокруг звезды каждый кадр описывается суммой
    PRF звезд Gaia (положения фиксированы) + плоский фон. Потоки звезд линейны, решение
    точное и быстрое. На выходе отдельные кривые SS 397 и соседа (CSV в формате пайплайна).
 3. Карты амплитуд: в каждом пикселе подгоняются синусоиды на частотах FREQS
    (+ квадратичный тренд). Комплексная карта амплитуды раскладывается по PRF звезд:
    видно, какая звезда несет каждую частоту.

Выход: out_localize/
    amp_maps.pdf              — карты амплитуд с положениями звезд
    attribution.txt           — доли амплитуды по звездам для каждой частоты
    lc_deblended_<id>.csv     — PRF-кривые (MJD, FLUX) для SS 397 и соседа
    lc_deblended.pdf          — сравнение кривых
Если RUN_TESS_AFTER = True, сразу после этого разделенные кривые прогоняются через
ss397_tess.py (частоты, ошибки, динамический спектр) — результаты в out_tess/.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# НАСТРОЙКИ — правь здесь и жми Run
# =============================================================================
STAR_NAME = "SS 397"
SECTOR = 80
CUT = 50
BOX = 13                 # размер окна для PRF-фотометрии (нечетный), пикс
DT_MAX = 3.5             # включать звезды Gaia не слабее T_target + DT_MAX
MERGE_PX = 1.0           # звезды ближе MERGE_PX к более яркой не фитуются отдельно (вырождение)
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
QUALITY_BITMASK = 175
RUN_TESS_AFTER = True     # сразу прогнать ss397_tess.analyse по разделенным кривым
SHOW_PLOTS = True         # открыть окна с рисунками в конце (PDF сохраняются всегда)


# =============================================================================
# ядро (без ввода-вывода — тестируется на синтетике)
# =============================================================================
def tmag_from_gaia(G, bp_rp):
    """Stassun et al. (2019), TIC-8: T из G и BP-RP."""
    x = np.nan_to_num(np.asarray(bp_rp, float), nan=1.0)
    return G - 0.00522555 * x ** 3 + 0.0891337 * x ** 2 - 0.633923 * x + 0.0324473


def select_stars(gaia, target_id, box_x0, box_y0, box, dt_max=DT_MAX, merge_px=MERGE_PX):
    """Звезды для PRF-модели окна: в окне ±2 пикс, T < T_target + dt_max,
    слабые звезды ближе merge_px к более яркой исключаются (их свет уходит в яркую)."""
    g = gaia.copy()
    g["T"] = tmag_from_gaia(g["phot_g_mean_mag"], g["bp_rp"])
    t = g[g["source_id"] == target_id].iloc[0]
    inbox = ((g["px_x"] > box_x0 - 2) & (g["px_x"] < box_x0 + box + 2) &
             (g["px_y"] > box_y0 - 2) & (g["px_y"] < box_y0 + box + 2) &
             (g["T"] < t["T"] + dt_max))
    g = g[inbox].sort_values("T").reset_index(drop=True)
    keep = []
    for i, s in g.iterrows():
        if any(np.hypot(s.px_x - g.loc[j, "px_x"], s.px_y - g.loc[j, "px_y"]) < merge_px for j in keep):
            continue
        keep.append(i)
    return g.loc[keep].reset_index(drop=True)


def prf_models(prf, stars, cut_w, cut_h, box_x0, box_y0, box):
    """PRF каждой звезды, вырезанный в окно. Ось 0 = x, ось 1 = y (как frame(i) пайплайна)."""
    from isolated.psf import add_prf_cut
    models = []
    for _, s in stars.iterrows():
        c = np.zeros((cut_w, cut_h))
        add_prf_cut(c, 1.0, prf, cut_w, cut_h, s["px_x"], s["px_y"])
        models.append(c[box_x0:box_x0 + box, box_y0:box_y0 + box])
    return np.array(models)                      # (nstar, box, box)


def design_matrix(models):
    nstar, bx, by = models.shape
    xx, yy = np.meshgrid(np.arange(bx) - bx / 2, np.arange(by) - by / 2, indexing="ij")
    cols = [m.ravel() for m in models] + [np.ones(bx * by), xx.ravel(), yy.ravel()]
    return np.array(cols).T                      # (npix, nstar + 3)


def deblend(cube_box, models):
    """cube_box: (nt, box, box). Возвращает потоки звезд (nt, nstar) и фон (nt, 3)."""
    X = design_matrix(models)
    D = cube_box.reshape(cube_box.shape[0], -1).T          # (npix, nt)
    ok = np.all(np.isfinite(D), axis=1)
    coef, *_ = np.linalg.lstsq(X[ok], D[ok], rcond=None)   # (npar, nt)
    nstar = models.shape[0]
    return coef[:nstar].T, coef[nstar:].T


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


def local_p2p_ppt(t, y_ppt, win=0.25):
    """Локальный шум точка-к-точке (ppt) в скользящем окне win сут — оценка ошибки точки."""
    d = np.diff(y_ppt)
    lo = np.searchsorted(t, t - win / 2)
    hi = np.searchsorted(t, t + win / 2)
    out = np.array([1.4826 * np.median(np.abs(d[a:min(b, d.size)] - np.median(d[a:min(b, d.size)])))
                    / np.sqrt(2) if min(b, d.size) - a > 10 else np.nan for a, b in zip(lo, hi)])
    out[np.isnan(out)] = np.nanmedian(out)
    return out


def export_clean_lc(t, flux, dat_path, csv_path=None, star="SS 397", sector=80, note=""):
    """
    Очищенная кривая блеска для коллег (ASCII, для CLEAN/LS в любых программах):
        BJD-2457000   dF/F[ppt]   sigma[ppt]   dmag[mag]   FLUX[e/s]
    и, если задан csv_path, копия в формате пайплайна (MJD, FLUX, MAG) —
    её можно подать в Peridogram_compute.py.
    """
    from isolated.geometry import calc_tess_magnitude
    med = np.median(flux)
    ppt = (flux / med - 1.0) * 1e3
    err = local_p2p_ppt(t, ppt)
    dmag = -2.5 * np.log10(flux / med)
    hdr = (f"{star}, TESS sector {sector}: PRF-photometry light curve (deblended from Gaia neighbours)\n"
           f"frames: TESScut FFI, cadence 200 s; QUALITY bitmask 175 removed; points with local\n"
           f"point-to-point noise > 2.5 x median removed. {note}\n"
           f"N = {t.size}, median flux = {med:.1f} e/s, median sigma = {np.median(err):.2f} ppt\n"
           f"columns: BJD-2457000  dF/F[ppt]  sigma[ppt]  dmag[mag]  FLUX[e/s]")
    np.savetxt(dat_path, np.c_[t, ppt, err, dmag, flux], fmt="%.6f %9.3f %7.3f %9.5f %11.3f", header=hdr)
    if csv_path:
        pd.DataFrame({"MJD": t, "FLUX": flux, "MAG": calc_tess_magnitude(flux)}).to_csv(csv_path, index=False)
    return dat_path


def local_noise_mask(t, y, win=0.25, kappa=2.5):
    d = np.diff(y)
    lo = np.searchsorted(t, t - win / 2)
    hi = np.searchsorted(t, t + win / 2)
    sig = np.array([1.4826 * np.median(np.abs(d[a:min(b, d.size)] - np.median(d[a:min(b, d.size)])))
                    / np.sqrt(2) if min(b, d.size) - a > 10 else np.nan for a, b in zip(lo, hi)])
    sig[np.isnan(sig)] = np.nanmedian(sig)
    return sig < kappa * np.median(sig)


# =============================================================================
# запуск на реальных данных
# =============================================================================
def main():
    from isolated.data_io import load_tess_cutouts, load_gaia_stars_in_view_data
    from isolated.psf import get_tesscut_prf_supersampled
    from isolated.geometry import get_nospace_star_name

    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # пути пакета isolated относительные
    os.makedirs(OUTDIR, exist_ok=True)
    cut_fits = load_tess_cutouts(STAR_NAME, CUT, CUT, sector=SECTOR)[SECTOR]
    data = cut_fits[1].data
    t_all = np.asarray(data["TIME"], float)
    q_all = np.asarray(data["QUALITY"], int)
    cube = np.transpose(np.asarray(data["FLUX"], float), (0, 2, 1))   # (nt, x, y), как frame(i)
    prf = get_tesscut_prf_supersampled(cut_fits)

    gaia = load_gaia_stars_in_view_data(STAR_NAME, cut_fits, d_mag_r=6.0)
    tgt = gaia[gaia["source_id"] == TARGET_ID].iloc[0]
    x0 = int(round(tgt["px_x"])) - 1 - BOX // 2       # 0-based начало окна
    y0 = int(round(tgt["px_y"])) - 1 - BOX // 2
    stars = select_stars(gaia, TARGET_ID, x0 + 1, y0 + 1, BOX)
    print(f"Звезд в модели окна: {len(stars)}")
    print(stars[["source_id", "T", "px_x", "px_y"]].to_string())

    # --- выбор кадров ---
    clean = pd.read_csv(os.path.join("stars_python", get_nospace_star_name(STAR_NAME),
                                     f"{CUT}x{CUT}", f"light_curve_sector_{SECTOR}_clean.csv"))
    in_clean = np.isin(np.round(t_all, 6), np.round(clean["MJD"].to_numpy(), 6))
    good = in_clean & np.isfinite(t_all)
    bits = [b for b in range(20) if np.any(q_all & (1 << b))]
    print("Флаги QUALITY в секторе (бит: число кадров): " +
          ", ".join(f"{1 << b}: {np.count_nonzero(q_all & (1 << b))}" for b in bits))
    if QUALITY_BITMASK is None:
        good &= (q_all == 0)
    elif QUALITY_BITMASK:
        good &= (q_all & QUALITY_BITMASK) == 0
    print(f"Кадров: всего {t_all.size}, выброшено флагами: "
          f"{(in_clean & np.isfinite(t_all) & ~good).sum()}, "
          f"в clean.csv: {in_clean.sum()}, используем: {good.sum()}")

    box = cube[good, x0:x0 + BOX, y0:y0 + BOX]
    t = t_all[good]
    models = prf_models(prf, stars, CUT, CUT, x0, y0, BOX)
    fluxes, bkg = deblend(box, models)

    it = int(np.where(stars["source_id"] == TARGET_ID)[0][0])
    inb = np.where(stars["source_id"] == NEIGHBOUR_ID)[0]
    iN = int(inb[0]) if inb.size else None

    # отбраковка по шуму PRF-кривой цели
    m = local_noise_mask(t, fluxes[:, it] / np.median(fluxes[:, it]))
    print(f"После отбраковки шумных участков: {m.sum()} кадров")
    t, box, fluxes = t[m], box[m], fluxes[m]

    for j, tag in [(it, "SS397"), (iN, "neighbour")]:
        if j is None:
            continue
        pd.DataFrame({"MJD": t, "FLUX": fluxes[:, j]}).to_csv(
            os.path.join(OUTDIR, f"lc_deblended_{tag}.csv"), index=False)
        print(f"{tag}: медианный поток {np.median(fluxes[:, j]):.0f} e/s "
              f"(ожидание по Gaia {10 ** (0.4 * (20.44 - stars.loc[j, 'T'])):.0f})")

    # очищенная кривая SS 397 для коллег + копия в формате пайплайна
    star_dir = os.path.join("stars_python", get_nospace_star_name(STAR_NAME), f"{CUT}x{CUT}")
    dat = export_clean_lc(t, fluxes[:, it], os.path.join(OUTDIR, f"SS397_TESS_S{SECTOR}_PRF.dat"),
                          csv_path=os.path.join(star_dir, f"light_curve_sector_{SECTOR}_prf_clean.csv"),
                          star=STAR_NAME, sector=SECTOR)
    print(f"Кривая для коллег: {dat}")

    # --- карты амплитуд и разложение по звездам ---
    # вычитаем подогнанный плоский фон (иначе медленные вариации рассеянного света
    # ложатся на все пиксели и мешают на низких частотах)
    _, bkg = deblend(box, models)
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
        fo.write(stars[["source_id", "T", "px_x", "px_y"]].to_string() + "\n\n" + txt + "\n")

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
        ss397_tess.run_all(runs=[(os.path.join(OUTDIR, "lc_deblended_SS397.csv"), "ss397_prf"),
                                 (os.path.join(OUTDIR, "lc_deblended_neighbour.csv"), "neighbour")],
                           show=False)
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
