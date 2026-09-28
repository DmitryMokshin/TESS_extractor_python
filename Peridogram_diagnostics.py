"""
`Peridogram_compute.py`'s частотный анализ, но подробнее (ROADMAP.md Этап 8):

  - объективный критерий отбраковки инструментальных участков по локальному
    шуму точка-к-точке (а не "на глаз") + типичная ошибка потока;
  - ошибки частот бутстрепом по остаткам и по двум половинам ряда — проверка
    формул Montgomery & O'Donoghue (1999) с поправкой на корреляцию остатков
    (Schwarzenberg-Czerny 1991); значимость — S/N >= 4 относительно локального
    шума (Breger et al. 1993) и модели красного шума (Bowman et al. 2019);
  - та же процедура для любой другой кривой: `analyse(<csv>)`;
  - комбинационные частоты с допуском по ошибкам частот, а не просто 0.5/T;
  - опция CLEAN (Roberts et al. 1987) для сравнения с программами коллег;
  - динамический амплитудный спектр (скользящее окно) для заданного диапазона
    частот + трекинг самого сильного пика в нём, амплитуды в ppt/mmag;
  - опционально: отметка окна внешних (например, наземных спектроскопических)
    наблюдений на кривой и её фрагмент вокруг этого окна;
  - опционально: отметка частот для сравнения на рисунках (из другой
    публикации/каталога по этой же звезде, `COMPARISON_FREQS`).

Входной формат: CSV со столбцами MJD (на деле BTJD = BJD-2457000), FLUX, [SN].
`run_all()` по умолчанию прогоняет `analyse()` по `RUNS`, собранному из
`run_config.py`: текущий режим фотометрии, альтернативный режим (если
посчитан) и деблендированная кривая соседа (`PRF_cleaning.py`, если
считался) — три прогона подряд, результаты сравнимы напрямую.

Запуск из PyCharm: поправь блок НАСТРОЙКИ ниже и нажми Run (зеленый треугольник).
Файл должен лежать в корне проекта TESS_extractor_python (рядом с run.py).

Числовая часть (спектр амплитуд, выбеливание, ошибки, модель красного шума)
живёт в isolated/prewhitening.py и просто вызывается отсюда — раньше была
продублирована почти дословно в этом файле (и одна из копий незаметно
расходилась с другой в параметризации red_noise_fit). Здесь остались только
загрузка/отбраковка кадров, отчеты, графики и настройки запуска.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from isolated.lightcurve_tools import local_point_to_point_sigma
from isolated.prewhitening import (amp_spectrum, window_function, prewhiten, red_noise_fit,
                                   find_combination_frequencies, bootstrap_frequency_errors,
                                   split_half_frequency_errors, frequency_stability,
                                   plot_frequency_stability, clean_periodogram, plot_clean_spectrum)
from run_config import CONFIG

# =============================================================================
# НАСТРОЙКИ — звезда/сектор/вырезка/режим и параметры выбеливания теперь в
# run_config.py (Этап 4); RUNS ниже строится из него же, чтобы не разъезжаться.
# =============================================================================
STAR_DIR = CONFIG.star_dir
# какие кривые анализировать: (путь к CSV, метка для имен файлов).
# Отсутствующие файлы пропускаются с сообщением. Первая запись -- текущий
# режим по run_config.py; вторая -- сравнение с другим режимом фотометрии;
# третья -- кривая яркого соседа (PRF_cleaning.py) для сравнения по частотам.
RUNS = [
    (f"{STAR_DIR}/light_curve_sector_{CONFIG.sector}{CONFIG.lc_suffix}.csv", "primary"),
    (f"{STAR_DIR}/light_curve_sector_{CONFIG.sector}"
     f"{'_clean' if CONFIG.photometry_mode == 'prf' else '_prf_clean'}.csv",
     "aperture_compare" if CONFIG.photometry_mode == "prf" else "prf_compare"),
    ("out_localize/lc_deblended_neighbour.csv", "neighbour"),
]
OUTDIR = "out_tess"
FMAX = CONFIG.prewhiten_fmax   # верхняя частота поиска, 1/сут
N_MAX = CONFIG.n_max_freq      # максимум частот при выбеливании
DYN_WINDOW = CONFIG.dyn_window  # окно динамического спектра, сут
KAPPA = CONFIG.local_noise_kappa  # отбраковка: локальный шум > KAPPA × медиана
# диапазон частот для "зум"-панели амплитудного спектра, динамического спектра
# и трекинга её самого сильного пика (1/сут) -- сузьте под свою группу частот
DYN_FRANGE = CONFIG.dyn_frange
DYN_FREQ_STEP = 0.005   # шаг сетки частот динамического спектра, 1/сут
# окно внешних (напр. наземных спектроскопических) наблюдений (UT); None -- не рисовать
GROUND_OBS_UT = None        # например ("2024-06-24T19:30", "2024-06-24T22:30")
# частоты для сравнения на рисунках (напр. из другой публикации/каталога по
# этой же звезде); пусто -- не рисовать
COMPARISON_FREQS = []
SHOW_PLOTS = True    # открыть окна с рисунками в конце (PDF сохраняются всегда)

# --- частотный анализ, дальше (ROADMAP.md Этап 8) -- специфично для этого
# шага (как COMPARISON_FREQS/GROUND_OBS_UT выше), не в run_config.py.
# По умолчанию в analyse() всё это выключено (не меняет скорость/поведение,
# пока явно не включено) -- здесь включено, чтобы реально проверять.
N_SIGMA_COMBINATION = 3.0    # допуск комбинационных частот, x sigma_f вместо 0.5/T
DO_BOOTSTRAP = True          # ошибки частот бутстрепом + по двум половинам ряда
N_BOOT = 200
DO_STABILITY = True          # амплитуда/фаза найденных частот по скользящему окну
STABILITY_WINDOW = 10.0      # сут
STABILITY_STEP = 1.0         # сут
DO_CLEAN = True               # опция CLEAN (Roberts et al. 1987)
CLEAN_GAIN = 0.2
CLEAN_NITER = 150

# =============================================================================
PPT2MMAG = 2.5 / np.log(10)   # 1 ppt ≈ 1.086 mmag


# =============================================================================
# данные и отбраковка
# =============================================================================
def load_lc(path):
    from isolated.data_io import read_light_curve_csv
    d = read_light_curve_csv(path)
    tcol = "BTJD" if "BTJD" in d else d.columns[0]  # read_light_curve_csv already renamed old "MJD" files
    fcol = "FLUX" if "FLUX" in d else d.columns[1]
    d = d[np.isfinite(d[tcol]) & np.isfinite(d[fcol])]
    t = d[tcol].to_numpy(float)
    f = d[fcol].to_numpy(float)
    o = np.argsort(t)
    return t[o], f[o]


def quality_mask(t, y, win=0.25, kappa=2.5):
    """Точка хорошая, если локальный шум < kappa * медианный шум по сектору."""
    sig = local_point_to_point_sigma(t, y, win)
    return sig < kappa * np.median(sig), sig


def segments(t, mask):
    """Непрерывные интервалы, где mask == False (для отчета/закраски)."""
    out, i = [], 0
    bad = ~mask
    while i < t.size:
        if bad[i]:
            j = i
            while j + 1 < t.size and bad[j + 1]:
                j += 1
            out.append((t[i], t[j], j - i + 1))
            i = j + 1
        else:
            i += 1
    return out


# =============================================================================
# основной анализ
# =============================================================================
def analyse(path, tag="target", outdir="out_tess", fmax=5.0, nmax=15, dyn_win=10.0,
            dyn_frange=(0.5, 5.0), dyn_freq_step=0.005, comparison_freqs=(),
            ground_obs_ut=None, kappa=2.5, n_sigma_combination=3.0,
            do_bootstrap=False, n_boot=200,
            do_stability=False, stability_window=10.0, stability_step=1.0,
            do_clean=False, clean_gain=0.2, clean_niter=150):
    os.makedirs(outdir, exist_ok=True)
    t, f = load_lc(path)
    y = (f / np.median(f) - 1) * 1e3                        # ppt
    good, sig = quality_mask(t, y, kappa=kappa)
    T_all = t.max() - t.min()

    rep = []
    P = rep.append
    P(f"# {tag}: {path}")
    P(f"N = {t.size}, T = {T_all:.2f} сут, каденс = {np.median(np.diff(t)) * 1440:.1f} мин")
    P(f"Шум точка-к-точке: медиана {np.median(sig):.2f} ppt ({np.median(sig) * PPT2MMAG:.2f} mmag) на точку")
    P(f"Отбраковка: локальный шум > {kappa} × медиана -> исключено {np.sum(~good)} точек:")
    for a, b, n in segments(t, good):
        if n >= 5:
            P(f"    {a:.3f} – {b:.3f}  ({n} точек, шум до {sig[(t >= a) & (t <= b)].max():.1f} ppt)")
    P(f"Разрешение по частоте: 1/T = {1 / T_all:.4f}, 1.5/T (Loumos & Deeming 1978) = {1.5 / T_all:.4f} 1/сут")

    results = {}
    for label, m in [("all", np.ones_like(good)), ("strict", good)]:
        tt, yy = t[m], y[m] - y[m].mean()
        T = tt.max() - tt.min()
        peaks, extra = prewhiten(tt, yy, fmax, fmin=0.5 / T, nmax=nmax)
        fr = np.array([p["frequency"] for p in peaks])
        am = np.array([p["amplitude"] for p in peaks])
        sf = np.array([p["frequency_err"] for p in peaks])
        sa = np.array([p["amplitude_err"] for p in peaks])
        snr_loc = np.array([p["snr_local"] for p in peaks])
        snr_red = np.array([p["snr_red"] for p in peaks])
        res, fg, Ares = extra["residuals"], extra["grid"], extra["amp_residuals"]
        rn = extra.get("noise_model") or red_noise_fit(fg, Ares)[0]
        D, sN = extra["D"], extra["sigma_res"]
        results[label] = dict(t=tt, y=yy, fr=fr, am=am, sf=sf, sa=sa, res=res, fg=fg, Ares=Ares, rn=rn,
                              peaks=peaks, extra=extra)

        P(f"\n## набор '{label}': N = {tt.size}, σ(остатков) = {sN:.2f} ppt, D = {D:.1f}")
        P("   nu, 1/сут        P, сут            A, ppt        A, mmag   S/N_лок  S/N_красн  примечание")
        for i, (x, a) in enumerate(zip(fr, am)):
            note = []
            if x < 2.0 / T:
                note.append("< 2/T: не разрешено, тренд")
            for j, x2 in enumerate(fr):
                if j != i and abs(x - x2) < 1.0 / T:
                    note.append(f"НЕ разрешена с {x2:.3f} (<1/T)")
                elif j != i and abs(x - x2) < 1.5 / T:
                    note.append(f"на грани с {x2:.3f} (<1.5/T)")
            per = 1 / x
            sper = sf[i] / x ** 2
            P(f"   {x:.4f}±{sf[i]:.4f}   {per:8.4f}±{sper:6.4f}   {a:6.2f}±{sa[i]:.2f}   {a * PPT2MMAG:6.2f}"
              f"   {snr_loc[i]:6.1f}   {snr_red[i]:7.1f}   {'; '.join(note)}")
        # гармоники и комбинации: 5 самых сильных "родителей", допуск по ошибкам
        # частот (ROADMAP.md Этап 8), не захардкоженный 0.5/T
        for c in find_combination_frequencies(peaks, n_top=5, n_sigma=n_sigma_combination):
            i, j, k = c["parent1"], c["parent2"], c["child"]
            P(f"   комбинация? ν={fr[k]:.4f} ≈ {fr[i]:.4f} {c['kind']} {fr[j]:.4f} = "
              f"{c['predicted']:.4f} (допуск ±{c['tolerance']:.4f})")
        results[label]["table"] = (fr, sf, am, sa, snr_loc, snr_red)

    # ---------------- Этап 8: бутстреп/сплит-half, стабильность, CLEAN ----------------
    # все три -- на наборе 'strict' (научно значимый), опциональны (см. do_bootstrap/
    # do_stability/do_clean) -- не меняют время выполнения/поведение по умолчанию
    Rs = results["strict"]
    if do_bootstrap and len(Rs["peaks"]):
        boot = bootstrap_frequency_errors(Rs["t"], Rs["y"], Rs["peaks"], Rs["extra"], n_boot=n_boot)
        first, second = split_half_frequency_errors(Rs["t"], Rs["y"], Rs["peaks"])
        P("\nПроверка ошибок Montgomery & O'Donoghue (1999) (набор 'strict', бутстреп "
          f"n={n_boot}, по двум половинам ряда):")
        P("   nu, 1/сут   sigma_f формула   sigma_f бутстреп   |half1 - half2|")
        for i, p in enumerate(Rs["peaks"]):
            half_diff = abs(first["frequency"][i] - second["frequency"][i])
            P(f"   {p['frequency']:.4f}      {p['frequency_err']:.5f}         "
              f"{boot['frequency'][i]:.5f}          {half_diff:.5f}")

    if do_stability and len(Rs["peaks"]):
        group_freqs = [p["frequency"] for p in Rs["peaks"] if p["frequency"] > 2.0 / T_all]
        if group_freqs:
            centers, amps_w, phases_w = frequency_stability(Rs["t"], Rs["y"], group_freqs,
                                                             window=stability_window, step=stability_step)
            plot_frequency_stability(centers, amps_w, phases_w, group_freqs, units="ppt",
                                     window=stability_window, title=f"{tag}: amplitude/phase stability",
                                     out_path=os.path.join(outdir, f"{tag}_stability.pdf"))
            P(f"\nСтабильность по скользящему окну ({stability_window:.0f} сут): "
              f"{outdir}/{tag}_stability.pdf")

    if do_clean and len(Rs["peaks"]):
        clean_freqs = np.arange(0.5 / T_all, fmax, 1.0 / (3 * T_all))
        components, _clean_resid, dirty = clean_periodogram(Rs["t"], Rs["y"], clean_freqs,
                                                             gain=clean_gain, n_iter=clean_niter)
        plot_clean_spectrum(clean_freqs, dirty, components, units="ppt", title=f"{tag}: CLEAN",
                            out_path=os.path.join(outdir, f"{tag}_clean.pdf"))
        top_components = sorted(components, key=lambda c: -c["amplitude"])[:10]
        P(f"\nCLEAN (Roberts et al. 1987): {len(components)} компонент "
          f"(gain={clean_gain}, до {clean_niter} итераций), сильнейшие:")
        for c in top_components:
            P(f"   ν={c['frequency']:.4f} 1/сут   A={c['amplitude']:.2f} ppt")

    # ---------------- рисунки ----------------
    R = results["strict"]
    # 1) кривая блеска + шум
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].plot(t[good], y[good], ".", ms=1.5, color="k", label="используется")
    ax[0].plot(t[~good], y[~good], ".", ms=1.5, color="C3", label="исключено (шум)")
    ax[1].plot(t, sig, "-", color="0.3", lw=0.8)
    ax[1].axhline(kappa * np.median(sig), color="C3", ls="--", lw=0.8)
    for a, b, n in segments(t, good):
        for axx in ax:
            axx.axvspan(a, b, color="C3", alpha=0.12, lw=0)
    if ground_obs_ut:
        from astropy.time import Time
        g1, g2 = [Time(s, scale="utc").jd - 2457000 for s in ground_obs_ut]
        for axx in ax:
            axx.axvspan(g1, g2, color="C0", alpha=0.3, lw=0)
        ax[0].text(g1, ax[0].get_ylim()[1], " внеш. набл.", color="C0", va="top")
    ax[0].set_ylabel("ΔF, ppt"); ax[0].legend(fontsize=8, markerscale=6)
    ax[1].set_ylabel("шум, ppt"); ax[1].set_xlabel("BJD − 2457000")
    ax[1].set_yscale("log")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{tag}_lc_quality.pdf"))

    # 2) амплитудный спектр, остатки, красный шум, окно
    fg = np.arange(0.5 / T_all, fmax, 0.1 / T_all)
    A0 = amp_spectrum(R["t"], R["y"], fg)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5))
    ax[0].plot(fg, A0, "k-", lw=0.7, label="исходный")
    ax[0].plot(R["fg"], R["Ares"], "-", color="0.6", lw=0.7, label="остатки после выбеливания")
    ax[0].plot(fg, 4 * R["rn"](fg), "C3--", lw=1, label="4 × модель красного шума")
    for x in R["fr"]:
        ax[0].axvline(x, color="C0", lw=0.5, alpha=0.6)
    ax[0].set_xlim(0, fmax); ax[0].set_ylabel("A, ppt"); ax[0].legend(fontsize=8)
    z = (fg > dyn_frange[0]) & (fg < dyn_frange[1])
    ax[1].plot(fg[z], A0[z], "k-", lw=1, label=tag)
    main = R["fr"][(R["fr"] > dyn_frange[0]) & (R["fr"] < dyn_frange[1])]
    if main.size:
        f0 = main[np.argmax(R["am"][(R["fr"] > dyn_frange[0]) & (R["fr"] < dyn_frange[1])])]
        W = window_function(R["t"], fg[z], f0)
        ax[1].plot(fg[z], W / W.max() * A0[z].max(), "-", color="C2", lw=0.8,
                   label=f"спектральное окно на {f0:.3f}")
    for x in comparison_freqs:
        if dyn_frange[0] < x < dyn_frange[1]:
            ax[1].axvline(x, color="C1", ls=":", lw=1)
    for x in main:
        ax[1].axvline(x, color="C0", lw=0.8)
    ax[1].set_xlabel("Частота, 1/сут"); ax[1].set_ylabel("A, ppt")
    ax[1].legend(fontsize=8)
    if len(comparison_freqs):
        ax[1].text(0.01, 0.95, "синие — найденные здесь, оранжевые — переданные для сравнения (COMPARISON_FREQS)",
                   transform=ax[1].transAxes, fontsize=7, va="top")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{tag}_amplitude_spectrum.pdf"))

    # 3) динамический амплитудный спектр
    tt, yy = R["t"], R["y"]
    centers = np.arange(tt.min() + dyn_win / 2, tt.max() - dyn_win / 2 + 1e-6, 0.25)
    fd = np.arange(dyn_frange[0], dyn_frange[1], dyn_freq_step)
    dyn = np.full((centers.size, fd.size), np.nan)
    track = []
    for i, c in enumerate(centers):
        m = np.abs(tt - c) < dyn_win / 2
        if m.sum() < 0.6 * dyn_win / np.median(np.diff(tt)):   # слишком большие пробелы
            continue
        yy_w = yy[m] - np.polyval(np.polyfit(tt[m] - c, yy[m], 2), tt[m] - c)  # убираем медленный тренд
        dyn[i] = amp_spectrum(tt[m], yy_w, fd)
        k = np.argmax(dyn[i])
        track.append((c, fd[k], dyn[i][k]))
    track = np.array(track)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw=dict(width_ratios=[3, 1]))
    im = ax[0].pcolormesh(fd, centers, dyn, shading="auto", cmap="viridis")
    for x in comparison_freqs:
        if dyn_frange[0] < x < dyn_frange[1]:
            ax[0].axvline(x, color="w", ls=":", lw=0.8)
    ax[0].set_xlabel("Частота, 1/сут"); ax[0].set_ylabel(f"центр окна ({dyn_win:.0f} сут), BJD−2457000")
    plt.colorbar(im, ax=ax[0], label="A, ppt")
    if track.size:
        ax[1].plot(track[:, 2], track[:, 0], "k.-")
        ax[1].set_xlabel(f"A пика в {dyn_frange[0]:.2f}–{dyn_frange[1]:.2f}, ppt")
        ax2 = ax[1].twiny()
        ax2.plot(track[:, 1], track[:, 0], "C3.", ms=3)
        ax2.set_xlabel("ν пика", color="C3")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{tag}_dynamic_spectrum.pdf"))
    if track.size:
        P(f"\nДинамический спектр (окно {dyn_win} сут, разрешение ~{1 / dyn_win:.2f} 1/сут — "
          f"близкие частоты в одном окне НЕ разделяются):")
        P(f"   частота главного пика в {dyn_frange[0]:.2f}–{dyn_frange[1]:.2f}: "
          f"{track[:, 1].min():.3f}–{track[:, 1].max():.3f} (медиана {np.median(track[:, 1]):.3f}); "
          f"амплитуда {track[:, 2].min():.1f}–{track[:, 2].max():.1f} ppt")

    # 4) фрагмент вокруг окна внешних наблюдений
    if ground_obs_ut:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        s = (t > g1 - 1.0) & (t < g2 + 1.0)
        ax.plot(t[s & good], y[s & good], ".", ms=2, color="k")
        ax.plot(t[s & ~good], y[s & ~good], ".", ms=2, color="C3")
        ax.axvspan(g1, g2, color="C0", alpha=0.3)
        ax.set_xlabel("BJD − 2457000"); ax.set_ylabel("ΔF, ppt")
        ax.set_title("TESS вокруг окна внешних наблюдений")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{tag}_ground_obs.pdf"))
        sb = (t >= g1) & (t <= g2)
        P(f"\nВнешние наблюдения {ground_obs_ut[0]} – {ground_obs_ut[1]} UT = BTJD {g1:.3f}–{g2:.3f}: "
          f"точек TESS {sb.sum()}, из них хороших {np.sum(sb & good)}; "
          f"локальный шум {np.median(sig[sb]) if sb.any() else np.nan:.1f} ppt")

    txt = "\n".join(rep)
    print(txt)
    with open(os.path.join(outdir, f"{tag}_report.txt"), "w") as fo:
        fo.write(txt + "\n")

    # LaTeX-фрагмент таблицы частот (набор strict), готовый для вставки в текст
    fr, sf, am, sa, sl, sr = results["strict"]["table"]
    with open(os.path.join(outdir, f"{tag}_freqs.tex"), "w") as fo:
        fo.write("% nu [1/d] & P [d] & A [ppt] & S/N_loc & S/N_red\n")
        for i in range(fr.size):
            fo.write(f"{i + 1} & ${fr[i]:.4f}\\pm{sf[i]:.4f}$ & ${1 / fr[i]:.4f}\\pm{sf[i] / fr[i] ** 2:.4f}$ & "
                     f"${am[i]:.2f}\\pm{sa[i]:.2f}$ & {sl[i]:.1f} & {sr[i]:.1f} \\\\\n")
    return results


def run_all(runs=None, show=SHOW_PLOTS):
    """Прогнать analyse() по списку RUNS. Работает из корня проекта."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    done = []
    for path, tag in (runs or RUNS):
        if not os.path.isfile(path):
            print(f"[пропуск] нет файла {path}")
            continue
        print("\n" + "=" * 70 + f"\n{tag}: {path}\n" + "=" * 70)
        analyse(path, tag, OUTDIR, FMAX, nmax=N_MAX, dyn_win=DYN_WINDOW,
               dyn_frange=DYN_FRANGE, dyn_freq_step=DYN_FREQ_STEP, comparison_freqs=COMPARISON_FREQS,
               ground_obs_ut=GROUND_OBS_UT, kappa=KAPPA,
               n_sigma_combination=N_SIGMA_COMBINATION,
               do_bootstrap=DO_BOOTSTRAP, n_boot=N_BOOT,
               do_stability=DO_STABILITY, stability_window=STABILITY_WINDOW, stability_step=STABILITY_STEP,
               do_clean=DO_CLEAN, clean_gain=CLEAN_GAIN, clean_niter=CLEAN_NITER)
        done.append(tag)
    print(f"\nГотово: {', '.join(done) or 'ничего'} -> {OUTDIR}/")
    if show and done:
        plt.show()


if __name__ == "__main__":
    run_all()
