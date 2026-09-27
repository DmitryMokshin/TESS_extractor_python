"""
SS 397, TESS сектор 80: анализ под замечания рецензента к разделу 5.

  п.1  инструментальные участки (≈3491, ≈3504): объективный критерий отбраковки
       по локальному шуму точка-к-точке + типичная ошибка потока;
  п.2  ошибки частот (Montgomery & O'Donoghue 1999 с поправкой на корреляцию
       остатков, Schwarzenberg-Czerny 1991) и значимость: S/N >= 4 относительно
       локального шума (Breger et al. 1993) и относительно модели красного шума;
  п.3  та же процедура для звезд сравнения: analyse(<csv>) — для любой кривой;
  п.4  динамический амплитудный спектр (скользящее окно) для группы 1.5–1.65 1/сут
       + амплитуды в ppt/mmag для обсуждения "что может дать такую амплитуду".

Входной формат: CSV со столбцами MJD (на деле BTJD = BJD-2457000), FLUX, [SN].

Запуск из PyCharm: поправь блок НАСТРОЙКИ ниже и нажми Run (зеленый треугольник).
Файл должен лежать в корне проекта TESS_extractor_python (рядом с run.py).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# =============================================================================
# НАСТРОЙКИ — правь здесь и жми Run
# =============================================================================
STAR_DIR = "stars_python/SS_397/50x50"
# какие кривые анализировать: (путь к CSV, метка для имен файлов).
# Отсутствующие файлы пропускаются с сообщением.
RUNS = [
    (f"{STAR_DIR}/light_curve_sector_80_clean.csv", "ss397_clean"),   # кривая из статьи
    ("out_localize/lc_deblended_SS397.csv", "ss397_prf"),             # после ss397_localize.py
    ("out_localize/lc_deblended_neighbour.csv", "neighbour"),         # яркий сосед
]
OUTDIR = "out_tess"
FMAX = 10.0          # верхняя частота поиска, 1/сут
N_MAX = 15           # максимум частот при выбеливании
DYN_WINDOW = 10.0    # окно динамического спектра, сут
KAPPA = 2.5          # отбраковка: локальный шум > KAPPA × медиана
# время наблюдений на БТА (UT); None — не рисовать
BTA_UT = None        # например ("2024-06-24T19:30", "2024-06-24T22:30")
SHOW_PLOTS = True    # открыть окна с рисунками в конце (PDF сохраняются всегда)

# =============================================================================
PPT2MMAG = 2.5 / np.log(10)   # 1 ppt ≈ 1.086 mmag

# частоты из табл. 2 статьи (для сравнения на рисунках)
PAPER_FREQS = [0.068, 0.132, 0.189, 1.516, 1.580, 1.625]


# =============================================================================
# данные и отбраковка
# =============================================================================
def load_lc(path):
    import pandas as pd
    d = pd.read_csv(path)
    tcol = "MJD" if "MJD" in d else d.columns[0]
    fcol = "FLUX" if "FLUX" in d else d.columns[1]
    d = d[np.isfinite(d[tcol]) & np.isfinite(d[fcol])]
    t = d[tcol].to_numpy(float)
    f = d[fcol].to_numpy(float)
    o = np.argsort(t)
    return t[o], f[o]


def local_p2p(t, y, win=0.25):
    """Шум точка-к-точке (робастный, через MAD разностей) в скользящем окне win сут."""
    d = np.diff(y)
    sig = np.empty_like(y)
    half = win / 2
    lo = np.searchsorted(t, t - half)
    hi = np.searchsorted(t, t + half)
    for i in range(t.size):
        a, b = lo[i], min(hi[i], t.size - 1)
        dd = d[a:b]
        if dd.size < 10:
            sig[i] = np.nan
            continue
        sig[i] = 1.4826 * np.median(np.abs(dd - np.median(dd))) / np.sqrt(2)
    sig[np.isnan(sig)] = np.nanmedian(sig)
    return sig


def quality_mask(t, y, win=0.25, kappa=2.5):
    """Точка хорошая, если локальный шум < kappa * медианный шум по сектору."""
    sig = local_p2p(t, y, win)
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
# спектр амплитуд, выбеливание, ошибки
# =============================================================================
def amp_spectrum(t, y, freqs, chunk=400):
    """Амплитудный спектр (ДФТ) в единицах y: A(f) = 2/N |Σ y exp(-2πift)|."""
    y = y - y.mean()
    out = np.empty(freqs.size)
    for k in range(0, freqs.size, chunk):
        f = freqs[k:k + chunk]
        ph = np.exp(-2j * np.pi * np.outer(f, t))
        out[k:k + chunk] = 2.0 / t.size * np.abs(ph @ y)
    return out


def window_function(t, freqs, f0):
    """Спектральное окно, центрированное на f0 (для рисунка)."""
    return amp_spectrum(t, np.sin(2 * np.pi * f0 * t), freqs)


NPOLY = 3   # полином 2-й степени по времени — медленный тренд за сектор


def model(p, t, nf, t0):
    x = t - t0
    y = p[0] + p[1] * x + p[2] * x ** 2
    for k in range(nf):
        f, a, ph = p[NPOLY + 3 * k:NPOLY + 3 * k + 3]
        y = y + a * np.sin(2 * np.pi * (f * t + ph))
    return y


def detrend_poly(t, y):
    t0 = t.mean()
    c = np.polyfit(t - t0, y, 2)
    return y - np.polyval(c, t - t0)


def fit_all(t, y, freqs, amps, phases, fmin=0.0, fmax=np.inf, anchors=None, halfwidth=None):
    t0 = t.mean()
    c = np.polyfit(t - t0, y, 2)[::-1]
    p0 = list(c)
    lo = [-np.inf] * NPOLY
    hi = [np.inf] * NPOLY
    for i, (f, a, ph) in enumerate(zip(freqs, amps, phases)):
        p0 += [f, a, ph]
        if anchors is not None:     # частота может сместиться не более чем на halfwidth от момента обнаружения
            lo += [max(fmin, anchors[i] - halfwidth), -np.inf, -np.inf]
            hi += [min(fmax, anchors[i] + halfwidth), np.inf, np.inf]
        else:
            lo += [fmin, -np.inf, -np.inf]
            hi += [fmax, np.inf, np.inf]
    p0 = np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
    r = least_squares(lambda p: model(p, t, len(freqs), t0) - y, p0, bounds=(lo, hi), method="trf",
                      x_scale="jac")
    p = r.x
    fr = p[NPOLY::3].copy(); am = p[NPOLY + 1::3].copy(); ph = p[NPOLY + 2::3].copy()
    neg = am < 0
    am[neg] *= -1; ph[neg] += 0.5
    return fr, am, ph % 1, y - model(p, t, len(freqs), t0)


def phase_guess(t, y, f):
    X = np.c_[np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)]
    s, c = np.linalg.lstsq(X, y - y.mean(), rcond=None)[0]
    return np.hypot(s, c), (np.arctan2(c, s) / (2 * np.pi)) % 1


def local_noise(freqs_grid, amp_res, f, box=1.0):
    m = (freqs_grid > max(freqs_grid[0], f - box / 2)) & (freqs_grid < f + box / 2)
    return amp_res[m].mean()


def run_lengths_D(res):
    """Средняя длина серии остатков одного знака (Schwarzenberg-Czerny 1991)."""
    s = np.sign(res)
    changes = np.count_nonzero(s[1:] != s[:-1])
    return max(1.0, res.size / (changes + 1))


def prewhiten(t, y, fmin, fmax, df, nmax=15, snr_stop=4.0, box=1.0):
    """
    Итеративное выбеливание. На каждом шаге к амплитудному спектру остатков
    подгоняется модель красного + белого шума, и берется пик с НАИБОЛЬШИМ S/N
    относительно этой модели (а не просто самый высокий: у Be-звезд на низких
    частотах сильный стохастический красный шум). Остановка при S/N < snr_stop.
    После добавления каждой частоты все параметры + полиномиальный тренд
    подгоняются заново одновременно.
    """
    freqs_grid = np.arange(fmin, fmax, df)
    fr, am, ph, anchors = [], [], [], []
    T = t.max() - t.min()
    res = detrend_poly(t, y)
    for _ in range(nmax):
        A = amp_spectrum(t, res, freqs_grid)
        rn, _ = red_noise_fit(freqs_grid, A)
        snr = A / rn(freqs_grid)
        # только локальные максимумы
        peak = np.r_[False, (A[1:-1] > A[:-2]) & (A[1:-1] > A[2:]), False]
        # запрет частот ближе 1/T (рэлеевское разрешение) к уже найденным:
        # иначе МНК начинает описывать амплитудную/частотную модуляцию парой
        # почти равных частот с огромными противофазными амплитудами
        for f0 in fr:
            peak &= np.abs(freqs_grid - f0) > 1.0 / T
        snr[~peak] = 0
        j = np.argmax(snr)
        if snr[j] < snr_stop:
            break
        fine = np.linspace(freqs_grid[j] - df, freqs_grid[j] + df, 41)
        fnew = fine[np.argmax(amp_spectrum(t, res, fine))]
        a0, p0 = phase_guess(t, res, fnew)
        fr.append(fnew); am.append(a0); ph.append(p0); anchors.append(fnew)
        f_, a_, p_, res = fit_all(t, y, fr, am, ph, fmin=fmin, fmax=fmax,
                                  anchors=anchors, halfwidth=0.25 / T)
        fr, am, ph = list(f_), list(a_), list(p_)
    A_res = amp_spectrum(t, res, freqs_grid)
    return np.array(fr), np.array(am), np.array(ph), res, freqs_grid, A_res


def errors_mo99(t, res, amps):
    """Montgomery & O'Donoghue (1999) × sqrt(D) на коррелированные остатки."""
    N, T = t.size, t.max() - t.min()
    sN = res.std()
    D = run_lengths_D(res)
    sf = np.sqrt(6.0) / (np.pi * T) * sN / (amps * np.sqrt(N)) * np.sqrt(D)
    sa = np.sqrt(2.0 / N) * sN * np.sqrt(D) * np.ones_like(amps)
    sp = sa / amps / (2 * np.pi)          # в долях периода
    return sf, sa, sp, D, sN


def red_noise_fit(freqs, amp, smooth=1.0):
    """
    Модель шума A(f) = A0/(1+(f/fc)^g) + Cw (Bowman et al. 2019), подгоняется к
    амплитудному спектру остатков, сглаженному СКОЛЬЗЯЩЕЙ МЕДИАНОЙ шириной smooth 1/сут
    (× 1.0645 — переход от медианы к среднему для шумового спектра). Медиана устойчива
    к еще не вычтенным сильным пикам: иначе мощная группа частот сама задирает
    "шум" и выбеливание останавливается слишком рано.
    """
    from scipy.ndimage import median_filter
    df = freqs[1] - freqs[0]
    n = max(3, int(round(smooth / df)) | 1)
    sm = 1.0645 * median_filter(amp, size=n, mode="reflect")
    m = freqs > 2 * df
    x, yv = freqs[m], sm[m]

    def mdl(p, x):
        A0, fc, g, Cw = np.exp(p[0]), np.exp(p[1]), p[2], np.exp(p[3])
        return A0 / (1 + (x / fc) ** g) + Cw

    hi = x > np.percentile(x, 80)
    lo_b, hi_b = [-20, np.log(0.02), 0.5, -20], [10, np.log(10), 6, 10]
    p0 = [np.log(yv[x < 1].mean() + 1e-9), np.log(1.0), 2.0, np.log(np.median(yv[hi]) + 1e-9)]
    p0 = list(np.clip(p0, np.array(lo_b) + 1e-6, np.array(hi_b) - 1e-6))
    r = least_squares(lambda p: np.log(mdl(p, x)) - np.log(yv), p0, bounds=(lo_b, hi_b))
    return lambda f: mdl(r.x, f), r.x


# =============================================================================
# основной анализ
# =============================================================================
def analyse(path, tag="ss397", outdir="out_tess", fmax=5.0, nmax=15, dyn_win=10.0,
            bta_ut=None, kappa=2.5):
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
        df = 0.1 / T
        fr, am, ph, res, fg, Ares = prewhiten(tt, yy, 0.5 / T, fmax, df, nmax=nmax)
        order = np.argsort(fr)
        fr, am, ph = fr[order], am[order], ph[order]
        sf, sa, sp, D, sN = errors_mo99(tt, res, am)
        rn, _ = red_noise_fit(fg, Ares)
        snr_loc = np.array([a / local_noise(fg, Ares, x) for x, a in zip(fr, am)])
        snr_red = am / rn(fr)
        results[label] = dict(t=tt, y=yy, fr=fr, am=am, sf=sf, sa=sa, res=res, fg=fg, Ares=Ares, rn=rn)

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
        # гармоники и комбинации: только для 5 самых сильных "родителей",
        # допуск 0.5/T, "ребенок" слабее обоих родителей
        top = np.argsort(am)[::-1][:5]
        for ii, i in enumerate(top):
            for j in top[ii:]:
                for s_, lab in [(fr[i] + fr[j], "+"), (abs(fr[i] - fr[j]), "-")]:
                    for k in range(len(fr)):
                        if (k not in (i, j) and abs(fr[k] - s_) < 0.5 / T
                                and am[k] < min(am[i], am[j])):
                            P(f"   комбинация? ν={fr[k]:.4f} ≈ {fr[i]:.4f} {lab} {fr[j]:.4f} = {s_:.4f}")
        results[label]["table"] = (fr, sf, am, sa, snr_loc, snr_red)

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
    if bta_ut:
        from astropy.time import Time
        b1, b2 = [Time(s, scale="utc").jd - 2457000 for s in bta_ut]
        for axx in ax:
            axx.axvspan(b1, b2, color="C0", alpha=0.3, lw=0)
        ax[0].text(b1, ax[0].get_ylim()[1], " БТА", color="C0", va="top")
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
    z = (fg > 1.2) & (fg < 2.0)
    ax[1].plot(fg[z], A0[z], "k-", lw=1, label="SS 397")
    main = R["fr"][(R["fr"] > 1.2) & (R["fr"] < 2)]
    if main.size:
        f0 = main[np.argmax(R["am"][(R["fr"] > 1.2) & (R["fr"] < 2)])]
        W = window_function(R["t"], fg[z], f0)
        ax[1].plot(fg[z], W / W.max() * A0[z].max(), "-", color="C2", lw=0.8,
                   label=f"спектральное окно на {f0:.3f}")
    for x in PAPER_FREQS:
        if 1.2 < x < 2.0:
            ax[1].axvline(x, color="C1", ls=":", lw=1)
    for x in main:
        ax[1].axvline(x, color="C0", lw=0.8)
    ax[1].set_xlabel("Частота, 1/сут"); ax[1].set_ylabel("A, ppt")
    ax[1].legend(fontsize=8)
    ax[1].text(0.01, 0.95, "синие — найденные здесь, оранжевые — табл. 2 статьи",
               transform=ax[1].transAxes, fontsize=7, va="top")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{tag}_amplitude_spectrum.pdf"))

    # 3) динамический амплитудный спектр
    tt, yy = R["t"], R["y"]
    centers = np.arange(tt.min() + dyn_win / 2, tt.max() - dyn_win / 2 + 1e-6, 0.25)
    fd = np.arange(0.8, 2.5, 0.005)
    dyn = np.full((centers.size, fd.size), np.nan)
    track = []
    for i, c in enumerate(centers):
        m = np.abs(tt - c) < dyn_win / 2
        if m.sum() < 0.6 * dyn_win / np.median(np.diff(tt)):   # слишком большие пробелы
            continue
        yy_w = yy[m] - np.polyval(np.polyfit(tt[m] - c, yy[m], 2), tt[m] - c)  # убираем медленный тренд
        dyn[i] = amp_spectrum(tt[m], yy_w, fd)
        g = (fd > 1.3) & (fd < 1.9)
        k = np.argmax(dyn[i][g])
        track.append((c, fd[g][k], dyn[i][g][k]))
    track = np.array(track)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw=dict(width_ratios=[3, 1]))
    im = ax[0].pcolormesh(fd, centers, dyn, shading="auto", cmap="viridis")
    for x in PAPER_FREQS[3:]:
        ax[0].axvline(x, color="w", ls=":", lw=0.8)
    ax[0].set_xlabel("Частота, 1/сут"); ax[0].set_ylabel(f"центр окна ({dyn_win:.0f} сут), BJD−2457000")
    plt.colorbar(im, ax=ax[0], label="A, ppt")
    if track.size:
        ax[1].plot(track[:, 2], track[:, 0], "k.-")
        ax[1].set_xlabel("A пика в 1.3–1.9, ppt")
        ax2 = ax[1].twiny()
        ax2.plot(track[:, 1], track[:, 0], "C3.", ms=3)
        ax2.set_xlabel("ν пика", color="C3")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{tag}_dynamic_spectrum.pdf"))
    if track.size:
        P(f"\nДинамический спектр (окно {dyn_win} сут, разрешение ~{1 / dyn_win:.2f} 1/сут — "
          f"три частоты группы в одном окне НЕ разделяются):")
        P(f"   частота главного пика в 1.3–1.9: {track[:, 1].min():.3f}–{track[:, 1].max():.3f} "
          f"(медиана {np.median(track[:, 1]):.3f}); амплитуда {track[:, 2].min():.1f}–{track[:, 2].max():.1f} ppt")

    # 4) фрагмент вокруг ночи БТА
    if bta_ut:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        s = (t > b1 - 1.0) & (t < b2 + 1.0)
        ax.plot(t[s & good], y[s & good], ".", ms=2, color="k")
        ax.plot(t[s & ~good], y[s & ~good], ".", ms=2, color="C3")
        ax.axvspan(b1, b2, color="C0", alpha=0.3)
        # модель из найденных частот (без трендов < 2/T) для наглядности
        ax.set_xlabel("BJD − 2457000"); ax.set_ylabel("ΔF, ppt")
        ax.set_title("TESS вокруг ночи наблюдений на БТА")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{tag}_bta_night.pdf"))
        sb = (t >= b1) & (t <= b2)
        P(f"\nНочь БТА {bta_ut[0]} – {bta_ut[1]} UT = BTJD {b1:.3f}–{b2:.3f}: точек TESS {sb.sum()}, "
          f"из них хороших {np.sum(sb & good)}; локальный шум {np.median(sig[sb]) if sb.any() else np.nan:.1f} ppt")

    txt = "\n".join(rep)
    print(txt)
    with open(os.path.join(outdir, f"{tag}_report.txt"), "w") as fo:
        fo.write(txt + "\n")

    # LaTeX-таблица для статьи (набор strict)
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
        analyse(path, tag, OUTDIR, FMAX, nmax=N_MAX, dyn_win=DYN_WINDOW, bta_ut=BTA_UT, kappa=KAPPA)
        done.append(tag)
    print(f"\nГотово: {', '.join(done) or 'ничего'} -> {OUTDIR}/")
    if show and done:
        plt.show()


if __name__ == "__main__":
    run_all()
