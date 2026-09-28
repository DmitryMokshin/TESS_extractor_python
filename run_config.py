"""
Единая точка настройки пайплайна: звезда, сектор, размер вырезки, режим
фотометрии и параметры чистки/частотного анализа. Смена звезды/сектора --
одна правка здесь; `run.py`, `TESS_cleaning.py`, `Peridogram_compute.py`,
`PRF_cleaning.py`, `Peridogram_diagnostics.py` читают этот файл вместо собственных
констант.

Не сюда: то, что специфично для одного скрипта/конкретного исследования,
а не для пайплайна в целом (id звезды сравнения, частоты для карт
амплитуд, сравнение со статьёй, флаги показа/сохранения графиков) --
это остаётся локальными константами в соответствующем скрипте.

Запуск из PyCharm кнопкой Run сохраняется: правите значения в `RunConfig()`
ниже и жмёте Run на любом скрипте пайплайна.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RunConfig:
    # --- звезда / сектор / вырезка ---
    star_name: str = "SS 397"
    sector: int = 80
    # несколько секторов (ROADMAP.md Этап 7): если задано (>=2 значений),
    # Peridogram_compute.py анализирует объединенный (склеенный) ряд вместо
    # одного `sector`; каждый сектор из списка должен уже иметь свою
    # `_clean`/`_prf_clean` кривую (обычный прогон TESS_cleaning.py/
    # PRF_cleaning.py на каждый сектор по отдельности). По умолчанию не
    # задано -- поведение всех скриптов не меняется, как и раньше один `sector`.
    sectors: Optional[tuple[int, ...]] = None
    cut_width: int = 50
    cut_height: int = 50
    # "prf" -- PRF_cleaning.py-стиль PRF-деблендинг (isolated.prf_photometry),
    # "aperture" -- TESS_cleaning.py-стиль апертурная фотометрия (load_light_curve)
    photometry_mode: str = "prf"

    # --- апертурная фотометрия (используется, если photometry_mode == "aperture") ---
    aperture_radius: int = 3
    d_mag_r: float = 5.0
    # порог правила №2 автоочистки (ROADMAP.md Этап 5): кадры со STAR_BKG_RATIO
    # ниже этого выбрасываются. По умолчанию выключено (ROADMAP не дает числа,
    # а порог напрямую меняет официальную _clean-кривую) -- включать осознанно
    # для конкретной звезды, см. isolated.cleaning.auto_clean_light_curve.
    star_bkg_ratio_min: Optional[float] = None

    # --- PRF-фотометрия (используется, если photometry_mode == "prf") ---
    prf_box: int = 13
    prf_dt_max: float = 3.5
    prf_merge_px: float = 1.0

    # --- отбраковка кадров ---
    quality_bitmask: int = 175  # см. isolated.prf_photometry/PRF_cleaning.py
    local_noise_kappa: float = 2.5  # см. isolated.lightcurve_tools.local_point_to_point_sigma

    # --- частотный анализ (Peridogram_compute.py / Peridogram_diagnostics.py) ---
    nyquist_factor: float = 2.0
    max_period_fraction: float = 0.5
    detrend_deg: int = 2
    prewhiten_fmax: float = 10.0
    snr_stop: float = 4.0
    n_max_freq: int = 15
    dyn_window: float = 10.0
    dyn_step: float = 0.25
    dyn_frange: tuple = (0.8, 2.5)
    image_format: str = "png"
    max_peaks_analyse: int = 10

    def __post_init__(self):
        if self.photometry_mode not in ("aperture", "prf"):
            raise ValueError(f'photometry_mode must be "aperture" or "prf", got {self.photometry_mode!r}')

    @property
    def lc_suffix(self) -> str:
        """Какой файл кривой блеска соответствует текущему `photometry_mode`."""
        return "_prf_clean" if self.photometry_mode == "prf" else "_clean"

    @property
    def star_dir(self) -> str:
        """`stars_python/{звезда}/{ширина}x{высота}/`, куда ложится всё для этой звезды/вырезки."""
        from isolated.geometry import get_nospace_star_name
        return f"stars_python/{get_nospace_star_name(self.star_name)}/{self.cut_width}x{self.cut_height}"


CONFIG = RunConfig()
