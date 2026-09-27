"""
Полная чистка кривой блеска сектора: сначала точечная чистка (провалы,
нули, явные выбросы) прямо на кривой блеска, затем чистка на уровне
кадров (полосы/засветка матрицы) -- чтобы поймать то, что точечная чистка
могла не тронуть (например мощный, но короткий выброс, который "на глаз"
сложно отличить от настоящей вспышки звезды).

Важно: оба набора окон ищутся по ИСХОДНОЙ, ещё не урезанной кривой, и
вырезаются ОДНИМ вызовом exclude_frame_windows в конце. Так и должно быть:
plot_stray_light_diagnostics и df_monitor построены по полному, нетронутому
набору кадров сектора, и номера кадров в windows_frames заданы относительно
него. Если сначала урезать кривую по windows_lc, а потом резать по
windows_frames уже урезанную кривую -- номера кадров разъедутся (после
удаления строк позиция строки больше не равна номеру кадра), и вырежутся не
те кадры. Резать сразу оба набора окон за один проход -- единственный
надёжный способ.
"""
import pandas as pd

from isolated.data_io import load_star_gaia_data, load_light_curve
from isolated.tess_point import find_tess_sectors
from isolated.config import TESS_MAX_SECTORS
from isolated.cleaning import auto_clean_light_curve, save_cleaning_log, summarize_cleaning_log
from isolated.lightcurve_tools import exclude_frame_windows, pick_exclusion_windows, save_clean_light_curve, \
    save_trash_light_curve, windows_to_log
from isolated.stray_light import pick_stripe_pixels, stray_light_monitor_curve, plot_stray_light_diagnostics
from run_config import CONFIG

# Как решить, дефект это или настоящая вспышка -- по панелям plot_stray_light_diagnostics:
#
# Синхронность (панели 1-2). Если скачок в кривой звезды по времени точно совпадает со скачком
# в кривой-мониторе (пустой фон) — это почти наверняка дефект: звезда физически не может заставить
# светиться пиксели, где её нет.
#
# Корреляция (панель 3). Если точки из подозрительного окна выстраиваются в чёткую диагональ (избыток
# звезды растёт вместе с избытком монитора) — сильный признак общей причины (засветка всего кадра),
# а не независимого события на звезде. Разбросанные, некоррелирующие точки — довод за реальную вспышку.
#
# Пространственная форма (панель 4). Дефект (полосы/засветка) обычно виден как явная линия/полоса через
# весь кадр, не привязанная к положению звезды (крестик). Настоящая вспышка увеличивает яркость только
# в районе апертуры звезды, остальной кадр не меняется.
#
# Если все три признака указывают на дефект — вырезаем через exclude_frame_windows. Если хотя бы
# синхронность/корреляция отсутствуют, а яркость локализована именно на звезде — это, скорее всего,
# настоящая вспышка, точки лучше оставить (или сохранить отдельно для дальнейшего изучения).

STAR_NAME = CONFIG.star_name
CUT_WIDTH = CONFIG.cut_width
CUT_HEIGHT = CONFIG.cut_height
TESS_SECTOR = CONFIG.sector


def clean_sector(star_name, sector, cut_width, cut_height=None):
    if cut_height is None:
        cut_height = cut_width

    df_lc = load_light_curve(star_name, sector, cut_width, cut_height)

    # 0. Автоочистка (ROADMAP.md Этап 5): QUALITY -> STAR_BKG_RATIO -> локальный шум,
    # по порядку, каждое правило -- на выживших после предыдущего
    df_auto, log_auto = auto_clean_light_curve(
        df_lc, quality_bitmask=CONFIG.quality_bitmask, star_bkg_ratio_min=CONFIG.star_bkg_ratio_min,
        local_noise_kappa=CONFIG.local_noise_kappa)
    print(f"Автоочистка: {len(df_lc)} -> {len(df_auto)} кадров ("
          + ", ".join(f"{reason}: {count}" for reason, count in log_auto["REASON"].value_counts().items()) + ")")

    # 1. Точечная чистка: провалы, нули, явные выбросы -- выделяем мышью (drag) прямо на кривой блеска,
    # уже без того, что отсеяла автоочистка (меньше шума на глаз)
    windows_lc = pick_exclusion_windows(df_auto, star_name=star_name, sector=sector)
    print("Окна точечной чистки:", windows_lc)

    # 2. Чистка на уровне кадров: засветка/полосы, которые точечная чистка могла не поймать
    pixels = pick_stripe_pixels(star_name, sector, cut_width, cut_height)
    df_monitor = stray_light_monitor_curve(star_name, sector, cut_width, cut_height, pixels=pixels)

    windows_frames = pick_exclusion_windows(df_monitor, star_name=star_name, sector=sector)
    print("Окна чистки по кадрам:", windows_frames)

    # диагностика по ИСХОДНОЙ df_lc (см. пояснение в докстринге модуля выше)
    plot_stray_light_diagnostics(star_name, sector, cut_width, cut_height, df_lc, df_monitor,
                                  windows=windows_frames)

    # 3. Вырезаем оба набора ручных окон одним проходом по уже автоочищенной кривой
    all_windows = windows_lc + windows_frames
    df_clean, df_trash = exclude_frame_windows(df_auto, all_windows)

    save_clean_light_curve(df_clean, star_name, sector, cut_width, cut_height)
    save_trash_light_curve(df_trash, star_name, sector, all_windows, cut_width, cut_height)

    # общий лог (автоочистка + ручные окна) -- по нему можно дословно восстановить,
    # почему выброшен каждый кадр (ROADMAP.md Этап 5, критерий готовности)
    log_manual = pd.concat([
        windows_to_log(df_auto, windows_lc, "manual:point-picker"),
        windows_to_log(df_auto, windows_frames, "manual:stray-light"),
    ], ignore_index=True)
    log_combined = pd.concat([log_auto, log_manual], ignore_index=True).sort_values("FRAME").reset_index(drop=True)
    log_path = save_cleaning_log(log_combined, star_name, sector, cut_width, cut_height)
    print(f"\nЛог чистки: {log_path}\n{summarize_cleaning_log(log_combined, total_frames=len(df_lc))}")

    return df_clean


if __name__ == "__main__":

    gaia = load_star_gaia_data(STAR_NAME)
    print(gaia)

    sectors = find_tess_sectors(float(gaia["ra"]), float(gaia["dec"]), TESS_MAX_SECTORS)
    print("Секторы:", sectors)

    print("Обработка сектора:", TESS_SECTOR)
    print("=" * 70)
    clean_sector(STAR_NAME, TESS_SECTOR, CUT_WIDTH, CUT_HEIGHT)
