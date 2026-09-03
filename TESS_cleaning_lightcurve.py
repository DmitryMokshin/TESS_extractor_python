from isolated.data_io import load_star_gaia_data, load_light_curve
from isolated.tess_point import find_tess_sectors
from isolated.config import TESS_MAX_SECTORS
from isolated.lightcurve_tools import exclude_frame_windows, pick_exclusion_windows, save_clean_light_curve, \
    save_trash_light_curve

# Просто алгоритм вырезать кадры из кривой блеска

star_name = "EM* AS 14"

gaia = load_star_gaia_data(star_name)
print(gaia)

sectors = find_tess_sectors(float(gaia["ra"]), float(gaia["dec"]), TESS_MAX_SECTORS)
print("Секторы:", sectors)

for sector in [sectors[1]]:
    print("Обработка сектора:", sector)
    print("=" * 70)
    lc = load_light_curve(star_name, sector, cut_width=50)

    windows = pick_exclusion_windows(lc, star_name, sector)

    print("Вырезанные окна:", windows)
    for window in windows:
        print("Окно:", window)

    df_clean, df_trash = exclude_frame_windows(lc, windows)

    save_clean_light_curve(df_clean, star_name, sector, cut_width=50)
    save_trash_light_curve(df_trash, star_name, sector, windows, cut_width=50)
