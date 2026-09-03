from isolated.lightcurve_tools import pick_exclusion_windows, exclude_frame_windows, save_clean_light_curve, \
    save_trash_light_curve
from isolated.stray_light import (pick_stripe_pixels, stray_light_monitor_curve,
                                   plot_stray_light_diagnostics)
from isolated.data_io import load_light_curve

# Обработчик для поиска аномалий и устранения их на кадрах TESS

# Синхронность (панели 1-2). Если скачок в кривой звезды по времени точно совпадает со скачком
# в кривой-мониторе (пустой фон) — это почти наверняка дефект: звезда физически не может заставить светиться пиксели, где её нет.

# Корреляция (панель 3). Если точки из подозрительного окна выстраиваются в чёткую диагональ (избыток звезды растёт вместе с избытком монитора) — сильный признак общей причины
# (засветка всего кадра), а не независимого события на звезде. Разбросанные, некоррелирующие точки — довод за реальную вспышку.

# Пространственная форма (панель 4). Дефект (полосы/засветка) обычно виден как явная линия/полоса через весь кадр, не привязанная к положению звезды (крестик).
# Настоящая вспышка увеличивает яркость только в районе апертуры звезды, остальной кадр не меняется.

# Если все три признака указывают на дефект — смело вырезайте через exclude_frame_windows. Если хотя бы синхронность/корреляция отсутствуют,
# а яркость локализована именно на звезде — это, скорее всего, настоящая вспышка, и лучше оставить точки как есть (или сохранить отдельно для дальнейшего изучения).

pixels = pick_stripe_pixels("EM* AS 14", sector=18, cut_width=50, cut_height=50)

df_monitor = stray_light_monitor_curve("EM* AS 14", sector=18, cut_width=50, cut_height=50, pixels=pixels)
df_lc = load_light_curve("EM* AS 14", sector=18, cut_width=50)

windows = pick_exclusion_windows(df_monitor, star_name="EM* AS 14", sector=18)
plot_stray_light_diagnostics("EM* AS 14", 18, 50, 50, df_lc, df_monitor, windows=windows)

# Шаг с вырезом окошка из кадра.

df_clean, df_trash = exclude_frame_windows(df_lc, windows)

save_clean_light_curve(df_clean, "EM* AS 14", sector=18, cut_width=50)
save_trash_light_curve(df_trash, star_name="EM* AS 14", sector=18, windows=windows, cut_width=50)

