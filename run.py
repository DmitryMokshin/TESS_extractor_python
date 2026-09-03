from isolated.data_io import load_star_gaia_data, load_light_curve
from isolated.tess_point import find_tess_sectors
from isolated.config import TESS_MAX_SECTORS
import matplotlib.pyplot as plt
from isolated.viewer import plot_cuts
from isolated.lightcurve_tools import save_lc_figure
from isolated.viewer import save_cutout_video

star_name = "EM* AS 14"

# 1. Найти звезду в Gaia (по имени через Simbad), закэшировать в stars_python/
gaia = load_star_gaia_data(star_name)
print(gaia)

# 2. Узнать, в каких секторах TESS наблюдал эту точку неба
sectors = find_tess_sectors(float(gaia["ra"]), float(gaia["dec"]), TESS_MAX_SECTORS)
print("Секторы:", sectors)

# 3. Скачать TESS-вырезку и построить кривую блеска для первого сектора и как итог pandas таблица
lc = load_light_curve(star_name, sectors[0], cut_width=50)
print(lc.head())
print(len(lc))

# # 4 Интерактивный просмотр кривой блеска сектора
viewer = plot_cuts(star_name=star_name, sector=sectors[0], cut_width=50, cut_height=50)
plt.show()

# 5 Вывод кривой блеска звезда в png картинку
save_lc_figure(star_name=star_name, sector=sectors[0], cut_size=50, day_step=2, jd_box=0.3, sigma_tol=5, n_out=10)

# 6 Сохранение видео отрывков для заданной звезды, в заданном секторе, с заданным размером изображения, с заданным ФПС, в заданном диапазоне кадров
# Для определения диапазона кадров, нужно смотреть шаг 4 с интерактивной кривой блеска. Путь выведется на экран. Стандарт сохранение plots.

# path = save_cutout_video(star_name, sector=sectors[0], cut_width=50, cut_height=50,
#                           frame_range=(2500, 2700), fps=10)
# print(path)
#
# path = save_cutout_video(star_name, sector=sectors[0], cut_width=50, cut_height=50,
#                           frame_range=(4400, 5000), fps=10)
# print(path)
#
# path = save_cutout_video(star_name, sector=sectors[0], cut_width=50, cut_height=50,
#                           frame_range=(10300, 10650), fps=10)
# print(path)