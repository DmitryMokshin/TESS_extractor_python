"""
Интерактивная обработка одной звезды: скачивание данных, выбор сектора
TESS с клавиатуры (после показа списка доступных), интерактивный просмотр
кадров, сохранение кривой блеска в PNG и, по желанию, видео-отрывков.

Запуск: python run.py -- дальше отвечаете на вопросы в терминале
(Enter -- принять значение по умолчанию в квадратных скобках).
"""
import matplotlib.pyplot as plt

from isolated.data_io import load_star_gaia_data, load_light_curve
from isolated.tess_point import find_tess_sectors
from isolated.config import TESS_MAX_SECTORS
from isolated.viewer import plot_cuts, save_cutout_video
from isolated.lightcurve_tools import save_lc_figure

STAR_NAME = "EM* AS 14"
CUT_WIDTH = 15
CUT_HEIGHT = 15


def print_stage(title, **settings):
    """Заголовок этапа + текущие настройки, с которыми он выполняется."""
    print()
    print("=" * 60)
    print(title)
    for key, value in settings.items():
        print(f"  {key}: {value}")
    print("=" * 60)


def ask_int(prompt, default=None, choices=None):
    """Запросить целое число с клавиатуры; Enter -- взять default.
    Если задан `choices`, значение обязано в них входить."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        try:
            raw = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            if default is not None:
                return default
            raise
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Нужно целое число.")
            continue
        if choices is not None and value not in choices:
            print(f"Нужно одно из: {choices}")
            continue
        return value


def ask_yes_no(prompt, default=False):
    """Запросить да/нет с клавиатуры; Enter -- взять default."""
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        raw = input(f"{prompt} {suffix}: ").strip().lower()
    except EOFError:
        return default
    if not raw:
        return default
    return raw in ("y", "yes", "д", "да")


def ask_frame_ranges():
    """Запросить один или несколько диапазонов кадров для видео, вида
    '2500-2700, 4400-5000'. Пустой ввод -- один клип на весь сектор."""
    try:
        raw = input("Диапазоны кадров для видео через запятую (напр. 2500-2700, 4400-5000), "
                     "Enter -- весь сектор одним клипом: ").strip()
    except EOFError:
        return [None]
    if not raw:
        return [None]
    ranges = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        start_s, _, end_s = chunk.partition("-")
        ranges.append((int(start_s), int(end_s)))
    return ranges or [None]


def main():
    print_stage("Шаг 1-2: звезда в Gaia + секторы TESS", звезда=STAR_NAME)

    # 1. Найти звезду в Gaia (по имени через Simbad), закэшировать в stars_python/
    gaia = load_star_gaia_data(STAR_NAME)
    print(gaia)

    # 2. Узнать, в каких секторах TESS наблюдал эту точку неба -- и выбрать нужный с клавиатуры
    sectors = find_tess_sectors(float(gaia["ra"]), float(gaia["dec"]), TESS_MAX_SECTORS)
    print("Доступные секторы:", sectors)
    sector = ask_int("Какой сектор скачать и обработать", default=sectors[0], choices=sectors)

    print_stage("Шаг 3: загрузка/кэширование кривой блеска",
                звезда=STAR_NAME, сектор=sector, размер_кадра=f"{CUT_WIDTH}x{CUT_HEIGHT}")

    # 3. Скачать TESS-вырезку и построить кривую блеска выбранного сектора
    lc = load_light_curve(STAR_NAME, sector, cut_width=CUT_WIDTH, cut_height=CUT_HEIGHT)
    print(lc.head())
    print(len(lc))

    print_stage("Шаг 4: интерактивный просмотр кадров", звезда=STAR_NAME, сектор=sector)

    # 4. Интерактивный просмотр кадров + кривой блеска сектора
    viewer = plot_cuts(star_name=STAR_NAME, sector=sector, cut_width=CUT_WIDTH, cut_height=CUT_HEIGHT)
    plt.show()

    print_stage("Шаг 5: сохранение PNG кривой блеска", звезда=STAR_NAME, сектор=sector)

    # 5. Вывод кривой блеска звезды в png картинку
    save_lc_figure(star_name=STAR_NAME, sector=sector, cut_size=CUT_WIDTH, day_step=2,
                    jd_box=0.3, sigma_tol=5, n_out=10)

    # 6. Сохранение видео-отрывков -- по желанию
    save_video = ask_yes_no("Сохранить видео-отрывки по кадрам сектора?", default=False)
    fps, frame_ranges = None, []
    if save_video:
        fps = ask_int("FPS видео", default=10)
        frame_ranges = ask_frame_ranges()

    print_stage("Шаг 6: сохранение видео",
                звезда=STAR_NAME, сектор=sector, видео=save_video, fps=fps, диапазоны=frame_ranges)

    if save_video:
        for frame_range in frame_ranges:
            path = save_cutout_video(STAR_NAME, sector=sector, cut_width=CUT_WIDTH, cut_height=CUT_HEIGHT,
                                      frame_range=frame_range, fps=fps)
            print("Сохранено:", path)


if __name__ == "__main__":
    main()
