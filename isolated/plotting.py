"""
Common figure handling: save to file, print where it went, close -- instead of
opening a window for every figure.

By default (`SHOW = False`) figures are only saved and closed, and each saved
path is printed to the console. Scripts set the switch from
`run_config.CONFIG.show_plots` via `set_show_plots(...)`; with `True` saved
figures stay open and the script calls `plt.show()` once at the end (the old
behaviour). Interactive tools (frame viewer, window pickers) are not affected:
they need a window by design.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

SHOW = False
SAVED: list[str] = []


def set_show_plots(flag) -> None:
    """Switch between "save only" (False, default) and "save and keep open for plt.show()" (True)."""
    global SHOW
    SHOW = bool(flag)


def finish_figure(fig, out_path=None, dpi=150, verbose=True):
    """
    Save `fig` to `out_path` (if given), print the absolute path, and close the
    figure unless `SHOW` is on. A figure without `out_path` is left open (it
    was not saved anywhere, closing it would lose it). Returns `out_path` if
    the figure was saved, otherwise the figure itself.
    """
    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if dpi is None:
            fig.savefig(out_path)
        else:
            fig.savefig(out_path, dpi=dpi)
        path = os.path.abspath(out_path)
        SAVED.append(path)
        if verbose:
            print(f"   рисунок: {path}")
        if not SHOW:
            plt.close(fig)
        return out_path
    return fig


def print_saved_summary(title="Рисунки сохранены", extra_files=(), since=0) -> None:
    """Print where the figures saved during this run (from index `since` of `SAVED` on) and
    `extra_files` went, grouped by folder."""
    files = list(dict.fromkeys(SAVED[since:] + [os.path.abspath(p) for p in extra_files if p and os.path.isfile(p)]))
    if not files:
        return
    by_dir = {}
    for p in files:
        by_dir.setdefault(os.path.dirname(p), []).append(os.path.basename(p))
    print("\n" + "=" * 70 + f"\n{title}:")
    for d, names in by_dir.items():
        print(f"  {d}/")
        for n in names:
            print(f"      {n}")
    if not SHOW:
        print("(окна не открывались; чтобы открывать -- show_plots=True в run_config.py)")
