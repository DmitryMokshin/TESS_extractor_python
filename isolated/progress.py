"""
A tiny helper for showing *some* sign of life during blocking network calls
that don't expose incremental progress (a single synchronous TAP query, or
astroquery's `Tesscut.get_cutouts`). Rather than a byte/item progress bar
(not available for these calls), this shows an elapsed-time "heartbeat" via
tqdm so a long-running call doesn't look like the script has hung.

For calls that *do* expose real progress (chunked HTTP downloads), prefer a
normal tqdm progress bar over this.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

from tqdm.auto import tqdm


@contextmanager
def heartbeat(desc: str, interval: float = 0.5):
    """
    Usage:
        with heartbeat("Querying Gaia TAP"):
            result = some_blocking_call()

    Shows a tqdm bar with unit "s" that just counts up elapsed seconds while
    the `with` block runs, so the person watching knows it's still working.
    """
    bar = tqdm(desc=desc, unit="s", bar_format="{desc}: {n:.0f}s elapsed")
    stop_event = threading.Event()

    def _tick():
        start = time.monotonic()
        while not stop_event.wait(interval):
            bar.n = time.monotonic() - start
            bar.refresh()

    thread = threading.Thread(target=_tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()
        bar.close()
