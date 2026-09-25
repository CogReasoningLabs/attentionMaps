"""Terminal progress bars for standalone dataset inspection."""

from contextlib import nullcontext
import sys


class _NoProgress:
    disable = True

    def update(self, _count):
        pass


_NO_PROGRESS = _NoProgress()


def inspection_progress(label: str, *, total: int | None = None, unit: str = "rows", enabled: bool = True):
    if not enabled or not sys.stderr.isatty():
        return nullcontext(_NO_PROGRESS)
    from tqdm import tqdm

    return tqdm(
        total=total, desc=label, unit=unit, unit_scale=True,
        dynamic_ncols=True, mininterval=0.2, file=sys.stderr,
    )
