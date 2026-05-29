"""Object-oriented profiling utility with composable, rich-formatted reports.

Provides a single :class:`Profiler` that supports both a context-manager
API for ad-hoc blocks and a :func:`timeit.Timer`-style entry point for
micro-benchmarking. Formatting is decoupled from measurement through a
:class:`ProfileReportBase` strategy; the default :class:`ProfileReport`
renders a coloured table via :mod:`rich`.

Example
-------
>>> import time, numpy as np
>>> from profiler import Profiler, ProfileReport
>>> rep = ProfileReport(time_unit="us", precision=2)
>>> p = Profiler("rng", report=rep).timeit(
...     lambda: np.random.default_rng().normal(size=1000),
...     number=200, repeat=7,
... )
>>> print(p)
"""

from __future__ import annotations

import copy
import gc
import io
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import (
    Any,
    Final,
    Literal,
    Self,
)

import numpy as np
from numpy.typing import NDArray
from rich.box import SIMPLE_HEAD
from rich.console import Console, RenderableType
from rich.table import Table
from rich.text import Text

__all__ = [
    "LineProfiler",
    "ProfileReport",
    "ProfileReportBase",
    "TimingResult",
    "TimeUnit",
]


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Nullary monotonic timer returning seconds.
TimerFn = Callable[[], float]

#: Nullary callable to be timed.
TimedFn = Callable[[], Any]

#: Optional pre-measurement hook.
SetupFn = Callable[[], None]

TimeUnit = Literal["s", "ms", "us", "ns"]

_UNIT_SCALE: Final[dict[TimeUnit, float]] = {
    "s": 1.0,
    "ms": 1e3,
    "us": 1e6,
    "ns": 1e9,
}


def _auto_unit(seconds: float) -> TimeUnit:
    if not np.isfinite(seconds) or seconds >= 1.0:
        return "s"
    if seconds >= 1e-3:
        return "ms"
    if seconds >= 1e-6:
        return "us"
    return "ns"


def _fmt(seconds: float, unit: TimeUnit, precision: int) -> str:
    if not np.isfinite(seconds):
        return "n/a"
    return f"{seconds * _UNIT_SCALE[unit]:.{precision}f} {unit}"


@contextmanager
def _gc_disabled() -> Iterator[None]:
    """Disable cyclic GC for the duration of the block.

    Matches :mod:`timeit` semantics: GC is paused so independent
    measurements are comparable. Original state is restored on exit
    even if the body raises.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


@dataclass(frozen=True, slots=True)
class TimingResult:
    """Immutable snapshot of a profiling session.

    Each entry of :attr:`times` is the total wall time for ``number``
    consecutive calls of the timed function (matching :mod:`timeit`
    ``Timer.repeat`` semantics). Per-call statistics are exposed as
    derived properties so the raw data and its summaries cannot drift
    out of sync.
    """

    name: str
    times: NDArray[np.float64]
    number: int = 1

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError(f"number must be >= 1, got {self.number}")
        if self.times.ndim != 1:
            raise ValueError(f"times must be 1-D; got shape {self.times.shape}")

    @property
    def n_repeats(self) -> int:
        return int(self.times.size)

    def __len__(self) -> int:
        return self.n_repeats

    @property
    def is_empty(self) -> bool:
        return self.n_repeats == 0

    @property
    def per_loop(self) -> NDArray[np.float64]:
        """Per-call elapsed time (seconds), i.e. ``times / number``."""
        return self.times / self.number

    @property
    def best(self) -> float:
        """Min per-call time -- the recommended benchmark summary."""
        return float(self.per_loop.min()) if not self.is_empty else float("nan")

    @property
    def worst(self) -> float:
        return float(self.per_loop.max()) if not self.is_empty else float("nan")

    @property
    def mean(self) -> float:
        return float(self.per_loop.mean()) if not self.is_empty else float("nan")

    @property
    def median(self) -> float:
        return float(np.median(self.per_loop)) if not self.is_empty else float("nan")

    @property
    def std(self) -> float:
        """Sample std (``ddof=1``); ``0.0`` when fewer than 2 repeats."""
        if self.n_repeats < 2:
            return 0.0
        return float(self.per_loop.std(ddof=1))

    def percentile(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-call percentiles. Returns NaN array when result is empty."""
        if self.is_empty:
            return np.full(q.shape, np.nan, dtype=np.float64)
        return np.asarray(np.percentile(self.per_loop, q), dtype=np.float64)

    @property
    def total(self) -> float:
        """Total wall time across all measurements (seconds)."""
        return float(self.times.sum()) if not self.is_empty else 0.0


class ProfileReportBase(ABC):
    """Abstract strategy for formatting a :class:`TimingResult`.

    Subclasses implement :meth:`render` (plain/ANSI string) and may
    override :meth:`as_rich` to expose a structured rich renderable.
    Instances are reusable across results; use :meth:`for_result` to
    obtain a copy bound to a specific result whose ``str(...)`` and
    ``rich.print(...)`` invoke the rendering automatically.
    """

    _bound: TimingResult | None = None

    @abstractmethod
    def render(self, result: TimingResult) -> str:
        """Format ``result`` as text (may include ANSI escape codes)."""

    def as_rich(self, result: TimingResult) -> RenderableType:
        """Return a rich-renderable for ``result``.

        Default falls back to the rendered string. Subclasses that
        produce structured content (tables, panels, ...) should
        override this for fidelity when printed via
        :class:`rich.console.Console`.
        """
        return Text.from_ansi(self.render(result))

    def for_result(self, result: TimingResult) -> Self:
        """Return a copy of this report bound to ``result``.

        The returned instance is a thin view: ``str(...)``, ``repr``,
        and ``rich.print(...)`` all dispatch to the bound result. The
        original is unmodified.
        """
        bound = copy.copy(self)
        bound._bound = result
        return bound

    def __call__(self, result: TimingResult) -> str:
        return self.render(result)

    def __str__(self) -> str:
        if self._bound is None:
            return repr(self)
        return self.render(self._bound)

    def __rich__(self) -> RenderableType:
        if self._bound is None:
            return Text(repr(self))
        return self.as_rich(self._bound)


class ProfileReport(ProfileReportBase):
    """Default profile report rendered as a :mod:`rich` table.

    Configure once at construction and pass to :class:`Profiler` to
    control the format of all of its results.

    Args:
        time_unit: Display unit. ``"auto"`` (default) picks based on the best
                per-call time so the values stay legible.
        precision: Decimal places for time formatting.
        show_percentiles: Include p25/p75/p95 rows when at least four repeats are
                        available.
        use_color: Emit ANSI colour codes in :meth:`render` (default ``True``).
                    Disable for plain-text logs / files.
        width: Optional fixed console width; ``None`` lets rich auto-size.
    """

    def __init__(
        self,
        *,
        time_unit: TimeUnit | Literal["auto"] = "auto",
        precision: int = 3,
        show_percentiles: bool = True,
        use_color: bool = True,
        width: int | None = None,
    ) -> None:
        if precision < 0:
            raise ValueError(f"precision must be >= 0, got {precision}")
        self.time_unit: TimeUnit | Literal["auto"] = time_unit
        self.precision = precision
        self.show_percentiles = show_percentiles
        self.use_color = use_color
        self.width = width

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(time_unit={self.time_unit!r}, "
            f"precision={self.precision}, "
            f"show_percentiles={self.show_percentiles}, "
            f"use_color={self.use_color}, width={self.width!r})"
        )

    def _resolve_unit(self, result: TimingResult) -> TimeUnit:
        if self.time_unit == "auto":
            return _auto_unit(result.best if not result.is_empty else 1.0)
        return self.time_unit

    def _build_table(self, result: TimingResult) -> Table:
        unit = self._resolve_unit(result)
        table = Table(
            title=Text(f"Profile: {result.name}", style="bold cyan"),
            box=SIMPLE_HEAD,
            show_header=True,
            header_style="bold magenta",
            title_justify="left",
            expand=False,
            pad_edge=False,
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        if result.is_empty:
            table.add_row("status", Text("no trials recorded", style="yellow"))
            return table

        def f(t: float) -> str:
            return _fmt(t, unit, self.precision)

        table.add_row("repeats", str(result.n_repeats))
        if result.number != 1:
            table.add_row("loops / repeat", str(result.number))
        table.add_row("best", Text(f(result.best), style="bold green"))
        table.add_row("mean", f(result.mean))
        table.add_row("median", f(result.median))
        table.add_row("std (ddof=1)", f(result.std))
        table.add_row("worst", Text(f(result.worst), style="red"))

        if self.show_percentiles and result.n_repeats >= 4:
            qs = result.percentile(np.array([25.0, 75.0, 95.0]))
            table.add_row("p25 / p75", f"{f(float(qs[0]))} / {f(float(qs[1]))}")
            table.add_row("p95", f(float(qs[2])))

        table.add_row("total wall", f(result.total))
        return table

    def as_rich(self, result: TimingResult) -> RenderableType:
        return self._build_table(result)

    def render(self, result: TimingResult) -> str:
        table = self._build_table(result)
        console = Console(
            file=io.StringIO(),
            width=self.width,
            force_terminal=self.use_color,
            no_color=not self.use_color,
            color_system="truecolor" if self.use_color else None,
            legacy_windows=False,
        )
        with console.capture() as capture:
            console.print(table)
        return capture.get().rstrip()


class LineProfiler:
    """Object-oriented profiler with context-manager and ``timeit`` APIs.

    Wall time is measured with a user-supplied monotonic clock
    (default: :func:`time.perf_counter`). Formatting is delegated to a
    :class:`ProfileReportBase` instance composed in at construction.

    Two complementary usage modes:

    Context manager
        For ad-hoc, single-shot measurement of arbitrary blocks. Each
        ``__enter__``/``__exit__`` cycle appends one sample::

            with Profiler("matmul") as p:
                C = A @ B
            print(p)

    timeit-style
        For micro-benchmarks where overhead matters. Uses ``number``
        loops per measurement and ``repeat`` independent measurements;
        cyclic GC is paused during each measurement following
        :mod:`timeit` convention::

            p = Profiler("fft").timeit(
                lambda: np.fft.fft(x), number=1000, repeat=7,
            )
            print(p)
    """

    def __init__(
        self,
        name: str,
        *,
        report: ProfileReportBase | None = None,
        timer: TimerFn = time.perf_counter,
    ) -> None:
        self._name = name
        self._report: ProfileReportBase = report if report is not None else ProfileReport()
        self._timer = timer
        self._times: list[float] = []
        self._number: int = 1
        self._t0: float | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def report(self) -> ProfileReportBase:
        return self._report

    @report.setter
    def report(self, value: ProfileReportBase) -> None:
        self._report = value

    @property
    def result(self) -> TimingResult:
        """Immutable snapshot of currently-recorded measurements."""
        return TimingResult(
            name=self._name,
            times=np.asarray(self._times),
            number=self._number,
        )

    def __len__(self) -> int:
        return len(self._times)

    def start(self) -> None:
        if self._t0 is not None:
            raise RuntimeError("LineProfiler.start(): a measurement is already in progress")
        self._number = 1
        self._t0 = self._timer()

    def stop(self) -> None:
        t1 = self._timer()
        if self._t0 is None:
            raise RuntimeError("Profiler.stop() called before start()")
        self._times.append(t1 - self._t0)
        self._t0 = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def timeit(
        self,
        fn: TimedFn,
        *,
        number: int | None = None,
        repeat: int = 5,
        setup: SetupFn | None = None,
        disable_gc: bool = True,
        clear: bool = True,
    ) -> Self:
        """Time ``fn`` with :mod:`timeit` semantics.

        Args:
            fn: Nullary callable to time.
            number: Loops per measurement. When ``None``, :meth:`autorange`
                    picks a value that yields at least 0.2 s per measurement,
                    matching :meth:`timeit.Timer.autorange`.
            repeat: Number of independent measurements.
            setup: Optional callable invoked once before each repeat.
            disable_gc: Pause cyclic GC during each measurement (default ``True``,
                matching :mod:`timeit`).
            clear: Discard any previously-recorded samples before timing.

        Returns:
            Profiler

        """
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")
        if number is not None and number < 1:
            raise ValueError(f"number must be >= 1, got {number}")

        if clear:
            self.reset()

        if number is None:
            number, _ = self.autorange(fn)

        timer = self._timer  # local binding -> fewer attribute lookups

        def _gc_ctx() -> AbstractContextManager[None]:
            # @contextmanager generators are single-use, so we build a
            # fresh one per measurement.
            return _gc_disabled() if disable_gc else nullcontext()

        for _ in range(repeat):
            if setup is not None:
                setup()
            with _gc_ctx():
                t0 = timer()
                for _ in range(number):
                    fn()
                self._times.append(timer() - t0)

        self._number = number
        return self

    def autorange(
        self,
        fn: TimedFn,
        *,
        min_time: float = 0.2,
    ) -> tuple[int, float]:
        """Pick ``number`` such that timing ``fn`` takes >= ``min_time`` s.

        Returns:
            The chosen ``number`` and the elapsed time of the run that
            crossed the threshold.
        """
        if min_time <= 0:
            raise ValueError(f"min_time must be > 0, got {min_time}")

        i = 1
        timer = self._timer
        while True:
            for j in (1, 2, 5):
                number = i * j
                t0 = timer()
                for _ in range(number):
                    fn()
                dt = timer() - t0
                if dt >= min_time:
                    return number, dt
            i *= 10

    def reset(self) -> Self:
        """Discard all measurements and any in-flight start time."""
        self._times.clear()
        self._number = 1
        self._t0 = None
        return self

    def __str__(self) -> str:
        return self._report.render(self.result)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self._name!r}, "
            f"n_repeats={len(self._times)}, number={self._number})"
        )

    def __rich__(self) -> RenderableType:
        return self._report.as_rich(self.result)
