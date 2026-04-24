import warnings
import weakref
from collections.abc import Callable
from functools import wraps
from typing import Any, Protocol


class MonitoredMethod(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def get_stats(self, obj: Any) -> int: ...


# Global cache for tracking calls per object
_OBJECT_CALL_COUNTS: weakref.WeakKeyDictionary[Any, dict[str, Any]] = (
    weakref.WeakKeyDictionary()
)
_CACHED_RESULTS: weakref.WeakKeyDictionary[Any, dict[Any, Any]] = (
    weakref.WeakKeyDictionary()
)


def performance_monitor(
    warn_threshold: int = 3, enable_warnings: bool = True, monitor: bool = True
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Monitor repeated calls to expensive methods using __id__ for state tracking."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if monitor:
                # Initialize tracking for this object if needed
                if self not in _OBJECT_CALL_COUNTS:
                    _OBJECT_CALL_COUNTS[self] = {"call_count": 0, "last_id": None}
                    _CACHED_RESULTS[self] = {}

                obj_data = _OBJECT_CALL_COUNTS[self]
                cache = _CACHED_RESULTS[self]

                # Get current ID using the __id__ method
                try:
                    current_id = self.__id__()
                except AttributeError:
                    # Fallback if __id__ not implemented
                    warnings.warn(
                        f"{self.__class__.__name__} does not implement __id__(). "
                        "Performance monitoring may be inaccurate.",
                        UserWarning,
                        stacklevel=2,
                    )
                    current_id = (id(self), 0)

                # Check if underlying data has changed
                if obj_data["last_id"] != current_id:
                    # Data has changed, reset counters and cache
                    obj_data["call_count"] = 0
                    obj_data["last_id"] = current_id
                    cache.clear()

                obj_data["call_count"] += 1

                # Warn about repeated calls
                if (
                    enable_warnings
                    and obj_data["call_count"] >= warn_threshold
                    and obj_data["call_count"] == warn_threshold
                ):
                    warnings.warn(
                        f"{func.__name__} called {warn_threshold} times on same "
                        f"{self.__class__.__name__} instance with unchanged data. "
                        f"Consider caching the result.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            return func(self, *args, **kwargs)

        return wrapper

    return decorator
