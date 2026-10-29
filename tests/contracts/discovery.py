# ================================================================
# 0. Section: IMPORTS
# ================================================================
import inspect

from typing import Any, get_args, get_origin
from contextlib import contextmanager
from collections.abc import Iterator


# ================================================================
# 1. Section: Registry access
# ================================================================
def registry_entries(registry: type) -> dict[str, type]:
    """Return the live key -> class mapping of any registry family.

    Supports the shared `bmiemg.utils.Registry` (`_entries`) and registries that keep their own dict (download's `_strategies`), so contracts survive a registry swap.
    """
    for attr in ("_entries", "_strategies"):
        entries = getattr(registry, attr, None)
        if isinstance(entries, dict):
            return entries
    raise AssertionError(
        f"{registry.__name__} exposes no entries dict (_entries or _strategies)"
    )


def registry_keys(registry: type) -> list[str]:
    known_keys = getattr(registry, "known_keys", None)
    keys = known_keys() if callable(known_keys) else registry_entries(registry)
    return sorted(keys)


def registered_type(registry: type, key: str) -> type:
    get_type = getattr(registry, "get_type", None)
    if callable(get_type):
        return get_type(key)
    return registry_entries(registry)[key]


@contextmanager
def temporary_entry(registry: type, key: str, target: type) -> Iterator[type]:
    """Register `target` under `key` through the public API, then remove it."""
    registry.register(key)(target)
    try:
        yield target
    finally:
        registry_entries(registry).pop(key, None)


# ================================================================
# 2. Section: Class discovery
# ================================================================
def all_subclasses(base: type) -> list[type]:
    found: list[type] = []
    stack = list(base.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in found:
            found.append(cls)
            stack.extend(cls.__subclasses__())
    return sorted(found, key=lambda cls: cls.__qualname__)


def leaf_subclasses(base: type) -> list[type]:
    """Concrete implementations: subclasses nobody else extends (intermediate bases are skipped)."""
    return [cls for cls in all_subclasses(base) if not cls.__subclasses__()]


def generic_arg(cls: type, base: type, index: int = 0) -> Any:
    """Return the `index`-th type argument `cls` passed to the generic `base`, or None."""
    for klass in inspect.getmro(cls):
        for orig in getattr(klass, "__orig_bases__", ()):
            if get_origin(orig) is base:
                args = get_args(orig)
                return args[index] if len(args) > index else None
    return None
