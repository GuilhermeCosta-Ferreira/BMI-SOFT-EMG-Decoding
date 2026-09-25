# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass
from collections.abc import Callable

from .protocol import Protocol



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ProtocolRegistry:
    _strategies: ClassVar[dict[str, type[Protocol]]] = {}

    @classmethod
    def register(cls, source: str) -> Callable[[type[Protocol]], type[Protocol]]:
        def decorator(strategy_cls: type[Protocol]) -> type[Protocol]:
            if source in cls._strategies:
                raise ValueError(f"Source {source!r} already registered")
            cls._strategies[source] = strategy_cls
            return strategy_cls
        return decorator

    @classmethod
    def get(cls, source: str) -> Protocol:
        try:
            strategy_cls = cls._strategies[source]
        except KeyError:
            raise ValueError(
                f"Unknown name {source!r}. Registered: {sorted(cls._strategies)}"
            ) from None
        return strategy_cls()
