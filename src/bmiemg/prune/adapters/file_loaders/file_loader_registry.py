# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import TYPE_CHECKING, ClassVar
from dataclasses import dataclass
from collections.abc import Callable

if TYPE_CHECKING:
    from .file_loader import FileLoader


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass
class FileLoaderRegistry:
    _loaders: ClassVar[dict[str, type["FileLoader"]]] = {}

    @classmethod
    def register(
        cls, suffix: str
    ) -> Callable[[type["FileLoader"]], type["FileLoader"]]:
        def decorator(loader_cls: type["FileLoader"]) -> type["FileLoader"]:
            if suffix in cls._loaders:
                raise ValueError(f"Suffix {suffix!r} already registered")
            cls._loaders[suffix] = loader_cls
            return loader_cls

        return decorator

    @classmethod
    def get(cls, suffix: str) -> "FileLoader":
        try:
            loader_cls = cls._loaders[suffix]
        except KeyError:
            raise ValueError(
                f"Unknown suffix {suffix!r}. Registered: {sorted(cls._loaders)}"
            ) from None
        return loader_cls()

    @classmethod
    def known_suffixes(cls) -> set[str]:
        return set(cls._loaders)
