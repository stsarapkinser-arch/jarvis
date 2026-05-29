from __future__ import annotations

from typing import Any


class Singleton(type):
    _instances: dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    @classmethod
    def reset(mcs, target: type | None = None) -> None:
        if target is None:
            mcs._instances.clear()
        else:
            mcs._instances.pop(target, None)
