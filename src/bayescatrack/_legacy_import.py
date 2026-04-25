"""Temporary aliases used while loading moved implementation modules."""

from __future__ import annotations

from contextlib import contextmanager
import sys
from types import ModuleType
from typing import Iterator, Mapping

import bayescatrack as _package
from bayescatrack.core import _bridge_impl

_MISSING = object()


@contextmanager
def bridge_alias(extra_modules: Mapping[str, ModuleType] | None = None) -> Iterator[None]:
    """Expose the old module name only while importing moved implementation code."""

    bridge_module = ModuleType("track2p_pyrecest_bridge")
    for name in _package.__all__:
        setattr(bridge_module, name, getattr(_package, name))
    setattr(
        bridge_module,
        "_suite2p_kwargs_from_args",
        getattr(_bridge_impl, "_suite2p_kwargs_from_args"),
    )
    setattr(bridge_module, "__path__", [])

    previous_bridge = sys.modules.get("track2p_pyrecest_bridge", _MISSING)
    previous_submodules: dict[str, object] = {}
    sys.modules["track2p_pyrecest_bridge"] = bridge_module

    for module_name, module in (extra_modules or {}).items():
        full_name = f"track2p_pyrecest_bridge.{module_name}"
        previous_submodules[full_name] = sys.modules.get(full_name, _MISSING)
        setattr(bridge_module, module_name, module)
        sys.modules[full_name] = module

    try:
        yield
    finally:
        for full_name, previous in previous_submodules.items():
            if previous is _MISSING:
                sys.modules.pop(full_name, None)
            else:
                sys.modules[full_name] = previous
        if previous_bridge is _MISSING:
            sys.modules.pop("track2p_pyrecest_bridge", None)
        else:
            sys.modules["track2p_pyrecest_bridge"] = previous_bridge
