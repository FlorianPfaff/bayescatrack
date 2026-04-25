"""Track2p benchmark metric facade."""

# pylint: disable=undefined-all-variable

from __future__ import annotations

from . import complete_track_scores as _scores

__all__ = _scores.__all__
globals().update({name: getattr(_scores, name) for name in __all__})
