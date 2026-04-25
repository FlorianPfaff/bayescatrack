"""Ground-truth evaluation helpers for Track2p-style track tables."""

from . import matching as _matching
from ._legacy_import import bridge_alias

with bridge_alias({"matching": _matching}):
    from ._ground_truth_eval_impl import (
        TrackEvaluation,
        TrackTable,
        complete_tracks_score,
        evaluate_track_table_prediction,
        load_track2p_ground_truth_csv,
        load_track_table_csv,
        main,
        proportion_correct_by_horizon,
        tracks_from_consecutive_matches,
    )

__all__ = (
    "TrackEvaluation",
    "TrackTable",
    "complete_tracks_score",
    "evaluate_track_table_prediction",
    "load_track2p_ground_truth_csv",
    "load_track_table_csv",
    "main",
    "proportion_correct_by_horizon",
    "tracks_from_consecutive_matches",
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
