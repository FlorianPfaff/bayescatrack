"""Global multi-session tracking helpers for BayesCaTrack."""

from ._legacy_import import bridge_alias

with bridge_alias():
    from ._multisession_tracking_impl import (
        LongitudinalTrackingResult,
        MultisessionTrackingConfig,
        PairwiseTrackingBundle,
        build_multisession_pairwise_costs,
        main,
        save_tracking_result_npz,
        track_sessions_multisession,
        track_subject_multisession,
    )

__all__ = (
    "LongitudinalTrackingResult",
    "MultisessionTrackingConfig",
    "PairwiseTrackingBundle",
    "build_multisession_pairwise_costs",
    "main",
    "save_tracking_result_npz",
    "track_sessions_multisession",
    "track_subject_multisession",
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
