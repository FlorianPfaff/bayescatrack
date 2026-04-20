"""CLI entry point for ``python -m track2p_pyrecest_bridge``."""

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from track2p_pyrecest_bridge import main
else:
    from . import main

raise SystemExit(main())
