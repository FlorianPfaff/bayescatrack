from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from bayescatrack.experiments.registration_qa_report import (
    RegistrationQAConfig,
    RegistrationQACost,
    run_registration_qa_report,
    summarize_registration_qa_links,
    write_registration_qa_results,
)
from bayescatrack.experiments.track2p_benchmark import ReferenceKind


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run registration QA once and write link and summary artifacts."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--reference-kind",
        required=True,
        choices=("manual-gt", "track2p-output", "aligned-subject-rows"),
    )
    parser.add_argument(
        "--input-format",
        default="suite2p",
        choices=("auto", "suite2p", "npy"),
    )
    parser.add_argument("--cost", required=True, choices=("registered-iou", "roi-aware"))
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--cost-threshold", type=float, default=6.0)
    parser.add_argument("--no-cost-threshold", action="store_true")
    parser.add_argument(
        "--include-behavior",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--include-non-cells", action="store_true")
    parser.add_argument("--weighted-masks", action="store_true")
    parser.add_argument("--weighted-centroids", action="store_true")
    parser.add_argument("--links-output", required=True, type=Path)
    parser.add_argument("--summary-csv-output", required=True, type=Path)
    parser.add_argument("--summary-table-output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = run_registration_qa_report(
        RegistrationQAConfig(
            data=args.data,
            reference=args.reference,
            reference_kind=cast(ReferenceKind, args.reference_kind),
            input_format=args.input_format,
            max_gap=args.max_gap,
            cost=cast(RegistrationQACost, args.cost),
            cost_threshold=None if args.no_cost_threshold else args.cost_threshold,
            include_behavior=args.include_behavior,
            include_non_cells=args.include_non_cells,
            weighted_masks=args.weighted_masks,
            weighted_centroids=args.weighted_centroids,
        )
    )
    summary_rows = summarize_registration_qa_links(rows)
    write_registration_qa_results(rows, args.links_output, "csv")
    write_registration_qa_results(summary_rows, args.summary_csv_output, "csv")
    write_registration_qa_results(summary_rows, args.summary_table_output, "table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
