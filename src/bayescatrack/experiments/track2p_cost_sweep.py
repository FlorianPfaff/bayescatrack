"""Cost-scale and threshold sweeps for Track2p global-assignment benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bayescatrack.association.pyrecest_global_assignment import (
    _load_pyrecest_multisession_solver,
    build_registered_pairwise_costs,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    ProgressReporter,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    _variant_name,
    discover_subject_dirs,
)

# pylint: disable=protected-access,too-many-locals


@dataclass(frozen=True)
class CostSweepConfig:
    """Configuration for a Track2p global-assignment cost sweep."""

    benchmark: Track2pBenchmarkConfig
    cost_scales: tuple[float, ...]
    cost_thresholds: tuple[float | None, ...]


@dataclass(frozen=True)
class CostSweepRun:
    """One subject and one cost-scale/threshold setting."""

    scale: float
    threshold: float | None
    sweep_index: int
    sweep_count: int


# pylint: disable=too-many-branches
def run_track2p_cost_sweep(config: CostSweepConfig) -> list[SubjectBenchmarkResult]:
    """Run global-assignment benchmark rows over cost scales and thresholds."""

    benchmark = config.benchmark
    if benchmark.method != "global-assignment":
        raise ValueError("Track2p cost sweeps require method='global-assignment'")
    if benchmark.split != "subject":
        raise ValueError("Track2p cost sweeps currently support split='subject' only")
    if benchmark.cost == "calibrated":
        raise ValueError("cost='calibrated' requires LOSO training and is not supported by this sweep")

    runs = _sweep_runs(config.cost_scales, config.cost_thresholds)
    subject_dirs = discover_subject_dirs(benchmark.data)
    if not subject_dirs:
        raise ValueError(f"No Track2p-style subject directories found under {benchmark.data}")

    results: list[SubjectBenchmarkResult] = []
    progress = ProgressReporter(
        len(subject_dirs) * len(runs), enabled=benchmark.progress, label="cost-sweep"
    )
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=benchmark.data, config=benchmark
        )
        _validate_reference_for_benchmark(reference, subject_dir=subject_dir, config=benchmark)
        sessions = _load_subject_sessions(subject_dir, benchmark)
        if reference.source == GROUND_TRUTH_REFERENCE_SOURCE:
            _validate_reference_roi_indices(reference, sessions)

        base_costs = build_registered_pairwise_costs(
            sessions,
            max_gap=benchmark.max_gap,
            cost=benchmark.cost,
            transform_type=benchmark.transform_type,
            order=benchmark.order,
            weighted_centroids=benchmark.weighted_centroids,
            velocity_variance=benchmark.velocity_variance,
            regularization=benchmark.regularization,
            pairwise_cost_kwargs=benchmark.pairwise_cost_kwargs,
        )
        session_sizes = tuple(int(session.plane_data.n_rois) for session in sessions)

        for run in runs:
            progress.step(
                f"running {subject_dir.name} scale={run.scale:g} threshold={_threshold_label(run.threshold)}"
            )
            scaled_costs = _scaled_pairwise_costs(base_costs, run.scale)
            solver_result = _load_pyrecest_multisession_solver()(
                scaled_costs,
                session_sizes=session_sizes,
                start_cost=float(benchmark.start_cost),
                end_cost=float(benchmark.end_cost),
                gap_penalty=float(benchmark.gap_penalty),
                cost_threshold=run.threshold,
            )
            predicted = tracks_to_suite2p_index_matrix(solver_result.tracks, sessions)
            scores = _score_prediction_against_reference(predicted, reference, config=benchmark)
            scores = {
                **scores,
                "sweep_index": int(run.sweep_index),
                "sweep_count": int(run.sweep_count),
                "cost_scale": float(run.scale),
                "cost_threshold": _threshold_label(run.threshold),
                **_pairwise_cost_statistics(scaled_costs, run.threshold),
            }
            results.append(
                SubjectBenchmarkResult(
                    subject=subject_dir.name,
                    variant=_variant_name(benchmark.cost),
                    method=benchmark.method,
                    scores=scores,
                    n_sessions=reference.n_sessions,
                    reference_source=reference.source,
                )
            )
    return results


def _sweep_runs(
    cost_scales: Sequence[float], cost_thresholds: Sequence[float | None]
) -> tuple[CostSweepRun, ...]:
    scales = _normalise_cost_scales(cost_scales)
    thresholds = _normalise_cost_thresholds(cost_thresholds)
    sweep_count = len(scales) * len(thresholds)
    runs: list[CostSweepRun] = []
    for scale in scales:
        for threshold in thresholds:
            runs.append(
                CostSweepRun(
                    scale=scale,
                    threshold=threshold,
                    sweep_index=len(runs) + 1,
                    sweep_count=sweep_count,
                )
            )
    return tuple(runs)


def _scaled_pairwise_costs(
    pairwise_costs: Mapping[tuple[int, int], np.ndarray], scale: float
) -> dict[tuple[int, int], np.ndarray]:
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Cost scales must be positive finite numbers")
    return {
        edge: np.asarray(costs, dtype=float) * scale
        for edge, costs in pairwise_costs.items()
    }


def _pairwise_cost_statistics(
    pairwise_costs: Mapping[tuple[int, int], np.ndarray], threshold: float | None
) -> dict[str, float | int | str]:
    total_values = 0
    finite_chunks: list[np.ndarray] = []
    for matrix in pairwise_costs.values():
        values = np.asarray(matrix, dtype=float).reshape(-1)
        total_values += int(values.size)
        finite_chunks.append(values[np.isfinite(values)])

    finite_values = (
        np.concatenate(finite_chunks) if finite_chunks else np.empty((0,), dtype=float)
    )
    finite_count = int(finite_values.size)
    stats: dict[str, float | int | str] = {
        "cost_edges": int(len(pairwise_costs)),
        "cost_values": int(total_values),
        "cost_finite_values": finite_count,
        "cost_nonfinite_values": int(total_values - finite_count),
    }
    if finite_count == 0:
        stats.update(
            {
                "cost_min": "nan",
                "cost_p10": "nan",
                "cost_median": "nan",
                "cost_p90": "nan",
                "cost_max": "nan",
                "cost_threshold_admitted_count": 0,
                "cost_threshold_admitted_fraction": 0.0,
            }
        )
        return stats

    admitted_count = (
        finite_count
        if threshold is None
        else int(np.count_nonzero(finite_values <= float(threshold)))
    )
    stats.update(
        {
            "cost_min": float(np.min(finite_values)),
            "cost_p10": float(np.percentile(finite_values, 10)),
            "cost_median": float(np.median(finite_values)),
            "cost_p90": float(np.percentile(finite_values, 90)),
            "cost_max": float(np.max(finite_values)),
            "cost_threshold_admitted_count": admitted_count,
            "cost_threshold_admitted_fraction": float(admitted_count / finite_count),
        }
    )
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-sweep",
        description="Sweep cost scaling and solver thresholds for Track2p global-assignment benchmarks.",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Track2p dataset root or one subject directory",
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", default="auto", choices=("auto", "suite2p", "npy")
    )
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        default="auto",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
    )
    parser.add_argument("--allow-track2p-as-reference-for-smoke-test", action="store_true")
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cost", default="registered-iou", choices=("registered-iou", "roi-aware")
    )
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument(
        "--transform-type", default="affine", choices=("affine", "rigid", "none")
    )
    parser.add_argument("--start-cost", type=float, default=5.0)
    parser.add_argument("--end-cost", type=float, default=5.0)
    parser.add_argument("--gap-penalty", type=float, default=1.0)
    parser.add_argument(
        "--cost-scales",
        "--cost-scale-sweep",
        dest="cost_scales",
        required=True,
        help="Comma-separated cost scales, e.g. 0.25,0.5,1,2,4",
    )
    parser.add_argument(
        "--cost-thresholds",
        "--cost-threshold-sweep",
        dest="cost_thresholds",
        required=True,
        help="Comma-separated thresholds; use none to disable thresholding",
    )
    parser.add_argument("--include-behavior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-non-cells", action="store_true")
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--weighted-masks", action="store_true")
    parser.add_argument(
        "--exclude-overlapping-pixels", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--order", default="xy", choices=("xy", "yx"))
    parser.add_argument("--weighted-centroids", action="store_true")
    parser.add_argument("--velocity-variance", type=float, default=25.0)
    parser.add_argument("--regularization", type=float, default=1.0e-6)
    parser.add_argument("--pairwise-cost-kwargs-json", default=None)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    rows = [result.to_dict() for result in run_track2p_cost_sweep(config)]
    if args.output is not None:
        write_sweep_results(rows, args.output, args.format)
    else:
        _write_sweep_stdout(rows, args.format)
    return 0


def _config_from_args(args: argparse.Namespace) -> CostSweepConfig:
    pairwise_cost_kwargs = None
    if args.pairwise_cost_kwargs_json is not None:
        parsed = json.loads(args.pairwise_cost_kwargs_json)
        if not isinstance(parsed, dict):
            raise ValueError("--pairwise-cost-kwargs-json must decode to a JSON object")
        pairwise_cost_kwargs = parsed
    benchmark = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        curated_only=args.curated_only,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        cost=args.cost,
        max_gap=args.max_gap,
        transform_type=args.transform_type,
        start_cost=args.start_cost,
        end_cost=args.end_cost,
        gap_penalty=args.gap_penalty,
        include_behavior=args.include_behavior,
        include_non_cells=args.include_non_cells,
        cell_probability_threshold=args.cell_probability_threshold,
        weighted_masks=args.weighted_masks,
        exclude_overlapping_pixels=args.exclude_overlapping_pixels,
        order=args.order,
        weighted_centroids=args.weighted_centroids,
        velocity_variance=args.velocity_variance,
        regularization=args.regularization,
        pairwise_cost_kwargs=pairwise_cost_kwargs,
        progress=args.progress,
    )
    return CostSweepConfig(
        benchmark=benchmark,
        cost_scales=_parse_cost_scales(args.cost_scales),
        cost_thresholds=_parse_thresholds(args.cost_thresholds),
    )


def _parse_cost_scales(raw: str) -> tuple[float, ...]:
    return _normalise_cost_scales(tuple(_parse_float_tokens(raw, name="--cost-scales")))


def _parse_thresholds(raw: str) -> tuple[float | None, ...]:
    values: list[float | None] = []
    for token in _split_tokens(raw, name="--cost-thresholds"):
        if token.casefold() in {"none", "null", "off", "disabled", "unbounded"}:
            values.append(None)
        else:
            values.append(_parse_finite_float(token, name="--cost-thresholds"))
    return _normalise_cost_thresholds(tuple(values))


def _parse_float_tokens(raw: str, *, name: str) -> tuple[float, ...]:
    return tuple(
        _parse_finite_float(token, name=name)
        for token in _split_tokens(raw, name=name)
    )


def _split_tokens(raw: str, *, name: str) -> tuple[str, ...]:
    tokens = tuple(token.strip() for token in raw.split(","))
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{name} must be a comma-separated list with no empty entries")
    return tokens


def _parse_finite_float(token: str, *, name: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise ValueError(f"{name} contains a non-numeric value: {token!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} values must be finite")
    return value


def _normalise_cost_scales(values: Sequence[float]) -> tuple[float, ...]:
    scales = tuple(float(value) for value in values)
    if not scales:
        raise ValueError("At least one cost scale is required")
    if any((not np.isfinite(scale)) or scale <= 0.0 for scale in scales):
        raise ValueError("Cost scales must be positive finite numbers")
    return scales


def _normalise_cost_thresholds(
    values: Sequence[float | None],
) -> tuple[float | None, ...]:
    thresholds = tuple(None if value is None else float(value) for value in values)
    if not thresholds:
        raise ValueError("At least one cost threshold is required")
    if any(value is not None and not np.isfinite(value) for value in thresholds):
        raise ValueError("Cost thresholds must be finite numbers or none")
    return thresholds


def write_sweep_results(
    rows: Sequence[dict[str, float | int | str]], output_path: Path, output_format: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    if output_format == "csv":
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_sweep_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        return
    output_path.write_text(format_sweep_table(rows) + "\n", encoding="utf-8")


def _write_sweep_stdout(
    rows: Sequence[dict[str, float | int | str]], output_format: str
) -> None:
    if output_format == "json":
        print(json.dumps(list(rows), indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_sweep_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        return
    print(format_sweep_table(rows))


def format_sweep_table(rows: Sequence[dict[str, float | int | str]]) -> str:
    columns = [
        "subject",
        "cost_scale",
        "cost_threshold",
        "cost_median",
        "cost_p90",
        "cost_threshold_admitted_fraction",
        "pairwise_f1",
        "complete_track_f1",
        "pairwise_precision",
        "pairwise_recall",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |"
    body = [header, separator]
    for row in rows:
        body.append(
            "| "
            + " | ".join(_format_value(row.get(column, "")) for column in columns)
            + " |"
        )
    return "\n".join(body)


def _sweep_fieldnames(rows: Sequence[dict[str, float | int | str]]) -> list[str]:
    preferred = [
        "subject",
        "variant",
        "method",
        "n_sessions",
        "reference_source",
        "sweep_index",
        "sweep_count",
        "cost_scale",
        "cost_threshold",
        "cost_edges",
        "cost_values",
        "cost_finite_values",
        "cost_nonfinite_values",
        "cost_min",
        "cost_p10",
        "cost_median",
        "cost_p90",
        "cost_max",
        "cost_threshold_admitted_count",
        "cost_threshold_admitted_fraction",
        "pairwise_f1",
        "complete_track_f1",
        "pairwise_precision",
        "pairwise_recall",
        "complete_tracks",
        "mean_track_length",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining


def _format_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def _threshold_label(threshold: float | None) -> float | str:
    return "none" if threshold is None else float(threshold)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
