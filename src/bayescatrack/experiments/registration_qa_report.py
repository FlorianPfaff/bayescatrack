"""Registration quality report for Track2p-style benchmark subjects."""

# pylint: disable=protected-access,too-many-locals,too-many-arguments,too-many-positional-arguments
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from bayescatrack.association.pyrecest_global_assignment import (
    registered_iou_cost_kwargs,
    roi_aware_cost_kwargs,
    session_edge_pairs,
)
from bayescatrack.association.registered_masks import replace_empty_registered_masks
from bayescatrack.core.bridge import (
    Track2pSession,
    build_session_pair_association_bundle,
)
from bayescatrack.experiments.track2p_benchmark import (
    ReferenceKind,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
)
from bayescatrack.track2p_registration import register_plane_pair

RegistrationQACost = Literal["registered-iou", "roi-aware"]
RegistrationQALevel = Literal["summary", "links"]
OutputFormat = Literal["table", "json", "csv"]


@dataclass(frozen=True)
class RegistrationQAConfig:
    """Configuration for a registration QA report."""

    data: Path
    reference: Path | None = None
    reference_kind: ReferenceKind = "auto"
    allow_track2p_as_reference_for_smoke_test: bool = False
    curated_only: bool = False
    plane_name: str = "plane0"
    input_format: str = "auto"
    max_gap: int = 2
    transform_type: str = "affine"
    cost: RegistrationQACost = "registered-iou"
    cost_threshold: float | None = 6.0
    include_behavior: bool = True
    include_non_cells: bool = False
    cell_probability_threshold: float = 0.5
    weighted_masks: bool = False
    exclude_overlapping_pixels: bool = True
    order: str = "xy"
    weighted_centroids: bool = False
    velocity_variance: float = 25.0
    regularization: float = 1.0e-6
    pairwise_cost_kwargs: dict[str, Any] | None = None
    progress: bool = False


def run_registration_qa_report(config: RegistrationQAConfig) -> list[dict[str, Any]]:
    """Return one diagnostics row for each manual-GT link and audited edge."""

    subject_dirs = discover_subject_dirs(config.data)
    if not subject_dirs:
        raise ValueError(f"No Track2p-style subject directories found under {config.data}")

    benchmark_config = _benchmark_config(config)
    rows: list[dict[str, Any]] = []
    for subject_dir in subject_dirs:
        if config.progress:
            print(f"registration-qa: {subject_dir.name}", file=sys.stderr, flush=True)
        reference = _load_reference_for_subject(
            subject_dir,
            data_root=config.data,
            config=benchmark_config,
        )
        _validate_reference_for_benchmark(
            reference,
            subject_dir=subject_dir,
            config=benchmark_config,
        )
        sessions = _load_subject_sessions(subject_dir, benchmark_config)
        _validate_reference_roi_indices(reference, sessions)
        reference_matrix = _reference_matrix(
            reference,
            curated_only=config.curated_only,
        )
        rows.extend(
            _audit_subject(
                subject_dir.name,
                sessions,
                reference_matrix,
                config,
            )
        )
    return rows


def summarize_registration_qa_links(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate link-level QA rows by subject and session edge."""

    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["subject"]),
            str(row["source_session_name"]),
            str(row["target_session_name"]),
            int(row["session_gap"]),
        )
        grouped[key].append(row)

    summary: list[dict[str, Any]] = []
    for (subject, source_name, target_name, session_gap), group in sorted(grouped.items()):
        summary.append(
            {
                "subject": subject,
                "source_session_name": source_name,
                "target_session_name": target_name,
                "session_gap": session_gap,
                "n_gt_links": len(group),
                "registration_backend": _mode(group, "registration_backend"),
                "transform_type": _mode(group, "transform_type"),
                "median_registered_iou": _stat(group, "registered_iou"),
                "p10_registered_iou": _stat(group, "registered_iou", 10),
                "p90_registered_iou": _stat(group, "registered_iou", 90),
                "median_registered_centroid_distance": _stat(
                    group,
                    "registered_centroid_distance",
                ),
                "p90_registered_centroid_distance": _stat(
                    group,
                    "registered_centroid_distance",
                    90,
                ),
                "empty_registered_rois": int(
                    max(int(row["empty_registered_rois"]) for row in group)
                ),
                "empty_registered_fraction": float(
                    max(float(row["empty_registered_fraction"]) for row in group)
                ),
                "gt_top1_rate": _mean_bool(group, "gt_is_top1"),
                "gt_admissible_rate": _mean_bool(group, "gt_candidate_admissible"),
                "empty_gt_mask_rate": _mean_bool(group, "target_empty_registered_mask"),
                "gated_gt_rate": _mean_bool(group, "target_gated"),
                "median_gt_rank": _stat(group, "gt_rank"),
                "p90_gt_rank": _stat(group, "gt_rank", 90),
                "median_cost_margin": _stat(group, "cost_margin"),
            }
        )
    return summary


def format_registration_qa_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Format summary rows as a compact Markdown table."""

    columns = [
        "subject",
        "source_session_name",
        "target_session_name",
        "n_gt_links",
        "registration_backend",
        "median_registered_iou",
        "median_registered_centroid_distance",
        "gt_top1_rate",
        "gt_admissible_rate",
        "empty_registered_rois",
        "median_gt_rank",
        "median_cost_margin",
    ]
    body = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        body.append("| " + " | ".join(_format_value(row.get(col, "")) for col in columns) + " |")
    return "\n".join(body)


def write_registration_qa_results(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    output_format: OutputFormat,
) -> None:
    """Write registration QA rows as JSON, CSV, or Markdown."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(json.dumps(list(rows), indent=2) + "\n", encoding="utf-8")
        return
    if output_format == "csv":
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        return
    output_path.write_text(format_registration_qa_table(rows) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark registration-qa",
        description="Report registration quality on manual-GT Track2p links.",
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        default="auto",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
    )
    parser.add_argument("--allow-track2p-as-reference-for-smoke-test", action="store_true")
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument("--input-format", default="auto", choices=("auto", "suite2p", "npy"))
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--transform-type", default="affine", choices=("affine", "rigid", "none"))
    parser.add_argument("--cost", default="registered-iou", choices=("registered-iou", "roi-aware"))
    parser.add_argument("--cost-threshold", type=float, default=6.0)
    parser.add_argument("--no-cost-threshold", action="store_true")
    parser.add_argument("--include-behavior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-non-cells", action="store_true")
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--weighted-masks", action="store_true")
    parser.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--order", default="xy", choices=("xy", "yx"))
    parser.add_argument("--weighted-centroids", action="store_true")
    parser.add_argument("--velocity-variance", type=float, default=25.0)
    parser.add_argument("--regularization", type=float, default=1.0e-6)
    parser.add_argument("--pairwise-cost-kwargs-json", default=None)
    parser.add_argument("--level", default="summary", choices=("summary", "links"))
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", default="table", choices=("table", "json", "csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows: Sequence[Mapping[str, Any]] = run_registration_qa_report(_config_from_args(args))
    if args.level == "summary":
        rows = summarize_registration_qa_links(rows)

    if args.output is not None:
        write_registration_qa_results(rows, args.output, args.format)
    elif args.format == "json":
        print(json.dumps(list(rows), indent=2))
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    else:
        print(format_registration_qa_table(rows))
    return 0


def _audit_subject(
    subject: str,
    sessions: Sequence[Track2pSession],
    reference_matrix: np.ndarray,
    config: RegistrationQAConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index, target_index in session_edge_pairs(len(sessions), max_gap=config.max_gap):
        reference_session = sessions[source_index]
        target_session = sessions[target_index]
        registered_plane = register_plane_pair(
            reference_session.plane_data,
            target_session.plane_data,
            transform_type=config.transform_type,
        )
        registered_plane, empty_registered_rois = replace_empty_registered_masks(
            registered_plane
        )
        raw_bundle = _association_bundle(
            reference_session,
            target_session,
            target_session.plane_data,
            config,
        )
        registered_bundle = _association_bundle(
            reference_session,
            target_session,
            registered_plane,
            config,
        )
        rows.extend(
            _audit_reference_links(
                subject,
                source_index,
                target_index,
                reference_session,
                target_session,
                reference_matrix,
                raw_bundle.pairwise_components,
                registered_bundle.pairwise_components,
                np.asarray(registered_bundle.pairwise_cost_matrix, dtype=float),
                empty_registered_rois,
                _registration_backend(config.transform_type, registered_plane.source),
                config,
            )
        )
    return rows


def _audit_reference_links(
    subject: str,
    source_index: int,
    target_index: int,
    source_session: Track2pSession,
    target_session: Track2pSession,
    reference_matrix: np.ndarray,
    raw_components: Mapping[str, np.ndarray],
    registered_components: Mapping[str, np.ndarray],
    cost_matrix: np.ndarray,
    empty_registered_rois: np.ndarray,
    registration_backend: str,
    config: RegistrationQAConfig,
) -> list[dict[str, Any]]:
    source_lookup = _roi_lookup(source_session)
    target_lookup = _roi_lookup(target_session)
    target_roi_indices = _roi_indices(target_session)
    rows: list[dict[str, Any]] = []
    for track_index, track in enumerate(reference_matrix):
        source_roi = track[source_index]
        target_roi = track[target_index]
        if source_roi is None or target_roi is None:
            continue
        source_local = source_lookup[int(source_roi)]
        target_local = target_lookup[int(target_roi)]
        cost_row = cost_matrix[source_local]
        gt_cost = float(cost_row[target_local])
        gt_rank = int(1 + np.count_nonzero(cost_row < gt_cost))
        best_target_local = int(np.nanargmin(cost_row))
        false_costs = np.delete(cost_row, target_local)
        finite_false_costs = false_costs[np.isfinite(false_costs)]
        best_false_cost = (
            float(np.min(finite_false_costs)) if finite_false_costs.size else np.nan
        )
        target_empty = bool(empty_registered_rois[target_local])
        target_gated = bool(
            _component_value(registered_components, "gated", source_local, target_local, False)
        )
        below_threshold = (
            True
            if config.cost_threshold is None
            else bool(gt_cost <= float(config.cost_threshold))
        )
        rows.append(
            {
                "subject": subject,
                "source_session_index": source_index,
                "target_session_index": target_index,
                "source_session_name": source_session.session_name,
                "target_session_name": target_session.session_name,
                "session_gap": target_index - source_index,
                "track_index": track_index,
                "registration_backend": registration_backend,
                "transform_type": config.transform_type,
                "source_roi": int(source_roi),
                "target_roi": int(target_roi),
                "raw_iou": _component_value(raw_components, "iou", source_local, target_local),
                "registered_iou": _component_value(
                    registered_components,
                    "iou",
                    source_local,
                    target_local,
                ),
                "raw_centroid_distance": _component_value(
                    raw_components,
                    "centroid_distance",
                    source_local,
                    target_local,
                ),
                "registered_centroid_distance": _component_value(
                    registered_components,
                    "centroid_distance",
                    source_local,
                    target_local,
                ),
                "gt_cost": gt_cost,
                "gt_rank": gt_rank,
                "gt_is_top1": gt_rank == 1,
                "best_target_roi": int(target_roi_indices[best_target_local]),
                "best_cost": float(cost_row[best_target_local]),
                "best_false_cost": best_false_cost,
                "cost_margin": (
                    best_false_cost - gt_cost if np.isfinite(best_false_cost) else np.nan
                ),
                "target_empty_registered_mask": target_empty,
                "target_gated": target_gated,
                "target_below_cost_threshold": below_threshold,
                "gt_candidate_admissible": (
                    (not target_empty) and (not target_gated) and below_threshold
                ),
                "empty_registered_rois": int(np.count_nonzero(empty_registered_rois)),
                "empty_registered_fraction": (
                    float(np.mean(empty_registered_rois))
                    if empty_registered_rois.size
                    else 0.0
                ),
            }
        )
    return rows


def _association_bundle(
    source_session: Track2pSession,
    target_session: Track2pSession,
    target_plane: Any,
    config: RegistrationQAConfig,
) -> Any:
    return build_session_pair_association_bundle(
        source_session,
        target_session,
        measurement_plane_in_reference_frame=target_plane,
        order=config.order,
        weighted_centroids=config.weighted_centroids,
        velocity_variance=config.velocity_variance,
        regularization=config.regularization,
        pairwise_cost_kwargs=_cost_kwargs(config),
        return_pairwise_components=True,
    )


def _cost_kwargs(config: RegistrationQAConfig) -> dict[str, Any]:
    kwargs = (
        registered_iou_cost_kwargs()
        if config.cost == "registered-iou"
        else roi_aware_cost_kwargs()
    )
    kwargs.update(config.pairwise_cost_kwargs or {})
    return kwargs


def _benchmark_config(config: RegistrationQAConfig) -> Track2pBenchmarkConfig:
    return Track2pBenchmarkConfig(
        data=config.data,
        method="track2p-baseline",
        plane_name=config.plane_name,
        input_format=config.input_format,
        reference=config.reference,
        reference_kind=config.reference_kind,
        allow_track2p_as_reference_for_smoke_test=config.allow_track2p_as_reference_for_smoke_test,
        curated_only=config.curated_only,
        include_behavior=config.include_behavior,
        include_non_cells=config.include_non_cells,
        cell_probability_threshold=config.cell_probability_threshold,
        weighted_masks=config.weighted_masks,
        exclude_overlapping_pixels=config.exclude_overlapping_pixels,
        progress=config.progress,
    )


def _config_from_args(args: argparse.Namespace) -> RegistrationQAConfig:
    pairwise_cost_kwargs = None
    if args.pairwise_cost_kwargs_json is not None:
        pairwise_cost_kwargs = json.loads(args.pairwise_cost_kwargs_json)
        if not isinstance(pairwise_cost_kwargs, dict):
            raise ValueError("--pairwise-cost-kwargs-json must decode to a JSON object")
    return RegistrationQAConfig(
        data=args.data,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        curated_only=args.curated_only,
        plane_name=args.plane_name,
        input_format=args.input_format,
        max_gap=args.max_gap,
        transform_type=args.transform_type,
        cost=args.cost,
        cost_threshold=None if args.no_cost_threshold else args.cost_threshold,
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


def _roi_lookup(session: Track2pSession) -> dict[int, int]:
    return {int(value): index for index, value in enumerate(_roi_indices(session))}


def _roi_indices(session: Track2pSession) -> np.ndarray:
    plane = session.plane_data
    if plane.roi_indices is None:
        return np.arange(plane.n_rois, dtype=int)
    return np.asarray(plane.roi_indices, dtype=int)


def _component_value(
    components: Mapping[str, np.ndarray],
    key: str,
    row: int,
    column: int,
    default: Any = np.nan,
) -> Any:
    if key not in components:
        return default
    return np.asarray(components[key])[row, column].item()


def _registration_backend(transform_type: str, source: str) -> str:
    if transform_type == "none":
        return "none"
    if "fov_registered" in source:
        return "fov-translation"
    if "registered" in source:
        return "track2p-elastix"
    return "unknown"


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _stat(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    percentile: float | None = None,
) -> float:
    values = _finite_values(rows, key)
    if not values.size:
        return np.nan
    if percentile is None:
        return float(np.median(values))
    return float(np.percentile(values, percentile))


def _mean_bool(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return np.nan
    return float(np.mean([bool(row.get(key, False)) for row in rows]))


def _mode(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(key, ""))] += 1
    if not counts:
        return ""
    return max(counts, key=counts.get)


def _csv_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    return fieldnames


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.4g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
