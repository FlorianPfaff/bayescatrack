"""Registration quality report for Track2p-style benchmark subjects."""

# pylint: disable=protected-access,too-many-locals,too-many-arguments
from __future__ import annotations

import argparse, csv, json, sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from bayescatrack.association.pyrecest_global_assignment import registered_iou_cost_kwargs, roi_aware_cost_kwargs, session_edge_pairs
from bayescatrack.association.registered_masks import replace_empty_registered_masks
from bayescatrack.core.bridge import Track2pSession, build_session_pair_association_bundle
from bayescatrack.experiments.track2p_benchmark import (
    ReferenceKind, Track2pBenchmarkConfig, _load_reference_for_subject, _load_subject_sessions, _reference_matrix,
    _validate_reference_for_benchmark, _validate_reference_roi_indices, discover_subject_dirs,
)
from bayescatrack.track2p_registration import register_plane_pair

RegistrationQACost = Literal["registered-iou", "roi-aware"]
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
    out: list[dict[str, Any]] = []
    bench = _bench(config)
    for subject_dir in subject_dirs:
        if config.progress:
            print(f"registration-qa: {subject_dir.name}", file=sys.stderr, flush=True)
        ref = _load_reference_for_subject(subject_dir, data_root=config.data, config=bench)
        _validate_reference_for_benchmark(ref, subject_dir=subject_dir, config=bench)
        sessions = _load_subject_sessions(subject_dir, bench)
        _validate_reference_roi_indices(ref, sessions)
        out.extend(_audit_subject(subject_dir.name, sessions, _reference_matrix(ref, curated_only=config.curated_only), config))
    return out

def summarize_registration_qa_links(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate link-level QA rows by subject and session edge."""
    groups: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r["subject"]), str(r["source_session_name"]), str(r["target_session_name"]), int(r["session_gap"]))].append(r)
    return [{
        "subject": k[0], "source_session_name": k[1], "target_session_name": k[2], "session_gap": k[3], "n_gt_links": len(g),
        "registration_backend": _mode(g, "registration_backend"), "transform_type": _mode(g, "transform_type"),
        "median_registered_iou": _stat(g, "registered_iou"), "p10_registered_iou": _stat(g, "registered_iou", 10), "p90_registered_iou": _stat(g, "registered_iou", 90),
        "median_registered_centroid_distance": _stat(g, "registered_centroid_distance"), "p90_registered_centroid_distance": _stat(g, "registered_centroid_distance", 90),
        "empty_registered_rois": int(max(int(r["empty_registered_rois"]) for r in g)), "empty_registered_fraction": float(max(float(r["empty_registered_fraction"]) for r in g)),
        "gt_top1_rate": _mean_bool(g, "gt_is_top1"), "gt_admissible_rate": _mean_bool(g, "gt_candidate_admissible"),
        "empty_gt_mask_rate": _mean_bool(g, "target_empty_registered_mask"), "gated_gt_rate": _mean_bool(g, "target_gated"),
        "median_gt_rank": _stat(g, "gt_rank"), "p90_gt_rank": _stat(g, "gt_rank", 90), "median_cost_margin": _stat(g, "cost_margin"),
    } for k, g in sorted(groups.items())]

def format_registration_qa_table(rows: Sequence[Mapping[str, Any]]) -> str:
    cols = ["subject", "source_session_name", "target_session_name", "n_gt_links", "registration_backend", "median_registered_iou", "median_registered_centroid_distance", "gt_top1_rate", "gt_admissible_rate", "empty_registered_rois", "median_gt_rank", "median_cost_margin"]
    body = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    body += ["| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(body)

def write_registration_qa_results(rows: Sequence[Mapping[str, Any]], output_path: Path, output_format: OutputFormat) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json": output_path.write_text(json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"); return
    if output_format == "csv":
        with output_path.open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=_fields(rows)); w.writeheader(); w.writerows(rows)
        return
    output_path.write_text(format_registration_qa_table(rows) + "\n", encoding="utf-8")

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bayescatrack benchmark registration-qa", description="Report registration quality on manual-GT Track2p links.")
    p.add_argument("--data", required=True, type=Path); p.add_argument("--reference", type=Path); p.add_argument("--reference-kind", default="auto", choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows")); p.add_argument("--allow-track2p-as-reference-for-smoke-test", action="store_true")
    p.add_argument("--curated-only", action="store_true"); p.add_argument("--plane", dest="plane_name", default="plane0"); p.add_argument("--input-format", default="auto", choices=("auto", "suite2p", "npy")); p.add_argument("--max-gap", type=int, default=2); p.add_argument("--transform-type", default="affine", choices=("affine", "rigid", "none")); p.add_argument("--cost", default="registered-iou", choices=("registered-iou", "roi-aware")); p.add_argument("--cost-threshold", type=float, default=6.0); p.add_argument("--no-cost-threshold", action="store_true")
    p.add_argument("--include-behavior", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--include-non-cells", action="store_true"); p.add_argument("--cell-probability-threshold", type=float, default=0.5); p.add_argument("--weighted-masks", action="store_true"); p.add_argument("--exclude-overlapping-pixels", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--order", default="xy", choices=("xy", "yx")); p.add_argument("--weighted-centroids", action="store_true"); p.add_argument("--velocity-variance", type=float, default=25.0); p.add_argument("--regularization", type=float, default=1.0e-6); p.add_argument("--pairwise-cost-kwargs-json"); p.add_argument("--level", default="summary", choices=("summary", "links")); p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--output", type=Path); p.add_argument("--format", default="table", choices=("table", "json", "csv"))
    return p

def main(argv: list[str] | None = None) -> int:
    a = build_arg_parser().parse_args(argv); rows = run_registration_qa_report(_config(a)); rows = rows if a.level == "links" else summarize_registration_qa_links(rows)
    if a.output: write_registration_qa_results(rows, a.output, a.format)
    elif a.format == "json": print(json.dumps(list(rows), indent=2))
    elif a.format == "csv": w = csv.DictWriter(sys.stdout, fieldnames=_fields(rows)); w.writeheader(); w.writerows(rows)
    else: print(format_registration_qa_table(rows))
    return 0

def _audit_subject(subject: str, sessions: Sequence[Track2pSession], ref: np.ndarray, c: RegistrationQAConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s, t in session_edge_pairs(len(sessions), max_gap=c.max_gap):
        reg = register_plane_pair(sessions[s].plane_data, sessions[t].plane_data, transform_type=c.transform_type)
        reg, empty = replace_empty_registered_masks(reg)
        raw = _bundle(sessions[s], sessions[t], sessions[t].plane_data, c).pairwise_components
        bun = _bundle(sessions[s], sessions[t], reg, c)
        rows += _audit_links(subject, s, t, sessions[s], sessions[t], ref, raw, bun.pairwise_components, np.asarray(bun.pairwise_cost_matrix, float), empty, _backend(c.transform_type, reg.source), c)
    return rows

def _audit_links(subject: str, sidx: int, tidx: int, source: Track2pSession, target: Track2pSession, ref: np.ndarray, raw: Mapping[str, np.ndarray], comp: Mapping[str, np.ndarray], costs: np.ndarray, empty: np.ndarray, backend: str, c: RegistrationQAConfig) -> list[dict[str, Any]]:
    smap, tmap, troi = _lookup(source), _lookup(target), _roi_indices(target); rows = []
    for track_index, tr in enumerate(ref):
        if tr[sidx] is None or tr[tidx] is None: continue
        i, j = smap[int(tr[sidx])], tmap[int(tr[tidx])]; row = costs[i]; gt = float(row[j]); rank = int(1 + np.count_nonzero(row < gt)); best = int(np.nanargmin(row)); false = np.delete(row, j); false = false[np.isfinite(false)]; bf = float(np.min(false)) if false.size else np.nan
        gated = bool(_cv(comp, "gated", i, j, False)); emp = bool(empty[j]); below = True if c.cost_threshold is None else bool(gt <= float(c.cost_threshold))
        rows.append({"subject": subject, "source_session_index": sidx, "target_session_index": tidx, "source_session_name": source.session_name, "target_session_name": target.session_name, "session_gap": tidx - sidx, "track_index": track_index, "registration_backend": backend, "transform_type": c.transform_type, "source_roi": int(tr[sidx]), "target_roi": int(tr[tidx]), "raw_iou": _cv(raw, "iou", i, j), "registered_iou": _cv(comp, "iou", i, j), "raw_centroid_distance": _cv(raw, "centroid_distance", i, j), "registered_centroid_distance": _cv(comp, "centroid_distance", i, j), "gt_cost": gt, "gt_rank": rank, "gt_is_top1": rank == 1, "best_target_roi": int(troi[best]), "best_cost": float(row[best]), "best_false_cost": bf, "cost_margin": bf - gt if np.isfinite(bf) else np.nan, "target_empty_registered_mask": emp, "target_gated": gated, "target_below_cost_threshold": below, "gt_candidate_admissible": (not emp) and (not gated) and below, "empty_registered_rois": int(np.count_nonzero(empty)), "empty_registered_fraction": float(np.mean(empty)) if empty.size else 0.0})
    return rows

def _bundle(source: Track2pSession, target: Track2pSession, plane: Any, c: RegistrationQAConfig) -> Any:
    return build_session_pair_association_bundle(source, target, measurement_plane_in_reference_frame=plane, order=c.order, weighted_centroids=c.weighted_centroids, velocity_variance=c.velocity_variance, regularization=c.regularization, pairwise_cost_kwargs=_cost_kwargs(c), return_pairwise_components=True)

def _cost_kwargs(c: RegistrationQAConfig) -> dict[str, Any]:
    k = registered_iou_cost_kwargs() if c.cost == "registered-iou" else roi_aware_cost_kwargs(); k.update(c.pairwise_cost_kwargs or {}); return k

def _bench(c: RegistrationQAConfig) -> Track2pBenchmarkConfig:
    return Track2pBenchmarkConfig(data=c.data, method="track2p-baseline", plane_name=c.plane_name, input_format=c.input_format, reference=c.reference, reference_kind=c.reference_kind, allow_track2p_as_reference_for_smoke_test=c.allow_track2p_as_reference_for_smoke_test, curated_only=c.curated_only, include_behavior=c.include_behavior, include_non_cells=c.include_non_cells, cell_probability_threshold=c.cell_probability_threshold, weighted_masks=c.weighted_masks, exclude_overlapping_pixels=c.exclude_overlapping_pixels, progress=c.progress)

def _config(a: argparse.Namespace) -> RegistrationQAConfig:
    pk = json.loads(a.pairwise_cost_kwargs_json) if a.pairwise_cost_kwargs_json else None
    return RegistrationQAConfig(data=a.data, reference=a.reference, reference_kind=a.reference_kind, allow_track2p_as_reference_for_smoke_test=a.allow_track2p_as_reference_for_smoke_test, curated_only=a.curated_only, plane_name=a.plane_name, input_format=a.input_format, max_gap=a.max_gap, transform_type=a.transform_type, cost=a.cost, cost_threshold=None if a.no_cost_threshold else a.cost_threshold, include_behavior=a.include_behavior, include_non_cells=a.include_non_cells, cell_probability_threshold=a.cell_probability_threshold, weighted_masks=a.weighted_masks, exclude_overlapping_pixels=a.exclude_overlapping_pixels, order=a.order, weighted_centroids=a.weighted_centroids, velocity_variance=a.velocity_variance, regularization=a.regularization, pairwise_cost_kwargs=pk, progress=a.progress)

def _lookup(s: Track2pSession) -> dict[int, int]: return {int(v): i for i, v in enumerate(_roi_indices(s))}
def _roi_indices(s: Track2pSession) -> np.ndarray: return np.arange(s.plane_data.n_rois, dtype=int) if s.plane_data.roi_indices is None else np.asarray(s.plane_data.roi_indices, dtype=int)
def _cv(c: Mapping[str, np.ndarray], k: str, i: int, j: int, d: Any = np.nan) -> Any: return d if k not in c else np.asarray(c[k])[i, j].item()
def _backend(t: str, src: str) -> str: return "none" if t == "none" else ("fov-translation" if "fov_registered" in src else ("track2p-elastix" if "registered" in src else "unknown"))
def _values(rows: Sequence[Mapping[str, Any]], k: str) -> np.ndarray: v = np.asarray([r.get(k, np.nan) for r in rows], dtype=float); return v[np.isfinite(v)]
def _stat(rows: Sequence[Mapping[str, Any]], k: str, p: float | None = None) -> float: v = _values(rows, k); return float(np.percentile(v, p)) if v.size and p is not None else (float(np.median(v)) if v.size else np.nan)
def _mean_bool(rows: Sequence[Mapping[str, Any]], k: str) -> float: return float(np.mean([bool(r.get(k, False)) for r in rows])) if rows else np.nan
def _mode(rows: Sequence[Mapping[str, Any]], k: str) -> str: c: dict[str, int] = defaultdict(int); [c.__setitem__(str(r.get(k, "")), c[str(r.get(k, ""))] + 1) for r in rows]; return max(c, key=c.get) if c else ""
def _fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    f: list[str] = []
    for r in rows:
        for k in r:
            if k not in f: f.append(str(k))
    return f
def _fmt(v: Any) -> str: return "nan" if isinstance(v, float) and np.isnan(v) else (f"{v:.4g}" if isinstance(v, float) else str(v))

if __name__ == "__main__":
    raise SystemExit(main())
