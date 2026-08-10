"""Deterministic hard-example selection inside the reserved SciVer search split."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from meta_harness.schemas import canonical_json
from meta_harness.split_manager import (
    paper_group_identity,
    verify_split_manifest,
)


HARD_SEARCH_SCHEMA_VERSION = 1
HARD_SEARCH_ALGORITHM = "paper_first_baseline_hard_diverse_v1"


class HardSearchError(ValueError):
    """Raised when a leakage-safe hard-search set cannot be constructed."""


def build_hard_search_manifest(
    records: Sequence[Mapping[str, Any]],
    baseline_results: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    *,
    target_size: int = 80,
    guard_fraction: float = 0.25,
    seed: int | None = None,
) -> dict[str, Any]:
    """Select baseline errors plus guard examples from the reserved search split."""

    verify_split_manifest(split_manifest)
    if not 50 <= target_size <= 100:
        raise HardSearchError("target_size must be between 50 and 100")
    if not 0.20 <= float(guard_fraction) <= 0.30:
        raise HardSearchError("guard_fraction must be between 0.20 and 0.30")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise HardSearchError("records must be a sequence")
    if (
        isinstance(baseline_results, (str, bytes))
        or not isinstance(baseline_results, Sequence)
    ):
        raise HardSearchError("baseline_results must be a sequence")

    reserved_ids = {
        _identity(value)
        for value in split_manifest["splits"]["search"]["sample_ids"]
    }
    validation_papers = set(
        split_manifest["splits"]["validation"]["paper_group_ids"]
    )
    final_papers = set(
        split_manifest["splits"]["final_test"]["paper_group_ids"]
    )
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or "sample_id" not in record:
            raise HardSearchError(
                f"record at index {index} requires a sample_id"
            )
        marker = _identity(record["sample_id"])
        if marker in by_id:
            raise HardSearchError("records contain duplicate sample IDs")
        if marker not in reserved_ids:
            raise HardSearchError(
                "records must contain only the reserved search split"
            )
        paper = paper_group_identity(record)
        if paper in validation_papers or paper in final_papers:
            raise HardSearchError(
                "search record paper overlaps a protected split"
            )
        by_id[marker] = record
    if set(by_id) != reserved_ids:
        raise HardSearchError(
            "records must contain every reserved search-split sample"
        )

    result_by_id: dict[str, Mapping[str, Any]] = {}
    for index, result in enumerate(baseline_results):
        if not isinstance(result, Mapping) or "sample_id" not in result:
            raise HardSearchError(
                f"baseline result at index {index} requires a sample_id"
            )
        marker = _identity(result["sample_id"])
        if marker in result_by_id:
            raise HardSearchError(
                "baseline_results contain duplicate sample IDs"
            )
        if marker not in reserved_ids:
            raise HardSearchError(
                "baseline_results must contain only reserved search samples"
            )
        result_by_id[marker] = result
    if set(result_by_id) != reserved_ids:
        raise HardSearchError(
            "baseline_results must cover the complete reserved search split"
        )

    entries = [
        _entry(by_id[marker], result_by_id[marker])
        for marker in sorted(by_id)
    ]
    errors = [entry for entry in entries if not entry["baseline_correct"]]
    guards = [entry for entry in entries if entry["baseline_correct"]]
    guard_count = round(target_size * float(guard_fraction))
    error_count = target_size - guard_count
    if len(errors) < error_count:
        raise HardSearchError(
            "the reserved search split has too few baseline errors for the "
            "requested hard-search target"
        )
    if len(guards) < guard_count:
        raise HardSearchError(
            "the reserved search split has too few baseline-correct guard "
            "examples"
        )

    resolved_seed = split_manifest["seed"] if seed is None else seed
    selected: list[dict[str, Any]] = []
    selected.extend(
        _diverse_select(errors, error_count, selected, resolved_seed)
    )
    selected.extend(
        _diverse_select(guards, guard_count, selected, resolved_seed)
    )
    selected.sort(
        key=lambda entry: _seeded_key(
            resolved_seed,
            entry["sample_id_marker"],
        )
    )

    items = [
        {
            key: entry[key]
            for key in (
                "sample_id",
                "paper_group_id",
                "record_sha256",
                "baseline_result_sha256",
                "baseline_correct",
                "label",
                "evidence_modality",
                "reasoning_method",
            )
        }
        for entry in selected
    ]
    payload = {
        "schema_version": HARD_SEARCH_SCHEMA_VERSION,
        "algorithm": HARD_SEARCH_ALGORITHM,
        "seed": resolved_seed,
        "source_split_sha256": split_manifest["split_sha256"],
        "source_dataset_sha256": split_manifest["dataset_sha256"],
        "source_search_sample_ids_sha256": _sha256_json(
            split_manifest["splits"]["search"]["sample_ids"]
        ),
        "target_size": target_size,
        "guard_fraction": float(guard_fraction),
        "baseline_error_count": error_count,
        "guard_count": guard_count,
        "items": items,
        "diversity": {
            "labels": sorted({entry["label"] for entry in selected}),
            "evidence_modalities": sorted(
                {entry["evidence_modality"] for entry in selected}
            ),
            "reasoning_methods": sorted(
                {entry["reasoning_method"] for entry in selected}
            ),
            "paper_count": len(
                {entry["paper_group_id"] for entry in selected}
            ),
        },
        "baseline_results_sha256": _sha256_json(
            sorted(
                (
                    {
                        "sample_id": result["sample_id"],
                        "result_sha256": _sha256_json(result),
                    }
                    for result in baseline_results
                ),
                key=lambda item: _identity(item["sample_id"]),
            )
        ),
    }
    manifest = {**payload, "hard_search_sha256": _sha256_json(payload)}
    verify_hard_search_manifest(manifest, split_manifest=split_manifest)
    return manifest


def materialize_hard_search_records(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return selected records in immutable manifest order."""

    verify_hard_search_manifest(manifest)
    wanted = [_identity(item["sample_id"]) for item in manifest["items"]]
    by_id = {_identity(record.get("sample_id")): record for record in records}
    if len(by_id) != len(records) or any(marker not in by_id for marker in wanted):
        raise HardSearchError("records do not cover the hard-search manifest")
    selected = [dict(by_id[marker]) for marker in wanted]
    for record, item in zip(selected, manifest["items"]):
        if _sha256_json(record) != item["record_sha256"]:
            raise HardSearchError("hard-search source record hash mismatch")
    return selected


def verify_hard_search_manifest(
    manifest: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any] | None = None,
) -> None:
    """Verify immutable identities and source split membership."""

    if not isinstance(manifest, Mapping):
        raise HardSearchError("hard-search manifest must be a mapping")
    required = {
        "schema_version",
        "algorithm",
        "seed",
        "source_split_sha256",
        "source_dataset_sha256",
        "source_search_sample_ids_sha256",
        "target_size",
        "guard_fraction",
        "baseline_error_count",
        "guard_count",
        "items",
        "diversity",
        "baseline_results_sha256",
        "hard_search_sha256",
    }
    if set(manifest) != required:
        raise HardSearchError(
            "hard-search manifest has unexpected or missing fields"
        )
    if manifest["schema_version"] != HARD_SEARCH_SCHEMA_VERSION:
        raise HardSearchError("unsupported hard-search schema_version")
    if manifest["algorithm"] != HARD_SEARCH_ALGORITHM:
        raise HardSearchError("unsupported hard-search algorithm")
    if manifest["target_size"] != len(manifest["items"]):
        raise HardSearchError("hard-search item count is inconsistent")
    if (
        manifest["baseline_error_count"] + manifest["guard_count"]
        != manifest["target_size"]
    ):
        raise HardSearchError("hard-search guard counts are inconsistent")
    markers = [_identity(item.get("sample_id")) for item in manifest["items"]]
    if len(markers) != len(set(markers)):
        raise HardSearchError("hard-search sample IDs must be unique")
    without_hash = {
        key: value
        for key, value in manifest.items()
        if key != "hard_search_sha256"
    }
    if _sha256_json(without_hash) != manifest["hard_search_sha256"]:
        raise HardSearchError("hard-search manifest hash mismatch")
    if split_manifest is not None:
        verify_split_manifest(split_manifest)
        if manifest["source_split_sha256"] != split_manifest["split_sha256"]:
            raise HardSearchError("hard-search source split hash mismatch")
        reserved = {
            _identity(value)
            for value in split_manifest["splits"]["search"]["sample_ids"]
        }
        if not set(markers).issubset(reserved):
            raise HardSearchError(
                "hard-search manifest contains a protected-split sample"
            )


def save_hard_search_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Create an immutable manifest or accept byte-equivalent content."""

    verify_hard_search_manifest(manifest)
    return _atomic_create_json(Path(path), manifest)


def load_hard_search_manifest(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HardSearchError(
            "hard-search manifest must be readable valid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise HardSearchError("hard-search manifest must contain an object")
    verify_hard_search_manifest(value)
    return value


def _entry(
    record: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    gold = _binary_label(
        result.get("gold_label", record.get("gold_label")),
        "gold_label",
    )
    prediction = result.get("prediction")
    parsed_prediction = (
        None
        if prediction is None
        else _binary_label(prediction, "prediction")
    )
    request_status = result.get("request_status", "success")
    correct = request_status == "success" and parsed_prediction == gold
    method = result.get(
        "reasoning_method",
        result.get("method", record.get("type", record.get("claim_type"))),
    )
    if not isinstance(method, str) or not method.strip():
        raise HardSearchError("reasoning method is missing")
    image_value = record.get("image_path", record.get("image_paths"))
    has_image = bool(image_value)
    has_text = any(
        bool(record.get(field))
        for field in ("section", "context", "caption", "captions", "item")
    )
    modality = (
        "multimodal"
        if has_image and has_text
        else "image"
        if has_image
        else "text"
    )
    return {
        "sample_id": record["sample_id"],
        "sample_id_marker": _identity(record["sample_id"]),
        "paper_group_id": paper_group_identity(record),
        "record_sha256": _sha256_json(record),
        "baseline_result_sha256": _sha256_json(result),
        "baseline_correct": correct,
        "label": gold,
        "evidence_modality": modality,
        "reasoning_method": method,
    }


def _diverse_select(
    pool: Sequence[dict[str, Any]],
    count: int,
    already_selected: Sequence[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    remaining = list(pool)
    selected = list(already_selected)
    added: list[dict[str, Any]] = []
    counts = {
        field: Counter(entry[field] for entry in selected)
        for field in (
            "label",
            "evidence_modality",
            "reasoning_method",
            "paper_group_id",
        )
    }
    for _ in range(count):
        chosen = min(
            remaining,
            key=lambda entry: (
                counts["paper_group_id"][entry["paper_group_id"]],
                counts["label"][entry["label"]],
                counts["evidence_modality"][entry["evidence_modality"]],
                counts["reasoning_method"][entry["reasoning_method"]],
                _seeded_key(seed, entry["sample_id_marker"]),
            ),
        )
        remaining.remove(chosen)
        added.append(chosen)
        for field in counts:
            counts[field][chosen[field]] += 1
    return added


def _binary_label(value: Any, field: str) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"yes", "true"}:
            return "yes"
        if normalized in {"no", "false"}:
            return "no"
    raise HardSearchError(f"{field} must be a normalized binary label")


def _seeded_key(seed: int, marker: str) -> tuple[str, str]:
    return (
        hashlib.sha256(f"{seed}\0{marker}".encode("utf-8")).hexdigest(),
        marker,
    )


def _identity(value: Any) -> str:
    try:
        return canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise HardSearchError("sample IDs must be JSON serializable") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> Path:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise HardSearchError(
                "refusing to replace a different hard-search manifest"
            )
        return path
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return _atomic_create_json(path, value)
    finally:
        temporary.unlink(missing_ok=True)
    return path


__all__ = [
    "HARD_SEARCH_ALGORITHM",
    "HARD_SEARCH_SCHEMA_VERSION",
    "HardSearchError",
    "build_hard_search_manifest",
    "materialize_hard_search_records",
    "load_hard_search_manifest",
    "save_hard_search_manifest",
    "verify_hard_search_manifest",
]
