#!/usr/bin/env python3
"""Inspect local dataset metadata without choosing a dataset field mapping."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable
from urllib.parse import urlparse


METADATA_FORMATS = {
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".csv": "csv",
    ".parquet": "parquet",
}
IMAGE_FORMATS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_EXAMPLE_CHARACTERS = 160
MAX_EXAMPLES_PER_FIELD = 3
MAX_INVENTORY_EXAMPLES = 10
PARQUET_SAMPLE_ROWS = 20

ROLE_TERMS = {
    "id": {"id", "identifier", "sample_id", "claim_id", "uid"},
    "claim": {"claim", "claim_text", "statement", "hypothesis"},
    "context": {
        "abstract",
        "context",
        "evidence",
        "passage",
        "rationale",
        "text",
    },
    "caption": {"caption", "figure_caption", "image_caption", "table_caption"},
    "image": {
        "figure",
        "figure_path",
        "image",
        "image_file",
        "image_filename",
        "image_path",
        "image_paths",
        "image_url",
        "image_urls",
        "images",
    },
    "label": {
        "answer",
        "gold_label",
        "ground_truth",
        "label",
        "labels",
        "target",
        "verdict",
    },
}


class InspectionError(ValueError):
    """Raised for invalid paths or inputs that cannot be inspected safely."""


def _primitive_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return _string_type(value)
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _string_type(value: str) -> str:
    stripped = value.strip()
    if not stripped or stripped.lower() in {"null", "none", "na", "n/a"}:
        return "null"
    if stripped.lower() in {"true", "false"}:
        return "boolean"
    if re.fullmatch(r"[+-]?(?:0|[1-9]\d*)", stripped):
        return "integer"
    try:
        number = float(stripped)
    except ValueError:
        return "string"
    return "number" if math.isfinite(number) else "string"


def _looks_like_base64(value: str) -> bool:
    compact = "".join(value.split())
    return len(compact) >= 64 and len(compact) % 4 == 0 and bool(
        re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", compact)
    )


def _safe_string(value: str) -> str:
    if re.search(r"data:[^,;]+;base64,", value, flags=re.IGNORECASE):
        return "<base64 data omitted>"
    if _looks_like_base64(value):
        return "<possible base64 data omitted>"
    if len(value) <= MAX_EXAMPLE_CHARACTERS:
        return value
    return value[: MAX_EXAMPLE_CHARACTERS - 1] + "…"


def _safe_example(value: Any, depth: int = 0) -> Any:
    if isinstance(value, str):
        return _safe_string(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "<binary data omitted>"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if depth >= 2:
        return f"<{type(value).__name__} omitted>"
    if isinstance(value, list):
        example = [_safe_example(item, depth + 1) for item in value[:3]]
        if len(value) > 3:
            example.append(f"<{len(value) - 3} more items>")
        return example
    if isinstance(value, dict):
        items = list(value.items())
        example = {
            str(key): _safe_example(item, depth + 1)
            for key, item in items[:6]
        }
        if len(items) > 6:
            example["<omitted>"] = f"{len(items) - 6} more keys"
        return example
    return _safe_string(str(value))


def _example_marker(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class FieldObservation:
    def __init__(self, name: str):
        self.name = name
        self.types: set[str] = set()
        self.examples: list[Any] = []
        self._example_markers: set[str] = set()

    def observe(self, value: Any) -> None:
        self.types.add(_primitive_type(value))
        if len(self.examples) >= MAX_EXAMPLES_PER_FIELD:
            return
        example = _safe_example(value)
        marker = _example_marker(example)
        if marker not in self._example_markers:
            self.examples.append(example)
            self._example_markers.add(marker)

    def report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "types": sorted(self.types),
            "examples": self.examples,
        }


def _normalise_field_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _role_matches(field_name: str) -> dict[str, list[str]]:
    normalised = _normalise_field_name(field_name)
    matches: dict[str, list[str]] = {}
    for role, terms in ROLE_TERMS.items():
        matched = sorted(
            term
            for term in terms
            if normalised == term
            or normalised.endswith("_" + term)
        )
        if role == "id" and normalised.endswith("_id"):
            matched.append("*_id")
        if matched:
            matches[role] = sorted(set(matched))
    return matches


def _field_candidates(field_names: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    candidates = {role: [] for role in ROLE_TERMS}
    for field_name in field_names:
        for role, terms in _role_matches(field_name).items():
            normalised = _normalise_field_name(field_name)
            exact = normalised in ROLE_TERMS[role]
            candidates[role].append(
                {
                    "field": field_name,
                    "confidence": "strong_name_match" if exact else "name_match",
                    "matched_terms": terms,
                }
            )
    return {
        role: sorted(values, key=lambda item: item["field"])
        for role, values in candidates.items()
    }


def _record_from_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {"_value": value}


def _flatten_image_references(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        references: list[str] = []
        for item in value:
            references.extend(_flatten_image_references(item))
        return references
    if isinstance(value, dict):
        references = []
        for key, item in value.items():
            key_matches_image = bool(_role_matches(str(key)).get("image"))
            key_matches_path = _normalise_field_name(str(key)) in {
                "file",
                "filename",
                "path",
                "url",
            }
            if key_matches_image or key_matches_path:
                references.extend(_flatten_image_references(item))
        return references
    return []


def _is_remote_reference(reference: str) -> bool:
    try:
        parsed = urlparse(reference.strip())
        return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)
    except ValueError:
        return False


class ImageStatistics:
    def __init__(self, metadata_path: Path, dataset_root: Path, scope: str):
        self.metadata_path = metadata_path
        self.dataset_root = dataset_root
        self.scope = scope
        self.samples_observed = 0
        self.counts: Counter[int] = Counter()
        self.resolution: Counter[str] = Counter()

    def observe(self, record: dict[str, Any]) -> None:
        references: list[str] = []
        for field_name, value in record.items():
            if _role_matches(str(field_name)).get("image"):
                references.extend(_flatten_image_references(value))
        self.samples_observed += 1
        self.counts[len(references)] += 1
        for reference in references:
            self.resolution[self._resolution_status(reference)] += 1

    def _resolution_status(self, reference: str) -> str:
        stripped = reference.strip()
        if re.match(r"^data:[^,;]+;base64,", stripped, flags=re.IGNORECASE):
            return "embedded_data_omitted"
        if _is_remote_reference(stripped):
            return "remote_reference_not_accessed"
        path = Path(stripped).expanduser()
        candidates = [path] if path.is_absolute() else [
            self.metadata_path.parent / path,
            self.dataset_root / path,
        ]
        try:
            if any(candidate.is_file() for candidate in candidates):
                return "resolved"
        except (OSError, ValueError):
            return "invalid_reference"
        return "missing"

    def report(self) -> dict[str, Any]:
        observed_counts = sorted(self.counts.elements())
        return {
            "scope": self.scope,
            "samples_observed": self.samples_observed,
            "possible_image_count_per_sample": {
                "minimum": min(observed_counts) if observed_counts else None,
                "maximum": max(observed_counts) if observed_counts else None,
                "distribution": {
                    str(count): frequency
                    for count, frequency in sorted(self.counts.items())
                },
            },
            "path_resolution": {
                "references_checked": sum(self.resolution.values()),
                **{
                    status: self.resolution.get(status, 0)
                    for status in (
                        "resolved",
                        "missing",
                        "remote_reference_not_accessed",
                        "embedded_data_omitted",
                        "invalid_reference",
                    )
                },
            },
        }


def _inspect_records(
    records: Iterable[Any],
    metadata_path: Path,
    dataset_root: Path,
    location: str,
    scope: str = "all_rows",
    declared_columns: Iterable[str] = (),
) -> dict[str, Any]:
    fields = {name: FieldObservation(name) for name in declared_columns}
    image_statistics = ImageStatistics(metadata_path, dataset_root, scope)
    row_count = 0
    for value in records:
        record = _record_from_value(value)
        row_count += 1
        for raw_name, item in record.items():
            name = str(raw_name)
            fields.setdefault(name, FieldObservation(name)).observe(item)
        image_statistics.observe(record)
    field_names = sorted(fields)
    return {
        "location": location,
        "row_count": row_count,
        "columns": [fields[name].report() for name in field_names],
        "possible_fields": _field_candidates(field_names),
        "image_statistics": image_statistics.report(),
    }


def _inspect_json(path: Path, dataset_root: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_JSON_BYTES:
        raise InspectionError(
            f"JSON file exceeds the {MAX_JSON_BYTES}-byte safe parsing limit"
        )
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return {
            "top_level_type": "array",
            "top_level_keys": [],
            "sections": [_inspect_records(value, path, dataset_root, "$")],
        }
    if isinstance(value, dict):
        sections = [_inspect_records([value], path, dataset_root, "$")]
        for key, candidate in value.items():
            if isinstance(candidate, list) and (
                not candidate or any(isinstance(item, dict) for item in candidate)
            ):
                sections.append(
                    _inspect_records(candidate, path, dataset_root, f"$.{key}")
                )
        return {
            "top_level_type": "object",
            "top_level_keys": sorted(str(key) for key in value),
            "sections": sections,
        }
    return {
        "top_level_type": _primitive_type(value),
        "top_level_keys": [],
        "sections": [_inspect_records([value], path, dataset_root, "$")],
    }


def _iter_jsonl(path: Path) -> Iterable[Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise InspectionError(f"invalid JSON on line {line_number}") from exc


def _inspect_jsonl(path: Path, dataset_root: Path) -> dict[str, Any]:
    return {
        "top_level_type": "line_delimited_records",
        "top_level_keys": [],
        "sections": [_inspect_records(_iter_jsonl(path), path, dataset_root, "$")],
    }


def _inspect_csv(path: Path, dataset_root: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise InspectionError("CSV file has no header row")
        section = _inspect_records(
            reader,
            path,
            dataset_root,
            "$",
            declared_columns=(str(name) for name in reader.fieldnames),
        )
    return {
        "top_level_type": "tabular_records",
        "top_level_keys": [],
        "sections": [section],
    }


def _arrow_type_name(arrow_type: Any) -> str:
    try:
        import pyarrow.types as arrow_types
    except ImportError:
        return "unknown"
    if arrow_types.is_boolean(arrow_type):
        return "boolean"
    if arrow_types.is_integer(arrow_type):
        return "integer"
    if arrow_types.is_floating(arrow_type) or arrow_types.is_decimal(arrow_type):
        return "number"
    if arrow_types.is_string(arrow_type) or arrow_types.is_large_string(arrow_type):
        return "string"
    if arrow_types.is_list(arrow_type) or arrow_types.is_large_list(arrow_type):
        return "array"
    if arrow_types.is_fixed_size_list(arrow_type):
        return "array"
    if arrow_types.is_struct(arrow_type) or arrow_types.is_map(arrow_type):
        return "object"
    if arrow_types.is_dictionary(arrow_type):
        return _arrow_type_name(arrow_type.value_type)
    if arrow_types.is_null(arrow_type):
        return "null"
    return "string"


def _arrow_contains_binary(arrow_type: Any) -> bool:
    import pyarrow.types as arrow_types

    if (
        arrow_types.is_binary(arrow_type)
        or arrow_types.is_large_binary(arrow_type)
        or arrow_types.is_fixed_size_binary(arrow_type)
    ):
        return True
    if (
        arrow_types.is_list(arrow_type)
        or arrow_types.is_large_list(arrow_type)
        or arrow_types.is_fixed_size_list(arrow_type)
    ):
        return _arrow_contains_binary(arrow_type.value_type)
    if arrow_types.is_struct(arrow_type):
        return any(_arrow_contains_binary(field.type) for field in arrow_type)
    if arrow_types.is_map(arrow_type):
        return _arrow_contains_binary(arrow_type.key_type) or _arrow_contains_binary(
            arrow_type.item_type
        )
    if arrow_types.is_dictionary(arrow_type):
        return _arrow_contains_binary(arrow_type.value_type)
    return False


def _inspect_parquet(path: Path, dataset_root: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise InspectionError(
            "Parquet format detected but a local Parquet reader is unavailable"
        ) from exc

    try:
        parquet_file = parquet.ParquetFile(path)
        schema = parquet_file.schema_arrow
        sample_rows: list[dict[str, Any]] = []
        sample_columns = [
            field.name for field in schema if not _arrow_contains_binary(field.type)
        ]
        if parquet_file.metadata.num_rows and sample_columns:
            for batch in parquet_file.iter_batches(
                batch_size=PARQUET_SAMPLE_ROWS,
                columns=sample_columns,
            ):
                sample_rows.extend(batch.to_pylist())
                break
    except Exception as exc:
        raise InspectionError("unable to read Parquet metadata safely") from exc
    section = _inspect_records(
        sample_rows,
        path,
        dataset_root,
        "$",
        scope=f"first_{len(sample_rows)}_rows",
        declared_columns=schema.names,
    )
    section["row_count"] = parquet_file.metadata.num_rows
    observations = {column["name"]: column for column in section["columns"]}
    for field in schema:
        observation = observations[field.name]
        declared_type = _arrow_type_name(field.type)
        if not observation["types"]:
            observation["types"] = [declared_type]
        elif declared_type not in observation["types"]:
            observation["types"].append(declared_type)
            observation["types"].sort()
    return {
        "top_level_type": "tabular_records",
        "top_level_keys": [],
        "sections": [section],
    }


def _display_path(path: Path, dataset_path: Path, dataset_root: Path) -> str:
    if dataset_path.is_file():
        return path.name
    return path.relative_to(dataset_root).as_posix()


def _discover_files(dataset_path: Path) -> tuple[list[Path], list[Path]]:
    candidates = [dataset_path] if dataset_path.is_file() else dataset_path.rglob("*")
    metadata_files: list[Path] = []
    image_files: list[Path] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        suffix = candidate.suffix.lower()
        if suffix in METADATA_FORMATS:
            metadata_files.append(candidate)
        if suffix in IMAGE_FORMATS:
            image_files.append(candidate)
    return sorted(metadata_files), sorted(image_files)


def inspect_dataset(dataset_name: str, dataset_path: Path | str) -> dict[str, Any]:
    path = Path(dataset_path).expanduser()
    if not dataset_name.strip():
        raise InspectionError("dataset name must not be empty")
    if not path.exists():
        raise InspectionError("dataset path does not exist")
    if not path.is_file() and not path.is_dir():
        raise InspectionError("dataset path must be a file or directory")

    resolved_path = path.resolve()
    dataset_root = resolved_path.parent if resolved_path.is_file() else resolved_path
    metadata_files, image_files = _discover_files(resolved_path)
    metadata_reports = []
    aggregate_candidates = {role: [] for role in ROLE_TERMS}

    inspectors = {
        "json": _inspect_json,
        "jsonl": _inspect_jsonl,
        "csv": _inspect_csv,
        "parquet": _inspect_parquet,
    }
    for metadata_path in metadata_files:
        file_format = METADATA_FORMATS[metadata_path.suffix.lower()]
        display_path = _display_path(metadata_path, resolved_path, dataset_root)
        report: dict[str, Any] = {
            "path": display_path,
            "format": file_format,
            "size_bytes": metadata_path.stat().st_size,
        }
        try:
            report.update(inspectors[file_format](metadata_path, dataset_root))
        except (
            InspectionError,
            OSError,
            UnicodeError,
            csv.Error,
            json.JSONDecodeError,
        ) as exc:
            report["inspection_error"] = str(exc)
            report["sections"] = []
        for section in report["sections"]:
            for role, candidates in section["possible_fields"].items():
                aggregate_candidates[role].extend(
                    {
                        "metadata_file": display_path,
                        "section": section["location"],
                        **candidate,
                    }
                    for candidate in candidates
                )
        metadata_reports.append(report)

    image_formats = Counter(path.suffix.lower().lstrip(".") for path in image_files)
    return {
        "dataset": dataset_name,
        "dataset_path": str(resolved_path),
        "discovered_metadata_files": [
            report["path"] for report in metadata_reports
        ],
        "file_formats": sorted({report["format"] for report in metadata_reports}),
        "metadata_files": metadata_reports,
        "possible_fields": aggregate_candidates,
        "image_files": {
            "count": len(image_files),
            "formats": dict(sorted(image_formats.items())),
            "examples": [
                _display_path(image_path, resolved_path, dataset_root)
                for image_path in image_files[:MAX_INVENTORY_EXAMPLES]
            ],
            "images_were_opened": False,
        },
        "mapping_status": "not_selected",
        "notes": [
            "Field candidates are name-based possibilities, not final mappings.",
            "Remote image references were recorded but never accessed.",
        ],
    }


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def write_report(report: dict[str, Any], output_path: Path | str, dataset_path: Path | str) -> None:
    output = Path(output_path).expanduser().resolve()
    source = Path(dataset_path).expanduser().resolve()
    if output == source or (source.is_dir() and _is_within(output, source)):
        raise InspectionError("output must be outside the source dataset path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a read-only schema report for local dataset metadata."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name for the report")
    parser.add_argument(
        "--dataset-path", required=True, type=Path, help="Local dataset file or directory"
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    try:
        report = inspect_dataset(arguments.dataset, arguments.dataset_path)
        write_report(report, arguments.output, arguments.dataset_path)
    except InspectionError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
