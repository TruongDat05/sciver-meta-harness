"""Trusted preparation artifacts for the isolated full SEARCH V3 protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from meta_harness.config import (
    FULL_SEARCH_V3_PROTOCOL_ID,
    FullSearchV3Config,
    canonical_full_search_v3_config,
    load_full_search_v3_config,
)
from meta_harness.full_search_v3 import (
    FULL_SEARCH_V3_ALLOCATION_ALGORITHM,
    FULL_SEARCH_V3_PAPER_IDENTITY_VERSION,
    FULL_SEARCH_V3_SAMPLE_IDENTITY_VERSION,
    FULL_SEARCH_V3_SPLIT_SCHEMA_VERSION,
    ValidatedSciVerV3Record,
    build_full_search_v3_split,
    validate_sciver_full_search_v3_records,
    verify_full_search_v3_split,
)
from utils.dataset_adapters import get_dataset_adapter


FULL_SEARCH_V3_PRIVATE_MANIFEST_SCHEMA = (
    "sciver_full_search_v3_private_manifest_v1"
)
FULL_SEARCH_V3_SEARCH_SAFE_MANIFEST_SCHEMA = (
    "sciver_full_search_v3_search_safe_manifest_v1"
)
FULL_SEARCH_V3_SEARCH_DATASET_SCHEMA = "sciver_full_search_v3_search_dataset_v1"
FULL_SEARCH_V3_PREPARATION_IDENTITY_SCHEMA = (
    "sciver_full_search_v3_preparation_identity_v1"
)
FULL_SEARCH_V3_FINAL_COMMITMENT_SCHEMA = (
    "sciver_full_search_v3_final_membership_commitment_v1"
)
PRIVATE_MANIFEST_FILENAME = "private_split_manifest.json"
SEARCH_SAFE_MANIFEST_FILENAME = "search_safe_manifest.json"
SEARCH_DATASET_FILENAME = "search_records.json"


class FullSearchV3PreparationError(ValueError):
    """Raised when a V3 preparation artifact is invalid or incompatible."""


@dataclass(frozen=True)
class FullSearchV3PreparationArtifacts:
    """Trusted and SEARCH-facing locations created by one offline preparation."""

    private_manifest_path: Path
    search_safe_manifest_path: Path
    search_dataset_path: Path
    summary: Mapping[str, Any]


def prepare_full_search_v3(
    *,
    source_path: str | Path,
    private_directory: str | Path,
    search_directory: str | Path,
    config_path: str | Path | None = None,
) -> FullSearchV3PreparationArtifacts:
    """Create or verify V3 artifacts without constructing a solver or client."""

    source = Path(source_path)
    config = (
        canonical_full_search_v3_config()
        if config_path is None
        else load_full_search_v3_config(config_path)
    )
    raw_records = _load_source_records(source)
    records = validate_sciver_full_search_v3_records(
        raw_records,
        source_path=source,
    )
    split = build_full_search_v3_split(records, config=config)
    verify_full_search_v3_split(split, records, config=config)
    private_manifest = build_full_search_v3_private_manifest(
        split,
        source_dataset_sha256=source_dataset_sha256(source),
        config=config,
    )
    private_path = Path(private_directory) / PRIVATE_MANIFEST_FILENAME
    save_full_search_v3_private_manifest(private_path, private_manifest)
    loaded_private = load_trusted_full_search_v3_private_manifest(private_path)
    verify_full_search_v3_private_manifest(
        loaded_private,
        records=records,
        source_path=source,
        config=config,
    )

    search_safe_manifest = derive_full_search_v3_search_safe_manifest(loaded_private)
    search_safe_path = Path(search_directory) / SEARCH_SAFE_MANIFEST_FILENAME
    save_full_search_v3_search_safe_manifest(search_safe_path, search_safe_manifest)
    loaded_search_safe = load_full_search_v3_search_safe_manifest(search_safe_path)
    verify_full_search_v3_search_safe_manifest(loaded_search_safe, loaded_private)

    search_records = materialize_full_search_v3_search_records(
        records,
        loaded_private,
        source_path=source,
    )
    search_dataset_path = Path(search_directory) / SEARCH_DATASET_FILENAME
    save_full_search_v3_search_dataset(search_dataset_path, search_records)
    loaded_search_records = load_full_search_v3_search_dataset(search_dataset_path)
    verify_full_search_v3_search_dataset(
        loaded_search_records,
        loaded_search_safe,
    )

    split_data = loaded_private["split"]
    summary = {
        "protocol_id": FULL_SEARCH_V3_PROTOCOL_ID,
        "source_record_count": split_data["eligible_sample_count"],
        "paper_count": split_data["eligible_paper_count"],
        "split_sha256": split_data["split_sha256"],
        "preparation_identity_sha256": loaded_private[
            "preparation_identity_sha256"
        ],
        "SEARCH": {
            "sample_count": split_data["SEARCH"]["sample_count"],
            "paper_count": split_data["SEARCH"]["paper_count"],
        },
        "FINAL": {
            "sample_count": split_data["FINAL"]["sample_count"],
            "paper_count": split_data["FINAL"]["paper_count"],
        },
        "sample_overlap_count": 0,
        "paper_overlap_count": 0,
        "private_manifest_path": str(private_path),
        "search_safe_manifest_path": str(search_safe_path),
        "search_dataset_path": str(search_dataset_path),
    }
    return FullSearchV3PreparationArtifacts(
        private_manifest_path=private_path,
        search_safe_manifest_path=search_safe_path,
        search_dataset_path=search_dataset_path,
        summary=summary,
    )


def source_dataset_sha256(path: str | Path) -> str:
    """Return the immutable SHA-256 identity of exact source file bytes."""

    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise FullSearchV3PreparationError(
            "V3 source dataset must be a readable local file"
        ) from exc


def build_full_search_v3_private_manifest(
    split: Mapping[str, Any],
    *,
    source_dataset_sha256: str,
    config: FullSearchV3Config | None = None,
) -> dict[str, Any]:
    """Build trusted state that binds an exact split to source and configuration."""

    active_config = config or canonical_full_search_v3_config()
    _require_source_digest(source_dataset_sha256)
    _verify_private_split_shape(split, active_config)
    preparation_identity_sha256 = _preparation_identity_sha256(
        source_dataset_sha256=source_dataset_sha256,
        config=active_config,
    )
    manifest = {
        "schema_version": FULL_SEARCH_V3_PRIVATE_MANIFEST_SCHEMA,
        "artifact_type": "trusted_private_split_manifest",
        "protocol_id": FULL_SEARCH_V3_PROTOCOL_ID,
        "source_dataset_sha256": source_dataset_sha256,
        "source_record_count": split["eligible_sample_count"],
        "source_paper_count": split["eligible_paper_count"],
        "split_seed": active_config.split_seed,
        "search_size": active_config.search_size,
        "final_size": active_config.final_size,
        "config_sha256": active_config.sha256(),
        "sample_identity_version": FULL_SEARCH_V3_SAMPLE_IDENTITY_VERSION,
        "paper_identity_version": FULL_SEARCH_V3_PAPER_IDENTITY_VERSION,
        "split_schema_version": FULL_SEARCH_V3_SPLIT_SCHEMA_VERSION,
        "split_algorithm": FULL_SEARCH_V3_ALLOCATION_ALGORITHM,
        "preparation_identity_schema_version": (
            FULL_SEARCH_V3_PREPARATION_IDENTITY_SCHEMA
        ),
        "preparation_identity_sha256": preparation_identity_sha256,
        "split": dict(split),
        "final_membership_commitment": compute_full_search_v3_final_commitment(split),
        "private_manifest_sha256": "",
    }
    manifest["private_manifest_sha256"] = _sha256_json(
        dict(manifest, private_manifest_sha256="")
    )
    verify_full_search_v3_private_manifest(manifest, config=active_config)
    return manifest


def compute_full_search_v3_final_commitment(split: Mapping[str, Any]) -> str:
    """Return the opaque one-way commitment to private FINAL membership."""

    if not isinstance(split, Mapping) or not isinstance(split.get("FINAL"), Mapping):
        raise FullSearchV3PreparationError("V3 split must contain a FINAL pool")
    return _sha256_json(
        {
            "commitment_schema_version": FULL_SEARCH_V3_FINAL_COMMITMENT_SCHEMA,
            "protocol_id": FULL_SEARCH_V3_PROTOCOL_ID,
            "config_sha256": split.get("config_sha256"),
            "split_sha256": split.get("split_sha256"),
            "FINAL": split["FINAL"],
        }
    )


def verify_full_search_v3_private_manifest(
    manifest: Mapping[str, Any],
    *,
    records: Sequence[ValidatedSciVerV3Record] | None = None,
    source_path: str | Path | None = None,
    config: FullSearchV3Config | None = None,
) -> None:
    """Verify trusted private state, optionally against the checked source."""

    active_config = config or canonical_full_search_v3_config()
    if not isinstance(manifest, Mapping):
        raise FullSearchV3PreparationError("private V3 manifest must be an object")
    expected = {
        "schema_version",
        "artifact_type",
        "protocol_id",
        "source_dataset_sha256",
        "source_record_count",
        "source_paper_count",
        "split_seed",
        "search_size",
        "final_size",
        "config_sha256",
        "sample_identity_version",
        "paper_identity_version",
        "split_schema_version",
        "split_algorithm",
        "preparation_identity_schema_version",
        "preparation_identity_sha256",
        "split",
        "final_membership_commitment",
        "private_manifest_sha256",
    }
    _require_exact_fields(manifest, expected, "private V3 manifest")
    locked = {
        "schema_version": FULL_SEARCH_V3_PRIVATE_MANIFEST_SCHEMA,
        "artifact_type": "trusted_private_split_manifest",
        "protocol_id": FULL_SEARCH_V3_PROTOCOL_ID,
        "config_sha256": active_config.sha256(),
        "sample_identity_version": FULL_SEARCH_V3_SAMPLE_IDENTITY_VERSION,
        "paper_identity_version": FULL_SEARCH_V3_PAPER_IDENTITY_VERSION,
        "split_schema_version": FULL_SEARCH_V3_SPLIT_SCHEMA_VERSION,
        "split_algorithm": FULL_SEARCH_V3_ALLOCATION_ALGORITHM,
        "preparation_identity_schema_version": (
            FULL_SEARCH_V3_PREPARATION_IDENTITY_SCHEMA
        ),
    }
    for field, expected_value in locked.items():
        if manifest[field] != expected_value:
            raise FullSearchV3PreparationError(
                f"private V3 manifest has invalid {field}"
            )
    _require_source_digest(manifest["source_dataset_sha256"])
    for field, expected_value in (
        ("split_seed", active_config.split_seed),
        ("search_size", active_config.search_size),
        ("final_size", active_config.final_size),
    ):
        if type(manifest[field]) is not int or manifest[field] != expected_value:
            raise FullSearchV3PreparationError(
                f"private V3 manifest has invalid {field}"
            )
    expected_preparation_identity = _preparation_identity_sha256(
        source_dataset_sha256=manifest["source_dataset_sha256"],
        config=active_config,
    )
    if manifest["preparation_identity_sha256"] != expected_preparation_identity:
        raise FullSearchV3PreparationError(
            "private V3 manifest preparation identity is invalid"
        )
    _verify_private_split_shape(manifest["split"], active_config)
    if manifest["source_record_count"] != manifest["split"]["eligible_sample_count"]:
        raise FullSearchV3PreparationError(
            "private V3 manifest source record count is invalid"
        )
    if manifest["source_paper_count"] != manifest["split"]["eligible_paper_count"]:
        raise FullSearchV3PreparationError(
            "private V3 manifest source paper count is invalid"
        )
    if manifest["final_membership_commitment"] != compute_full_search_v3_final_commitment(
        manifest["split"]
    ):
        raise FullSearchV3PreparationError(
            "private V3 manifest FINAL commitment is invalid"
        )
    expected_hash = _sha256_json(dict(manifest, private_manifest_sha256=""))
    if manifest["private_manifest_sha256"] != expected_hash:
        raise FullSearchV3PreparationError("private V3 manifest hash is invalid")
    if records is not None:
        verify_full_search_v3_split(manifest["split"], records, config=active_config)
    if (
        source_path is not None
        and manifest["source_dataset_sha256"]
        != source_dataset_sha256(source_path)
    ):
        raise FullSearchV3PreparationError("private V3 manifest source dataset hash differs")


def derive_full_search_v3_search_safe_manifest(
    private_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the only V3 manifest that SEARCH-facing code may load."""

    verify_full_search_v3_private_manifest(private_manifest)
    split = private_manifest["split"]
    search_pool = split["SEARCH"]
    manifest = {
        "schema_version": FULL_SEARCH_V3_SEARCH_SAFE_MANIFEST_SCHEMA,
        "artifact_type": "search_safe_manifest",
        "protocol_id": FULL_SEARCH_V3_PROTOCOL_ID,
        "split_seed": split["split_seed"],
        "source_dataset_sha256": private_manifest["source_dataset_sha256"],
        "config_sha256": private_manifest["config_sha256"],
        "sample_identity_version": private_manifest["sample_identity_version"],
        "paper_identity_version": private_manifest["paper_identity_version"],
        "split_schema_version": private_manifest["split_schema_version"],
        "split_algorithm": private_manifest["split_algorithm"],
        "preparation_identity_schema_version": private_manifest[
            "preparation_identity_schema_version"
        ],
        "preparation_identity_sha256": private_manifest[
            "preparation_identity_sha256"
        ],
        "split_sha256": split["split_sha256"],
        "search_materialization_schema_version": FULL_SEARCH_V3_SEARCH_DATASET_SCHEMA,
        "search_membership_sha256": _sha256_json(search_pool),
        "SEARCH": dict(search_pool),
        "final_membership_commitment": private_manifest["final_membership_commitment"],
        "search_safe_manifest_sha256": "",
    }
    manifest["search_safe_manifest_sha256"] = _sha256_json(
        dict(manifest, search_safe_manifest_sha256="")
    )
    verify_full_search_v3_search_safe_manifest(manifest)
    return manifest


def verify_full_search_v3_search_safe_manifest(
    manifest: Mapping[str, Any],
    private_manifest: Mapping[str, Any] | None = None,
) -> None:
    """Verify a SEARCH-safe artifact without returning private FINAL members."""

    if not isinstance(manifest, Mapping):
        raise FullSearchV3PreparationError("SEARCH-safe V3 manifest must be an object")
    expected = {
        "schema_version",
        "artifact_type",
        "protocol_id",
        "split_seed",
        "source_dataset_sha256",
        "config_sha256",
        "sample_identity_version",
        "paper_identity_version",
        "split_schema_version",
        "split_algorithm",
        "preparation_identity_schema_version",
        "preparation_identity_sha256",
        "split_sha256",
        "search_materialization_schema_version",
        "search_membership_sha256",
        "SEARCH",
        "final_membership_commitment",
        "search_safe_manifest_sha256",
    }
    _require_exact_fields(manifest, expected, "SEARCH-safe V3 manifest")
    if manifest["schema_version"] != FULL_SEARCH_V3_SEARCH_SAFE_MANIFEST_SCHEMA:
        raise FullSearchV3PreparationError("unsupported SEARCH-safe V3 manifest schema")
    if manifest["artifact_type"] != "search_safe_manifest":
        raise FullSearchV3PreparationError("SEARCH-safe V3 manifest artifact type is invalid")
    if manifest["protocol_id"] != FULL_SEARCH_V3_PROTOCOL_ID:
        raise FullSearchV3PreparationError("SEARCH-safe V3 manifest protocol is invalid")
    if (
        manifest["search_materialization_schema_version"]
        != FULL_SEARCH_V3_SEARCH_DATASET_SCHEMA
    ):
        raise FullSearchV3PreparationError("SEARCH-safe V3 dataset schema is invalid")
    if (
        manifest["preparation_identity_schema_version"]
        != FULL_SEARCH_V3_PREPARATION_IDENTITY_SCHEMA
    ):
        raise FullSearchV3PreparationError(
            "SEARCH-safe V3 preparation identity schema is invalid"
        )
    _require_source_digest(manifest["source_dataset_sha256"])
    _require_sha256(
        manifest["preparation_identity_sha256"],
        "preparation identity SHA-256",
    )
    _verify_search_pool(manifest["SEARCH"])
    if manifest["search_membership_sha256"] != _sha256_json(manifest["SEARCH"]):
        raise FullSearchV3PreparationError("SEARCH-safe V3 membership hash is invalid")
    _require_sha256(manifest["final_membership_commitment"], "FINAL commitment")
    if manifest["search_safe_manifest_sha256"] != _sha256_json(
        dict(manifest, search_safe_manifest_sha256="")
    ):
        raise FullSearchV3PreparationError("SEARCH-safe V3 manifest hash is invalid")
    if private_manifest is not None:
        verify_full_search_v3_private_manifest(private_manifest)
        expected_safe = derive_full_search_v3_search_safe_manifest(private_manifest)
        if manifest != expected_safe:
            raise FullSearchV3PreparationError(
                "SEARCH-safe V3 manifest differs from trusted private state"
            )


def materialize_full_search_v3_search_records(
    records: Sequence[ValidatedSciVerV3Record],
    private_manifest: Mapping[str, Any],
    *,
    source_path: str | Path,
) -> list[dict[str, Any]]:
    """Build evaluator-side SEARCH records in immutable manifest order."""

    verify_full_search_v3_private_manifest(
        private_manifest,
        records=records,
        source_path=source_path,
    )
    adapted = get_dataset_adapter("SciVer").load(source_path)
    if len(adapted) != len(records):
        raise FullSearchV3PreparationError(
            "V3 source normalization count differs from validated source"
        )
    by_sample_id: dict[str, dict[str, Any]] = {}
    for checked, normalized in zip(records, adapted):
        materialized = dict(normalized.record)
        materialized["sample_id"] = checked.sample_id
        if checked.sample_id in by_sample_id:
            raise FullSearchV3PreparationError("V3 source produced duplicate sample IDs")
        by_sample_id[checked.sample_id] = materialized
    search_ids = private_manifest["split"]["SEARCH"]["sample_ids"]
    try:
        result = [by_sample_id[sample_id] for sample_id in search_ids]
    except KeyError as exc:
        raise FullSearchV3PreparationError(
            "V3 SEARCH membership references an absent normalized source record"
        ) from exc
    safe = derive_full_search_v3_search_safe_manifest(private_manifest)
    verify_full_search_v3_search_dataset(result, safe)
    return result


def verify_full_search_v3_search_dataset(
    records: Sequence[Mapping[str, Any]], search_safe_manifest: Mapping[str, Any]
) -> None:
    """Verify ordered SEARCH materialization against the safe membership only."""

    verify_full_search_v3_search_safe_manifest(search_safe_manifest)
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise FullSearchV3PreparationError("V3 SEARCH dataset must be a list of records")
    expected_ids = tuple(search_safe_manifest["SEARCH"]["sample_ids"])
    if len(records) != len(expected_ids):
        raise FullSearchV3PreparationError("V3 SEARCH dataset count is invalid")
    observed_ids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise FullSearchV3PreparationError(
                f"V3 SEARCH dataset record {index} must be an object"
            )
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise FullSearchV3PreparationError(
                f"V3 SEARCH dataset record {index} has an invalid sample ID"
            )
        observed_ids.append(sample_id)
    if len(observed_ids) != len(set(observed_ids)):
        raise FullSearchV3PreparationError("V3 SEARCH dataset has duplicate sample IDs")
    if tuple(observed_ids) != expected_ids:
        raise FullSearchV3PreparationError(
            "V3 SEARCH dataset order does not match immutable membership"
        )


def save_full_search_v3_private_manifest(
    path: str | Path, manifest: Mapping[str, Any]
) -> Path:
    verify_full_search_v3_private_manifest(manifest)
    return _atomic_create_or_verify_json(
        Path(path), dict(manifest), load_trusted_full_search_v3_private_manifest
    )


def load_trusted_full_search_v3_private_manifest(path: str | Path) -> dict[str, Any]:
    payload = _load_json_object(Path(path), "private V3 manifest")
    verify_full_search_v3_private_manifest(payload)
    return payload


def save_full_search_v3_search_safe_manifest(
    path: str | Path, manifest: Mapping[str, Any]
) -> Path:
    verify_full_search_v3_search_safe_manifest(manifest)
    return _atomic_create_or_verify_json(
        Path(path), dict(manifest), load_full_search_v3_search_safe_manifest
    )


def load_full_search_v3_search_safe_manifest(path: str | Path) -> dict[str, Any]:
    payload = _load_json_object(Path(path), "SEARCH-safe V3 manifest")
    verify_full_search_v3_search_safe_manifest(payload)
    return payload


def save_full_search_v3_search_dataset(
    path: str | Path, records: Sequence[Mapping[str, Any]]
) -> Path:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise FullSearchV3PreparationError("V3 SEARCH dataset must be a list")
    payload = [dict(record) for record in records]
    return _atomic_create_or_verify_json(
        Path(path), payload, load_full_search_v3_search_dataset
    )


def load_full_search_v3_search_dataset(path: str | Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullSearchV3PreparationError(
            "V3 SEARCH dataset must be readable valid JSON"
        ) from exc
    if not isinstance(payload, list) or any(
        not isinstance(row, dict) for row in payload
    ):
        raise FullSearchV3PreparationError(
            "V3 SEARCH dataset must be a JSON list of objects"
        )
    return payload


def _load_source_records(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullSearchV3PreparationError(
            "V3 source dataset must be readable valid JSON"
        ) from exc
    if not isinstance(payload, list) or any(
        not isinstance(record, Mapping) for record in payload
    ):
        raise FullSearchV3PreparationError(
            "V3 source dataset must be a JSON list of objects"
        )
    return payload


def _preparation_identity_sha256(
    *,
    source_dataset_sha256: str,
    config: FullSearchV3Config,
) -> str:
    """Bind all experiment-defining preparation inputs without timestamps."""

    return _sha256_json(
        {
            "schema_version": FULL_SEARCH_V3_PREPARATION_IDENTITY_SCHEMA,
            "protocol_id": FULL_SEARCH_V3_PROTOCOL_ID,
            "source_dataset_sha256": source_dataset_sha256,
            "config_sha256": config.sha256(),
            "sample_identity_version": FULL_SEARCH_V3_SAMPLE_IDENTITY_VERSION,
            "paper_identity_version": FULL_SEARCH_V3_PAPER_IDENTITY_VERSION,
            "split_schema_version": FULL_SEARCH_V3_SPLIT_SCHEMA_VERSION,
            "split_algorithm": FULL_SEARCH_V3_ALLOCATION_ALGORITHM,
            "split_seed": config.split_seed,
            "search_size": config.search_size,
            "final_size": config.final_size,
        }
    )


def _verify_private_split_shape(
    split: Mapping[str, Any], config: FullSearchV3Config
) -> None:
    if not isinstance(split, Mapping):
        raise FullSearchV3PreparationError("private V3 manifest split must be an object")
    locked = {
        "schema_version": FULL_SEARCH_V3_SPLIT_SCHEMA_VERSION,
        "protocol_id": FULL_SEARCH_V3_PROTOCOL_ID,
        "split_seed": config.split_seed,
        "config_sha256": config.sha256(),
        "sample_identity_version": FULL_SEARCH_V3_SAMPLE_IDENTITY_VERSION,
        "paper_identity_version": FULL_SEARCH_V3_PAPER_IDENTITY_VERSION,
        "allocation_algorithm": FULL_SEARCH_V3_ALLOCATION_ALGORITHM,
    }
    for field, expected in locked.items():
        if split.get(field) != expected:
            raise FullSearchV3PreparationError(f"private V3 manifest split has invalid {field}")
    _verify_search_pool(split.get("SEARCH"))
    final_pool = split.get("FINAL")
    if not isinstance(final_pool, Mapping):
        raise FullSearchV3PreparationError("private V3 manifest split has no FINAL pool")
    _verify_pool(final_pool, config.final_size, "FINAL")
    if set(split["SEARCH"]["sample_ids"]) & set(final_pool["sample_ids"]):
        raise FullSearchV3PreparationError("private V3 manifest split sample IDs overlap")
    if set(split["SEARCH"]["paper_identities"]) & set(final_pool["paper_identities"]):
        raise FullSearchV3PreparationError("private V3 manifest split paper identities overlap")


def _verify_search_pool(pool: Any) -> None:
    _verify_pool(pool, canonical_full_search_v3_config().search_size, "SEARCH")


def _verify_pool(pool: Any, expected_count: int, pool_name: str) -> None:
    if not isinstance(pool, Mapping):
        raise FullSearchV3PreparationError(f"V3 {pool_name} pool must be an object")
    expected = {"sample_ids", "paper_identities", "sample_count", "paper_count"}
    _require_exact_fields(pool, expected, f"V3 {pool_name} pool")
    sample_ids = pool["sample_ids"]
    paper_ids = pool["paper_identities"]
    if not isinstance(sample_ids, list) or not isinstance(paper_ids, list):
        raise FullSearchV3PreparationError(f"V3 {pool_name} membership must use lists")
    if any(not isinstance(value, str) or not value for value in sample_ids + paper_ids):
        raise FullSearchV3PreparationError(f"V3 {pool_name} membership must use non-empty text")
    if len(sample_ids) != len(set(sample_ids)) or len(paper_ids) != len(set(paper_ids)):
        raise FullSearchV3PreparationError(f"V3 {pool_name} membership contains duplicates")
    if pool["sample_count"] != expected_count or len(sample_ids) != expected_count:
        raise FullSearchV3PreparationError(f"V3 {pool_name} count is invalid")
    if pool["paper_count"] != len(paper_ids):
        raise FullSearchV3PreparationError(f"V3 {pool_name} paper count is invalid")


def _require_source_digest(value: Any) -> None:
    _require_sha256(value, "source dataset SHA-256")


def _require_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise FullSearchV3PreparationError(f"{field} must be a lowercase SHA-256")


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing: {', '.join(missing)}")
    if unexpected:
        details.append(f"unexpected: {', '.join(unexpected)}")
    raise FullSearchV3PreparationError(f"{context} fields are invalid ({'; '.join(details)})")


def _atomic_create_or_verify_json(
    path: Path,
    value: Any,
    loader: Callable[[Path], Any],
) -> Path:
    encoded = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = loader(path)
        if _json_bytes(existing) != encoded:
            raise FullSearchV3PreparationError(
                f"refusing to replace a different immutable V3 artifact: {path.name}"
            )
        return path
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError:
            return _atomic_create_or_verify_json(path, value, loader)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _load_json_object(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullSearchV3PreparationError(
            f"{context} must be readable valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise FullSearchV3PreparationError(f"{context} JSON must be an object")
    return payload


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "FULL_SEARCH_V3_FINAL_COMMITMENT_SCHEMA",
    "FULL_SEARCH_V3_PRIVATE_MANIFEST_SCHEMA",
    "FULL_SEARCH_V3_PREPARATION_IDENTITY_SCHEMA",
    "FULL_SEARCH_V3_SEARCH_DATASET_SCHEMA",
    "FULL_SEARCH_V3_SEARCH_SAFE_MANIFEST_SCHEMA",
    "FullSearchV3PreparationArtifacts",
    "FullSearchV3PreparationError",
    "build_full_search_v3_private_manifest",
    "compute_full_search_v3_final_commitment",
    "derive_full_search_v3_search_safe_manifest",
    "load_full_search_v3_search_dataset",
    "load_full_search_v3_search_safe_manifest",
    "load_trusted_full_search_v3_private_manifest",
    "materialize_full_search_v3_search_records",
    "prepare_full_search_v3",
    "save_full_search_v3_private_manifest",
    "save_full_search_v3_search_dataset",
    "save_full_search_v3_search_safe_manifest",
    "source_dataset_sha256",
    "verify_full_search_v3_private_manifest",
    "verify_full_search_v3_search_dataset",
    "verify_full_search_v3_search_safe_manifest",
]
