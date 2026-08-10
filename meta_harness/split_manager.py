"""Deterministic, leakage-safe paper-group splits for normalized SciVer data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from meta_harness.config import MetaHarnessConfig


SPLIT_NAMES = ("search", "validation", "final_test")
SPLIT_ALGORITHM = "sha256_seeded_paper_groups_v1"


class SplitManagerError(ValueError):
    """Raised when paper identity or a split manifest is invalid."""


def build_split_manifest(
    records: Sequence[Mapping[str, Any]],
    config: MetaHarnessConfig | None = None,
) -> dict[str, Any]:
    """Build a deterministic manifest without exposing paper file paths."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise SplitManagerError("records must be a sequence of mappings")
    resolved_config = config or MetaHarnessConfig()
    if not isinstance(resolved_config, MetaHarnessConfig):
        raise SplitManagerError("config must be a MetaHarnessConfig")
    if not records:
        raise SplitManagerError("at least one SciVer record is required")

    groups: dict[str, dict[str, Any]] = {}
    seen_samples: set[str] = set()
    dataset_entries: list[dict[str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise SplitManagerError(f"record at index {index} must be a mapping")
        if "sample_id" not in record:
            raise SplitManagerError(
                f"record at index {index} requires a stable sample_id"
            )
        sample_marker = _json_identity(record["sample_id"], "sample_id", index)
        if sample_marker in seen_samples:
            raise SplitManagerError(f"duplicate sample_id at record index {index}")
        seen_samples.add(sample_marker)

        group_id, identity_source = _paper_group(record, index)
        group = groups.setdefault(
            group_id,
            {
                "group_id": group_id,
                "identity_source": identity_source,
                "samples": [],
            },
        )
        if group["identity_source"] != identity_source:
            raise SplitManagerError("paper identity source collision")
        group["samples"].append(
            {
                "sample_id": record["sample_id"],
                "sample_marker": sample_marker,
            }
        )
        dataset_entries.append(
            {
                "sample": sample_marker,
                "paper_group": group_id,
                "record_sha256": _sha256_json(record),
            }
        )

    if len(groups) < len(SPLIT_NAMES):
        raise SplitManagerError(
            "at least three distinct paper groups are required for non-empty "
            "search, validation, and final_test splits"
        )

    ordered_group_ids = sorted(
        groups,
        key=lambda group_id: _seeded_order_key(
            resolved_config.seed,
            group_id,
        ),
    )
    group_counts = _allocate_group_counts(
        len(ordered_group_ids),
        resolved_config.split_ratios.as_dict(),
    )

    splits: dict[str, dict[str, Any]] = {}
    cursor = 0
    total_samples = len(records)
    for split_name in SPLIT_NAMES:
        split_group_ids = ordered_group_ids[
            cursor : cursor + group_counts[split_name]
        ]
        cursor += group_counts[split_name]
        samples = sorted(
            (
                sample
                for group_id in split_group_ids
                for sample in groups[group_id]["samples"]
            ),
            key=lambda sample: sample["sample_marker"],
        )
        sample_ids = [sample["sample_id"] for sample in samples]
        identity_sources = {
            "paper_id": sum(
                groups[group_id]["identity_source"] == "paper_id"
                for group_id in split_group_ids
            ),
            "paper_sha256": sum(
                groups[group_id]["identity_source"] == "paper_sha256"
                for group_id in split_group_ids
            ),
        }
        splits[split_name] = {
            "sample_ids": sample_ids,
            "paper_group_ids": split_group_ids,
            "sample_count": len(sample_ids),
            "paper_count": len(split_group_ids),
            "achieved_sample_ratio": len(sample_ids) / total_samples,
            "paper_identity_sources": identity_sources,
        }

    dataset_payload = {
        "samples": sorted(
            dataset_entries,
            key=lambda entry: (entry["sample"], entry["paper_group"]),
        )
    }
    dataset_sha256 = _sha256_json(dataset_payload)
    manifest_without_hash = {
        "schema_version": 1,
        "algorithm": SPLIT_ALGORITHM,
        "seed": resolved_config.seed,
        "configured_ratios": resolved_config.split_ratios.as_dict(),
        "dataset_sha256": dataset_sha256,
        "config_sha256": resolved_config.sha256(),
        "total_samples": total_samples,
        "total_papers": len(groups),
        "splits": splits,
    }
    manifest = {
        **manifest_without_hash,
        "split_sha256": _sha256_json(manifest_without_hash),
    }
    verify_split_manifest(manifest)
    return manifest


create_split_manifest = build_split_manifest


def verify_split_manifest(manifest: Mapping[str, Any]) -> None:
    """Verify hashes, counts, and pairwise paper/sample disjointness."""

    if not isinstance(manifest, Mapping):
        raise SplitManagerError("split manifest must be a mapping")
    required = {
        "schema_version",
        "algorithm",
        "seed",
        "configured_ratios",
        "dataset_sha256",
        "config_sha256",
        "total_samples",
        "total_papers",
        "splits",
        "split_sha256",
    }
    missing = sorted(required - set(manifest))
    extra = sorted(set(manifest) - required, key=str)
    if missing or extra:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(map(str, extra)))
        raise SplitManagerError(
            "split manifest fields are invalid (" + "; ".join(details) + ")"
        )
    if manifest["schema_version"] != 1:
        raise SplitManagerError("unsupported split manifest schema_version")
    if manifest["algorithm"] != SPLIT_ALGORITHM:
        raise SplitManagerError("unsupported split algorithm")
    for hash_field in ("dataset_sha256", "config_sha256", "split_sha256"):
        value = manifest[hash_field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise SplitManagerError(f"{hash_field} must be a hexadecimal SHA-256")

    splits = manifest["splits"]
    if not isinstance(splits, Mapping) or set(splits) != set(SPLIT_NAMES):
        raise SplitManagerError(
            "manifest splits require exactly search, validation, and final_test"
        )

    seen_samples: set[str] = set()
    seen_groups: set[str] = set()
    sample_total = 0
    paper_total = 0
    for split_name in SPLIT_NAMES:
        split = splits[split_name]
        if not isinstance(split, Mapping):
            raise SplitManagerError(f"{split_name} split must be a mapping")
        sample_ids = split.get("sample_ids")
        group_ids = split.get("paper_group_ids")
        if (
            isinstance(sample_ids, (str, bytes))
            or not isinstance(sample_ids, Sequence)
            or isinstance(group_ids, (str, bytes))
            or not isinstance(group_ids, Sequence)
        ):
            raise SplitManagerError(
                f"{split_name} split requires sample_ids and paper_group_ids"
            )
        sample_markers = {
            _json_identity(value, f"{split_name}.sample_ids", index)
            for index, value in enumerate(sample_ids)
        }
        if len(sample_markers) != len(sample_ids):
            raise SplitManagerError(f"{split_name} contains duplicate sample IDs")
        if any(not isinstance(value, str) or not value for value in group_ids):
            raise SplitManagerError(
                f"{split_name} paper_group_ids must be non-empty strings"
            )
        group_markers = set(group_ids)
        if len(group_markers) != len(group_ids):
            raise SplitManagerError(f"{split_name} contains duplicate paper groups")
        if seen_samples.intersection(sample_markers):
            raise SplitManagerError("sample IDs overlap across splits")
        if seen_groups.intersection(group_markers):
            raise SplitManagerError("paper groups overlap across splits")
        seen_samples.update(sample_markers)
        seen_groups.update(group_markers)
        if split.get("sample_count") != len(sample_ids):
            raise SplitManagerError(f"{split_name} sample_count is inconsistent")
        if split.get("paper_count") != len(group_ids):
            raise SplitManagerError(f"{split_name} paper_count is inconsistent")
        sample_total += len(sample_ids)
        paper_total += len(group_ids)

    if manifest["total_samples"] != sample_total:
        raise SplitManagerError("manifest total_samples is inconsistent")
    if manifest["total_papers"] != paper_total:
        raise SplitManagerError("manifest total_papers is inconsistent")

    without_hash = {
        key: value for key, value in manifest.items() if key != "split_sha256"
    }
    if _sha256_json(without_hash) != manifest["split_sha256"]:
        raise SplitManagerError("split manifest hash mismatch")


def proposer_split_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return aggregate search/validation metadata with no final-test data."""

    verify_split_manifest(manifest)
    safe_splits = {}
    for split_name in ("search", "validation"):
        split = manifest["splits"][split_name]
        safe_splits[split_name] = {
            "sample_count": split["sample_count"],
            "paper_count": split["paper_count"],
        }
    return {
        "schema_version": manifest["schema_version"],
        "algorithm": manifest["algorithm"],
        "splits": safe_splits,
    }


def save_split_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Atomically create an immutable manifest or accept an identical one."""

    verify_split_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            manifest,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")

    if destination.exists():
        existing = load_split_manifest(destination)
        if existing["split_sha256"] != manifest["split_sha256"]:
            raise SplitManagerError(
                "refusing to replace a different split manifest"
            )
        return destination

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = load_split_manifest(destination)
            if existing["split_sha256"] != manifest["split_sha256"]:
                raise SplitManagerError(
                    "refusing to replace a concurrently created split manifest"
                )
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SplitManagerError("split manifest must be readable valid JSON") from exc
    if not isinstance(payload, dict):
        raise SplitManagerError("split manifest JSON must be an object")
    verify_split_manifest(payload)
    return payload


def _paper_group(
    record: Mapping[str, Any],
    index: int,
) -> tuple[str, str]:
    paper_id = record.get("paper_id")
    if paper_id is not None:
        if isinstance(paper_id, bool) or not isinstance(paper_id, (str, int)):
            raise SplitManagerError(
                f"record at index {index} paper_id must be non-empty text or an integer"
            )
        if isinstance(paper_id, str) and not paper_id.strip():
            raise SplitManagerError(
                f"record at index {index} paper_id must not be empty"
            )
        encoded_id = _json_identity(paper_id, "paper_id", index)
        digest = hashlib.sha256(encoded_id.encode("utf-8")).hexdigest()
        return f"paper_id:{digest}", "paper_id"

    paper_path = record.get("paper_path")
    if not isinstance(paper_path, str) or not paper_path.strip():
        raise SplitManagerError(
            f"record at index {index} requires paper_id or a local paper_path; "
            "sample_id fallback is forbidden"
        )
    path = Path(paper_path)
    try:
        paper_bytes = path.read_bytes()
        decoded = paper_bytes.decode("utf-8")
        paper = json.loads(decoded)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SplitManagerError(
            f"record at index {index} paper_path must reference readable valid JSON"
        ) from exc
    if not isinstance(paper, Mapping):
        raise SplitManagerError(
            f"record at index {index} paper JSON must contain an object"
        )
    digest = hashlib.sha256(paper_bytes).hexdigest()
    return f"paper_sha256:{digest}", "paper_sha256"


def paper_group_identity(record: Mapping[str, Any]) -> str:
    """Return the verified paper identity used by split construction."""

    if not isinstance(record, Mapping):
        raise SplitManagerError("record must be a mapping")
    return _paper_group(record, 0)[0]


def _allocate_group_counts(
    group_total: int,
    ratios: Mapping[str, float],
) -> dict[str, int]:
    raw = {name: ratios[name] * group_total for name in SPLIT_NAMES}
    if group_total < len(SPLIT_NAMES):
        raise SplitManagerError(
            "configured ratios cannot produce three non-empty paper splits"
        )
    counts = {name: 1 for name in SPLIT_NAMES}
    for _ in range(group_total - len(SPLIT_NAMES)):
        selected = min(
            SPLIT_NAMES,
            key=lambda name: (
                -(raw[name] - counts[name]),
                SPLIT_NAMES.index(name),
            ),
        )
        counts[selected] += 1
    return counts


def _seeded_order_key(seed: int, group_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{group_id}".encode("utf-8")).hexdigest()
    return digest, group_id


def _json_identity(value: Any, field: str, index: int) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise SplitManagerError(
            f"record at index {index} {field} must be JSON serializable"
        ) from exc


def _sha256_json(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SplitManagerError("split data must be JSON serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SPLIT_ALGORITHM",
    "SPLIT_NAMES",
    "SplitManagerError",
    "build_split_manifest",
    "create_split_manifest",
    "load_split_manifest",
    "paper_group_identity",
    "proposer_split_summary",
    "save_split_manifest",
    "verify_split_manifest",
]
