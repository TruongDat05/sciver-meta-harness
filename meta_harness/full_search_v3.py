"""Strict data identity and paper-exclusive splitting for full SEARCH V3.

This module deliberately does not reuse the staged split result shapes.  It
accepts only released SciVer source records, derives content-addressed
identities, and creates the two V3 membership pools needed by later
preparation code.
"""

from __future__ import annotations

from array import array
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from meta_harness.config import (
    FULL_SEARCH_V3_PROTOCOL_ID,
    FullSearchV3Config,
    canonical_full_search_v3_config,
)


FULL_SEARCH_V3_RECORD_SCHEMA_VERSION = "sciver_source_record_v1"
FULL_SEARCH_V3_SAMPLE_IDENTITY_VERSION = "sciver_sample_content_v1"
FULL_SEARCH_V3_PAPER_IDENTITY_VERSION = "sciver_paper_reference_v1"
FULL_SEARCH_V3_SPLIT_SCHEMA_VERSION = "sciver_paper_split_v1"
FULL_SEARCH_V3_ALLOCATION_ALGORITHM = "seeded_exact_paper_dp_v1"
FULL_SEARCH_V3_POOLS = ("SEARCH", "FINAL")

_COMMON_FIELDS = frozenset(
    {
        "request_id",
        "claim",
        "claim_type",
        "paper_path",
        "section",
        "label",
        "origin_statement",
        "perturbed_explanation",
        "perturbed_statement",
    }
)
_OPTIONAL_COMMON_FIELDS = frozenset({"paperid"})
_ONE_EVIDENCE_FIELDS = frozenset({"type", "item", "image_path"})
_TWO_EVIDENCE_FIELDS = frozenset(
    {"item1_type", "item1", "item1_path", "item2_type", "item2", "item2_path"}
)
_ONE_EVIDENCE_TYPES = frozenset({"direct", "analytical"})
_TWO_EVIDENCE_TYPES = frozenset({"parallel", "sequential"})
_EVIDENCE_TYPES = frozenset({"chart", "table"})


class FullSearchV3DataError(ValueError):
    """Raised when a SciVer source record is not safe for the V3 contract."""


class FullSearchV3SplitError(ValueError):
    """Raised when exact paper-exclusive membership cannot be established."""


@dataclass(frozen=True)
class ValidatedSciVerV3Record:
    """A strictly checked source record with immutable V3 identities."""

    sample_id: str
    paper_identity: str
    paper_identity_source: str
    record: Mapping[str, Any]


@dataclass(frozen=True)
class _PaperGroup:
    paper_identity: str
    sample_ids: tuple[str, ...]


def validate_sciver_full_search_v3_records(
    records: Sequence[Mapping[str, Any]], *, source_path: str | Path
) -> tuple[ValidatedSciVerV3Record, ...]:
    """Check released SciVer records and return their stable V3 identities.

    The source path anchors every local paper and evidence reference.  No
    permissive normalized-record path is accepted here because its generated
    row-based IDs are not valid V3 membership identities.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise FullSearchV3DataError("SciVer V3 source records must be a sequence")
    resolved_source = Path(source_path).expanduser().resolve()
    if not resolved_source.is_file():
        raise FullSearchV3DataError("SciVer V3 source path must name an existing file")

    validated: list[ValidatedSciVerV3Record] = []
    seen_sample_ids: dict[str, int] = {}
    for index, record in enumerate(records):
        checked = _validate_one_record(record, resolved_source, index)
        prior_index = seen_sample_ids.get(checked.sample_id)
        if prior_index is not None:
            raise FullSearchV3DataError(
                "SciVer V3 sample identity collision between source records "
                f"{prior_index} and {index}; refine immutable source fields"
            )
        seen_sample_ids[checked.sample_id] = index
        validated.append(checked)
    return tuple(validated)


def build_full_search_v3_split(
    records: Sequence[ValidatedSciVerV3Record],
    *,
    config: FullSearchV3Config | None = None,
) -> dict[str, Any]:
    """Allocate exact V3 pools by whole paper groups with seeded DP ties."""

    active_config = config or canonical_full_search_v3_config()
    _require_v3_config(active_config)
    sample_index = _validated_sample_index(records)
    groups = _paper_groups(sample_index)
    ordered_groups = tuple(
        sorted(
            groups,
            key=lambda group: _seeded_paper_key(
                active_config.split_seed, group.paper_identity
            ),
        )
    )
    search_group_indexes, final_group_indexes = _allocate_paper_groups(
        ordered_groups,
        search_size=active_config.search_size,
        final_size=active_config.final_size,
    )
    search_groups = tuple(ordered_groups[index] for index in search_group_indexes)
    final_groups = tuple(ordered_groups[index] for index in final_group_indexes)
    source_fingerprint = _source_fingerprint(sample_index)
    result = {
        "schema_version": FULL_SEARCH_V3_SPLIT_SCHEMA_VERSION,
        "protocol_id": active_config.protocol_id,
        "split_seed": active_config.split_seed,
        "config_sha256": active_config.sha256(),
        "sample_identity_version": FULL_SEARCH_V3_SAMPLE_IDENTITY_VERSION,
        "paper_identity_version": FULL_SEARCH_V3_PAPER_IDENTITY_VERSION,
        "allocation_algorithm": FULL_SEARCH_V3_ALLOCATION_ALGORITHM,
        "eligible_sample_count": len(sample_index),
        "eligible_paper_count": len(ordered_groups),
        "source_fingerprint": source_fingerprint,
        "SEARCH": _pool_payload(search_groups),
        "FINAL": _pool_payload(final_groups),
    }
    result["split_sha256"] = _split_fingerprint(result)
    verify_full_search_v3_split(result, records, config=active_config)
    return result


def verify_full_search_v3_split(
    result: Mapping[str, Any],
    records: Sequence[ValidatedSciVerV3Record],
    *,
    config: FullSearchV3Config | None = None,
) -> None:
    """Raise a clear error unless a result is an exact V3 paper split."""

    active_config = config or canonical_full_search_v3_config()
    _require_v3_config(active_config)
    if not isinstance(result, Mapping):
        raise FullSearchV3SplitError("V3 split result must be an object")
    expected_fields = {
        "schema_version",
        "protocol_id",
        "split_seed",
        "config_sha256",
        "sample_identity_version",
        "paper_identity_version",
        "allocation_algorithm",
        "eligible_sample_count",
        "eligible_paper_count",
        "source_fingerprint",
        "SEARCH",
        "FINAL",
        "split_sha256",
    }
    _require_exact_result_fields(result, expected_fields, "V3 split result")
    locked_values = {
        "schema_version": FULL_SEARCH_V3_SPLIT_SCHEMA_VERSION,
        "protocol_id": FULL_SEARCH_V3_PROTOCOL_ID,
        "split_seed": active_config.split_seed,
        "config_sha256": active_config.sha256(),
        "sample_identity_version": FULL_SEARCH_V3_SAMPLE_IDENTITY_VERSION,
        "paper_identity_version": FULL_SEARCH_V3_PAPER_IDENTITY_VERSION,
        "allocation_algorithm": FULL_SEARCH_V3_ALLOCATION_ALGORITHM,
    }
    for field, expected in locked_values.items():
        actual = result[field]
        if type(actual) is not type(expected) or actual != expected:
            raise FullSearchV3SplitError(f"V3 split result has invalid {field}")

    sample_index = _validated_sample_index(records)
    groups = _paper_groups(sample_index)
    ordered_groups = tuple(
        sorted(
            groups,
            key=lambda group: _seeded_paper_key(
                active_config.split_seed, group.paper_identity
            ),
        )
    )
    if result["eligible_sample_count"] != len(sample_index):
        raise FullSearchV3SplitError("V3 split eligible sample count does not match source")
    if result["eligible_paper_count"] != len(ordered_groups):
        raise FullSearchV3SplitError("V3 split eligible paper count does not match source")
    if result["source_fingerprint"] != _source_fingerprint(sample_index):
        raise FullSearchV3SplitError("V3 split source fingerprint does not match source")

    pool_memberships: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for pool_name, expected_size in (
        ("SEARCH", active_config.search_size),
        ("FINAL", active_config.final_size),
    ):
        pool = result[pool_name]
        if not isinstance(pool, Mapping):
            raise FullSearchV3SplitError(f"V3 {pool_name} pool must be an object")
        _require_exact_result_fields(
            pool,
            {"sample_ids", "paper_identities", "sample_count", "paper_count"},
            f"V3 {pool_name} pool",
        )
        sample_ids = _string_tuple(pool["sample_ids"], f"V3 {pool_name} sample IDs")
        paper_ids = _string_tuple(
            pool["paper_identities"], f"V3 {pool_name} paper identities"
        )
        if len(sample_ids) != len(set(sample_ids)):
            raise FullSearchV3SplitError(f"V3 {pool_name} sample IDs are not unique")
        if len(paper_ids) != len(set(paper_ids)):
            raise FullSearchV3SplitError(f"V3 {pool_name} paper identities are not unique")
        if pool["sample_count"] != expected_size or len(sample_ids) != expected_size:
            raise FullSearchV3SplitError(
                f"V3 {pool_name} must contain exactly {expected_size} samples"
            )
        if pool["paper_count"] != len(paper_ids):
            raise FullSearchV3SplitError(f"V3 {pool_name} paper count is invalid")
        if any(sample_id not in sample_index for sample_id in sample_ids):
            raise FullSearchV3SplitError(
                f"V3 {pool_name} references a sample absent from validated source"
            )
        if any(paper_id not in {group.paper_identity for group in groups} for paper_id in paper_ids):
            raise FullSearchV3SplitError(
                f"V3 {pool_name} references a paper absent from validated source"
            )
        expected_papers = tuple(
            group.paper_identity
            for group in ordered_groups
            if group.paper_identity in set(paper_ids)
        )
        if paper_ids != expected_papers:
            raise FullSearchV3SplitError(
                f"V3 {pool_name} paper ordering is not deterministic"
            )
        expected_samples = tuple(
            sample_id
            for group in ordered_groups
            if group.paper_identity in set(paper_ids)
            for sample_id in group.sample_ids
        )
        if sample_ids != expected_samples:
            raise FullSearchV3SplitError(
                f"V3 {pool_name} sample ordering or paper ownership is invalid"
            )
        pool_memberships[pool_name] = (sample_ids, paper_ids)

    search_samples, search_papers = pool_memberships["SEARCH"]
    final_samples, final_papers = pool_memberships["FINAL"]
    if set(search_samples) & set(final_samples):
        raise FullSearchV3SplitError("V3 SEARCH and FINAL sample IDs overlap")
    if set(search_papers) & set(final_papers):
        raise FullSearchV3SplitError("V3 SEARCH and FINAL paper identities overlap")
    if any(sample_index[sample_id].paper_identity not in set(search_papers) for sample_id in search_samples):
        raise FullSearchV3SplitError("V3 SEARCH sample has a mismatched paper identity")
    if any(sample_index[sample_id].paper_identity not in set(final_papers) for sample_id in final_samples):
        raise FullSearchV3SplitError("V3 FINAL sample has a mismatched paper identity")
    if result["split_sha256"] != _split_fingerprint(dict(result, split_sha256="")):
        raise FullSearchV3SplitError("V3 split fingerprint is invalid")


def _validate_one_record(
    record: Mapping[str, Any], source_path: Path, index: int
) -> ValidatedSciVerV3Record:
    context = f"SciVer V3 source record {index}"
    if not isinstance(record, Mapping):
        raise FullSearchV3DataError(f"{context} must be an object")
    claim_type = record.get("claim_type")
    if claim_type not in _ONE_EVIDENCE_TYPES | _TWO_EVIDENCE_TYPES:
        raise FullSearchV3DataError(
            f"{context} claim_type must be one of direct, analytical, parallel, sequential"
        )
    required_evidence = (
        _ONE_EVIDENCE_FIELDS if claim_type in _ONE_EVIDENCE_TYPES else _TWO_EVIDENCE_FIELDS
    )
    _require_source_fields(
        record,
        required=_COMMON_FIELDS | required_evidence,
        allowed=_COMMON_FIELDS | _OPTIONAL_COMMON_FIELDS | required_evidence,
        context=context,
    )
    _require_nonempty_text(record["claim"], f"{context} claim")
    _require_nonempty_text(record["paper_path"], f"{context} paper_path")
    _require_nonempty_text(record["origin_statement"], f"{context} origin_statement")
    _require_nonempty_text(
        record["perturbed_explanation"], f"{context} perturbed_explanation"
    )
    _require_nonempty_text(record["perturbed_statement"], f"{context} perturbed_statement")
    if isinstance(record["request_id"], bool) or not isinstance(record["request_id"], int):
        raise FullSearchV3DataError(f"{context} request_id must be an integer")
    if not isinstance(record["label"], bool):
        raise FullSearchV3DataError(f"{context} label must be a JSON boolean")
    sections = _section_list(record["section"], context)
    if "paperid" in record:
        _require_nonempty_text(record["paperid"], f"{context} paperid")

    paper_path = _resolve_local_reference(
        record["paper_path"], source_path, f"{context} paper_path"
    )
    paper_bytes, paper_document = _read_paper_document(paper_path, context)
    paper_identity, paper_identity_source = _paper_identity(
        record.get("paperid"), paper_path, paper_bytes
    )
    _check_sections_exist(paper_document, sections, context)

    evidence_descriptors: list[dict[str, Any]] = []
    if claim_type in _ONE_EVIDENCE_TYPES:
        evidence_descriptors.append(
            _check_evidence(
                paper_document,
                record["type"],
                record["item"],
                record["image_path"],
                source_path,
                f"{context} evidence",
            )
        )
    else:
        for number in (1, 2):
            evidence_descriptors.append(
                _check_evidence(
                    paper_document,
                    record[f"item{number}_type"],
                    record[f"item{number}"],
                    record[f"item{number}_path"],
                    source_path,
                    f"{context} evidence {number}",
                )
            )

    identity_payload = {
        "identity_version": FULL_SEARCH_V3_SAMPLE_IDENTITY_VERSION,
        "paper_identity": paper_identity,
        "request_id": record["request_id"],
        "claim_type": claim_type,
        "claim": record["claim"],
        "section": sections,
        "evidence": evidence_descriptors,
    }
    sample_id = (
        f"{FULL_SEARCH_V3_SAMPLE_IDENTITY_VERSION}:"
        f"{_canonical_sha256(identity_payload)}"
    )
    return ValidatedSciVerV3Record(
        sample_id=sample_id,
        paper_identity=paper_identity,
        paper_identity_source=paper_identity_source,
        record=MappingProxyType(_json_clone(record)),
    )


def _require_source_fields(
    record: Mapping[str, Any],
    *,
    required: frozenset[str],
    allowed: frozenset[str],
    context: str,
) -> None:
    actual = set(record)
    missing = sorted(required - actual)
    unexpected = sorted(actual - allowed)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected or hybrid: {', '.join(unexpected)}")
        raise FullSearchV3DataError(f"{context} has invalid fields ({'; '.join(details)})")


def _require_nonempty_text(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise FullSearchV3DataError(f"{field} must be non-empty text")


def _section_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise FullSearchV3DataError(f"{context} section must be a non-empty list")
    sections = list(value)
    for section in sections:
        _require_nonempty_text(section, f"{context} section entry")
    return sections


def _resolve_local_reference(value: str, source_path: Path, context: str) -> Path:
    if "://" in value or value.startswith("data:"):
        raise FullSearchV3DataError(f"{context} must be a local path")
    candidate = Path(value)
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend((source_path.parent / candidate, source_path.parent.parent / candidate))
        parts = candidate.parts
        if parts and parts[0] in {".", ""}:
            parts = parts[1:]
        if parts and parts[0].casefold() in {"sciver", source_path.parent.name.casefold()}:
            parts = parts[1:]
        if parts:
            candidates.append(source_path.parent.joinpath(*parts))
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved.is_file():
            return resolved
    raise FullSearchV3DataError(f"{context} does not resolve to a local file")


def _read_paper_document(path: Path, context: str) -> tuple[bytes, Mapping[str, Any]]:
    try:
        paper_bytes = path.read_bytes()
        document = json.loads(paper_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullSearchV3DataError(f"{context} must reference a valid JSON document") from exc
    if not isinstance(document, Mapping):
        raise FullSearchV3DataError(f"{context} paper JSON must be an object")
    return paper_bytes, document


def _paper_identity(
    source_paperid: Any, paper_path: Path, paper_bytes: bytes
) -> tuple[str, str]:
    # The released paperid is accepted only after its basename check.  The
    # canonical group identity stays content-addressed so a verified row and a
    # fallback row for the same JSON can never split that paper across pools.
    paper_identity = f"paper_sha256:{hashlib.sha256(paper_bytes).hexdigest()}"
    if isinstance(source_paperid, str) and source_paperid and paper_path.stem == source_paperid:
        return paper_identity, "verified_paperid"
    return paper_identity, "paper_sha256"


def _check_sections_exist(
    paper_document: Mapping[str, Any], sections: Sequence[str], context: str
) -> None:
    paper_sections = paper_document.get("sections")
    if isinstance(paper_sections, (str, bytes)) or not isinstance(paper_sections, Sequence):
        raise FullSearchV3DataError(f"{context} paper JSON has no valid sections list")
    available: set[str] = set()
    requested = set(sections)
    for section in paper_sections:
        if not isinstance(section, Mapping):
            raise FullSearchV3DataError(f"{context} paper JSON has a malformed section")
        section_id = section.get("section_id")
        _require_nonempty_text(section_id, f"{context} paper section_id")
        available.add(section_id)
    missing = sorted(requested - available)
    if missing:
        raise FullSearchV3DataError(
            f"{context} section references absent paper sections: {', '.join(missing)}"
        )


def _check_evidence(
    paper_document: Mapping[str, Any],
    evidence_type: Any,
    item: Any,
    evidence_path: Any,
    source_path: Path,
    context: str,
) -> dict[str, Any]:
    if evidence_type not in _EVIDENCE_TYPES:
        raise FullSearchV3DataError(f"{context} type must be chart or table")
    if isinstance(item, bool) or not isinstance(item, (int, str)):
        raise FullSearchV3DataError(f"{context} item must be an integer or string")
    _require_nonempty_text(evidence_path, f"{context} path")
    evidence_file = _resolve_local_reference(evidence_path, source_path, f"{context} path")
    catalog_name = "image_paths" if evidence_type == "chart" else "tables"
    catalog = paper_document.get(catalog_name)
    if not isinstance(catalog, Mapping):
        raise FullSearchV3DataError(f"{context} paper JSON has no valid {catalog_name}")
    entry = catalog.get(str(item))
    if not isinstance(entry, Mapping):
        raise FullSearchV3DataError(f"{context} item is absent from paper {catalog_name}")
    caption = entry.get("caption", entry.get("capture"))
    _require_nonempty_text(caption, f"{context} caption")
    try:
        content_sha256 = hashlib.sha256(evidence_file.read_bytes()).hexdigest()
    except OSError as exc:
        raise FullSearchV3DataError(f"{context} path cannot be read") from exc
    return {
        "type": evidence_type,
        "item": item,
        "content_sha256": content_sha256,
    }


def _validated_sample_index(
    records: Sequence[ValidatedSciVerV3Record],
) -> dict[str, ValidatedSciVerV3Record]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise FullSearchV3SplitError("V3 split source must be validated record sequence")
    result: dict[str, ValidatedSciVerV3Record] = {}
    for index, record in enumerate(records):
        if not isinstance(record, ValidatedSciVerV3Record):
            raise FullSearchV3SplitError(
                f"V3 split source record {index} was not strictly checked"
            )
        if record.sample_id in result:
            raise FullSearchV3SplitError("V3 split source contains duplicate sample IDs")
        result[record.sample_id] = record
    return result


def _paper_groups(
    sample_index: Mapping[str, ValidatedSciVerV3Record],
) -> tuple[_PaperGroup, ...]:
    grouped: dict[str, list[str]] = {}
    for sample_id, record in sample_index.items():
        grouped.setdefault(record.paper_identity, []).append(sample_id)
    return tuple(
        _PaperGroup(paper_identity, tuple(sorted(sample_ids)))
        for paper_identity, sample_ids in grouped.items()
    )


def _seeded_paper_key(seed: int, paper_identity: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{paper_identity}".encode("utf-8")).hexdigest()
    return digest, paper_identity


def _allocate_paper_groups(
    groups: Sequence[_PaperGroup], *, search_size: int, final_size: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    target_total = search_size + final_size
    width = search_size + 1
    state_count = (target_total + 1) * width
    predecessor = array("i", [-1]) * state_count
    predecessor_group = array("i", [-1]) * state_count
    choice = bytearray(state_count)
    reachable = [0] * (target_total + 1)
    reachable[0] = 1
    search_mask = (1 << width) - 1

    for group_index, group in enumerate(groups):
        size = len(group.sample_ids)
        if size > target_total:
            continue
        for total in range(target_total - size, -1, -1):
            source_bits = reachable[total]
            if not source_bits:
                continue
            destination_total = total + size
            destination_bits = reachable[destination_total]
            final_bits = source_bits & ~destination_bits
            if final_bits:
                _record_predecessors(
                    final_bits,
                    predecessor,
                    predecessor_group,
                    choice,
                    previous_total=total,
                    destination_total=destination_total,
                    width=width,
                    group_index=group_index,
                    assignment=2,
                )
                destination_bits |= final_bits
            search_bits = (source_bits << size) & search_mask & ~destination_bits
            if search_bits:
                _record_predecessors(
                    search_bits,
                    predecessor,
                    predecessor_group,
                    choice,
                    previous_total=total,
                    destination_total=destination_total,
                    width=width,
                    group_index=group_index,
                    assignment=1,
                )
                destination_bits |= search_bits
            reachable[destination_total] = destination_bits

    final_state = target_total * width + search_size
    if not (reachable[target_total] & (1 << search_size)):
        sizes = Counter(len(group.sample_ids) for group in groups)
        available = sum(size * count for size, count in sizes.items())
        raise FullSearchV3SplitError(
            "cannot allocate exact paper-exclusive V3 pools: "
            f"need SEARCH={search_size}, FINAL={final_size}, total={target_total}; "
            f"eligible_samples={available}, eligible_papers={len(groups)}, "
            f"paper_group_sizes={dict(sorted(sizes.items()))}"
        )

    search_indexes: list[int] = []
    final_indexes: list[int] = []
    current_state = final_state
    while current_state:
        group_index = predecessor_group[current_state]
        assignment = choice[current_state]
        previous_state = predecessor[current_state]
        if group_index < 0 or assignment not in {1, 2} or previous_state < 0:
            raise FullSearchV3SplitError("V3 exact allocator has an invalid DP path")
        if assignment == 1:
            search_indexes.append(group_index)
        else:
            final_indexes.append(group_index)
        current_state = previous_state
    return tuple(sorted(search_indexes)), tuple(sorted(final_indexes))


def _record_predecessors(
    bits: int,
    predecessor: array,
    predecessor_group: array,
    choice: bytearray,
    *,
    previous_total: int,
    destination_total: int,
    width: int,
    group_index: int,
    assignment: int,
) -> None:
    while bits:
        lowest_bit = bits & -bits
        destination_search = lowest_bit.bit_length() - 1
        if assignment == 1:
            previous_search = destination_search - (destination_total - previous_total)
        else:
            previous_search = destination_search
        state = destination_total * width + destination_search
        predecessor[state] = previous_total * width + previous_search
        predecessor_group[state] = group_index
        choice[state] = assignment
        bits ^= lowest_bit


def _pool_payload(groups: Sequence[_PaperGroup]) -> dict[str, Any]:
    sample_ids = [sample_id for group in groups for sample_id in group.sample_ids]
    paper_ids = [group.paper_identity for group in groups]
    return {
        "sample_ids": sample_ids,
        "paper_identities": paper_ids,
        "sample_count": len(sample_ids),
        "paper_count": len(paper_ids),
    }


def _source_fingerprint(sample_index: Mapping[str, ValidatedSciVerV3Record]) -> str:
    payload = [
        {"sample_id": sample_id, "paper_identity": record.paper_identity}
        for sample_id, record in sorted(sample_index.items())
    ]
    return _canonical_sha256(payload)


def _split_fingerprint(result: Mapping[str, Any]) -> str:
    payload = dict(result)
    payload["split_sha256"] = ""
    return _canonical_sha256(payload)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _json_clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        cloned = json.loads(
            json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        )
    except (TypeError, ValueError) as exc:
        raise FullSearchV3DataError("SciVer V3 source record must contain JSON values") from exc
    if not isinstance(cloned, dict):
        raise FullSearchV3DataError("SciVer V3 source record must be a JSON object")
    return cloned


def _require_v3_config(config: FullSearchV3Config) -> None:
    if not isinstance(config, FullSearchV3Config):
        raise FullSearchV3SplitError("V3 split requires FullSearchV3Config")
    if config.protocol_id != FULL_SEARCH_V3_PROTOCOL_ID:
        raise FullSearchV3SplitError("V3 split has an invalid protocol identity")


def _require_exact_result_fields(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise FullSearchV3SplitError(f"{context} fields are invalid ({'; '.join(details)})")


def _string_tuple(value: Any, context: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FullSearchV3SplitError(f"{context} must be a list")
    values = tuple(value)
    if any(not isinstance(item, str) or not item for item in values):
        raise FullSearchV3SplitError(f"{context} must contain non-empty text")
    return values


__all__ = [
    "FULL_SEARCH_V3_ALLOCATION_ALGORITHM",
    "FULL_SEARCH_V3_PAPER_IDENTITY_VERSION",
    "FULL_SEARCH_V3_POOLS",
    "FULL_SEARCH_V3_RECORD_SCHEMA_VERSION",
    "FULL_SEARCH_V3_SAMPLE_IDENTITY_VERSION",
    "FULL_SEARCH_V3_SPLIT_SCHEMA_VERSION",
    "FullSearchV3DataError",
    "FullSearchV3SplitError",
    "ValidatedSciVerV3Record",
    "build_full_search_v3_split",
    "validate_sciver_full_search_v3_records",
    "verify_full_search_v3_split",
]
