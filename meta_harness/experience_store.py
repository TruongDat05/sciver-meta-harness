"""Search-only experience artifacts with a separate proposer-safe view."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from difflib import unified_diff
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from meta_harness.candidate_store import CandidateStore
from meta_harness.schemas import canonical_json


EXPERIENCE_SCHEMA_VERSION = 1
_SENSITIVE = (
    re.compile(r"(?i)\bauthorization\s*:\s*[^\s,;]+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b(?:api[_ -]?key|api[_ -]?url)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)data:image/[^,;\s]+,[A-Za-z0-9+/=]+"),
    re.compile(r"\b[A-Za-z0-9+/]{128,}={0,2}\b"),
)


class ExperienceStoreError(ValueError):
    """Raised when a search trace is unsafe or inconsistent."""


class SearchExperienceStore:
    """Persist trusted traces and a label-safe directory for the proposer."""

    def __init__(
        self,
        repository_root: str | Path,
        run_id: str,
        candidate_store: CandidateStore,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.run_directory = candidate_store.run_directory
        self.trusted_directory = self.run_directory / "trusted_search_experience"
        self.proposer_directory = self.run_directory / "experience"
        self.candidate_store = candidate_store
        self.run_id = run_id

    def persist_candidate(
        self,
        *,
        candidate_id: str,
        samples: Sequence[Any],
        results_path: str | Path,
        parent_id: str,
    ) -> dict[str, Any]:
        """Create immutable per-candidate JSONL views from search results only."""

        candidate = self.candidate_store.load(candidate_id)
        parent = self.candidate_store.load(parent_id)
        prompt_diff = _prompt_diff(parent.templates, candidate.templates)
        samples_by_id = {
            _identity(_sample_id(sample)): _sample_record(sample)
            for sample in samples
        }
        results = _read_results(Path(results_path), candidate_id)
        trusted: list[dict[str, Any]] = []
        visible: list[dict[str, Any]] = []
        for result in results:
            marker = _identity(result.get("sample_id"))
            if marker not in samples_by_id:
                raise ExperienceStoreError(
                    "search result references a sample outside search data"
                )
            record = samples_by_id[marker]
            common = {
                "schema_version": EXPERIENCE_SCHEMA_VERSION,
                "stage": "search",
                "candidate_id": candidate_id,
                "candidate_prompt_sha256": candidate.source_sha256,
                "prompt_diff": prompt_diff,
                "claim": _safe_value(record.get("claim")),
                "evidence": _evidence(record, self.repository_root),
                "reasoning_method": _safe_text(
                    result.get(
                        "reasoning_method",
                        result.get("method", record.get("type", "")),
                    )
                ),
                "solver_response": _safe_text(result.get("raw_response")),
                "latency_seconds": result.get("latency"),
                "token_usage": _safe_usage(result.get("usage")),
                "request_status": _safe_text(result.get("request_status")),
                "parse_status": _safe_text(result.get("parse_status")),
            }
            prediction = result.get("prediction")
            gold_label = result.get("gold_label")
            correctness = (
                prediction == gold_label
                if prediction is not None and gold_label is not None
                else None
            )
            trusted.append(
                {
                    **common,
                    "sample_id": result.get("sample_id"),
                    "prediction": prediction,
                    "gold_label": gold_label,
                    "correctness": correctness,
                }
            )
            # AGENTS.md forbids gold labels in model-visible input. Prediction,
            # correctness, and an answer-bearing response together reconstruct
            # the binary gold label, so the safe view keeps the response but
            # omits the other two signals and uses an opaque stable trace key.
            visible.append(
                {
                    **common,
                    "trace_key": "trace_"
                    + hashlib.sha256(
                        f"{self.run_id}\0{marker}".encode("utf-8")
                    ).hexdigest()[:16],
                }
            )

        trusted_path = self.trusted_directory / f"{candidate_id}.jsonl"
        visible_path = self.proposer_directory / "traces" / f"{candidate_id}.jsonl"
        _atomic_create_jsonl(trusted_path, trusted)
        _atomic_create_jsonl(visible_path, visible)
        summary = {
            "schema_version": EXPERIENCE_SCHEMA_VERSION,
            "stage": "search",
            "candidate_id": candidate_id,
            "candidate_prompt_sha256": candidate.source_sha256,
            "trace_count": len(visible),
            "trace_file": str(visible_path.relative_to(self.proposer_directory)),
            "trace_sha256": _sha256_file(visible_path),
        }
        _atomic_create_json(
            self.proposer_directory / "index" / f"{candidate_id}.json",
            summary,
        )
        self._write_readme()
        return summary

    def _write_readme(self) -> None:
        _atomic_create_text(
            self.proposer_directory / "README.md",
            (
                "# Search experience\n\n"
                "This read-only directory contains search-stage traces only. "
                "Query `index/*.json` and `traces/*.jsonl`. Protected data, "
                "credentials, image bytes, labels, predictions, correctness, "
                "and answer keys are intentionally absent.\n"
            ),
        )


def _prompt_diff(
    parent_templates: Mapping[str, str],
    candidate_templates: Mapping[str, str],
) -> dict[str, str]:
    return {
        method: "\n".join(
            unified_diff(
                parent_templates[method].splitlines(),
                candidate_templates[method].splitlines(),
                fromfile="parent",
                tofile="candidate",
                lineterm="",
            )
        )
        for method in candidate_templates
    }


def _evidence(record: Mapping[str, Any], root: Path) -> dict[str, Any]:
    evidence = {}
    for field in ("context", "section", "caption", "captions", "item"):
        if field in record:
            evidence[field] = _safe_value(record[field])
    paths: list[str] = []
    for field in ("image_path", "image_paths"):
        raw = record.get(field)
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            if not isinstance(value, str) or not value:
                continue
            if value.startswith(("http://", "https://", "data:")):
                continue
            path = Path(value)
            resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
            if resolved.is_relative_to(root):
                paths.append(str(resolved))
    evidence["safe_local_paths"] = paths
    return evidence


def _sample_id(sample: Any) -> Any:
    if hasattr(sample, "sample_id"):
        return sample.sample_id
    if isinstance(sample, Mapping):
        return sample.get("sample_id")
    raise ExperienceStoreError("search sample has no sample_id")


def _sample_record(sample: Any) -> dict[str, Any]:
    if hasattr(sample, "record"):
        return dict(sample.record)
    if isinstance(sample, Mapping):
        return dict(sample)
    raise ExperienceStoreError("search sample must be adapted sample data")


def _read_results(path: Path, candidate_id: str) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperienceStoreError(
            "search results must be readable JSONL"
        ) from exc
    return [
        value
        for value in values
        if isinstance(value, dict)
        and value.get("candidate_id", value.get("prompt_variant"))
        == candidate_id
    ]


def _safe_usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        field: raw
        for field in ("input_tokens", "output_tokens", "total_tokens")
        if isinstance((raw := value.get(field)), int)
        and not isinstance(raw, bool)
        and raw >= 0
    }


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
            if str(key).casefold()
            not in {
                "gold_label",
                "label",
                "prediction",
                "authorization",
                "api_key",
                "api_url",
                "image_base64",
            }
        }
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _safe_text(value)


def _safe_text(value: Any) -> str:
    text = "" if value is None else str(value)
    for name in ("API_KEY", "API_URL"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    for pattern in _SENSITIVE:
        text = pattern.sub("[REDACTED]", text)
    return text


def _identity(value: Any) -> str:
    return canonical_json(value)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_create_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    encoded = "".join(canonical_json(value) + "\n" for value in values)
    _atomic_create_bytes(path, encoded.encode("utf-8"))


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_create_bytes(
        path,
        (
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _atomic_create_text(path: Path, value: str) -> None:
    _atomic_create_bytes(path, value.encode("utf-8"))


def _atomic_create_bytes(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ExperienceStoreError(
                "refusing to replace an immutable search experience artifact"
            )
        return
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
            _atomic_create_bytes(path, encoded)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "EXPERIENCE_SCHEMA_VERSION",
    "ExperienceStoreError",
    "SearchExperienceStore",
]
