"""Offline preflight and immutable run metadata for benchmark notebooks.

This module deliberately delegates request preparation to the production
remote adapter.  It never reads credentials, creates a client, or performs a
network request.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from model_inference.remote_api import InvalidRemoteInputError, prepare_remote_requests
from utils.constant import COT_PROMPT
from utils.dataset_adapters import get_dataset_adapter
from utils.remote_input_processing import (
    EmptyImageError,
    UnreadableImageError,
    UnsupportedImageFormatError,
)


_BINARY_STRING_LABELS = frozenset({"yes", "no", "true", "false"})
_SINGLE_IMAGE_METHODS = frozenset({"direct", "analytical"})
_MULTI_IMAGE_METHODS = frozenset({"parallel", "sequential"})


class BenchmarkWorkflowError(ValueError):
    """Raised when offline benchmark validation or resume metadata fails."""


def _safe_distribution(values: list[Any]) -> dict[str, int]:
    distribution: Counter[str] = Counter()
    for value in values:
        try:
            marker = json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            marker = f"<{type(value).__name__}>"
        if len(marker) > 80:
            marker = marker[:79] + "…"
        distribution[marker] += 1
    return dict(sorted(distribution.items()))


def _label_support(labels: list[Any]) -> str:
    scored_labels = [label for label in labels if label is not None]
    if not scored_labels:
        return "no_normalized_gold_labels"
    for label in scored_labels:
        if isinstance(label, bool):
            continue
        if isinstance(label, str) and label.strip().casefold() in _BINARY_STRING_LABELS:
            continue
        return "native_labels_preserved_unscored"
    return "binary_evaluation_ready"


def preflight_dataset(
    dataset_name: str,
    dataset_path: str | Path,
    *,
    selected_sample_count: int,
    evidence_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate every normalized record through production request preparation.

    Only aggregate metadata is returned.  Claims, captions, paper content,
    image bytes, and paths are never included in the report.
    """

    if (
        isinstance(selected_sample_count, bool)
        or not isinstance(selected_sample_count, int)
        or selected_sample_count < -1
    ):
        raise BenchmarkWorkflowError(
            "selected_sample_count must be -1 or a non-negative integer"
        )

    adapter = get_dataset_adapter(dataset_name)
    samples = adapter.load(dataset_path, evidence_dir=evidence_dir)
    total_samples = len(samples)
    selected_samples = (
        total_samples
        if selected_sample_count == -1
        else min(total_samples, selected_sample_count)
    )

    labels: list[Any] = []
    splits: list[Any] = []
    reasoning_types: list[Any] = []
    missing_context_count = 0
    missing_image_count = 0
    invalid_inputs = 0

    for index, sample in enumerate(samples):
        record = sample.record
        labels.append(record.get("gold_label"))
        if "split" in record:
            splits.append(record["split"])
        reasoning_types.append(record.get("claim_type"))
        try:
            messages = prepare_remote_requests([record], COT_PROMPT)[0]
            content = messages[-1].get("content")
            if not isinstance(content, list):
                raise BenchmarkWorkflowError("prepared request content is invalid")
            actual_images = sum(
                1
                for block in content
                if isinstance(block, Mapping) and block.get("type") == "image_url"
            )
            method = record.get("claim_type")
            normalized_method = method.strip().casefold() if isinstance(method, str) else ""
            expected_images = 1 if normalized_method in _SINGLE_IMAGE_METHODS else 2
            if normalized_method not in _SINGLE_IMAGE_METHODS | _MULTI_IMAGE_METHODS:
                raise BenchmarkWorkflowError("prepared request method is invalid")
            if actual_images != expected_images:
                raise BenchmarkWorkflowError("prepared request image count is invalid")
        except FileNotFoundError:
            missing_image_count += 1
            invalid_inputs += 1
        except InvalidRemoteInputError as exc:
            if "context" in str(exc).casefold():
                missing_context_count += 1
            invalid_inputs += 1
        except (EmptyImageError, UnreadableImageError, UnsupportedImageFormatError):
            invalid_inputs += 1
        except BenchmarkWorkflowError:
            invalid_inputs += 1

    if total_samples == 0:
        invalid_inputs = 1

    readiness = "ready" if invalid_inputs == 0 else f"not_ready ({invalid_inputs} invalid)"
    split_distribution = _safe_distribution(splits) if splits else {}
    return {
        "dataset": dataset_name,
        "split": split_distribution or "not_available",
        "total_samples": total_samples,
        "selected_samples": selected_samples,
        "label_distribution": _safe_distribution(labels),
        "label_support": _label_support(labels),
        "reasoning_or_claim_type_distribution": _safe_distribution(reasoning_types),
        "missing_context_count": missing_context_count,
        "missing_image_count": missing_image_count,
        "adapter_readiness": readiness,
        "invalid_inputs": invalid_inputs,
        "adapter": adapter.adapter_name,
    }


def require_adapter_ready(summary: Mapping[str, Any]) -> None:
    """Reject a dataset before credentials or request code can be reached."""

    if summary.get("adapter_readiness") != "ready":
        raise BenchmarkWorkflowError(
            "Dataset is not ready for the native adapter; inspect aggregate "
            "preflight counts and normalize the source schema without guessing mappings."
        )


def prompt_fingerprint() -> str:
    """Hash the existing prompt templates without exposing their text."""

    payload = {
        method: template.template
        for method, template in sorted(COT_PROMPT.items())
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(
    *,
    dataset_name: str,
    dataset_path: str | Path,
    summary: Mapping[str, Any],
    model_name: str,
    method: str,
    experiment_id: str,
    request_delay: float,
    git_sha: str,
) -> dict[str, Any]:
    """Build credential-free immutable metadata shared by all live run modes."""

    path = Path(dataset_path)
    stat = path.stat()
    immutable_configuration = {
        "dataset_identity": {
            "dataset": dataset_name,
            "file_name": path.name,
            "sha256": _file_sha256(path),
            "size_bytes": stat.st_size,
            "total_samples": summary["total_samples"],
        },
        "split": summary["split"],
        "model": model_name,
        "method": method,
        "experiment_id": experiment_id,
        "prompt_hash": prompt_fingerprint(),
        "generation_settings": {
            "n": 1,
            "stream": False,
            "request_delay_seconds": request_delay,
        },
        "git_sha": git_sha,
    }
    fingerprint_payload = json.dumps(
        immutable_configuration,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **immutable_configuration,
        "configuration_fingerprint": hashlib.sha256(fingerprint_payload).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def ensure_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> str:
    """Create a manifest once or reject a mismatched resume configuration."""

    manifest_path = Path(path)
    expected_fingerprint = manifest.get("configuration_fingerprint")
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        raise BenchmarkWorkflowError("manifest configuration fingerprint is missing")

    if manifest_path.exists():
        if not manifest_path.is_file():
            raise BenchmarkWorkflowError("run manifest path is not a file")
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BenchmarkWorkflowError("existing run manifest is invalid") from exc
        if existing.get("configuration_fingerprint") != expected_fingerprint:
            raise BenchmarkWorkflowError(
                "Resume rejected because the run configuration differs from the manifest"
            )
        return "matched"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, manifest_path)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()
    return "created"


__all__ = [
    "BenchmarkWorkflowError",
    "build_run_manifest",
    "ensure_run_manifest",
    "preflight_dataset",
    "prompt_fingerprint",
    "require_adapter_ready",
]
