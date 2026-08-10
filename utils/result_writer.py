"""Durable, resumable JSONL storage for per-sample inference results.

The writer deliberately has no inference or network dependencies.  A completed
sample is appended before :meth:`write_result` returns, flushed, and synced to
disk so an interrupted notebook can reconstruct its resume state.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Set, Tuple


SUCCESS = "success"
API_FAILURE = "api_failure"
PARSE_FAILURE = "parse_failure"
INVALID_INPUT = "invalid_input"
REQUEST_STATUSES = frozenset(
    {SUCCESS, API_FAILURE, PARSE_FAILURE, INVALID_INPUT}
)

_MISSING = object()
_REQUIRED_FIELDS = frozenset(
    {
        "run_id",
        "sample_id",
        "dataset",
        "model",
        "method",
        "prediction",
        "parse_status",
        "raw_response",
        "request_status",
        "error_type",
        "error_message",
        "timestamp",
        "attempt_count",
    }
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_AUTHORIZATION_RE = re.compile(
    r"(?i)(authorization(?:_header)?\s*[:=]\s*)([^\r\n,;\]}]+)"
)
_DATA_IMAGE_RE = re.compile(
    r"(?i)data:image/[a-z0-9.+-]+;base64,[a-z0-9+/=\s]+"
)
# Long uninterrupted base64 strings are not useful diagnostics and may contain
# image bytes.  The threshold avoids changing ordinary short model responses.
_LONG_BASE64_RE = re.compile(
    r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{128,}={0,2}(?![A-Za-z0-9+/=])"
)
_REDACTED = "[REDACTED]"

RecordKey = Tuple[str, str, str, str, str, str]


class ResultWriterError(ValueError):
    """Raised when a result file or record cannot be handled safely."""


def _identity_part(value: Any) -> str:
    """Create a stable, type-preserving key component for JSON-compatible IDs."""

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ResultWriterError("result identity values must be JSON serializable") from exc


def _record_key(record: Mapping[str, Any]) -> RecordKey:
    try:
        return (
            _identity_part(record["run_id"]),
            _identity_part(record["sample_id"]),
            _identity_part(record["dataset"]),
            _identity_part(record["model"]),
            _identity_part(record.get("prompt_variant", "cot")),
            _identity_part(record["method"]),
        )
    except KeyError as exc:
        raise ResultWriterError(
            f"result record is missing required field: {exc.args[0]}"
        ) from exc


def _sanitize_text(value: str) -> str:
    sanitized = value
    for environment_name in ("API_KEY", "API_URL"):
        sensitive_value = os.environ.get(environment_name)
        if sensitive_value:
            sanitized = sanitized.replace(sensitive_value, _REDACTED)
    sanitized = _DATA_IMAGE_RE.sub(_REDACTED, sanitized)
    sanitized = _BEARER_RE.sub(f"Bearer {_REDACTED}", sanitized)
    sanitized = _AUTHORIZATION_RE.sub(rf"\1{_REDACTED}", sanitized)
    sanitized = _LONG_BASE64_RE.sub(_REDACTED, sanitized)
    return sanitized


def _sanitize(value: Any) -> Any:
    """Recursively remove credential/header/image material before serialization."""

    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, Mapping):
        clean: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            normalized = key_text.lower().replace("-", "_")
            if normalized in {
                "api_key",
                "authorization",
                "authorization_header",
                "image_base64",
                "base64_image",
            }:
                clean[key_text] = _REDACTED
            else:
                clean[key_text] = _sanitize(item)
        return clean
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _sanitize_text(str(value))


def sanitize_result_data(value: Any) -> Any:
    """Return a JSON-compatible copy with sensitive result data redacted."""

    return _sanitize(value)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ResultWriter:
    """Append inference outcomes and expose resume state from a JSONL file."""

    def __init__(self, output_path: os.PathLike[str] | str):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._successful: Set[RecordKey] = set()
        self._attempts: Dict[RecordKey, int] = {}
        self._needs_separator = False
        self._load_existing_file()

    def _load_existing_file(self) -> None:
        if not self.path.exists():
            return
        if not self.path.is_file():
            raise ResultWriterError("result output path is not a regular file")

        data = self.path.read_bytes()
        if not data:
            return

        lines = data.splitlines(keepends=True)
        valid_end = 0
        for index, line in enumerate(lines):
            line_end = valid_end + len(line)
            stripped = line.strip()
            if not stripped:
                valid_end = line_end
                continue
            try:
                decoded = stripped.decode("utf-8")
                record = json.loads(decoded)
                if not isinstance(record, dict):
                    raise ValueError("record is not an object")
                self._index_record(record)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                if index != len(lines) - 1:
                    raise ResultWriterError(
                        f"invalid JSONL record at line {index + 1}"
                    ) from exc
                # An interrupted append may leave only the final line malformed.
                # Remove that fragment so the next append starts at a clean line.
                self._truncate_to(valid_end)
                self._needs_separator = bool(valid_end and not data[:valid_end].endswith(b"\n"))
                return
            valid_end = line_end

        self._needs_separator = bool(data and not data.endswith((b"\n", b"\r")))

    def _truncate_to(self, length: int) -> None:
        with self.path.open("r+b") as output:
            output.truncate(length)
            output.flush()
            os.fsync(output.fileno())

    def _index_record(self, record: Mapping[str, Any]) -> None:
        missing = _REQUIRED_FIELDS.difference(record)
        if missing:
            raise ResultWriterError(
                "result record is missing required fields: " + ", ".join(sorted(missing))
            )
        prompt_variant = record.get("prompt_variant", "cot")
        if not isinstance(prompt_variant, str) or not prompt_variant:
            raise ResultWriterError(
                "prompt_variant must be a non-empty string when present"
            )
        status = record["request_status"]
        if status not in REQUEST_STATUSES:
            raise ResultWriterError(f"invalid request_status: {status!r}")
        attempt_count = record["attempt_count"]
        if not isinstance(attempt_count, int) or isinstance(attempt_count, bool) or attempt_count < 1:
            raise ResultWriterError("attempt_count must be a positive integer")

        key = _record_key(record)
        self._attempts[key] = max(self._attempts.get(key, 0), attempt_count)
        if status == SUCCESS:
            self._successful.add(key)

    @staticmethod
    def _key(
        *,
        run_id: Any,
        sample_id: Any,
        dataset: Any,
        model: Any,
        method: Any,
        prompt_variant: Any = "cot",
    ) -> RecordKey:
        return _record_key(
            {
                "run_id": _sanitize(run_id),
                "sample_id": _sanitize(sample_id),
                "dataset": _sanitize(dataset),
                "model": _sanitize(model),
                "prompt_variant": _sanitize(prompt_variant),
                "method": _sanitize(method),
            }
        )

    def is_successful(
        self,
        *,
        run_id: Any,
        sample_id: Any,
        dataset: Any,
        model: Any,
        method: Any,
        prompt_variant: Any = "cot",
    ) -> bool:
        """Return whether this exact run/sample configuration already succeeded."""

        key = self._key(
            run_id=run_id,
            sample_id=sample_id,
            dataset=dataset,
            model=model,
            method=method,
            prompt_variant=prompt_variant,
        )
        with self._lock:
            return key in self._successful

    should_skip = is_successful

    def successful_sample_ids(
        self,
        *,
        run_id: Any,
        dataset: Any,
        model: Any,
        method: Any,
        prompt_variant: Any = "cot",
    ) -> Set[Any]:
        """Return successful sample IDs for one run configuration."""

        wanted = (
            _identity_part(_sanitize(run_id)),
            _identity_part(_sanitize(dataset)),
            _identity_part(_sanitize(model)),
            _identity_part(_sanitize(prompt_variant)),
            _identity_part(_sanitize(method)),
        )
        with self._lock:
            encoded_ids = {
                key[1]
                for key in self._successful
                if (key[0], key[2], key[3], key[4], key[5]) == wanted
            }
        return {json.loads(encoded) for encoded in encoded_ids}

    def next_attempt_count(
        self,
        *,
        run_id: Any,
        sample_id: Any,
        dataset: Any,
        model: Any,
        method: Any,
        prompt_variant: Any = "cot",
    ) -> int:
        key = self._key(
            run_id=run_id,
            sample_id=sample_id,
            dataset=dataset,
            model=model,
            method=method,
            prompt_variant=prompt_variant,
        )
        with self._lock:
            return self._attempts.get(key, 0) + 1

    def write_result(
        self,
        *,
        run_id: Any,
        sample_id: Any,
        dataset: str,
        model: str,
        method: str,
        prompt_variant: str = "cot",
        prediction: Any,
        parse_status: str,
        raw_response: Any,
        request_status: str,
        gold_label: Any = _MISSING,
        error: Optional[BaseException] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        timestamp: Optional[str] = None,
        attempt_count: Optional[int] = None,
        candidate_id: Any = _MISSING,
        split_sha256: Any = _MISSING,
        reasoning_method: Any = _MISSING,
        latency: Any = _MISSING,
        usage: Any = _MISSING,
        http_status_code: Any = _MISSING,
        response_error_code: Any = _MISSING,
        response_error_message: Any = _MISSING,
    ) -> bool:
        """Append one result and sync it to disk.

        Returns ``False`` without writing when the sample already has a success.
        Failed outcomes remain appendable and therefore retryable.
        """

        if request_status not in REQUEST_STATUSES:
            raise ResultWriterError(
                "request_status must be one of: " + ", ".join(sorted(REQUEST_STATUSES))
            )
        if not all(
            isinstance(item, str) and item
            for item in (dataset, model, method, prompt_variant)
        ):
            raise ResultWriterError(
                "dataset, model, method, and prompt_variant must be non-empty strings"
            )
        if not isinstance(parse_status, str) or not parse_status:
            raise ResultWriterError("parse_status must be a non-empty string")
        if request_status == SUCCESS and parse_status != "parsed":
            raise ResultWriterError("successful results must have parse_status='parsed'")

        key = self._key(
            run_id=run_id,
            sample_id=sample_id,
            dataset=dataset,
            model=model,
            method=method,
            prompt_variant=prompt_variant,
        )
        with self._lock:
            if key in self._successful:
                return False

            next_attempt = self._attempts.get(key, 0) + 1
            if attempt_count is None:
                attempt_count = next_attempt
            if (
                not isinstance(attempt_count, int)
                or isinstance(attempt_count, bool)
                or attempt_count < next_attempt
            ):
                raise ResultWriterError(
                    f"attempt_count must be an integer of at least {next_attempt}"
                )

            if error is not None:
                if error_type is None:
                    error_type = type(error).__name__
                if error_message is None:
                    error_message = str(error)

            record: Dict[str, Any] = {
                "run_id": run_id,
                "sample_id": sample_id,
                "dataset": dataset,
                "model": model,
                "prompt_variant": prompt_variant,
                "method": method,
                "prediction": prediction,
                "parse_status": parse_status,
                "raw_response": raw_response,
                "request_status": request_status,
                "error_type": error_type,
                "error_message": error_message,
                "timestamp": timestamp or _utc_timestamp(),
                "attempt_count": attempt_count,
            }
            if gold_label is not _MISSING:
                record["gold_label"] = gold_label
            if candidate_id is not _MISSING:
                record["candidate_id"] = candidate_id
            if split_sha256 is not _MISSING:
                record["split_sha256"] = split_sha256
            if reasoning_method is not _MISSING:
                record["reasoning_method"] = reasoning_method
            if latency is not _MISSING:
                record["latency"] = latency
            if usage is not _MISSING:
                record["usage"] = usage
            if http_status_code is not _MISSING:
                record["http_status_code"] = http_status_code
            if response_error_code is not _MISSING:
                record["response_error_code"] = response_error_code
            if response_error_message is not _MISSING:
                record["response_error_message"] = response_error_message

            clean_record = _sanitize(record)
            self._validate_serializable(clean_record)
            self._append_record(clean_record)
            self._index_record(clean_record)
            return True

    def _validate_serializable(self, record: Mapping[str, Any]) -> None:
        timestamp = record.get("timestamp")
        if not isinstance(timestamp, str) or not timestamp:
            raise ResultWriterError("timestamp must be a non-empty string")
        try:
            json.dumps(record, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ResultWriterError("result record must be JSON serializable") from exc

    def _append_record(self, record: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        if self._needs_separator:
            encoded = b"\n" + encoded

        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written == 0:
                    raise OSError("could not append result record")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._needs_separator = False


def iter_result_records(output_path: os.PathLike[str] | str) -> Iterable[Dict[str, Any]]:
    """Yield complete records, ignoring only a malformed final JSONL fragment."""

    path = Path(output_path)
    if not path.exists():
        return
    lines = path.read_bytes().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            if index == len(lines) - 1:
                return
            raise ResultWriterError(f"invalid JSONL record at line {index + 1}") from exc
        record.setdefault("prompt_variant", "cot")
        yield record


__all__ = [
    "API_FAILURE",
    "INVALID_INPUT",
    "PARSE_FAILURE",
    "REQUEST_STATUSES",
    "SUCCESS",
    "ResultWriter",
    "ResultWriterError",
    "iter_result_records",
    "sanitize_result_data",
]
