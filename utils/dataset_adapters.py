"""Strict, provider-neutral adapters for supported claim-verification datasets.

Raw mappings in this module follow the released schemas for SciVer,
SciAtomicBench, MuSciClaims, and SciClaimEval Task 1.  They intentionally copy
only fields needed for inference and offline scoring; source rationales and
other gold-only annotations are never retained in normalized records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


SUPPORTED_DATASETS = (
    "SciVer",
    "SciAtomicBench",
    "MuSciClaims",
    "SciClaimEval",
)

_NO_CONTEXT = "No additional context is provided."
_NORMALIZED_REQUIRED = frozenset({"claim_type", "claim"})
_REMOTE_REFERENCE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_SAFE_FILE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class DatasetAdapterError(ValueError):
    """Raised when local benchmark input cannot be normalized safely."""


@dataclass(frozen=True)
class AdaptedSample:
    """One stable sample identity and its normalized inference record."""

    sample_id: Any
    record: Mapping[str, Any]


def _read_records(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        raise DatasetAdapterError("dataset path does not exist")
    if not path.is_file():
        raise DatasetAdapterError("dataset metadata path must be a file")
    try:
        if path.suffix.casefold() == ".jsonl":
            values = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif path.suffix.casefold() == ".json":
            values = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise DatasetAdapterError("dataset metadata must be JSON or JSONL")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetAdapterError("dataset metadata must contain valid JSON") from exc
    if not isinstance(values, list):
        raise DatasetAdapterError("dataset JSON must contain a top-level list")
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise DatasetAdapterError(f"dataset record at index {index} must be an object")
    return values


def _required_text(record: Mapping[str, Any], field: str, index: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DatasetAdapterError(
            f"dataset record at index {index} requires non-empty {field!r} text"
        )
    return value.strip()


def _optional_context(value: Any, index: int) -> str:
    if value is None or value == "":
        return _NO_CONTEXT
    if not isinstance(value, str):
        raise DatasetAdapterError(
            f"dataset record at index {index} field 'context' must be text or null"
        )
    return value.strip() or _NO_CONTEXT


def _split_name(path: Path) -> str:
    """Return the released split name used to namespace generated IDs."""

    stem = path.stem.casefold()
    for split in ("train", "dev", "validation", "val", "test"):
        if stem == split or stem.startswith(f"{split}_") or stem.startswith(
            f"{split}-"
        ) or stem.startswith(f"{split}set"):
            return split
    return stem


def _generated_sample_id(
    dataset: str,
    namespace: str,
    raw_id: Any,
    row_index: int,
) -> str:
    """Keep source IDs readable while making repeated raw IDs collision-safe."""

    return f"{dataset}:{namespace}:{raw_id}:{row_index}"


def _label(value: Any, mapping: Mapping[str, str], field: str, index: int) -> str:
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in mapping:
            return mapping[normalized]
    expected = ", ".join(sorted(mapping))
    raise DatasetAdapterError(
        f"dataset record at index {index} has unsupported {field!r}; expected: {expected}"
    )


def _local_path(reference: Any, metadata_path: Path, index: int, field: str) -> str:
    if not isinstance(reference, str) or not reference.strip():
        raise DatasetAdapterError(
            f"dataset record at index {index} requires a non-empty local {field!r}"
        )
    raw = reference.strip()
    if _REMOTE_REFERENCE.match(raw) or raw.casefold().startswith("data:"):
        raise DatasetAdapterError(
            f"dataset record at index {index} field {field!r} must reference a local file"
        )
    path = Path(raw).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [metadata_path.parent / path]
        if metadata_path.parent.parent != metadata_path.parent:
            candidates.append(metadata_path.parent.parent / path)
        parts = path.parts
        if parts and (
            parts[0] in {".", metadata_path.parent.name}
            or parts[0].casefold() in {name.casefold() for name in SUPPORTED_DATASETS}
        ):
            trimmed = Path(*[part for part in parts[1:] if part != "."])
            candidates.append(metadata_path.parent / trimmed)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return str(candidate.resolve())
        except OSError:
            continue
    raise DatasetAdapterError(
        f"dataset record at index {index} field {field!r} does not resolve to a local file"
    )


def _unique_samples(records: Sequence[Mapping[str, Any]]) -> list[AdaptedSample]:
    samples: list[AdaptedSample] = []
    identities: set[str] = set()
    for index, record in enumerate(records):
        sample_id = record.get("sample_id", index)
        try:
            marker = json.dumps(
                sample_id, sort_keys=True, ensure_ascii=False, allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise DatasetAdapterError(
                f"sample_id at index {index} must be JSON serializable"
            ) from exc
        if marker in identities:
            raise DatasetAdapterError(f"duplicate sample_id at index {index}")
        identities.add(marker)
        samples.append(AdaptedSample(sample_id=sample_id, record=record))
    return samples


def _is_pre_normalized(records: Sequence[Mapping[str, Any]]) -> bool:
    return bool(records) and all(
        _NORMALIZED_REQUIRED.issubset(record)
        and (
            "image_path" in record
            or ("item1_path" in record and "item2_path" in record)
        )
        and ("sample_id" in record or "gold_label" in record)
        for record in records
    )


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _table_rows(markdown: str, index: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        raise DatasetAdapterError(
            f"dataset record at index {index} table_content is not a Markdown table"
        )
    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def _wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    width: int,
) -> list[str]:
    words: list[str] = []
    for word in text.replace("\n", " ").split():
        if draw.textlength(word, font=font) <= width:
            words.append(word)
            continue
        chunk = ""
        for character in word:
            proposed = chunk + character
            if chunk and draw.textlength(proposed, font=font) > width:
                words.append(chunk)
                chunk = character
            else:
                chunk = proposed
        if chunk:
            words.append(chunk)
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        proposed = f"{current} {word}"
        if draw.textlength(proposed, font=font) <= width:
            current = proposed
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def render_markdown_table(markdown: str, output_path: str | Path, *, index: int = 0) -> Path:
    """Render the released SciAtomic Markdown table syntax to a readable PNG."""

    rows = _table_rows(markdown, index)
    font = _font(20)
    header_font = _font(20, bold=True)
    scratch = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(scratch)
    padding = 14
    line_height = 28
    column_widths: list[int] = []
    for column in range(len(rows[0])):
        longest = max(
            draw.textlength(row[column], font=header_font if row_index == 0 else font)
            for row_index, row in enumerate(rows)
        )
        column_widths.append(max(120, min(360, int(longest) + padding * 2)))
    wrapped: list[list[list[str]]] = []
    row_heights: list[int] = []
    for row_index, row in enumerate(rows):
        row_font = header_font if row_index == 0 else font
        wrapped_row = [
            _wrapped_lines(draw, cell, row_font, column_widths[column] - padding * 2)
            for column, cell in enumerate(row)
        ]
        wrapped.append(wrapped_row)
        row_heights.append(max(len(lines) for lines in wrapped_row) * line_height + padding * 2)
    image = Image.new(
        "RGB",
        (sum(column_widths) + 1, sum(row_heights) + 1),
        "white",
    )
    draw = ImageDraw.Draw(image)
    y = 0
    for row_index, (wrapped_row, row_height) in enumerate(zip(wrapped, row_heights)):
        x = 0
        fill = "#E8EEF7" if row_index == 0 else ("#F8FAFC" if row_index % 2 == 0 else "white")
        row_font = header_font if row_index == 0 else font
        for column, lines in enumerate(wrapped_row):
            width = column_widths[column]
            draw.rectangle((x, y, x + width, y + row_height), fill=fill, outline="#64748B", width=1)
            draw.multiline_text(
                (x + padding, y + padding),
                "\n".join(lines),
                fill="#111827",
                font=row_font,
                spacing=line_height - 20,
            )
            x += width
        y += row_height
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
    return destination.resolve()


class DatasetAdapter:
    """Base class retaining compatibility with already-normalized JSON input."""

    adapter_name = "normalized_json"

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

    def _metadata_paths(self, dataset_path: Path) -> list[Path]:
        if dataset_path.is_file():
            return [dataset_path]
        raise DatasetAdapterError("dataset path must be a JSON or JSONL file")

    def load(
        self,
        dataset_path: str | Path,
        *,
        evidence_dir: str | Path | None = None,
    ) -> list[AdaptedSample]:
        paths = self._metadata_paths(Path(dataset_path))
        records = [record for path in paths for record in _read_records(path)]
        if _is_pre_normalized(records):
            return _unique_samples(records)
        normalized = self._normalize(paths, evidence_dir=evidence_dir)
        return _unique_samples(normalized)

    def _normalize(
        self, paths: Sequence[Path], *, evidence_dir: str | Path | None
    ) -> list[Mapping[str, Any]]:
        raise DatasetAdapterError(
            f"{self.dataset_name} raw schema was not recognized; supply verified released data or normalized JSON"
        )


class NormalizedJSONDatasetAdapter(DatasetAdapter):
    """Backward-compatible name for callers using normalized SciVer JSON."""


class SciVerDatasetAdapter(DatasetAdapter):
    adapter_name = "normalized_json"

    def _normalize(self, paths: Sequence[Path], *, evidence_dir: str | Path | None) -> list[Mapping[str, Any]]:
        normalized: list[Mapping[str, Any]] = []
        label_map = {"entailed": "yes", "refuted": "no", "yes": "yes", "no": "no"}
        for path in paths:
            split = _split_name(path)
            for index, record in enumerate(_read_records(path)):
                raw_id = record.get("request_id")
                if raw_id is None:
                    raise DatasetAdapterError(
                        f"dataset record at index {index} requires released SciVer field 'request_id'"
                    )
                raw_label = record.get("label")
                if isinstance(raw_label, bool):
                    gold = "yes" if raw_label else "no"
                else:
                    gold = _label(raw_label, label_map, "label", index)
                paper_path = Path(_local_path(record.get("paper_path"), path, index, "paper_path"))
                try:
                    paper = json.loads(paper_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise DatasetAdapterError(
                        f"dataset record at index {index} paper_path is not valid JSON"
                    ) from exc
                _sciver_context(record, paper, index)
                method = _required_text(record, "claim_type", index).casefold()
                if method not in {"direct", "analytical", "parallel", "sequential"}:
                    raise DatasetAdapterError(
                        f"dataset record at index {index} has unsupported SciVer claim_type"
                    )
                normalized_record: dict[str, Any] = {
                    "sample_id": _generated_sample_id(
                        "SciVer", split, raw_id, index
                    ),
                    "claim_type": method,
                    "claim": _required_text(record, "claim", index),
                    "paper_path": str(paper_path.resolve()),
                    "section": list(record["section"]),
                    "gold_label": gold,
                }
                if method in {"direct", "analytical"}:
                    _sciver_caption(record, paper, index)
                    normalized_record.update(
                        {
                            "type": record.get("type"),
                            "item": record.get("item"),
                            "image_path": _local_path(
                                record.get("image_path"), path, index, "image_path"
                            ),
                        }
                    )
                else:
                    _sciver_caption(record, paper, index, prefix="item1_")
                    _sciver_caption(record, paper, index, prefix="item2_")
                    normalized_record.update(
                        {
                            "item1_type": record.get("item1_type"),
                            "item1": record.get("item1"),
                            "item1_path": _local_path(
                                record.get("item1_path"), path, index, "item1_path"
                            ),
                            "item2_type": record.get("item2_type"),
                            "item2": record.get("item2"),
                            "item2_path": _local_path(
                                record.get("item2_path"), path, index, "item2_path"
                            ),
                        }
                    )
                normalized.append(normalized_record)
        return normalized


def _sciver_context(record: Mapping[str, Any], paper: Any, index: int) -> str:
    sections = record.get("section")
    if isinstance(sections, (str, bytes)) or not isinstance(sections, Sequence) or not all(isinstance(item, str) for item in sections):
        raise DatasetAdapterError(f"dataset record at index {index} requires SciVer section IDs")
    if not isinstance(paper, Mapping) or not isinstance(paper.get("sections"), Sequence):
        raise DatasetAdapterError(f"dataset record at index {index} paper requires sections")
    top_sections = list(dict.fromkeys(item.split(".")[0] for item in sections))
    parts: list[str] = []
    for section in paper["sections"]:
        if not isinstance(section, Mapping):
            raise DatasetAdapterError(f"dataset record at index {index} paper section is invalid")
        section_id = section.get("section_id")
        name = section.get("section_name")
        text = section.get("text")
        if all(isinstance(value, str) for value in (section_id, name, text)) and section_id.split(".")[0] in top_sections:
            parts.append(f"{name}:\n{text}\n")
    context = "".join(parts)
    if not context.strip():
        raise DatasetAdapterError(f"dataset record at index {index} has no matching SciVer context")
    return context


def _sciver_caption(
    record: Mapping[str, Any],
    paper: Any,
    index: int,
    *,
    prefix: str = "",
) -> str:
    evidence_type = record.get(f"{prefix}type")
    if evidence_type not in {"chart", "table"} or not isinstance(paper, Mapping):
        raise DatasetAdapterError(f"dataset record at index {index} requires chart or table evidence")
    collection_name, caption_name = ("image_paths", "caption") if evidence_type == "chart" else ("tables", "capture")
    try:
        caption = paper[collection_name][record.get(f"{prefix.rstrip('_')}") if prefix else record.get("item")][caption_name]
    except (KeyError, IndexError, TypeError) as exc:
        raise DatasetAdapterError(f"dataset record at index {index} paper evidence is incomplete") from exc
    if not isinstance(caption, str) or not caption.strip():
        raise DatasetAdapterError(f"dataset record at index {index} evidence caption is empty")
    return caption.strip()


class SciAtomicDatasetAdapter(DatasetAdapter):
    adapter_name = "sciatomic"
    _FILES = ("fin.json", "mat.json", "med.json", "ml.json")

    def _metadata_paths(self, dataset_path: Path) -> list[Path]:
        if dataset_path.is_file():
            return [dataset_path]
        if dataset_path.is_dir():
            missing = [
                name for name in self._FILES if not (dataset_path / name).is_file()
            ]
            if missing:
                raise DatasetAdapterError(
                    "SciAtomicBench directory is missing required domain files: "
                    + ", ".join(missing)
                )
            return [dataset_path / name for name in self._FILES]
        raise DatasetAdapterError("SciAtomicBench path must be a released JSON file or directory")

    def _normalize(self, paths: Sequence[Path], *, evidence_dir: str | Path | None) -> list[Mapping[str, Any]]:
        destination = Path(evidence_dir) if evidence_dir is not None else paths[0].parent / ".sciver_evidence"
        normalized: list[Mapping[str, Any]] = []
        for path in paths:
            domain = path.stem.casefold()
            for index, record in enumerate(_read_records(path)):
                raw_id = _required_text(record, "id", index)
                table = _required_text(record, "table_content", index)
                caption = _required_text(record, "table_caption", index)
                digest = hashlib.sha256(table.encode("utf-8")).hexdigest()[:16]
                safe_id = _SAFE_FILE_NAME.sub("_", raw_id)[:80] or "sample"
                image_path = destination / f"{domain}-{safe_id}-{digest}.png"
                if image_path.exists():
                    try:
                        with Image.open(image_path) as image:
                            if image.format != "PNG":
                                raise DatasetAdapterError("cached SciAtomic evidence is not PNG")
                    except (OSError, UnidentifiedImageError) as exc:
                        raise DatasetAdapterError("cached SciAtomic evidence is unreadable") from exc
                else:
                    render_markdown_table(table, image_path, index=index)
                normalized.append(
                    {
                        "sample_id": _generated_sample_id(
                            "SciAtomicBench", domain, raw_id, index
                        ),
                        "claim_type": "analytical",
                        "claim": _required_text(record, "claim", index),
                        "context": _NO_CONTEXT,
                        "caption": caption,
                        "image_path": str(image_path.resolve()),
                        "gold_label": _label(record.get("label"), {"support": "yes", "refute": "no"}, "label", index),
                    }
                )
        return normalized


class MuSciClaimsDatasetAdapter(DatasetAdapter):
    adapter_name = "musciclaims"

    def _metadata_paths(self, dataset_path: Path) -> list[Path]:
        if dataset_path.is_file():
            return [dataset_path]
        candidate = dataset_path / "test_set.jsonl"
        if candidate.is_file():
            return [candidate]
        raise DatasetAdapterError("MuSciClaims path must be test_set.jsonl or its containing directory")

    def _normalize(self, paths: Sequence[Path], *, evidence_dir: str | Path | None) -> list[Mapping[str, Any]]:
        normalized: list[Mapping[str, Any]] = []
        for path in paths:
            split = _split_name(path)
            for index, record in enumerate(_read_records(path)):
                raw_id = _required_text(record, "claim_id", index)
                normalized.append(
                    {
                        "sample_id": _generated_sample_id(
                            "MuSciClaims", split, raw_id, index
                        ),
                        "claim_type": "analytical",
                        "claim": _required_text(record, "claim_text", index),
                        "context": _NO_CONTEXT,
                        "caption": _required_text(record, "caption", index),
                        "image_path": _local_path(record.get("associated_figure_filepath"), path, index, "associated_figure_filepath"),
                        "gold_label": _label(record.get("label_2class"), {"support": "yes", "non_support": "no"}, "label_2class", index),
                    }
                )
        return normalized


class SciClaimEvalDatasetAdapter(DatasetAdapter):
    adapter_name = "sciclaimeval_task1"

    def _metadata_paths(self, dataset_path: Path) -> list[Path]:
        if dataset_path.is_file():
            return [dataset_path]
        candidates = tuple(
            base / filename
            for base in (dataset_path / "data", dataset_path)
            for filename in (
                "dev_task1_release.json",
                "test_task1_release.json",
            )
        )
        for candidate in candidates:
            if candidate.is_file():
                return [candidate]
        raise DatasetAdapterError("SciClaimEval path must identify the released Task 1 JSON")

    def _normalize(self, paths: Sequence[Path], *, evidence_dir: str | Path | None) -> list[Mapping[str, Any]]:
        normalized: list[Mapping[str, Any]] = []
        for path in paths:
            split = _split_name(path)
            for index, record in enumerate(_read_records(path)):
                raw_id = _required_text(record, "claim_id", index)
                normalized_record = {
                    "sample_id": _generated_sample_id(
                        "SciClaimEval", split, raw_id, index
                    ),
                    "claim_type": "analytical",
                    "claim": _required_text(record, "claim", index),
                    "context": _optional_context(record.get("context"), index),
                    "caption": _required_text(record, "caption", index),
                    "image_path": _local_path(
                        record.get("evi_path"), path, index, "evi_path"
                    ),
                }
                if record.get("label") is not None:
                    normalized_record["gold_label"] = _label(
                        record.get("label"),
                        {"supported": "yes", "refuted": "no"},
                        "label",
                        index,
                    )
                normalized.append(normalized_record)
        return normalized


_ADAPTERS: dict[str, DatasetAdapter] = {
    "sciver": SciVerDatasetAdapter("SciVer"),
    "sciatomicbench": SciAtomicDatasetAdapter("SciAtomicBench"),
    "musciclaims": MuSciClaimsDatasetAdapter("MuSciClaims"),
    "sciclaimeval": SciClaimEvalDatasetAdapter("SciClaimEval"),
}


def get_dataset_adapter(dataset_name: str) -> DatasetAdapter:
    """Return the strict adapter for a supported benchmark name."""

    try:
        return _ADAPTERS[dataset_name.casefold()]
    except (AttributeError, KeyError) as exc:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise DatasetAdapterError(
            f"Unknown dataset {dataset_name!r}. Supported datasets: {supported}."
        ) from exc


__all__ = [
    "AdaptedSample",
    "DatasetAdapter",
    "DatasetAdapterError",
    "NormalizedJSONDatasetAdapter",
    "SUPPORTED_DATASETS",
    "get_dataset_adapter",
    "render_markdown_table",
]
