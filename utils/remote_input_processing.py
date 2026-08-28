"""Build provider-neutral multimodal messages from local image files."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import os
from os import PathLike
from pathlib import Path
import tempfile
from typing import Iterable

from PIL import Image, UnidentifiedImageError


_SUPPORTED_IMAGE_FORMATS = {
    "JPEG": (".jpg", "image/jpeg"),
    "PNG": (".png", "image/png"),
    "WEBP": (".webp", "image/webp"),
}
_EXTENSION_FORMATS = {
    ".jpeg": "JPEG",
    ".jpg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
}
_DEFAULT_IMAGE_CACHE = (
    Path(__file__).resolve().parents[1]
    / "workspace"
    / "meta_harness"
    / "image_cache"
    / "v1"
)

# Provider-neutral workaround for endpoints that accept only one image per
# request.  When exactly "1", multiple evidence images are fused side-by-side
# into a single image so the prompt still carries all evidence.  Off by
# default so multi-image-capable providers keep the standard per-image format.
FUSE_EVIDENCE_IMAGES_ENV = "SCIVER_FUSE_EVIDENCE_IMAGES"


class EmptyImageError(ValueError):
    """Raised when an image file contains no data."""


class UnsupportedImageFormatError(ValueError):
    """Raised when an image is not a supported PNG or JPEG file."""


class UnreadableImageError(OSError):
    """Raised when an image cannot be read or validated."""


def _image_to_data_uri(
    image_path: str | PathLike[str],
    *,
    cache_dir: str | PathLike[str] | None = None,
) -> str:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise UnreadableImageError(f"Image path is not a readable file: {path}")

    try:
        image_bytes = path.read_bytes()
    except OSError as exc:
        raise UnreadableImageError(f"Unable to read image file: {path}") from exc

    if not image_bytes:
        raise EmptyImageError(f"Image file is empty: {path}")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            actual_format = (image.format or "").upper()
            image.load()
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as exc:
        raise UnreadableImageError(f"Unable to decode image file: {path}") from exc

    format_details = _SUPPORTED_IMAGE_FORMATS.get(actual_format)
    if format_details is None:
        raise UnsupportedImageFormatError(
            f"Unsupported decoded image format for {path}; supported formats "
            "are JPEG, PNG, and WEBP"
        )
    _extension, mime_type = format_details

    expected_format = _EXTENSION_FORMATS.get(path.suffix.casefold())
    serialized_bytes = image_bytes
    if expected_format != actual_format:
        normalized_path = _normalized_cache_path(
            path,
            image_bytes,
            actual_format,
            Path(cache_dir) if cache_dir is not None else _DEFAULT_IMAGE_CACHE,
        )
        try:
            serialized_bytes = normalized_path.read_bytes()
        except OSError as exc:
            raise UnreadableImageError(
                f"Unable to read normalized image cache entry for: {path}"
            ) from exc

    encoded_image = base64.b64encode(serialized_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded_image}"


def _normalized_cache_path(
    source_path: Path,
    source_bytes: bytes,
    image_format: str,
    cache_dir: Path,
) -> Path:
    extension, _mime_type = _SUPPORTED_IMAGE_FORMATS[image_format]
    digest = hashlib.sha256(source_bytes).hexdigest()
    destination = cache_dir / f"{digest}{extension}"
    if destination.is_file():
        return destination

    try:
        with Image.open(BytesIO(source_bytes)) as image:
            image.load()
            normalized = _normalized_mode(image, image_format)
            encoded = BytesIO()
            normalized.save(
                encoded,
                format=image_format,
                **_deterministic_save_options(image_format),
            )
            normalized_bytes = encoded.getvalue()
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as exc:
        raise UnreadableImageError(
            f"Unable to normalize mislabeled image file: {source_path}"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{digest}.",
        suffix=".tmp",
        dir=cache_dir,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(normalized_bytes)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            pass
        return destination
    except OSError as exc:
        raise UnreadableImageError(
            f"Unable to write normalized image cache entry for: {source_path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _normalized_mode(image: Image.Image, image_format: str) -> Image.Image:
    if image_format == "JPEG":
        return image.convert("L" if image.mode == "L" else "RGB")
    if image.mode in {"1", "L", "LA", "RGB", "RGBA"}:
        return image.copy()
    return image.convert("RGBA" if "transparency" in image.info else "RGB")


def _deterministic_save_options(image_format: str) -> dict[str, object]:
    if image_format == "JPEG":
        return {
            "quality": 95,
            "subsampling": 0,
            "optimize": False,
            "progressive": False,
        }
    if image_format == "PNG":
        return {"compress_level": 9, "optimize": False}
    return {"lossless": True, "quality": 100, "method": 6, "exact": True}


def _build_image_url_block(data_uri: str) -> dict[str, object]:
    # The image request format must be validated with a one-sample live smoke test.
    return {
        "type": "image_url",
        "image_url": {"url": data_uri},
    }


def _fuse_evidence_images(
    image_paths: Iterable[str | PathLike[str]],
    *,
    cache_dir: str | PathLike[str] | None = None,
) -> str:
    """Combine evidence images side-by-side into one JPEG data URI.

    Order is preserved (first image left). This is the opt-in workaround for
    endpoints that allow at most one image per prompt; it keeps every piece of
    evidence visible in a single request image.
    """

    images = []
    for path in image_paths:
        data_uri = _image_to_data_uri(path, cache_dir=cache_dir)
        encoded = data_uri.split(",", 1)[1]
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            image.load()
            images.append(image.convert("RGB"))
    target_height = max(image.height for image in images)
    scaled = []
    for image in images:
        ratio = target_height / image.height
        width = max(1, round(image.width * ratio))
        scaled.append(image.resize((width, target_height), Image.BILINEAR))
    canvas = Image.new("RGB", (sum(im.width for im in scaled), target_height), (255, 255, 255))
    offset = 0
    for image in scaled:
        canvas.paste(image, (offset, 0))
        offset += image.width
    buffer = BytesIO()
    canvas.save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_remote_messages(
    user_prompt: str,
    image_paths: Iterable[str | PathLike[str]] = (),
    system_prompt: str | None = None,
    *,
    image_cache_dir: str | PathLike[str] | None = None,
) -> list[dict[str, object]]:
    """Build one request's messages without retaining dataset-level input."""
    messages: list[dict[str, object]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    path_list = list(image_paths)
    fuse = os.environ.get(FUSE_EVIDENCE_IMAGES_ENV) == "1"
    user_content: list[dict[str, object]] = []
    if fuse and len(path_list) > 1:
        user_content.append(
            _build_image_url_block(
                _fuse_evidence_images(path_list, cache_dir=image_cache_dir)
            )
        )
    else:
        for image_path in path_list:
            user_content.append(
                _build_image_url_block(
                    _image_to_data_uri(image_path, cache_dir=image_cache_dir)
                )
            )
    user_content.append({"type": "text", "text": user_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages
