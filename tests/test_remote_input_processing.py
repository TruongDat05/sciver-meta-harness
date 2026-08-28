import base64
import hashlib
from io import BytesIO

import pytest
from PIL import Image

from utils.remote_input_processing import (
    EmptyImageError,
    UnreadableImageError,
    UnsupportedImageFormatError,
    build_remote_messages,
)


def _write_image(path, image_format, color):
    Image.new("RGB", (2, 2), color=color).save(path, format=image_format)
    return path


def _image_url(messages, index=0):
    return messages[-1]["content"][index]["image_url"]["url"]


def test_text_only_message():
    messages = build_remote_messages(
        user_prompt="Check this claim.",
        system_prompt="Use the supplied evidence.",
    )

    assert messages == [
        {"role": "system", "content": "Use the supplied evidence."},
        {
            "role": "user",
            "content": [{"type": "text", "text": "Check this claim."}],
        },
    ]


def test_text_only_message_without_system_prompt():
    messages = build_remote_messages(user_prompt="Check this claim.")

    assert messages == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Check this claim."}],
        }
    ]


def test_one_png_image_has_correct_prefix_and_decodable_content(tmp_path):
    image_path = _write_image(tmp_path / "figure.png", "PNG", "red")
    expected_bytes = image_path.read_bytes()

    messages = build_remote_messages("Inspect the figure.", [image_path])

    data_uri = _image_url(messages)
    assert data_uri.startswith("data:image/png;base64,")
    assert base64.b64decode(data_uri.split(",", 1)[1], validate=True) == expected_bytes
    assert messages[-1]["content"][-1] == {
        "type": "text",
        "text": "Inspect the figure.",
    }


def test_one_jpeg_image_has_correct_prefix_and_decodable_content(tmp_path):
    image_path = _write_image(tmp_path / "figure.jpg", "JPEG", "blue")
    expected_bytes = image_path.read_bytes()

    messages = build_remote_messages("Inspect the figure.", [image_path])

    data_uri = _image_url(messages)
    assert data_uri.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(data_uri.split(",", 1)[1], validate=True) == expected_bytes


@pytest.mark.parametrize(
    ("image_format", "mime_type", "cache_extension"),
    [
        ("JPEG", "image/jpeg", ".jpg"),
        ("WEBP", "image/webp", ".webp"),
    ],
)
def test_mislabeled_png_uses_decoded_format_and_sha256_cache(
    tmp_path,
    image_format,
    mime_type,
    cache_extension,
):
    image_path = _write_image(
        tmp_path / "figure (panel-a).png",
        image_format,
        "purple",
    )
    source_bytes = image_path.read_bytes()
    source_before = source_bytes
    cache_dir = tmp_path / "normalized-cache"

    first = build_remote_messages(
        "Inspect the figure.",
        [image_path],
        image_cache_dir=cache_dir,
    )
    second = build_remote_messages(
        "Inspect the figure.",
        [image_path],
        image_cache_dir=cache_dir,
    )

    expected_cache_path = (
        cache_dir
        / f"{hashlib.sha256(source_bytes).hexdigest()}{cache_extension}"
    )
    assert expected_cache_path.is_file()
    assert list(cache_dir.iterdir()) == [expected_cache_path]
    assert image_path.read_bytes() == source_before
    assert _image_url(first).startswith(f"data:{mime_type};base64,")
    assert _image_url(second) == _image_url(first)
    normalized_bytes = base64.b64decode(
        _image_url(first).split(",", 1)[1],
        validate=True,
    )
    assert normalized_bytes == expected_cache_path.read_bytes()
    with Image.open(BytesIO(normalized_bytes)) as normalized:
        assert normalized.format == image_format
        normalized.load()


def test_two_images_preserve_input_order(tmp_path):
    first_path = _write_image(tmp_path / "first.png", "PNG", "red")
    second_path = _write_image(tmp_path / "second.jpg", "JPEG", "blue")

    messages = build_remote_messages(
        "Compare the figures.", [first_path, second_path]
    )

    content = messages[-1]["content"]
    assert len(content) == 3
    assert base64.b64decode(_image_url(messages, 0).split(",", 1)[1]) == (
        first_path.read_bytes()
    )
    assert base64.b64decode(_image_url(messages, 1).split(",", 1)[1]) == (
        second_path.read_bytes()
    )


def test_two_mislabeled_images_preserve_input_order(tmp_path):
    first_path = _write_image(
        tmp_path / "first (a).png",
        "WEBP",
        "red",
    )
    second_path = _write_image(
        tmp_path / "second (b).png",
        "JPEG",
        "blue",
    )

    messages = build_remote_messages(
        "Compare the figures.",
        [first_path, second_path],
        image_cache_dir=tmp_path / "cache",
    )

    assert _image_url(messages, 0).startswith("data:image/webp;base64,")
    assert _image_url(messages, 1).startswith("data:image/jpeg;base64,")


def test_multiple_images_are_supported(tmp_path):
    image_paths = [
        _write_image(tmp_path / f"figure-{index}.png", "PNG", color)
        for index, color in enumerate(("red", "green", "blue", "white"))
    ]

    messages = build_remote_messages("Compare all figures.", image_paths)

    image_blocks = messages[-1]["content"][:-1]
    assert len(image_blocks) == len(image_paths)
    assert all(block["type"] == "image_url" for block in image_blocks)


def test_fusion_flag_fuses_two_images_into_one(monkeypatch, tmp_path):
    first = _write_image(tmp_path / "first.png", "PNG", "red")
    second = _write_image(tmp_path / "second.jpg", "JPEG", "blue")
    monkeypatch.setenv("SCIVER_FUSE_EVIDENCE_IMAGES", "1")

    messages = build_remote_messages("Compare the figures.", [first, second])

    content = messages[-1]["content"]
    assert len(content) == 2  # one fused image + text
    data_uri = _image_url(messages, 0)
    assert data_uri.startswith("data:image/jpeg;base64,")
    with Image.open(BytesIO(base64.b64decode(data_uri.split(",", 1)[1]))) as fused:
        with Image.open(first) as a, Image.open(second) as b:
            assert fused.width == a.width + b.width
            assert fused.height == a.height == b.height


def test_fusion_flag_single_image_is_unchanged(monkeypatch, tmp_path):
    first = _write_image(tmp_path / "first.png", "PNG", "red")
    monkeypatch.setenv("SCIVER_FUSE_EVIDENCE_IMAGES", "1")

    messages = build_remote_messages("Inspect the figure.", [first])

    content = messages[-1]["content"]
    assert len(content) == 2  # one image + text
    assert _image_url(messages, 0).startswith("data:image/png;base64,")


def test_fusion_flag_off_by_default(tmp_path):
    first = _write_image(tmp_path / "first.png", "PNG", "red")
    second = _write_image(tmp_path / "second.png", "PNG", "blue")

    messages = build_remote_messages("Compare the figures.", [first, second])

    assert len(messages[-1]["content"]) == 3  # two images + text, unchanged


def test_fusion_flag_non_one_is_ignored(monkeypatch, tmp_path):
    first = _write_image(tmp_path / "first.png", "PNG", "red")
    second = _write_image(tmp_path / "second.png", "PNG", "blue")
    monkeypatch.setenv("SCIVER_FUSE_EVIDENCE_IMAGES", "0")

    messages = build_remote_messages("Compare the figures.", [first, second])

    assert len(messages[-1]["content"]) == 3


def test_missing_image_has_clear_error(tmp_path):
    missing_path = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        build_remote_messages("Inspect the figure.", [missing_path])


def test_empty_image_has_clear_error(tmp_path):
    empty_path = tmp_path / "empty.png"
    empty_path.touch()

    with pytest.raises(EmptyImageError, match="empty"):
        build_remote_messages("Inspect the figure.", [empty_path])


def test_unsupported_format_has_clear_error(tmp_path):
    unsupported_path = _write_image(tmp_path / "figure.gif", "GIF", "red")

    with pytest.raises(
        UnsupportedImageFormatError,
        match="Unsupported decoded image format",
    ):
        build_remote_messages("Inspect the figure.", [unsupported_path])


def test_unreadable_image_has_clear_error(tmp_path):
    invalid_path = tmp_path / "invalid.png"
    invalid_path.write_bytes(b"not an image")

    with pytest.raises(UnreadableImageError, match="Unable to decode"):
        build_remote_messages("Inspect the figure.", [invalid_path])
