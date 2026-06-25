import json
import pytest
import urllib.error
from unittest.mock import patch, MagicMock, call
from PIL import Image
import io


# ── helpers ──────────────────────────────────────────────────────────────────

VALID_JSON = json.dumps({
    "caption": "A sunny beach day",
    "scene": "outdoor",
    "location_type": "beach",
    "weather": "sunny",
    "season": "summer",
    "time_of_day": "afternoon",
    "occasion": "vacation",
    "group_size": "couple",
    "clothing_style": "swimwear",
    "mood": "happy",
    "objects": ["umbrella", "towel"],
    "people_description": "Two people relaxing"
})


def _make_tiny_image_bytes():
    img = Image.new("RGB", (10, 10), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _gemini_response(text: str):
    class _Resp:
        def read(self): return json.dumps({
            "candidates": [{"content": {"parts": [{"text": text}]}}]
        }).encode()
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return _Resp()


def _http_error(code: int):
    err = urllib.error.HTTPError(url="", code=code, msg="", hdrs=None, fp=None)
    err.read = lambda: b"error"
    return err


# ── encode_image ──────────────────────────────────────────────────────────────

def test_encode_image_returns_base64(tmp_path):
    from vision import encode_image
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(_make_tiny_image_bytes())
    result = encode_image(str(img_path))
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 10


def test_encode_image_missing_file_returns_none():
    from vision import encode_image
    result = encode_image("/nonexistent/path/image.jpg")
    assert result is None


def test_encode_image_respects_max_size(tmp_path):
    from vision import encode_image
    import base64
    img = Image.new("RGB", (2000, 2000), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_path = tmp_path / "large.jpg"
    img_path.write_bytes(buf.getvalue())
    result = encode_image(str(img_path), max_size=(64, 64))
    # decode and check size
    raw = base64.b64decode(result)
    decoded = Image.open(io.BytesIO(raw))
    assert decoded.width <= 64 and decoded.height <= 64


# ── _strip_markdown ────────────────────────────────────────────────────────────

def test_strip_markdown_removes_code_fence():
    from vision import _strip_markdown
    assert _strip_markdown("```json\n{\"a\":1}\n```") == '{"a":1}'


def test_strip_markdown_passthrough_plain():
    from vision import _strip_markdown
    assert _strip_markdown('{"a":1}') == '{"a":1}'


# ── _call_gemini ──────────────────────────────────────────────────────────────

def test_call_gemini_tries_lite_first():
    from vision import _call_gemini, GEMINI_VISION_MODELS
    called_models = []
    def fake_urlopen(req, timeout=None):
        model = req.full_url.split("/models/")[1].split(":")[0]
        called_models.append(model)
        return _gemini_response(VALID_JSON)
    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("vision.GEMINI_API_KEY", "fake-key"):
        _call_gemini("b64data")
    assert called_models[0] == GEMINI_VISION_MODELS[0], "Must try first (lite) model first"


def test_call_gemini_skips_429_tries_next():
    from vision import _call_gemini, GEMINI_VISION_MODELS
    call_count = [0]
    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] == 1:
            raise _http_error(429)
        return _gemini_response(VALID_JSON)
    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("vision.GEMINI_API_KEY", "fake-key"):
        text, model_used = _call_gemini("b64data")
    assert call_count[0] == 2
    assert "caption" in text or text == VALID_JSON
    assert model_used == GEMINI_VISION_MODELS[1]  # second model after the 429


def test_call_gemini_all_429_raises():
    from vision import _call_gemini
    with patch("urllib.request.urlopen", side_effect=_http_error(429)), \
         patch("vision.GEMINI_API_KEY", "fake-key"):
        with pytest.raises(RuntimeError, match="exhausted"):
            _call_gemini("b64data")


def test_call_gemini_no_key_raises():
    from vision import _call_gemini
    with patch("vision.GEMINI_API_KEY", ""):
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            _call_gemini("b64data")


def test_call_gemini_400_propagates():
    from vision import _call_gemini
    with patch("urllib.request.urlopen", side_effect=_http_error(400)), \
         patch("vision.GEMINI_API_KEY", "fake-key"):
        with pytest.raises(RuntimeError, match="400"):
            _call_gemini("b64data")


# ── get_image_caption ─────────────────────────────────────────────────────────

def test_get_image_caption_uses_lm_studio_first(tmp_path):
    from vision import get_image_caption
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(_make_tiny_image_bytes())

    mock_response = MagicMock()
    mock_response.choices[0].message.content = VALID_JSON
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("vision._get_lm_client", return_value=mock_client):
        result = get_image_caption(str(img_path))

    assert result == VALID_JSON
    mock_client.chat.completions.create.assert_called_once()


def test_get_image_caption_falls_back_on_connection_error(tmp_path):
    from vision import get_image_caption
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(_make_tiny_image_bytes())

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = ConnectionRefusedError("refused")

    with patch("vision._get_lm_client", return_value=mock_client), \
         patch("vision._call_gemini", return_value=(VALID_JSON, "gemini-2.0-flash-lite")) as mock_gemini:
        result = get_image_caption(str(img_path))

    mock_gemini.assert_called_once()
    assert result == VALID_JSON


def test_get_image_caption_no_fallback_on_non_connection_error(tmp_path):
    from vision import get_image_caption
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(_make_tiny_image_bytes())

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = ValueError("bad model name")

    with patch("vision._get_lm_client", return_value=mock_client), \
         patch("vision._call_gemini") as mock_gemini:
        result = get_image_caption(str(img_path))

    mock_gemini.assert_not_called()
    assert "error" in json.loads(result)


def test_get_image_caption_missing_image_returns_error():
    from vision import get_image_caption
    result = get_image_caption("/no/such/file.jpg")
    assert "error" in json.loads(result)


def test_get_image_caption_with_model_returns_tuple(tmp_path):
    from vision import get_image_caption
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(_make_tiny_image_bytes())

    mock_response = MagicMock()
    mock_response.choices[0].message.content = VALID_JSON
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_client.models.list.return_value.data = [MagicMock(id="qwen2-vl")]

    with patch("vision._get_lm_client", return_value=mock_client):
        text, label = get_image_caption(str(img_path), with_model=True)

    assert text == VALID_JSON
    assert label == "lm_studio:qwen2-vl"


def test_get_image_caption_with_model_gemini_label(tmp_path):
    from vision import get_image_caption
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(_make_tiny_image_bytes())

    with patch("vision._call_gemini", return_value=(VALID_JSON, "gemini-2.0-flash-lite")):
        text, label = get_image_caption(str(img_path), force_provider="gemini", with_model=True)

    assert text == VALID_JSON
    assert label == "gemini:gemini-2.0-flash-lite"


def test_get_image_caption_with_model_error_label():
    from vision import get_image_caption
    text, label = get_image_caption("/no/such/file.jpg", with_model=True)
    assert "error" in json.loads(text)
    assert label == "error"


def test_get_image_caption_lm_studio_explicit_model(tmp_path):
    from vision import get_image_caption
    img_path = tmp_path / "t.jpg"
    img_path.write_bytes(_make_tiny_image_bytes())
    with patch("vision._call_lm_studio", return_value=VALID_JSON) as call:
        text, label = get_image_caption(str(img_path), force_provider="lm_studio",
                                        with_model=True, model="qwen-vl")
    assert label == "lm_studio:qwen-vl"
    assert call.call_args[0][1] == "qwen-vl"  # model forwarded


def test_get_image_caption_gemini_explicit_model(tmp_path):
    from vision import get_image_caption
    img_path = tmp_path / "t.jpg"
    img_path.write_bytes(_make_tiny_image_bytes())
    with patch("vision._call_gemini", return_value=(VALID_JSON, "gemini-2.0-flash")) as call:
        text, label = get_image_caption(str(img_path), force_provider="gemini",
                                        with_model=True, model="gemini-2.0-flash")
    assert label == "gemini:gemini-2.0-flash"
    assert call.call_args[0][1] == "gemini-2.0-flash"


def test_call_gemini_single_model_when_specified():
    from vision import _call_gemini
    calls = []
    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url.split("/models/")[1].split(":")[0])
        return _gemini_response(VALID_JSON)
    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("vision.GEMINI_API_KEY", "fake-key"):
        _call_gemini("b64", model="gemini-2.5-flash")
    assert calls == ["gemini-2.5-flash"]  # only the requested model, no cascade


def test_list_lm_studio_models():
    from vision import list_lm_studio_models
    from unittest.mock import MagicMock
    client = MagicMock()
    client.models.list.return_value.data = [MagicMock(id="a"), MagicMock(id="b")]
    with patch("vision._get_lm_client", return_value=client):
        assert list_lm_studio_models() == ["a", "b"]


# ── parse_vision_attributes ───────────────────────────────────────────────────

def test_parse_valid_json():
    from vision import parse_vision_attributes
    attrs = parse_vision_attributes(VALID_JSON)
    assert attrs["caption"] == "A sunny beach day"
    assert attrs["scene"] == "outdoor"
    assert attrs["weather"] == "sunny"
    assert attrs["mood"] == "happy"


def test_parse_objects_list_to_string():
    from vision import parse_vision_attributes
    data = json.dumps({"objects": ["umbrella", "towel"], "caption": ""})
    attrs = parse_vision_attributes(data)
    assert attrs["objects"] == "umbrella, towel"


def test_parse_invalid_json_returns_defaults():
    from vision import parse_vision_attributes
    attrs = parse_vision_attributes("not valid json {{{")
    assert attrs["scene"] == "unknown"
    assert attrs["weather"] == "unknown"
    assert attrs["caption"] == ""


def test_parse_partial_json_fills_defaults():
    from vision import parse_vision_attributes
    attrs = parse_vision_attributes(json.dumps({"caption": "hello"}))
    assert attrs["caption"] == "hello"
    assert attrs["scene"] == "unknown"
    assert attrs["mood"] == "unknown"


def test_parse_ignores_unknown_keys():
    from vision import parse_vision_attributes
    data = json.dumps({"caption": "test", "unknown_key": "value"})
    attrs = parse_vision_attributes(data)
    assert "unknown_key" not in attrs
