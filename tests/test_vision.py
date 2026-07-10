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
    "person_count": 2,
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


def test_strip_markdown_case_insensitive_language_tag():
    from vision import _strip_markdown
    assert _strip_markdown('```JSON\n{"a":1}\n```') == '{"a":1}'
    assert _strip_markdown('```Json5\n{"a":1}\n```') == '{"a":1}'


def test_strip_markdown_does_not_eat_leading_json_content():
    """Regression: the old lstrip('json') stripped individual j/s/o/n
    characters, not the word — it would mangle real JSON content starting
    with those letters (e.g. a caption beginning with "no" or "on")."""
    from vision import _strip_markdown
    # No language tag on the fence — nothing should be stripped besides ``` .
    assert _strip_markdown('```\n{"caption":"nice"}\n```') == '{"caption":"nice"}'


# ── _salvage_json ─────────────────────────────────────────────────────────────

def test_salvage_json_extracts_from_preamble_and_postamble():
    from vision import _salvage_json
    assert _salvage_json('Here you go:\n{"a":1}\nHope that helps') == '{"a":1}'


def test_salvage_json_returns_none_on_truncated():
    from vision import _salvage_json
    # cut mid-object: no matching close brace → unrecoverable (escalation signal)
    assert _salvage_json('{"caption":"a long ca') is None


def test_salvage_json_passthrough_clean():
    from vision import _salvage_json
    assert _salvage_json('{"a":1}') == '{"a":1}'


# ── token-budget escalation ───────────────────────────────────────────────────

def test_escalation_grows_budget_on_truncation_then_succeeds():
    from vision import _caption_with_escalation
    seen = []
    def call(budget):
        seen.append(budget)
        # truncated at the first (default) budget, complete at the next
        if len(seen) == 1:
            return ('{"caption":"cut off mid fie', "max_tokens")
        return ('{"caption":"done"}', "stop")
    out = _caption_with_escalation(call, "test")
    assert out == '{"caption":"done"}'
    assert seen[1] == seen[0] * 2  # budget doubled on truncation


def test_escalation_raises_actionable_error_at_ceiling():
    from vision import _caption_with_escalation, VisionTruncated, _MAX_TOKENS_CEILING
    def call(budget):
        return ("{", "max_tokens")  # always truncated
    with pytest.raises(VisionTruncated, match="vision_max_tokens"):
        _caption_with_escalation(call, "test")


def test_escalation_no_retry_when_not_truncated():
    from vision import _caption_with_escalation
    calls = []
    def call(budget):
        calls.append(budget)
        return ("this is not json at all", "stop")
    with pytest.raises(RuntimeError, match="unparseable"):
        _caption_with_escalation(call, "test")
    assert len(calls) == 1  # a non-truncation malformation is not retried


def test_escalation_http_error_propagates_for_provider_fallback():
    import urllib.error
    from vision import _caption_with_escalation
    def call(budget):
        raise urllib.error.HTTPError("u", 429, "quota", {}, None)
    with pytest.raises(urllib.error.HTTPError):
        _caption_with_escalation(call, "test")


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

    with patch("vision._get_lm_client", return_value=mock_client), \
         patch("vision.list_lm_studio_models_v0", return_value=[]):
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
         patch("vision.list_lm_studio_models_v0", return_value=[]), \
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
         patch("vision.list_lm_studio_models_v0", return_value=[]), \
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

    with patch("vision._get_lm_client", return_value=mock_client), \
         patch("vision.list_lm_studio_models_v0", return_value=[]):
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
    assert attrs["person_count"] == 2


def test_parse_person_count_defaults_to_zero_when_missing():
    from vision import parse_vision_attributes
    attrs = parse_vision_attributes(json.dumps({"caption": "hello"}))
    assert attrs["person_count"] == 0


def test_parse_person_count_coerces_numeric_string():
    from vision import parse_vision_attributes
    attrs = parse_vision_attributes(json.dumps({"person_count": "3"}))
    assert attrs["person_count"] == 3


def test_parse_person_count_falls_back_to_zero_on_garbage():
    from vision import parse_vision_attributes
    attrs = parse_vision_attributes(json.dumps({"person_count": "a lot"}))
    assert attrs["person_count"] == 0


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


def test_parse_list_fields_join_to_comma_string():
    from vision import parse_vision_attributes
    data = json.dumps({"animals": ["dog", "cat"], "vehicles": ["car"], "dominant_colors": ["red", "blue"]})
    attrs = parse_vision_attributes(data)
    assert attrs["animals"] == "dog, cat"
    assert attrs["vehicles"] == "car"
    assert attrs["dominant_colors"] == "red, blue"


def test_parse_list_fields_default_empty_string_when_missing():
    from vision import parse_vision_attributes
    attrs = parse_vision_attributes(json.dumps({"caption": "x"}))
    assert attrs["animals"] == ""
    assert attrs["food_items"] == ""
    assert attrs["activities"] == ""


def test_parse_list_field_garbage_type_falls_back_to_empty():
    from vision import parse_vision_attributes
    attrs = parse_vision_attributes(json.dumps({"animals": {"weird": "dict"}}))
    assert attrs["animals"] == ""


def test_parse_scalar_field_with_unexpected_list_is_coerced_not_left_as_list():
    """Chroma metadata only accepts scalars. If a model hands back a list for
    a field we expect to be scalar (not one of the known _LIST_KEYS), it must
    be coerced to a string, not passed through — a raw list here would crash
    build_embed_payload's Chroma add()/upsert() call downstream."""
    from vision import parse_vision_attributes
    data = json.dumps({"caption": ["a", "b"], "scene": "outdoor"})
    attrs = parse_vision_attributes(data)
    assert isinstance(attrs["caption"], str)
    assert attrs["caption"] == "a, b"


def test_parse_scalar_field_with_unexpected_dict_is_coerced():
    from vision import parse_vision_attributes
    data = json.dumps({"mood": {"primary": "happy", "secondary": "excited"}})
    attrs = parse_vision_attributes(data)
    assert isinstance(attrs["mood"], str)
    assert "happy" in attrs["mood"]


def test_parse_new_scalar_fields():
    from vision import parse_vision_attributes
    data = json.dumps({
        "festival_name": "Diwali", "photo_type": "selfie",
        "text_in_image": "SALE 50%", "landmark": "Taj Mahal",
    })
    attrs = parse_vision_attributes(data)
    assert attrs["festival_name"] == "Diwali"
    assert attrs["photo_type"] == "selfie"
    assert attrs["text_in_image"] == "SALE 50%"
    assert attrs["landmark"] == "Taj Mahal"


# ── build_embedding_text ─────────────────────────────────────────────────────

def test_build_embedding_text_is_plain_sentences_not_json():
    from vision import parse_vision_attributes, build_embedding_text
    attrs = parse_vision_attributes(VALID_JSON)
    text = build_embedding_text(attrs)
    assert "{" not in text and "}" not in text and '"' not in text
    assert "A sunny beach day" in text


def test_build_embedding_text_includes_animals_and_activities():
    from vision import parse_vision_attributes, build_embedding_text
    data = json.dumps({
        "caption": "A dog runs on the beach.",
        "animals": ["dog"], "activities": ["running", "swimming"],
    })
    attrs = parse_vision_attributes(data)
    text = build_embedding_text(attrs)
    assert "dog" in text
    assert "running, swimming" in text


def test_build_embedding_text_skips_unknown_and_empty_fields():
    from vision import parse_vision_attributes, build_embedding_text
    attrs = parse_vision_attributes(json.dumps({"caption": "x"}))
    text = build_embedding_text(attrs)
    assert "unknown" not in text.lower()


def test_encode_image_converts_rgba_png(tmp_path):
    """RGBA (and P/LA) images must be converted before the JPEG encode —
    otherwise every screenshot-style PNG fails vision with 'encoding failed'."""
    from vision import encode_image
    import base64
    img = Image.new("RGBA", (32, 32), color=(255, 0, 0, 128))
    img_path = tmp_path / "shot.png"
    img.save(img_path, format="PNG")
    result = encode_image(str(img_path))
    assert result is not None
    decoded = Image.open(io.BytesIO(base64.b64decode(result)))
    assert decoded.format == "JPEG"


# ── 9Router ───────────────────────────────────────────────────────────────────

def _9r_chat_response(text: str, served_model: str = "gc/gemini-2.5-flash"):
    resp = MagicMock()
    resp.model = served_model
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    return resp


def test_call_9router_requires_model():
    from vision import _call_9router
    with pytest.raises(ValueError, match="explicit vision model"):
        _call_9router("base64data", None)


def test_call_9router_success_sends_stream_false():
    import vision
    client = MagicMock()
    client.chat.completions.create.return_value = _9r_chat_response(VALID_JSON)
    with patch.object(vision, "_get_9router_client", return_value=client), \
         patch.dict(vision._9r_cooldown, {}, clear=True):
        out, used = vision._call_9router("base64data", "gc/gemini-2.5-flash")
    assert out == VALID_JSON
    assert used == "gc/gemini-2.5-flash"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gc/gemini-2.5-flash"
    assert kwargs["stream"] is False


def test_call_9router_echoed_model_keeps_requested_id():
    """Served name is usually the requested id minus the provider prefix —
    keep the requested id (it carries strictly more information)."""
    import vision
    client = MagicMock()
    client.chat.completions.create.return_value = _9r_chat_response(
        VALID_JSON, served_model="gemini-2.5-flash")
    with patch.object(vision, "_get_9router_client", return_value=client), \
         patch.dict(vision._9r_cooldown, {}, clear=True):
        _, used = vision._call_9router("base64data", "gc/gemini-2.5-flash")
    assert used == "gc/gemini-2.5-flash"


def test_call_9router_substitution_reports_served_model():
    """The gateway substituted a different upstream model → the caption is
    kept but attributed to the model that actually produced it."""
    import vision
    client = MagicMock()
    client.chat.completions.create.return_value = _9r_chat_response(
        VALID_JSON, served_model="gemini-3.1-flash-lite")
    with patch.object(vision, "_get_9router_client", return_value=client), \
         patch.dict(vision._9r_cooldown, {}, clear=True):
        text, used = vision._call_9router("base64data", "gc/gemini-2.5-flash-lite")
    assert text == VALID_JSON
    assert used == "gemini-3.1-flash-lite"


def test_call_9router_429_sets_cooldown_and_skips_retry():
    import vision
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("Error code: 429 - quota")
    with patch.object(vision, "_get_9router_client", return_value=client), \
         patch.dict(vision._9r_cooldown, {}, clear=True):
        with pytest.raises(RuntimeError):
            vision._call_9router("base64data", "gc/gemini-2.5-flash")
        assert "gc/gemini-2.5-flash" in vision._9r_cooldown
        # second call short-circuits on the cooldown without hitting the client
        with pytest.raises(RuntimeError, match="cooldown"):
            vision._call_9router("base64data", "gc/gemini-2.5-flash")
    assert client.chat.completions.create.call_count == 1


def test_get_image_caption_9router_labels_requested_model(tmp_path):
    import vision
    img_path = tmp_path / "t.jpg"
    img_path.write_bytes(_make_tiny_image_bytes())
    client = MagicMock()
    client.chat.completions.create.return_value = _9r_chat_response(VALID_JSON)
    with patch.object(vision, "_get_9router_client", return_value=client), \
         patch.dict(vision._9r_cooldown, {}, clear=True):
        text, label = vision.get_image_caption(
            str(img_path), force_provider="9router",
            model="gc/gemini-2.5-flash", with_model=True,
        )
    assert text == VALID_JSON
    assert label == "9router:gc/gemini-2.5-flash"


def test_get_image_caption_9router_substitution_labels_served_model(tmp_path):
    """Caption from a substituted model is stored under the ACTUAL model's
    label — per-model bookkeeping must not lie about provenance."""
    import vision
    img_path = tmp_path / "t.jpg"
    img_path.write_bytes(_make_tiny_image_bytes())
    client = MagicMock()
    client.chat.completions.create.return_value = _9r_chat_response(
        VALID_JSON, served_model="gemini-3.1-flash-lite")
    with patch.object(vision, "_get_9router_client", return_value=client), \
         patch.dict(vision._9r_cooldown, {}, clear=True):
        text, label = vision.get_image_caption(
            str(img_path), force_provider="9router",
            model="gc/gemini-2.5-flash-lite", with_model=True,
        )
    assert text == VALID_JSON
    assert label == "9router:gemini-3.1-flash-lite"


def test_get_image_caption_9router_without_model_errors_no_fallback(tmp_path):
    """No silent auto-pick and no cross-provider fallback for 9Router."""
    import vision
    img_path = tmp_path / "t.jpg"
    img_path.write_bytes(_make_tiny_image_bytes())
    gem = MagicMock()
    with patch.object(vision, "_call_gemini", gem), \
         patch.object(vision, "_call_lm_studio", gem):
        text, label = vision.get_image_caption(
            str(img_path), force_provider="9router", with_model=True,
        )
    assert label == "error"
    assert "9Router" in json.loads(text)["error"]
    gem.assert_not_called()


def test_list_9router_vision_models_filters_and_excludes():
    import vision
    ids = [
        "gc/gemini-2.5-flash", "gemini/gemma-4-31b-it", "kr/claude-sonnet-4.5",
        "kr/claude-sonnet-4.5-thinking", "kr/claude-sonnet-4.5-agentic",
        "kr/deepseek-3.2", "oc/some-coder",
    ]
    class _Resp:
        def read(self): return json.dumps({"data": [{"id": i} for i in ids]}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): pass
    with patch("urllib.request.urlopen", return_value=_Resp()), \
         patch.object(vision, "_9r_models_cache", None):
        models = vision.list_9router_vision_models()
    assert "gc/gemini-2.5-flash" in models
    assert "gemini/gemma-4-31b-it" in models
    assert "kr/claude-sonnet-4.5" in models
    assert "kr/claude-sonnet-4.5-thinking" not in models
    assert "kr/claude-sonnet-4.5-agentic" not in models
    assert "kr/deepseek-3.2" not in models
    assert "oc/some-coder" not in models


def test_list_9router_vision_models_offline_returns_empty():
    import vision
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")), \
         patch.object(vision, "_9r_models_cache", None):
        assert vision.list_9router_vision_models() == []


# ── _lm_model_id honesty (bug: arbitrary model claimed when none loaded) ─────

def test_lm_model_id_no_arbitrary_claim_when_v0_says_nothing_loaded():
    """v0 API reachable + nothing loaded → bare 'lm_studio', never the first
    /v1/models entry (which can be a not-loaded embedding model)."""
    import vision
    v0 = [
        {"id": "text-embedding-foo", "type": "embeddings", "state": "not-loaded"},
        {"id": "some-vlm", "type": "vlm", "state": "not-loaded"},
    ]
    client = MagicMock()
    with patch.object(vision, "list_lm_studio_models_v0", return_value=v0), \
         patch.object(vision, "_get_lm_client", return_value=client):
        assert vision._lm_model_id() == "lm_studio"
    client.models.list.assert_not_called()


def test_lm_model_id_uses_v1_guess_only_when_v0_unavailable():
    import vision
    m = MagicMock(); m.id = "first-model"
    client = MagicMock()
    client.models.list.return_value = MagicMock(data=[m])
    with patch.object(vision, "list_lm_studio_models_v0", return_value=[]), \
         patch.object(vision, "_get_lm_client", return_value=client):
        assert vision._lm_model_id() == "lm_studio:first-model"
