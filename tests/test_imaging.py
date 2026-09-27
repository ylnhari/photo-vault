from concurrent.futures import ThreadPoolExecutor
import time

from PIL import Image

import imaging


def _jpeg(path, size=(600, 400), orientation=None):
    im = Image.new("RGB", size, (30, 100, 180))
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    im.save(path, "JPEG", exif=exif)


def test_derivative_applies_exif_orientation_and_caps_dimensions(tmp_path):
    src = tmp_path / "rotated.jpg"
    out = tmp_path / "thumb.webp"
    _jpeg(src, orientation=6)

    assert imaging.ensure_derivative(str(src), str(out), 100)
    with Image.open(out) as result:
        assert result.size == (67, 100)


def test_concurrent_derivative_requests_decode_once(tmp_path, monkeypatch):
    src = tmp_path / "photo.jpg"
    out = tmp_path / "thumb.webp"
    _jpeg(src)
    original = imaging.safe_open
    calls = 0

    def counted(path):
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return original(path)

    monkeypatch.setattr(imaging, "safe_open", counted)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(
            lambda _: imaging.ensure_derivative(str(src), str(out), 100), range(6)
        ))

    assert results == [True] * 6
    assert calls == 1
    with Image.open(out) as result:
        result.verify()


def test_failed_derivative_does_not_publish_partial_output(tmp_path, monkeypatch):
    src = tmp_path / "photo.jpg"
    out = tmp_path / "thumb.webp"
    _jpeg(src)
    original = Image.Image.save

    def fail_temp(self, fp, *args, **kwargs):
        if isinstance(fp, str) and ".pv_derivative_" in fp:
            raise OSError("synthetic write failure")
        return original(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", fail_temp)
    assert imaging.ensure_derivative(str(src), str(out), 100) is False
    assert not out.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["photo.jpg"]
