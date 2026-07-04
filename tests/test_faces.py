import json
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


def test_index_faces_upserts_each_face():
    import faces
    col = MagicMock()
    with patch("faces.db.faces_collection", return_value=col):
        faces.index_faces("img1", [
            {"embedding": [0.1, 0.2], "bbox": [0, 0, 10, 10]},
            {"embedding": [0.3, 0.4], "bbox": [5, 5, 15, 15]},
        ])
    col.delete.assert_called_once()  # clears prior entries for this image
    args = col.upsert.call_args[1]
    assert args["ids"] == ["img1:0", "img1:1"]
    assert args["metadatas"][0] == {"image_id": "img1", "face_index": 0}


def test_index_faces_empty_skips_upsert():
    import faces
    col = MagicMock()
    with patch("faces.db.faces_collection", return_value=col):
        faces.index_faces("img1", [])
    col.upsert.assert_not_called()


def test_query_person_faces_filters_by_distance():
    import faces
    col = MagicMock()
    col.count.return_value = 3
    col.query.return_value = {
        "metadatas": [[{"image_id": "a"}, {"image_id": "b"}, {"image_id": "c"}]],
        "distances": [[0.1, 0.9, 0.35]],
    }
    # SIMILARITY_THRESHOLD 0.6 → distance cutoff 0.4 → keep a (0.1) and c (0.35)
    with patch("faces.db.faces_collection", return_value=col):
        matched = faces.query_person_faces([0.1, 0.2])
    assert matched == {"a", "c"}


def test_query_person_faces_empty_index_returns_empty():
    import faces
    col = MagicMock()
    col.count.return_value = 0
    with patch("faces.db.faces_collection", return_value=col), \
         patch("faces.rebuild_face_index", return_value=0):
        assert faces.query_person_faces([0.1]) == set()


# ── EXIF orientation (fix #1) ───────────────────────────────────────────────
# All three decode paths in _read_image_bgr (cv2.imread, the cv2.imdecode
# fallback for unicode paths, and the PIL fallback for HEIC/anything cv2
# can't decode) must land in the SAME "display-correct" orientation — the
# one PIL.ImageOps.exif_transpose() produces — since that's the coordinate
# space face bboxes get computed in, and the contract the api.py face-crop
# endpoint depends on.

def _make_exif_image(tmp_path, orientation, w=60, h=40, name=None):
    from PIL import Image
    img = Image.new("RGB", (w, h), color=(10, 20, 30))
    exif = img.getexif()
    exif[274] = orientation
    path = tmp_path / (name or f"exif_{orientation}.jpg")
    img.save(str(path), exif=exif)
    return str(path)


def test_read_image_bgr_cv2imread_path_matches_exif_transpose(tmp_path):
    import faces
    from PIL import Image, ImageOps
    path = _make_exif_image(tmp_path, orientation=6)
    expected = ImageOps.exif_transpose(Image.open(path)).size  # (w, h)

    img = faces._read_image_bgr(path)

    assert img is not None
    assert (img.shape[1], img.shape[0]) == expected


def test_read_image_bgr_imdecode_fallback_matches_exif_transpose(tmp_path):
    """Forces cv2.imread to fail (as it does for non-ASCII Windows paths),
    exercising the cv2.imdecode fallback branch specifically."""
    import faces
    from PIL import Image, ImageOps
    for orientation in (1, 3, 6, 8):
        path = _make_exif_image(tmp_path, orientation, name=f"a_{orientation}.jpg")
        expected = ImageOps.exif_transpose(Image.open(path)).size

        with patch("faces.cv2.imread", return_value=None):
            img = faces._read_image_bgr(path)

        assert img is not None, f"orientation {orientation}"
        assert (img.shape[1], img.shape[0]) == expected, f"orientation {orientation}"


def test_read_image_bgr_pil_fallback_matches_exif_transpose(tmp_path):
    """Forces both cv2.imread and cv2.imdecode to fail, exercising the PIL
    (imaging.safe_open) fallback used for HEIC/anything cv2 can't decode."""
    import faces
    from PIL import Image, ImageOps
    for orientation in (1, 3, 6, 8):
        path = _make_exif_image(tmp_path, orientation, name=f"b_{orientation}.jpg")
        expected = ImageOps.exif_transpose(Image.open(path)).size

        with patch("faces.cv2.imread", return_value=None), \
             patch("faces.cv2.imdecode", return_value=None):
            img = faces._read_image_bgr(path)

        assert img is not None, f"orientation {orientation}"
        assert (img.shape[1], img.shape[0]) == expected, f"orientation {orientation}"


def test_detect_and_embed_faces_bbox_in_exif_corrected_space(tmp_path):
    """The image fed to the detector — and therefore the space face bboxes
    are computed in — must be the same size ImageOps.exif_transpose()
    would produce (fix #2's contract with api.py's face-crop endpoint)."""
    import faces
    from PIL import Image, ImageOps
    path = _make_exif_image(tmp_path, orientation=6)
    expected_w, expected_h = ImageOps.exif_transpose(Image.open(path)).size

    fake_face = MagicMock()
    fake_face.bbox.tolist.return_value = [5, 5, expected_w - 5, expected_h - 5]
    fake_face.embedding.tolist.return_value = [0.1, 0.2]
    fake_app = MagicMock()
    fake_app.get.return_value = [fake_face]

    with patch("faces._get_app", return_value=fake_app):
        result = faces.detect_and_embed_faces(path)

    assert len(result) == 1
    x1, y1, x2, y2 = result[0]["bbox"]
    assert 0 <= x1 < x2 <= expected_w
    assert 0 <= y1 < y2 <= expected_h
    fed_img = fake_app.get.call_args[0][0]
    assert (fed_img.shape[1], fed_img.shape[0]) == (expected_w, expected_h)


def test_detect_and_embed_faces_reraises_unexpected_detector_errors():
    """A genuinely unexpected error from the detector must propagate — not
    be swallowed into an indistinguishable-from-real-zero-faces []."""
    import faces
    fake_app = MagicMock()
    fake_app.get.side_effect = KeyError("unexpected")
    with patch("faces._get_app", return_value=fake_app), \
         patch("faces._read_image_bgr", return_value=np.zeros((10, 10, 3), dtype="uint8")):
        with pytest.raises(KeyError):
            faces.detect_and_embed_faces("whatever.jpg")


def test_detect_and_embed_faces_swallows_expected_detector_errors():
    import faces
    fake_app = MagicMock()
    fake_app.get.side_effect = RuntimeError("onnxruntime blew up")
    with patch("faces._get_app", return_value=fake_app), \
         patch("faces._read_image_bgr", return_value=np.zeros((10, 10, 3), dtype="uint8")):
        assert faces.detect_and_embed_faces("whatever.jpg") == []


# ── rebuild_face_index orphan cross-check (fix #12) ─────────────────────────

def test_rebuild_face_index_skips_and_removes_orphaned_files(tmp_path):
    import faces
    face_dir = tmp_path / "faces"
    face_dir.mkdir()
    (face_dir / "live.json").write_text(json.dumps([{"embedding": [0.1, 0.2], "bbox": [0, 0, 1, 1]}]))
    (face_dir / "ghost.json").write_text(json.dumps([{"embedding": [0.3, 0.4], "bbox": [0, 0, 1, 1]}]))

    col = MagicMock()
    col.get.return_value = {"ids": []}

    with patch("faces.FACE_DIR", str(face_dir)), \
         patch("faces.db.faces_collection", return_value=col), \
         patch("faces._live_catalog_ids", return_value={"live"}):
        total = faces.rebuild_face_index()

    assert total == 1
    upserted_image_ids = {m["image_id"] for c in col.upsert.call_args_list for m in c.kwargs["metadatas"]}
    assert upserted_image_ids == {"live"}
    assert not (face_dir / "ghost.json").exists()
    assert (face_dir / "live.json").exists()


def test_rebuild_face_index_skips_orphan_check_when_catalog_unreadable(tmp_path):
    """If the catalog can't be read at all, don't guess — keep every face
    file rather than risk mass-deleting valid ones."""
    import faces
    face_dir = tmp_path / "faces"
    face_dir.mkdir()
    (face_dir / "a.json").write_text(json.dumps([{"embedding": [0.1, 0.2], "bbox": [0, 0, 1, 1]}]))

    col = MagicMock()
    col.get.return_value = {"ids": []}

    with patch("faces.FACE_DIR", str(face_dir)), \
         patch("faces.db.faces_collection", return_value=col), \
         patch("faces._live_catalog_ids", return_value=None):
        total = faces.rebuild_face_index()

    assert total == 1
    assert (face_dir / "a.json").exists()
