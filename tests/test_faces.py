from unittest.mock import patch, MagicMock


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
