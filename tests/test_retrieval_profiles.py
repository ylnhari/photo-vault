"""Synthetic Chroma integration checks for the active retrieval contract."""
import json

import pytest


def test_profile_collection_uses_cosine_and_chroma_ids_limit_candidates(tmp_path, monkeypatch):
    import db
    import embeddings

    registry_path = tmp_path / "embedding_registry.json"
    monkeypatch.setattr(embeddings, "EMBEDDING_REGISTRY_PATH", str(registry_path))
    monkeypatch.setattr(db, "CHROMA_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(db, "_client", None)
    client = None
    try:
        embeddings.register_model("lm_studio", "nomic-embed-text", 2)
        info = embeddings.get_registry()["models"]["nomic-embed-text"]
        collection = db.collection("nomic-embed-text", model_info=info)
        assert collection.metadata["hnsw:space"] == "cosine"
        assert collection.metadata["embedding_profile_id"] == info["profile_id"]

        ids = [f"global-{n}" for n in range(8)] + ["person-close", "person-far"]
        vectors = [[1.0, n / 100.0] for n in range(8)] + [[0.7, 0.714], [0.1, 0.995]]
        collection.add(
            ids=ids,
            embeddings=vectors,
            metadatas=[{"kind": "global"}] * 8 + [
                {"kind": "person"}, {"kind": "person"}
            ],
        )
        # Without candidate constraints the top result comes from the global
        # distractors. With ids applied inside Chroma, top_k ranks only this
        # person's two photos before selecting the closest one.
        constrained = collection.query(
            query_embeddings=[[1.0, 0.0]],
            ids=["person-close", "person-far"],
            n_results=1,
            include=["metadatas", "distances"],
        )
        assert constrained["ids"][0] == ["person-close"]
    finally:
        client = db._client
        if client is not None:
            client.close()
        db._client = None
