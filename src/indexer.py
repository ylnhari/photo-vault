import os
import json
import chromadb
from vision import get_image_caption, parse_vision_attributes
from embeddings import get_embedding, collection_name_for, get_active_model, get_registry
from faces import detect_and_embed_faces, save_face_data
from scanner import scan_directory
from constants import IMAGE_CATALOG_PATH, CHROMA_DB_PATH

RICH_ATTRIBUTES = ["weather", "occasion", "location_type", "scene", "mood"]


def _caption_has_error(text: str) -> bool:
    try:
        return bool(json.loads(text).get("error"))
    except Exception:
        return False


def _record_caption_history(img_data: dict, model: str, text: str):
    """Keep one caption per vision model. Re-running with the same model replaces it;
    a different model is appended. caption_json/caption_model always hold the latest."""
    hist = [h for h in img_data.get("caption_history", []) if h.get("model") != model]
    hist.append({"model": model, "caption_json": text})
    img_data["caption_history"] = hist
    img_data["caption_json"] = text
    img_data["caption_model"] = model


class Indexer:
    def __init__(self, target_directories=None):
        self.target_directories = target_directories or []
        self.image_catalog = self._load_image_catalog()

    def _load_image_catalog(self):
        if os.path.exists(IMAGE_CATALOG_PATH):
            with open(IMAGE_CATALOG_PATH, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            if isinstance(data, list):
                return {"images": {img["path"]: img for img in data if "path" in img}}
        return {"images": {}}

    def _save_catalog(self):
        with open(IMAGE_CATALOG_PATH, "w") as f:
            json.dump(self.image_catalog, f, indent=2)

    def _collection(self, model_name: str = None):
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        if model_name is None:
            model_name = get_active_model()
        col_name = collection_name_for(model_name) if model_name else "images"
        return client.get_or_create_collection(name=col_name)

    # ── Scanning ──────────────────────────────────────────────────────────────

    def scan_only(self):
        for d in self.target_directories:
            scan_directory(d, IMAGE_CATALOG_PATH)
        self.image_catalog = self._load_image_catalog()

    # ── Gap detection ─────────────────────────────────────────────────────────

    def get_missing(self) -> list[tuple]:
        existing_ids = set(self._collection().get()["ids"])
        return [
            (img_id, img_data)
            for img_id, img_data in self.image_catalog["images"].items()
            if img_id not in existing_ids
        ]

    def get_missing_attributes(self) -> list[tuple]:
        result = self._collection().get(include=["metadatas"])
        catalog = self.image_catalog.get("images", {})
        stale = []
        for img_id, meta in zip(result["ids"], result["metadatas"]):
            all_unknown = all(
                meta.get(attr, "unknown") in ("unknown", "", None)
                for attr in RICH_ATTRIBUTES
            )
            if all_unknown and img_id in catalog:
                stale.append((img_id, catalog[img_id]))
        return stale

    def get_missing_files(self) -> list[tuple]:
        """Catalog entries whose file path no longer exists on disk."""
        return [
            (img_id, data)
            for img_id, data in self.image_catalog.get("images", {}).items()
            if not os.path.exists(data.get("path", ""))
        ]

    def get_vision_pending(self) -> list[tuple]:
        """Images that have not yet been through vision analysis."""
        return [
            (img_id, img_data)
            for img_id, img_data in self.image_catalog["images"].items()
            if not img_data.get("caption_json")
        ]

    def get_embed_pending(self) -> list[tuple]:
        """Images with caption_json but not yet in the active embedding collection."""
        existing_ids = set(self._collection().get()["ids"])
        return [
            (img_id, img_data)
            for img_id, img_data in self.image_catalog["images"].items()
            if img_data.get("caption_json") and img_id not in existing_ids
        ]

    def get_stage_stats(self) -> dict:
        catalog = self.image_catalog.get("images", {})
        captioned = sum(1 for img in catalog.values() if img.get("caption_json"))

        reg = get_registry()
        active_model = reg.get("active_model")
        active_embedded = 0
        model_stats = {}
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        for model_name, info in reg.get("models", {}).items():
            try:
                col = client.get_or_create_collection(name=collection_name_for(model_name))
                count = col.count()
            except Exception:
                count = 0
            model_stats[model_name] = {**info, "indexed_count": count}
            if model_name == active_model:
                active_embedded = count

        total = len(catalog)
        return {
            "total_scanned": total,
            "vision_done": captioned,
            "vision_pending": total - captioned,
            "active_model": active_model,
            "active_model_embedded": active_embedded,
            "embed_pending": captioned - active_embedded,
            "models": model_stats,
        }

    # ── Single-item ops (driven by the background job manager) ─────────────────

    def vision_one(self, img_id: str, force_provider: str = "auto") -> str:
        """Vision-only for one image. Stores caption + model, saves catalog. Raises on failure."""
        img_data = self.image_catalog["images"][img_id]
        text, vmodel = get_image_caption(
            img_data["path"], force_provider=force_provider, with_model=True
        )
        if _caption_has_error(text):
            raise RuntimeError(json.loads(text).get("error", "vision failed"))
        _record_caption_history(img_data, vmodel, text)
        self._save_catalog()
        return f"vision:{vmodel}"

    def embed_one(self, img_id: str, upsert: bool = False) -> str:
        """Embed one already-captioned image into the active model's collection."""
        img_data = self.image_catalog["images"][img_id]
        return _embed_one(img_id, img_data, upsert=upsert)

    def index_one_full(self, img_id: str, use_cached: bool = True, upsert: bool = False,
                       force_provider: str = "auto") -> str:
        """Vision (unless cached) + embed for one image. Saves catalog."""
        img_data = self.image_catalog["images"][img_id]
        note = _index_one(img_id, img_data, upsert=upsert, use_cached=use_cached,
                          force_provider=force_provider)
        self._save_catalog()
        return note

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_image(self, img_id: str) -> str | None:
        """Remove from all ChromaDB collections, catalog, and face data. Returns file path."""
        catalog = self.image_catalog.get("images", {})
        img_path = catalog.get(img_id, {}).get("path")

        reg = get_registry()
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        for model_name in reg.get("models", {}):
            try:
                col = client.get_or_create_collection(name=collection_name_for(model_name))
                col.delete(ids=[img_id])
            except Exception as e:
                print(f"[indexer] ChromaDB delete ({model_name}) warning: {e}")
        try:
            client.get_or_create_collection(name="images").delete(ids=[img_id])
        except Exception:
            pass

        if img_id in catalog:
            del self.image_catalog["images"][img_id]
            self._save_catalog()

        from constants import FACE_DIR
        face_file = os.path.join(FACE_DIR, f"{img_id}.json")
        if os.path.exists(face_file):
            try:
                os.remove(face_file)
            except Exception:
                pass

        return img_path


# ── Module-level helpers ──────────────────────────────────────────────────────

def _embed_one(img_id: str, img_data: dict, upsert: bool = False) -> str:
    """Embedding + face detection + ChromaDB store from cached caption_json."""
    caption_json = img_data["caption_json"]
    attrs = parse_vision_attributes(caption_json)

    try:
        parsed = json.loads(caption_json)
        if parsed.get("error", ""):
            raise RuntimeError(f"vision error: {parsed['error']}")
    except json.JSONDecodeError:
        pass

    embedding, model_name, embed_source = get_embedding(caption_json)
    if embedding is None:
        raise RuntimeError("embedding failed (LM Studio and Gemini both unavailable)")

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=collection_name_for(model_name))

    face_data = detect_and_embed_faces(img_data["path"])
    save_face_data(img_id, face_data)

    meta = img_data.get("metadata", {})
    year = meta.get("date", "")[:4] if meta.get("date", "") else "unknown"
    payload = {
        "path": img_data["path"],
        "filename": img_data["filename"],
        "caption": attrs["caption"],
        "scene": attrs["scene"],
        "location_type": attrs["location_type"],
        "weather": attrs["weather"],
        "season": attrs["season"],
        "time_of_day": attrs["time_of_day"],
        "occasion": attrs["occasion"],
        "group_size": attrs["group_size"],
        "clothing_style": attrs["clothing_style"],
        "mood": attrs["mood"],
        "objects": attrs["objects"],
        "people_description": attrs["people_description"],
        "year": year,
        "metadata_json": json.dumps(meta),
        "embedding_source": embed_source,
        "embedding_model": model_name,
    }

    if upsert:
        collection.upsert(ids=[img_id], embeddings=[embedding], metadatas=[payload])
    else:
        collection.add(ids=[img_id], embeddings=[embedding], metadatas=[payload])

    return f"embed:{embed_source}"


def _index_one(img_id: str, img_data: dict, upsert: bool = False, use_cached: bool = True,
               force_provider: str = "auto") -> str:
    """Vision + embed in one shot. use_cached=True reuses stored caption_json."""
    if not (use_cached and img_data.get("caption_json")):
        text, vmodel = get_image_caption(
            img_data["path"], force_provider=force_provider, with_model=True
        )
        _record_caption_history(img_data, vmodel, text)
    return _embed_one(img_id, img_data, upsert=upsert)
