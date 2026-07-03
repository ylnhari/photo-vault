import os
import json
import hashlib
import time
from pathlib import Path
import db
from vision import get_image_caption, parse_vision_attributes
from embeddings import get_embedding, collection_name_for, get_active_model, get_registry
from faces import detect_and_embed_faces, save_face_data, index_faces, delete_faces_for_image
from scanner import scan_directory
from constants import IMAGE_CATALOG_PATH, FACE_DIR, THUMB_DIR

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


def _path_under(path: str, folder: str) -> bool:
    """True if path is exactly folder or is inside folder. Case-insensitive on Windows."""
    p = os.path.normcase(path)
    f = os.path.normcase(folder)
    return p == f or p.startswith(f + os.sep) or p.startswith(f + "/")


# mtime-cached catalog read so hot paths (status polls, image serving, timeline)
# don't re-parse the whole images.json on every request. Keyed on (path, mtime)
# so it stays correct when tests point IMAGE_CATALOG_PATH at different files.
# This snapshot is shared read-only; mutators load their own private copy.
_catalog_cache: dict = {"key": None, "data": None}

# Short-TTL cache for the orphaned-file scan (see get_missing_files).
_missing_files_cache: dict = {"key": None, "at": 0.0, "data": []}


def load_catalog_cached() -> dict:
    """Return a shared read-only catalog snapshot, reloaded only when the file changes."""
    try:
        key = (IMAGE_CATALOG_PATH, os.path.getmtime(IMAGE_CATALOG_PATH))
    except OSError:
        return {"images": {}}
    if _catalog_cache["key"] != key or _catalog_cache["data"] is None:
        try:
            with open(IMAGE_CATALOG_PATH) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {"images": {}}
            _catalog_cache["data"] = data
            _catalog_cache["key"] = key
        except Exception:
            return {"images": {}}
    return _catalog_cache["data"]


def catalog_path_for(img_id: str) -> str | None:
    """Look up a scanned image's on-disk path by id (cached)."""
    entry = load_catalog_cached().get("images", {}).get(img_id)
    return entry.get("path") if entry else None


def _remove_derived_files(img_id: str):
    """Delete the face JSON, ANN face entries, and every thumbnail tier for one id."""
    face_file = os.path.join(FACE_DIR, f"{img_id}.json")
    thumb_h = hashlib.sha1(img_id.encode("utf-8")).hexdigest()
    paths = [
        face_file,
        os.path.join(THUMB_DIR, f"{thumb_h}.jpg"),     # thumb tier
        os.path.join(THUMB_DIR, f"{thumb_h}_m.jpg"),   # medium tier
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    delete_faces_for_image(img_id)
    try:
        import albums
        albums.remove_image_from_all(img_id)
    except Exception:
        pass


class Indexer:
    def __init__(self, target_directories=None, use_cache: bool = False):
        """
        use_cache=False (default): load a private, mutable copy of the catalog.
          Required for any code that mutates + saves (scan, jobs, delete, tests).
        use_cache=True: share the cached read-only snapshot — cheap, for hot
          read-only endpoints (status polls, timeline, explore).
        """
        self.target_directories = target_directories or []
        self.image_catalog = load_catalog_cached() if use_cache else self._load_image_catalog()

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
        tmp = IMAGE_CATALOG_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.image_catalog, f, indent=2)
        os.replace(tmp, IMAGE_CATALOG_PATH)

    def _collection(self, model_name: str = None):
        return db.collection(model_name)

    # ── Scanning ──────────────────────────────────────────────────────────────

    def scan_only(self) -> dict:
        """
        Scan all configured (or explicitly provided) directories.
        Uses folders.py registry when target_directories is empty.
        Returns per-folder breakdown + aggregate totals.
        """
        from folders import get_effective_scan_dirs, get_excluded_paths, update_scan_result

        if self.target_directories:
            scan_dirs = self.target_directories
        else:
            scan_dirs = get_effective_scan_dirs()

        excluded = get_excluded_paths()

        aggregate = {"added": 0, "moved": 0, "unchanged": 0, "total": 0}
        per_folder = {}

        for d in scan_dirs:
            d_norm = str(Path(d).resolve())
            if not os.path.isdir(d_norm):
                per_folder[d] = {"error": "directory not found"}
                continue

            s = scan_directory(
                d_norm,
                IMAGE_CATALOG_PATH,
                excluded_paths=excluded,
            )
            per_folder[d] = s
            for k in ("added", "moved", "unchanged"):
                aggregate[k] += s.get(k, 0)
            aggregate["total"] = s.get("total", aggregate["total"])

            # Update folder registry with latest scan stats
            update_scan_result(d_norm, s.get("scanned", 0))

        self.image_catalog = self._load_image_catalog()

        if aggregate["moved"]:
            aggregate["reconciled"] = self.reconcile_paths()

        return {"per_folder": per_folder, **aggregate}

    def get_folders(self) -> dict:
        """Scanned-folder registry from images.json (legacy, kept for compat)."""
        return self.image_catalog.get("folders", {})

    def reconcile_paths(self) -> int:
        """After a scan that detected moved files, update the stored path in every
        ChromaDB collection so search results still resolve to the new location."""
        catalog = self.image_catalog.get("images", {})
        reg = get_registry()
        client = db.client()
        fixed = 0
        for model_name in reg.get("models", {}):
            try:
                col = client.get_or_create_collection(name=collection_name_for(model_name))
                res = col.get(include=["metadatas"])
                for cid, meta in zip(res["ids"], res["metadatas"]):
                    cat = catalog.get(cid)
                    if cat and meta.get("path") != cat.get("path"):
                        meta["path"] = cat["path"]
                        meta["filename"] = cat.get("filename", "")
                        col.update(ids=[cid], metadatas=[meta])
                        fixed += 1
            except Exception as e:
                print(f"[indexer] reconcile ({model_name}) warning: {e}")
        return fixed

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

    def get_missing_files(self, use_cache: bool = False) -> list[tuple]:
        """Catalog entries whose file path no longer exists on disk (orphaned).
        Inherently one os.path.exists per image (~1s / 25k photos), so hot
        read-only callers (the status poll) pass use_cache=True for a short-TTL
        snapshot; mutating callers keep the exact live check."""
        global _missing_files_cache
        key = _catalog_cache["key"]
        if use_cache and _missing_files_cache["key"] == key and (
            time.time() - _missing_files_cache["at"] < 30
        ):
            return _missing_files_cache["data"]
        missing = [
            (img_id, data)
            for img_id, data in self.image_catalog.get("images", {}).items()
            if not os.path.exists(data.get("path", ""))
        ]
        if use_cache:
            _missing_files_cache = {"key": key, "at": time.time(), "data": missing}
        return missing

    # ── Faces ──────────────────────────────────────────────────────────────────

    def _face_data_ids(self) -> set[str]:
        """One directory listing instead of an os.path.exists per image —
        the status endpoint calls this on every poll, and per-file stats cost
        ~1s per 25k images."""
        try:
            return {f[:-5] for f in os.listdir(FACE_DIR) if f.endswith(".json")}
        except OSError:
            return set()

    def _has_face_data(self, img_id: str) -> bool:
        return os.path.exists(os.path.join(FACE_DIR, f"{img_id}.json"))

    def get_faces_pending(self) -> list[tuple]:
        """Images that have not yet had face detection run (no face JSON file)."""
        have = self._face_data_ids()
        return [
            (img_id, data)
            for img_id, data in self.image_catalog.get("images", {}).items()
            if img_id not in have
        ]

    def get_faces_stats(self) -> dict:
        catalog = self.image_catalog.get("images", {})
        total = len(catalog)
        have = self._face_data_ids()
        detected = sum(1 for img_id in catalog if img_id in have)
        return {"total": total, "detected": detected, "pending": total - detected}

    def detect_faces_one(self, img_id: str) -> str:
        """Run face detection for one image and persist its face JSON. Raises on missing file."""
        img_data = self.image_catalog["images"][img_id]
        path = img_data.get("path", "")
        if not os.path.exists(path):
            raise FileNotFoundError(f"file not on disk: {path}")
        data = detect_and_embed_faces(path)
        save_face_data(img_id, data)
        index_faces(img_id, data)
        return f"faces:{len(data)}"

    def get_vision_pending(self) -> list[tuple]:
        """Images that have not yet been through any vision analysis."""
        return [
            (img_id, img_data)
            for img_id, img_data in self.image_catalog["images"].items()
            if not img_data.get("caption_json")
        ]

    def get_vision_pending_for_model(self, model_label: str) -> list[tuple]:
        """Images that have not yet been captioned by the given model label."""
        return [
            (img_id, img_data)
            for img_id, img_data in self.image_catalog["images"].items()
            if not any(
                h.get("model") == model_label
                for h in img_data.get("caption_history", [])
            )
        ]

    def get_embed_eligible_ids(self, caption_source_model: str = None) -> set[str]:
        """
        IDs of images eligible for embedding.
        caption_source_model=None → any image with a caption (caption_json set).
        caption_source_model="X" → only images with a caption_history entry from model X.
        """
        catalog = self.image_catalog.get("images", {})
        if not caption_source_model:
            return {img_id for img_id, d in catalog.items() if d.get("caption_json")}
        return {
            img_id for img_id, d in catalog.items()
            if any(h.get("model") == caption_source_model for h in d.get("caption_history", []))
        }

    def get_embed_pending(self) -> list[tuple]:
        """Images with caption_json but not yet in the active embedding collection."""
        existing_ids = set(self._collection().get()["ids"])
        return [
            (img_id, img_data)
            for img_id, img_data in self.image_catalog["images"].items()
            if img_data.get("caption_json") and img_id not in existing_ids
        ]

    def get_embed_pending_for_model(self, embed_model_name: str,
                                    caption_source_model: str = None) -> list[tuple]:
        """
        Images eligible for embedding with embed_model_name that haven't been
        embedded yet. Eligibility is gated on caption_source_model.
        """
        eligible = self.get_embed_eligible_ids(caption_source_model)
        try:
            col = self._collection(embed_model_name)
            embedded = set(col.get()["ids"])
        except Exception:
            embedded = set()
        catalog = self.image_catalog.get("images", {})
        return [
            (img_id, catalog[img_id])
            for img_id in eligible
            if img_id not in embedded and img_id in catalog
        ]

    def get_vision_model_summary(self) -> dict[str, int]:
        """
        Count caption_history entries per model label across the whole catalog.
        Returns e.g. {"lm_studio:qwen2-vl-7b": 1200, "gemini:gemini-2.5-flash": 300}
        """
        counts: dict[str, int] = {}
        for img_data in self.image_catalog.get("images", {}).values():
            for h in img_data.get("caption_history", []):
                m = h.get("model")
                if m and m != "error":
                    counts[m] = counts.get(m, 0) + 1
        return counts

    def get_stage_stats(self) -> dict:
        catalog = self.image_catalog.get("images", {})
        captioned = sum(1 for img in catalog.values() if img.get("caption_json"))

        reg = get_registry()
        active_model = reg.get("active_model")
        active_embedded = 0
        model_stats = {}
        client = db.client()
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

    def compute_caption(self, img_id: str, force_provider: str = "auto",
                        model: str = None) -> tuple[str, str]:
        """
        Run vision for one image and RETURN (model_label, caption_json) WITHOUT
        mutating or saving. Safe to call from worker threads in parallel (it only
        reads the catalog and hits the network). Raises on failure.
        """
        img_data = self.image_catalog["images"][img_id]
        text, vmodel = get_image_caption(
            img_data["path"], force_provider=force_provider, with_model=True, model=model
        )
        if _caption_has_error(text):
            raise RuntimeError(json.loads(text).get("error", "vision failed"))
        return vmodel, text

    def record_caption(self, img_id: str, vmodel: str, text: str):
        """Apply a computed caption to the catalog in memory (caller persists)."""
        _record_caption_history(self.image_catalog["images"][img_id], vmodel, text)

    def vision_one(self, img_id: str, force_provider: str = "auto", model: str = None,
                   persist: bool = True) -> str:
        """Vision for one image. Stores caption + model. Persists unless batched by caller."""
        vmodel, text = self.compute_caption(img_id, force_provider=force_provider, model=model)
        self.record_caption(img_id, vmodel, text)
        if persist:
            self._save_catalog()
        return f"vision:{vmodel}"

    def embed_one(self, img_id: str, upsert: bool = False,
                  embed_provider: str = "auto", embed_model: str = None,
                  caption_source_model: str = None, detect_faces: bool = True) -> str:
        """Embed one already-captioned image into the chosen model's collection.
        (Does not write images.json — embeddings live in ChromaDB + face JSON.)"""
        img_data = self.image_catalog["images"][img_id]
        return _embed_one(img_id, img_data, upsert=upsert,
                          embed_provider=embed_provider, embed_model=embed_model,
                          caption_source_model=caption_source_model, detect_faces=detect_faces)

    def index_one_full(self, img_id: str, use_cached: bool = True, upsert: bool = False,
                       vision_provider: str = "auto", vision_model: str = None,
                       embed_provider: str = "auto", embed_model: str = None,
                       caption_source_model: str = None, persist: bool = True,
                       detect_faces: bool = True) -> str:
        """Vision (unless cached) + embed for one image. Persists unless batched by caller."""
        img_data = self.image_catalog["images"][img_id]
        note = _index_one(img_id, img_data, upsert=upsert, use_cached=use_cached,
                          vision_provider=vision_provider, vision_model=vision_model,
                          embed_provider=embed_provider, embed_model=embed_model,
                          caption_source_model=caption_source_model, detect_faces=detect_faces)
        if persist:
            self._save_catalog()
        return note

    # ── Folder-level purge ────────────────────────────────────────────────────

    def count_images_under(self, folder_path: str) -> int:
        """Count catalog entries whose path is inside folder_path."""
        folder_path = str(Path(folder_path).resolve())
        return sum(
            1 for data in self.image_catalog.get("images", {}).values()
            if _path_under(data.get("path", ""), folder_path)
        )

    def purge_folder(self, folder_path: str) -> int:
        """
        Remove all images under folder_path from every store:
        ChromaDB collections, face data files, thumbnails, and the catalog.
        Returns the number of images removed.
        """
        folder_path = str(Path(folder_path).resolve())
        catalog = self.image_catalog.get("images", {})

        to_remove = [
            img_id for img_id, data in catalog.items()
            if _path_under(data.get("path", ""), folder_path)
        ]
        if not to_remove:
            return 0

        # Batch-delete from all ChromaDB collections
        reg = get_registry()
        client = db.client()
        for model_name in reg.get("models", {}):
            try:
                col = client.get_or_create_collection(name=collection_name_for(model_name))
                col.delete(ids=to_remove)
            except Exception as e:
                print(f"[indexer] purge ChromaDB ({model_name}) warning: {e}")
        # Also clean the legacy "images" collection if it exists
        try:
            client.get_or_create_collection(name="images").delete(ids=to_remove)
        except Exception:
            pass

        for img_id in to_remove:
            # Face data
            _remove_derived_files(img_id)
            del self.image_catalog["images"][img_id]

        self._save_catalog()
        return len(to_remove)

    # ── Delete (single image) ─────────────────────────────────────────────────

    def delete_image(self, img_id: str) -> str | None:
        """Remove from all ChromaDB collections, catalog, and face data. Returns file path."""
        catalog = self.image_catalog.get("images", {})
        img_path = catalog.get(img_id, {}).get("path")

        reg = get_registry()
        client = db.client()
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

        _remove_derived_files(img_id)
        return img_path


# ── Module-level helpers ──────────────────────────────────────────────────────

def _embed_one(img_id: str, img_data: dict, upsert: bool = False,
               embed_provider: str = "auto", embed_model: str = None,
               caption_source_model: str = None, detect_faces: bool = True) -> str:
    """
    Embedding + (optional) face detection + ChromaDB store.
    caption_source_model: if set, use the caption from that specific model in
    caption_history; otherwise use the latest caption_json.
    detect_faces: when True, also run + persist face detection inline.
    """
    if caption_source_model:
        hist = img_data.get("caption_history", [])
        entry = next(
            (h for h in reversed(hist) if h.get("model") == caption_source_model), None
        )
        if not entry:
            raise RuntimeError(
                f"No caption from model '{caption_source_model}' — run vision with that model first"
            )
        caption_json = entry["caption_json"]
    else:
        caption_json = img_data.get("caption_json", "")
    if not caption_json:
        raise RuntimeError("No caption available — run vision analysis first")
    attrs = parse_vision_attributes(caption_json)

    try:
        parsed = json.loads(caption_json)
        if parsed.get("error", ""):
            raise RuntimeError(f"vision error: {parsed['error']}")
    except json.JSONDecodeError:
        pass

    embedding, model_name, embed_source = get_embedding(
        caption_json, force_provider=embed_provider, model=embed_model
    )
    if embedding is None:
        raise RuntimeError("embedding failed (LM Studio and Gemini both unavailable)")

    client = db.client()
    collection = client.get_or_create_collection(name=collection_name_for(model_name))

    if detect_faces:
        face_data = detect_and_embed_faces(img_data["path"])
        save_face_data(img_id, face_data)
        index_faces(img_id, face_data)

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
               vision_provider: str = "auto", vision_model: str = None,
               embed_provider: str = "auto", embed_model: str = None,
               caption_source_model: str = None, detect_faces: bool = True) -> str:
    """Vision + embed in one shot. use_cached=True reuses stored caption_json."""
    if not (use_cached and img_data.get("caption_json")):
        text, vmodel = get_image_caption(
            img_data["path"], force_provider=vision_provider, with_model=True, model=vision_model
        )
        _record_caption_history(img_data, vmodel, text)
    return _embed_one(img_id, img_data, upsert=upsert,
                      embed_provider=embed_provider, embed_model=embed_model,
                      caption_source_model=caption_source_model, detect_faces=detect_faces)
