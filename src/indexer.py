import os
import json
import hashlib
import time
from pathlib import Path
import db
import geocode
import catalog_db
from vision import get_image_caption, parse_vision_attributes, validate_vision_output, build_embedding_text
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
    """True if path is exactly folder or is inside folder. Case-insensitive on Windows.

    normpath() is required, not just normcase(): for a drive-root folder like
    "D:\\", Path.resolve() keeps the trailing separator, so a naive
    f + os.sep comparison doubles up ("D:\\\\") and never matches real child
    paths. normpath() collapses that trailing separator for every folder
    except a bare drive root, so stripping any remaining trailing separator
    before re-appending exactly one handles that edge case too.
    """
    p = os.path.normcase(os.path.normpath(path)) if path else ""
    f = os.path.normcase(os.path.normpath(folder)) if folder else ""
    if not p or not f:
        return False
    f_trimmed = f.rstrip(os.sep) or f
    return p == f or p.startswith(f_trimmed + os.sep)


# mtime-cached catalog read so hot paths (status polls, image serving, timeline)
# don't re-parse the whole images.json on every request. Keyed on (path, mtime)
# so it stays correct when tests point IMAGE_CATALOG_PATH at different files.
# This snapshot is shared read-only; mutators load their own private copy.
_catalog_cache: dict = {"key": None, "data": None}

# Short-TTL cache for the orphaned-file scan (see get_missing_files).
_missing_files_cache: dict = {"key": None, "at": 0.0, "data": []}


def load_catalog_cached() -> dict:
    """Return a shared read-only catalog snapshot, reloaded only when this
    process has written to the catalog since the last read (in-process write
    counter — see catalog_db.py for why this isn't mtime-keyed)."""
    key = (IMAGE_CATALOG_PATH, catalog_db.version(IMAGE_CATALOG_PATH))
    if _catalog_cache["key"] != key or _catalog_cache["data"] is None:
        try:
            data = catalog_db.load_all(IMAGE_CATALOG_PATH)
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
    paths = [face_file]
    for ext in ("webp", "jpg"):  # current WebP tiers + pre-WebP legacy JPEGs
        paths.append(os.path.join(THUMB_DIR, f"{thumb_h}.{ext}"))     # thumb
        paths.append(os.path.join(THUMB_DIR, f"{thumb_h}_m.{ext}"))   # medium
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
        # ids touched since the last _save_catalog() call. Mutating methods
        # mark ids here instead of every save rewriting the whole catalog —
        # see catalog_db.py for why (images.json was a full-file rewrite on
        # every batch of every job, ~6000+ times over one vision run).
        self._dirty_ids: set[str] = set()
        self._deleted_ids: set[str] = set()

    def _load_image_catalog(self):
        return catalog_db.load_all(IMAGE_CATALOG_PATH)

    def _mark_dirty(self, img_id: str):
        self._dirty_ids.add(img_id)
        self._deleted_ids.discard(img_id)

    def _mark_deleted(self, img_id: str):
        self._deleted_ids.add(img_id)
        self._dirty_ids.discard(img_id)

    def _save_catalog(self):
        if self._deleted_ids:
            catalog_db.delete_images(IMAGE_CATALOG_PATH, self._deleted_ids)
            self._deleted_ids.clear()
        if self._dirty_ids:
            images = self.image_catalog.get("images", {})
            dirty = {iid: images[iid] for iid in self._dirty_ids if iid in images}
            catalog_db.upsert_images(IMAGE_CATALOG_PATH, dirty)
            self._dirty_ids.clear()
        if self.image_catalog.get("folders"):
            catalog_db.save_folders(IMAGE_CATALOG_PATH, self.image_catalog["folders"])

    def _collection(self, model_name: str = None):
        # allow_default: these are read-only "what's already embedded" checks
        # (get_missing, get_missing_attributes, get_embed_pending) that must
        # degrade to "nothing embedded yet" before any model has ever been
        # selected, not raise — db.collection() otherwise raises ValueError
        # here specifically to stop embedding code from silently writing
        # into an ungoverned fallback collection.
        return db.collection(model_name, allow_default=True)

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
        all_moved_ids: list[str] = []

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
            all_moved_ids.extend(s.get("moved_ids", []))

            # Update folder registry with latest scan stats
            update_scan_result(d_norm, s.get("scanned", 0))

        self.image_catalog = self._load_image_catalog()

        if aggregate["moved"]:
            aggregate["reconciled"] = self.reconcile_paths(moved_ids=all_moved_ids or None)

        return {"per_folder": per_folder, **aggregate}

    def scan_folder_one(self, folder: str) -> str:
        """Scan one configured folder (driven by the background 'scan' job)."""
        from folders import get_excluded_paths, update_scan_result

        s = scan_directory(folder, IMAGE_CATALOG_PATH,
                           excluded_paths=get_excluded_paths())
        update_scan_result(folder, s.get("scanned", 0))
        self.image_catalog = self._load_image_catalog()
        if s.get("moved"):
            self.reconcile_paths(moved_ids=s.get("moved_ids") or None)
        return (f"+{s.get('added', 0)} new · {s.get('moved', 0)} moved · "
                f"{s.get('unchanged', 0)} unchanged")

    def get_folders(self) -> dict:
        """Scanned-folder registry from images.json (legacy, kept for compat)."""
        return self.image_catalog.get("folders", {})

    def reconcile_paths(self, moved_ids: list[str] | None = None) -> int:
        """After a scan that detected moved files, update the stored path in every
        ChromaDB collection so search results still resolve to the new location.

        moved_ids: when the caller knows exactly which ids moved (scan_directory
        tracks this), fetch only those from Chroma instead of the whole
        collection. Falls back to a full collection scan when unknown (e.g. an
        older/foreign caller) — still correct, just O(collection size)."""
        catalog = self.image_catalog.get("images", {})
        reg = get_registry()
        client = db.client()
        fixed = 0
        for model_name in reg.get("models", {}):
            try:
                col = client.get_or_create_collection(name=collection_name_for(model_name))
                if moved_ids:
                    res = col.get(ids=moved_ids, include=["metadatas"])
                else:
                    res = col.get(include=["metadatas"])
                # Collect every changed id/metadata pair and hand Chroma ONE
                # update() call per collection instead of one per id — with a
                # large collection and many moved files this turns an O(n)
                # sequence of API round-trips into O(1) per scan.
                changed_ids, changed_metas = [], []
                for cid, meta in zip(res["ids"], res["metadatas"]):
                    cat = catalog.get(cid)
                    if cat and meta.get("path") != cat.get("path"):
                        meta["path"] = cat["path"]
                        meta["filename"] = cat.get("filename", "")
                        changed_ids.append(cid)
                        changed_metas.append(meta)
                if changed_ids:
                    col.update(ids=changed_ids, metadatas=changed_metas)
                    fixed += len(changed_ids)
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
        """Run face detection for one image and persist its face JSON.
        Returns a 'skipped' note (not a failure, not a silent success) when the
        file is missing/moved — same convention as thumb_one/dhash_one, so a
        gap doesn't trip the job's consecutive-failure abort but also isn't
        indistinguishable from real face-detection output."""
        img_data = self.image_catalog["images"][img_id]
        path = img_data.get("path", "")
        if not os.path.exists(path):
            return "faces:skipped (file missing)"
        data = detect_and_embed_faces(path)
        save_face_data(img_id, data)
        index_faces(img_id, data)
        return f"faces:{len(data)}"

    # ── Thumbnails (pregeneration job) ────────────────────────────────────────

    @staticmethod
    def _existing_thumb_hashes() -> set[str]:
        """Hash part of every existing thumb file (either format), one listdir."""
        try:
            names = os.listdir(THUMB_DIR)
        except OSError:
            return set()
        return {
            n[:-5] for n in names if n.endswith(".webp") and len(n) == 45
        } | {
            n[:-4] for n in names if n.endswith(".jpg") and len(n) == 44
        }

    def get_thumbs_pending(self) -> list[tuple]:
        """Images with no grid thumbnail yet (WebP or legacy JPEG)."""
        have = self._existing_thumb_hashes()
        return [
            (img_id, data)
            for img_id, data in self.image_catalog.get("images", {}).items()
            if hashlib.sha1(img_id.encode("utf-8")).hexdigest() not in have
        ]

    def count_thumbs_missing(self) -> int:
        """Cheap approximation for the status endpoint: catalog size minus
        existing thumb files (no per-file stat calls)."""
        return max(0, len(self.image_catalog.get("images", {})) -
                   len(self._existing_thumb_hashes()))

    def thumb_one(self, img_id: str) -> str:
        """Generate the 400px grid thumbnail for one image."""
        from imaging import derivative_path, ensure_derivative, THUMB_PX
        data = self.image_catalog["images"][img_id]
        path = data.get("path", "")
        if not os.path.exists(path):
            return "thumb:skipped (file missing)"
        if not ensure_derivative(path, derivative_path(img_id), THUMB_PX):
            raise RuntimeError("thumbnail generation failed (corrupt/unreadable file)")
        return "thumb:ok"

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
            "embed_pending": max(0, captioned - active_embedded),
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
        validation = validate_vision_output(text)
        if not validation["valid"]:
            raise RuntimeError(validation["warning"])
        return vmodel, text

    def record_caption(self, img_id: str, vmodel: str, text: str):
        """Apply a computed caption to the catalog in memory (caller persists)."""
        _record_caption_history(self.image_catalog["images"][img_id], vmodel, text)
        self._mark_dirty(img_id)

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
        self._mark_dirty(img_id)
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
            self._mark_deleted(img_id)

        self._save_catalog()
        return len(to_remove)

    # ── Delete / trash (single image) ────────────────────────────────────────

    def _drop_from_collections(self, img_id: str):
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

    def delete_image(self, img_id: str, to_trash: bool = True,
                     file_deleted: bool = False) -> str | None:
        """
        Remove a photo from search. to_trash=True (default) soft-deletes: the
        catalog entry moves to the trash store and derived files (thumbs, face
        data) are KEPT so restore is cheap. to_trash=False removes permanently.
        Returns the file path.
        """
        import trash as trash_mod

        catalog = self.image_catalog.get("images", {})
        entry = catalog.get(img_id)
        img_path = entry.get("path") if entry else None

        self._drop_from_collections(img_id)

        if entry is not None:
            if to_trash:
                trash_mod.add(img_id, entry, file_deleted=file_deleted)
            del self.image_catalog["images"][img_id]
            self._mark_deleted(img_id)
            self._save_catalog()

        if not to_trash:
            _remove_derived_files(img_id)
        return img_path

    def restore_images(self, img_ids: list[str]) -> int:
        """Bring trashed photos back into the catalog. Their caption survives,
        so only the embed stage needs re-running (they show as embed-pending)."""
        import trash as trash_mod

        entries = trash_mod.take(img_ids)
        if not entries:
            return 0
        self.image_catalog.setdefault("images", {}).update(entries)
        for iid in entries:
            self._mark_dirty(iid)
        self._save_catalog()
        return len(entries)

    def purge_trash(self, img_ids: list[str] | None = None) -> int:
        """Permanently drop trashed photos (all when img_ids is None)."""
        import trash as trash_mod

        dropped = trash_mod.purge(img_ids)
        for iid in dropped:
            _remove_derived_files(iid)
        return len(dropped)

    # ── Perceptual hash (duplicates) ─────────────────────────────────────────

    def get_dhash_pending(self) -> list[tuple]:
        return [
            (img_id, data)
            for img_id, data in self.image_catalog.get("images", {}).items()
            if not data.get("dhash")
        ]

    def dhash_one(self, img_id: str) -> str:
        """Compute + store the perceptual hash for one image (caller persists)."""
        from dupes import dhash
        data = self.image_catalog["images"][img_id]
        path = data.get("path", "")
        if not os.path.exists(path):
            return "dhash:skipped (file missing)"
        data["dhash"] = dhash(path)
        self._mark_dirty(img_id)
        return "dhash:ok"


# ── Module-level helpers ──────────────────────────────────────────────────────

def resolve_caption_json(img_data: dict, caption_source_model: str = None) -> str:
    """The caption text to embed for one image. Raises when the image has no
    (matching) caption or the stored caption is a vision error record."""
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
    try:
        parsed = json.loads(caption_json)
    except json.JSONDecodeError:
        raise RuntimeError("Stored caption is not valid JSON — cannot build an embedding from it")
    if parsed.get("error", ""):
        raise RuntimeError(f"vision error: {parsed['error']}")
    return caption_json


def build_embed_payload(img_data: dict, caption_json: str,
                        embed_source: str, model_name: str) -> dict:
    """ChromaDB metadata payload for one embedded image."""
    attrs = parse_vision_attributes(caption_json)
    meta = img_data.get("metadata", {})
    date = meta.get("date", "")
    year = date[:4] if date and len(date) >= 4 else "unknown"
    month = date[5:7] if len(date) >= 7 else "unknown"
    lat, lon = meta.get("gps_lat"), meta.get("gps_lon")
    place = geocode.place_for(lat, lon) if lat is not None and lon is not None else None
    return {
        "path": img_data["path"],
        "filename": img_data["filename"],
        "caption": attrs["caption"],
        "scene": attrs["scene"],
        "location_type": attrs["location_type"],
        "weather": attrs["weather"],
        "season": attrs["season"],
        "time_of_day": attrs["time_of_day"],
        "occasion": attrs["occasion"],
        "festival_name": attrs["festival_name"],
        "group_size": attrs["group_size"],
        "person_count": attrs["person_count"],
        "clothing_style": attrs["clothing_style"],
        "mood": attrs["mood"],
        "objects": attrs["objects"],
        "animals": attrs["animals"],
        "vehicles": attrs["vehicles"],
        "food_items": attrs["food_items"],
        "activities": attrs["activities"],
        "photo_type": attrs["photo_type"],
        "text_in_image": attrs["text_in_image"],
        "landmark": attrs["landmark"],
        "dominant_colors": attrs["dominant_colors"],
        "people_description": attrs["people_description"],
        "year": year,
        "month": month,
        "place": place or "unknown",
        "metadata_json": json.dumps(meta),
        "embedding_source": embed_source,
        "embedding_model": model_name,
    }


def _embed_one(img_id: str, img_data: dict, upsert: bool = False,
               embed_provider: str = "auto", embed_model: str = None,
               caption_source_model: str = None, detect_faces: bool = True) -> str:
    """
    Embedding + (optional) face detection + ChromaDB store.
    caption_source_model: if set, use the caption from that specific model in
    caption_history; otherwise use the latest caption_json.
    detect_faces: when True, also run + persist face detection inline.
    """
    caption_json = resolve_caption_json(img_data, caption_source_model)
    embedding_text = build_embedding_text(parse_vision_attributes(caption_json))

    embedding, model_name, embed_source = get_embedding(
        embedding_text, force_provider=embed_provider, model=embed_model
    )
    if embedding is None:
        from embeddings import last_embed_error
        raise RuntimeError(
            f"embedding failed — {last_embed_error() or 'all embedding providers unavailable'}"
        )

    client = db.client()
    collection = client.get_or_create_collection(name=collection_name_for(model_name))

    if detect_faces:
        # Mirrors _run_embed_batched in jobs.py: a corrupt/unreadable image
        # must not discard an already-successful text embedding for this id.
        try:
            face_data = detect_and_embed_faces(img_data["path"])
            save_face_data(img_id, face_data)
            index_faces(img_id, face_data)
        except Exception as e:
            print(f"[indexer] face detection failed for {img_id}: {e}")

    payload = build_embed_payload(img_data, caption_json, embed_source, model_name)

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
        # Same schema check compute_caption() (the "vision" job path) already
        # does — without it, a bad/incomplete model response (missing keys, or
        # not JSON at all) was silently accepted here and embedded with
        # all-default attributes instead of failing loudly.
        validation = validate_vision_output(text)
        if not validation["valid"]:
            raise RuntimeError(validation["warning"])
        _record_caption_history(img_data, vmodel, text)
    return _embed_one(img_id, img_data, upsert=upsert,
                      embed_provider=embed_provider, embed_model=embed_model,
                      caption_source_model=caption_source_model, detect_faces=detect_faces)
