"""
Near-duplicate detection via perceptual difference-hash (dhash).

A 64-bit dhash is stored per image in the catalog (computed by the "dhash"
job). Grouping uses banding: two hashes within a small Hamming distance share
at least one of four 16-bit bands, so candidates are found in O(n) instead of
comparing all pairs, then verified with the real Hamming distance.
"""
from PIL import ImageOps

from imaging import safe_open

# Hamming distance (out of 64 bits) at or under which two photos count as
# near-duplicates. 0 = pixel-identical resaves; ~6 tolerates resizes/recompression.
DEFAULT_THRESHOLD = 6


def dhash(image_path: str) -> str:
    """64-bit difference hash as a 16-char hex string. Raises on unreadable files."""
    with safe_open(image_path) as im:
        # A camera original (EXIF rotation tag, unrotated pixels) and a
        # WhatsApp/resave copy of the same photo (rotation baked into pixels,
        # tag stripped) must hash identically to be recognized as duplicates.
        im = ImageOps.exif_transpose(im)
        gray = im.convert("L").resize((9, 8))
    px = list(gray.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return f"{bits:016x}"


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def group_duplicates(catalog_images: dict, threshold: int = DEFAULT_THRESHOLD) -> list[list[str]]:
    """
    Group image ids whose dhashes are within `threshold` bits of each other.
    Returns groups (each len >= 2), largest first. Images without a dhash are
    skipped — run the dhash job first.
    """
    entries = [
        (img_id, int(data["dhash"], 16))
        for img_id, data in catalog_images.items()
        if data.get("dhash")
    ]

    # Union-find over band-bucket candidates.
    parent = {img_id: img_id for img_id, _ in entries}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    bands: dict[tuple[int, int], list[int]] = {}
    for idx_e, (_, h) in enumerate(entries):
        for b in range(4):
            key = (b, (h >> (b * 16)) & 0xFFFF)
            bands.setdefault(key, []).append(idx_e)

    # Any two hashes within `threshold` (< 16) bits necessarily agree on at
    # least one full 16-bit band (pigeonhole), so pairwise checks inside each
    # bucket find every qualifying pair. Degenerate buckets (e.g. a flat-image
    # band value shared by thousands) are capped to keep this O(n).
    MAX_BUCKET = 500
    for bucket in bands.values():
        if len(bucket) < 2 or len(bucket) > MAX_BUCKET:
            continue
        for x in range(len(bucket) - 1):
            ia, ha = entries[bucket[x]]
            for y in range(x + 1, len(bucket)):
                ib, hb = entries[bucket[y]]
                if _hamming(ha, hb) <= threshold:
                    union(ia, ib)

    groups: dict[str, list[str]] = {}
    for img_id, _ in entries:
        groups.setdefault(find(img_id), []).append(img_id)
    out = [g for g in groups.values() if len(g) >= 2]
    out.sort(key=len, reverse=True)
    return out
