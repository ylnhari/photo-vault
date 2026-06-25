// Photo Vault service worker — app-shell + thumbnail caching for offline use.
const CACHE = "pv-shell-v1";
const ASSET_RE = /\/(assets|icon|manifest)/;

self.addEventListener("install", () => self.skipWaiting());

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  // Let cross-origin requests (e.g. OSM map tiles) go straight to the network.
  if (url.origin !== self.location.origin) return;

  // App-shell navigations: network-first, fall back to the cached shell offline.
  if (req.mode === "navigate") {
    e.respondWith((async () => {
      try {
        const fresh = await fetch(req);
        const c = await caches.open(CACHE);
        c.put("/", fresh.clone());
        return fresh;
      } catch {
        return (await caches.match("/")) || Response.error();
      }
    })());
    return;
  }

  // Thumbnails/photos and static build assets: stale-while-revalidate.
  if (url.pathname.startsWith("/api/image") || ASSET_RE.test(url.pathname)) {
    e.respondWith((async () => {
      const c = await caches.open(CACHE);
      const cached = await c.match(req);
      const network = fetch(req)
        .then((res) => { if (res && res.ok) c.put(req, res.clone()); return res; })
        .catch(() => null);
      return cached || (await network) || Response.error();
    })());
    return;
  }

  // Everything else (status/search/other JSON): network as usual, no caching.
});
