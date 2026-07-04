<script>
  import { onMount, onDestroy, createEventDispatcher, tick } from "svelte";
  import { api } from "./api.js";
  import L from "leaflet";
  import "leaflet/dist/leaflet.css";
  import "leaflet.markercluster";
  import "leaflet.markercluster/dist/MarkerCluster.css";
  import "leaflet.markercluster/dist/MarkerCluster.Default.css";

  export let indexedCount = 0;
  const dispatch = createEventDispatcher();

  let mapEl;
  let map = null;
  let points = [];
  let loading = true;
  let err = "";

  onMount(async () => {
    try {
      points = (await api.mapPhotos()).points || [];
    } catch (e) { err = e.message; }
    loading = false;
    if (points.length) {
      // Wait for Svelte to actually flush the DOM update that removes
      // .mapwrap's class:hidden (a bare microtask doesn't guarantee that —
      // tick() does), so the element has real layout dimensions before
      // Leaflet measures it.
      await tick();
      initMap();
    }
  });

  function initMap() {
    map = L.map(mapEl, { scrollWheelZoom: true }).setView([points[0].lat, points[0].lon], 4);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
      maxZoom: 19,
    }).addTo(map);

    const ids = points.map((p) => p.id);
    // divIcon (CSS dot) markers so they cluster without needing image assets.
    const dot = L.divIcon({ className: "pv-pin", iconSize: [14, 14] });
    const cluster = L.markerClusterGroup({ maxClusterRadius: 50, chunkedLoading: true });
    for (const p of points) {
      const marker = L.marker([p.lat, p.lon], { icon: dot });
      marker.bindTooltip(p.filename || p.id, { direction: "top" });
      marker.on("click", () => dispatch("select", { id: p.id, ids }));
      cluster.addLayer(marker);
    }
    map.addLayer(cluster);
    const b = cluster.getBounds();
    if (b.isValid()) map.fitBounds(b, { padding: [40, 40] });
    // Leaflet sometimes needs a nudge once the tab is visible.
    setTimeout(() => map && map.invalidateSize(), 50);
  }

  onDestroy(() => { if (map) { map.remove(); map = null; } });
</script>

{#if indexedCount === 0}
  <div class="card"><p>No photos indexed yet. Scan and index photos first.</p></div>
{:else if loading}
  <p class="muted">Loading map…</p>
{:else if err}
  <p style="color:var(--danger)">{err}</p>
{:else if points.length === 0}
  <div class="card">
    <p><b>No geotagged photos found.</b></p>
    <p class="muted" style="font-size:13px">
      Photos must contain GPS EXIF data, and the folder must be re-scanned after the EXIF
      upgrade so coordinates are extracted (Index &amp; Manage → Scan).
    </p>
  </div>
{/if}

<div class="mapwrap" bind:this={mapEl} class:hidden={loading || err || points.length === 0}></div>

<style>
  .mapwrap {
    height: calc(100vh - 170px); min-height: 400px;
    border-radius: 12px; overflow: hidden; border: 1px solid var(--border);
  }
  .hidden { display: none; }
  .card { background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px; }
  .muted { color: var(--muted); }
  /* Leaflet popups/controls inherit dark-ish app colors poorly; keep tiles bright. */
  :global(.leaflet-container) { background: #1a1d23; font: inherit; }
  :global(.pv-pin) {
    background: #5b8def; border: 2px solid #fff; border-radius: 50%;
    box-shadow: 0 0 0 1px rgba(0,0,0,.3);
  }
</style>
