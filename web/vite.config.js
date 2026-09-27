import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { readFileSync } from "node:fs";

// Match constants.SERVER_PORT: explicit process configuration, optional workspace
// registry, then the clone's documented fallback. Never silently choose a port.
function backendPort() {
  const explicit = Number(process.env.PHOTO_VAULT_PORT);
  if (Number.isInteger(explicit) && explicit > 0 && explicit <= 65535) return explicit;
  try {
    const registry = JSON.parse(readFileSync(new URL("../../ports.json", import.meta.url), "utf8"));
    const port = Number(registry.registry?.["photo-vault"]?.port);
    if (Number.isInteger(port) && port > 0 && port <= 65535) return port;
  } catch { /* External clones do not need a workspace registry. */ }
  return 8768;
}

// Dev server proxies /api to the configured FastAPI backend.
// Production build (npm run build) emits to dist/, served same-origin by FastAPI.
export default defineConfig({
  plugins: [svelte()],
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": `http://127.0.0.1:${backendPort()}`,
    },
  },
});
