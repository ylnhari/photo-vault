import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Dev server proxies /api to the FastAPI backend (port from ports.json → 8768).
// Production build (npm run build) emits to dist/, served same-origin by FastAPI.
export default defineConfig({
  plugins: [svelte()],
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8768",
    },
  },
});
