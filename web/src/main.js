import App from "./App.svelte";
import "./app.css";

const app = new App({ target: document.getElementById("app") });

// Register the PWA service worker (app shell + thumbnail offline cache).
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  });
}

export default app;
