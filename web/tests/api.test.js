import assert from "node:assert/strict";
import test from "node:test";
import { api } from "../src/lib/api.js";

test("API shares token bootstrap and forwards filters and abort signals", async () => {
  const originalFetch = globalThis.fetch;
  let tokenRequests = 0;
  const calls = [];
  try {
    globalThis.fetch = (url, options = {}) => {
      if (url === "/api/token") {
        tokenRequests += 1;
        return new Promise((resolve) => setTimeout(() => resolve(new Response(
          JSON.stringify({ token: "synthetic-test-token" }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        )), 15));
      }
      calls.push({ url, options });
      if (url.startsWith("/api/library?")) {
        return new Promise((resolve, reject) => {
          options.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
        });
      }
      return Promise.resolve(new Response(JSON.stringify({ total: 0 }), {
        status: 200, headers: { "Content-Type": "application/json" },
      }));
    };

    const controller = new AbortController();
    const pagePromise = api.library({
      offset: 120, limit: 60, q: "family + tea", media_type: "image", year: "2026",
      text_scope: "text", date_from: "2026-01-02", date_to: "2026-06-30", month: "6", day: "12",
      photo_type: "Screenshot", has_text: "yes", sort: "oldest",
    }, { signal: controller.signal });
    const summaryPromise = api.librarySummary();
    await new Promise((resolve) => setTimeout(resolve, 25));

    assert.equal(tokenRequests, 1);
    assert.equal(calls.length, 2);
    assert.equal(calls[0].url, "/api/library?offset=120&limit=60&q=family+%2B+tea&media_type=image&year=2026&text_scope=text&date_from=2026-01-02&date_to=2026-06-30&month=6&day=12&photo_type=Screenshot&has_text=yes&sort=oldest");
    assert.equal(calls[0].options.signal, controller.signal);
    assert.equal(calls[0].options.headers.Authorization, "Bearer synthetic-test-token");
    controller.abort();
    await assert.rejects(pagePromise, { name: "AbortError" });
    assert.deepEqual(await summaryPromise, { total: 0 });
  } finally {
    globalThis.fetch = originalFetch;
  }
});
