import assert from "node:assert/strict";
import test from "node:test";
import {
  BROWSE_SEARCH_STORAGE_KEY, MAX_SAVED_SEARCHES, buildBrowseParams,
  readSavedSearches, validateBrowseQuery, validateSavedSearch, writeSavedSearches,
} from "../src/lib/browseSearch.js";

const saved = (overrides = {}) => ({
  name: "Hikes", q: '"forest trail" -screenshot', mediaType: "all", year: "2026",
  filters: { text_scope: "all", date_from: "2026-02-01", date_to: "2026-02-28", month: "", day: "", photo_type: "", scene: "", weather: "", occasion: "", time_of_day: "", camera: "", has_text: "", has_location: "", has_caption: "", sort: "newest" },
  ...overrides,
});

test("Browse params preserve all exact local constraints", () => {
  assert.deepEqual(buildBrowseParams({ q: '"warm tea" -crowd', mediaType: "videos", year: "Unknown", filters: { text_scope: "details", has_location: "no", date_from: "2020-01-01", camera: "Fujifilm X100" } }, 60, 60), {
    offset: 60, limit: 60, q: '"warm tea" -crowd', media_type: "video", year: "Unknown", text_scope: "details", date_from: "2020-01-01", camera: "Fujifilm X100", has_location: "no",
  });
});

test("query syntax accepts whole terms and phrases, and rejects malformed or overlong searches", () => {
  assert.equal(validateBrowseQuery('"birthday cake" -screenshot'), "");
  assert.match(validateBrowseQuery('"unfinished'), /Finish the quoted phrase/);
  assert.match(validateBrowseQuery('foo"bar"'), /quotation marks around the whole phrase/);
  assert.match(validateBrowseQuery('"phrase"suffix'), /space after the closing quote/);
  assert.match(validateBrowseQuery("-"), /after the minus sign/);
  assert.match(validateBrowseQuery('""'), /at least one word/);
  assert.match(validateBrowseQuery('"   "'), /at least one word/);
  assert.match(validateBrowseQuery(Array(21).fill("word").join(" ")), /20 search terms/);
});

test("date validation rejects impossible dates and reversed ranges before fetching", () => {
  assert.equal(validateBrowseQuery("", { date_from: "2024-02-29" }), "");
  assert.match(validateBrowseQuery("", { date_from: "2026-02-29" }), /valid start date/);
  assert.match(validateBrowseQuery("", { date_to: "2026-02-31" }), /valid end date/);
  assert.match(validateBrowseQuery("", { date_from: "0000-01-01" }), /valid start date/);
  assert.match(validateBrowseQuery("", { date_from: "2026-06-02", date_to: "2026-05-30" }), /on or after/);
  assert.match(validateBrowseQuery("", { month: "2", day: "30" }), /does not exist/);
});

test("saved-search values enforce bounded schema and use the live query/date validator", () => {
  assert.equal(validateSavedSearch(saved()), true);
  assert.equal(validateSavedSearch(saved({ year: "Unknown" })), true);
  assert.equal(validateSavedSearch(saved({ q: 'word"tail' })), false);
  assert.equal(validateSavedSearch(saved({ q: Array(21).fill("word").join(" ") })), false);
  assert.equal(validateSavedSearch(saved({ extra: true })), false);
  assert.equal(validateSavedSearch(saved({ filters: { ...saved().filters, date_to: "2026-02-30" } })), false);
  assert.equal(validateSavedSearch(saved({ filters: { ...saved().filters, photo_type: "x".repeat(201) } })), false);
});

test("saved-search storage handles corrupt, oversized, unavailable, and bounded data", () => {
  const memory = new Map();
  const storage = {
    getItem: (key) => memory.get(key) ?? null,
    setItem: (key, value) => memory.set(key, value),
  };
  assert.equal(writeSavedSearches([saved()], storage), true);
  assert.equal(readSavedSearches(storage)[0].name, "Hikes");
  assert.equal(memory.has(BROWSE_SEARCH_STORAGE_KEY), true);
  memory.set(BROWSE_SEARCH_STORAGE_KEY, JSON.stringify([saved(), saved({ name: "hIKES" })]));
  assert.equal(readSavedSearches(storage).length, 1);
  assert.equal(readSavedSearches({ getItem: () => "{" }).length, 0);
  assert.equal(readSavedSearches({ getItem: () => " ".repeat(70_000) }).length, 0);
  assert.equal(writeSavedSearches(Array(MAX_SAVED_SEARCHES + 1).fill(saved()), storage), false);

  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  try {
    Object.defineProperty(globalThis, "localStorage", { configurable: true, get() { throw new Error("blocked storage"); } });
    assert.deepEqual(readSavedSearches(), []);
    assert.equal(writeSavedSearches([saved()]), false);
  } finally {
    if (descriptor) Object.defineProperty(globalThis, "localStorage", descriptor);
    else delete globalThis.localStorage;
  }
});
