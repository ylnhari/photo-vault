export const BROWSE_SEARCH_STORAGE_KEY = "photo-vault.saved-browse-searches.v1";
export const MAX_SAVED_SEARCHES = 12;
export const MAX_SAVED_SEARCHES_CHARS = 64_000;

export const BROWSE_FILTER_KEYS = Object.freeze([
  "text_scope", "date_from", "date_to", "month", "day", "photo_type", "scene",
  "weather", "occasion", "time_of_day", "camera", "has_text", "has_location",
  "has_caption", "sort",
]);

const TEXT_SCOPES = new Set(["all", "filename", "caption", "details", "text", "place", "camera"]);
const SORTS = new Set(["newest", "oldest", "added", "filename"]);
const PRESENCE = new Set(["", "yes", "no"]);
const FACET_KEYS = new Set(["photo_type", "scene", "weather", "occasion", "time_of_day", "camera"]);

export function buildBrowseParams({ q = "", mediaType = "all", year = "", filters = {} } = {}, offset = 0, limit = 60) {
  const params = { offset, limit, q: q.trim(), media_type: mediaType === "photos" ? "image" : mediaType === "videos" ? "video" : "" };
  if (year) params.year = year;
  for (const key of BROWSE_FILTER_KEYS) {
    const value = filters[key];
    if (value !== undefined && value !== null && value !== "" && !(key === "text_scope" && value === "all") && !(key === "sort" && value === "newest")) {
      params[key] = String(value);
    }
  }
  return params;
}

export function validateBrowseQuery(q, filters = {}) {
  if (q.length > 500) return "Search text must be 500 characters or fewer.";
  const parsed = scanSearchTerms(q);
  if (!parsed.ok) return parsed.reason;
  if (parsed.terms.length > 20) return "Use 20 search terms or fewer.";
  if (filters.date_from && !validISODate(filters.date_from)) return "Choose a valid start date.";
  if (filters.date_to && !validISODate(filters.date_to)) return "Choose a valid end date.";
  if (filters.date_from && filters.date_to && filters.date_from > filters.date_to) return "The end date must be on or after the start date.";
  if (filters.month && (!Number.isInteger(Number(filters.month)) || Number(filters.month) < 1 || Number(filters.month) > 12)) return "Choose a month from 1 to 12.";
  if (filters.day && (!filters.month || !Number.isInteger(Number(filters.day)) || Number(filters.day) < 1 || Number(filters.day) > 31)) return "Choose a day from 1 to 31 after selecting a month.";
  if (filters.month && filters.day && Number(filters.day) > new Date(2000, Number(filters.month), 0).getDate()) return "That day does not exist in the selected month.";
  return "";
}

function scanSearchTerms(query) {
  const terms = [];
  let index = 0;
  while (index < query.length) {
    while (index < query.length && /\s/.test(query[index])) index++;
    if (index >= query.length) break;
    let excluded = false;
    if (query[index] === "-") {
      excluded = true;
      index++;
      if (index >= query.length || /\s/.test(query[index])) return { ok: false, terms, reason: "Put a word or quoted phrase after the minus sign." };
    }
    if (query[index] === '"') {
      index++;
      const phraseStart = index;
      while (index < query.length && query[index] !== '"') index++;
      if (index >= query.length) return { ok: false, terms, reason: "Finish the quoted phrase before searching." };
      if (!query.slice(phraseStart, index).trim()) return { ok: false, terms, reason: "A quoted phrase needs at least one word." };
      index++;
      if (index < query.length && !/\s/.test(query[index])) return { ok: false, terms, reason: "Add a space after the closing quote." };
    } else {
      const wordStart = index;
      while (index < query.length && !/\s/.test(query[index])) {
        if (query[index] === '"') return { ok: false, terms, reason: "Put quotation marks around the whole phrase." };
        index++;
      }
      if (index === wordStart) return { ok: false, terms, reason: "Add a word after the minus sign." };
    }
    terms.push({ excluded });
  }
  return { ok: true, terms };
}

export function readableMatchFields(fields) {
  const labels = { filename: "Filename", caption: "Caption", details: "Photo details", text: "Text in photo", place: "Place", camera: "Camera" };
  return [...new Set((Array.isArray(fields) ? fields : []).map((field) => labels[field]).filter(Boolean))];
}

function validISODate(value) {
  if (!value) return true;
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return false;
  const year = Number(match[1]), month = Number(match[2]), day = Number(match[3]);
  if (year < 1 || month < 1 || month > 12 || day < 1) return false;
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  return day <= days[month - 1];
}

export function validateSavedSearch(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  if (Object.keys(value).some((key) => !["name", "q", "mediaType", "year", "filters"].includes(key))) return false;
  if (typeof value.name !== "string" || !value.name.trim() || value.name.length > 60) return false;
  if (typeof value.q !== "string" || value.q.length > 500) return false;
  if (validateBrowseQuery(value.q, value.filters || {})) return false;
  if (!new Set(["all", "photos", "videos"]).has(value.mediaType)) return false;
  if (typeof value.year !== "string" || (value.year !== "" && value.year !== "Unknown" && !/^\d{4}$/.test(value.year))) return false;
  if (!value.filters || typeof value.filters !== "object" || Array.isArray(value.filters)) return false;
  for (const key of Object.keys(value.filters)) {
    if (!BROWSE_FILTER_KEYS.includes(key) || typeof value.filters[key] !== "string" || value.filters[key].length > 500) return false;
    const field = value.filters[key];
    if (key === "text_scope" && !TEXT_SCOPES.has(field)) return false;
    if (key === "sort" && !SORTS.has(field)) return false;
    if (["has_text", "has_location", "has_caption"].includes(key) && !PRESENCE.has(field)) return false;
    if (FACET_KEYS.has(key) && field.length > 200) return false;
    if (["month", "day"].includes(key) && field && (!/^\d{1,2}$/.test(field) || Number(field) < 1 || Number(field) > (key === "month" ? 12 : 31))) return false;
    if (["date_from", "date_to"].includes(key) && !validISODate(field)) return false;
  }
  return true;
}

export function readSavedSearches(storage) {
  try {
    storage ??= globalThis.localStorage;
    const raw = storage?.getItem(BROWSE_SEARCH_STORAGE_KEY) || "[]";
    if (raw.length > MAX_SAVED_SEARCHES_CHARS) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const names = new Set();
    const valid = [];
    for (const item of parsed) {
      if (!validateSavedSearch(item)) continue;
      const normalizedName = item.name.trim().toLocaleLowerCase();
      if (names.has(normalizedName)) continue;
      names.add(normalizedName);
      valid.push(item);
      if (valid.length === MAX_SAVED_SEARCHES) break;
    }
    return valid;
  } catch { return []; }
}

export function writeSavedSearches(items, storage) {
  if (!Array.isArray(items) || items.length > MAX_SAVED_SEARCHES || !items.every(validateSavedSearch)) return false;
  try {
    storage ??= globalThis.localStorage;
    const serialized = JSON.stringify(items);
    if (serialized.length > MAX_SAVED_SEARCHES_CHARS) return false;
    storage?.setItem(BROWSE_SEARCH_STORAGE_KEY, serialized);
    return !!storage;
  } catch { return false; }
}
