# Feature Request: Integrating 9Router into photo-vault

> **Status: IMPLEMENTED 2026-07-08** — first-class `9router` provider shipped (vision +
> embeddings + health + UI selector), all Section 4 requirements met, verified live end-to-end
> (real photo captioned via `gc/gemini-2.5-flash-lite`, 3072-d single+batch embeds via
> `gemini/gemini-embedding-001`, honest LM Studio loaded-state reporting, no regressions —
> 359/359 tests green). Substitution policy (2026-07-08): 9Router may serve a request with a
> substituted upstream model (observed: `gc/gemini-2.5-flash-lite` answered by
> `gemini-3.1-flash-lite`). **Vision:** the caption is kept and stored under the model that
> ACTUALLY produced it; a 9Router vision run targets photos with no caption at all (coverage
> semantics), and the job panel shows a per-model "Models used" tally at the end of the run.
> **Embeddings:** a substituted model is REJECTED (same-dimension substitutes would silently
> poison the collection's vector space); the aborted job records {requested, served, suggested}
> and the Embed section offers a one-click "Switch to <served model> & re-run" (saves settings,
> restarts with correct pending accounting — decided over silently saving substituted vectors,
> which would fragment coverage across collections search can't use). 9Router is opt-in only —
> never part of the "auto" fallback chain, and always requires an explicit model id.
>
> Original header: Status PROPOSED, drafted 2026-07-06, design intentionally deferred.

---

## 1. What this is / why

[9Router](https://github.com/decolua/9router) is a local, self-hosted **OpenAI-compatible proxy**
that fronts 40+ AI providers with 3-tier fallback (subscription → cheap → free), round-robin
key/account pooling, and per-account quota tracking. Running locally:

- Endpoint: **`http://127.0.0.1:20128/v1`** (OpenAI-compatible: `/chat/completions`, `/embeddings`, `/models`)
- Dashboard: `http://127.0.0.1:20128/dashboard` (local password gate; default `123456` — **change it**)
- Bound to loopback (`127.0.0.1`) per the "never 0.0.0.0" house rule.
- Version tested: **v0.5.18**.

**Why photo-vault cares:** it needs *free* image captioning (vision) + text embeddings. 9Router turns
"one Gemini free key" into a pooled, auto-failing-over fleet of free vision + embedding capacity, and
becomes a **single local LLM gateway reusable by every project** (not just photo-vault).

> NOTE: 9Router's port is registered in `../ports.json` → `registry.9router.port` (20128).
> Read it from there — do not hardcode. See `../9ROUTER.md`.

---

## 2. Can photo-vault already call it? — Technically yes

Both AI paths already speak OpenAI-compatible against a base URL from `constants.py`:

```python
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
```

- `vision.py` → `OpenAI(base_url=LM_STUDIO_URL).chat.completions.create(model, messages=[text+image_url])`
- `embeddings.py` → `POST {LM_STUDIO_URL}/embeddings {"model","input"}`

Both match 9Router's API. So the *minimum* wiring is `LM_STUDIO_URL=http://127.0.0.1:20128/v1`.
**But do NOT just hijack that var** — there are real mismatches (Section 5) and design requirements
(Section 4) that call for a proper, first-class `9router` provider instead.

---

## 3. Verified findings (live tests, 2026-07-06)

All of the following were tested end-to-end through the running 9Router, not assumed.

### Providers connected during testing
Gemini (API key), Gemini CLI (`ylnharimailme@gmail.com`, OAuth), Kiro (AWS Builder ID),
OpenCode Free (no-auth), MiMo Code Free (no-auth).

### Vision models that WORK (captioned a test image correctly)
| Provider | Prefix | Working vision models | Notes |
|---|---|---|---|
| Gemini CLI | `gc/` | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-3.1-flash-lite-preview` | 3.x *pro/3-flash* previews **404** on CLI |
| Gemini API key | `gemini/` | `gemini-3.1-flash-lite-preview`, `gemini-3-flash-preview`, `gemma-4-31b-it`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` | `gemini-3.1-pro-preview` → **429** (near-zero free quota) |
| Kiro | `kr/` | `claude-sonnet-4.5` (+ other Claude) | premium; limited credits |
| OpenCode Free / MiMo | `oc/` `mmf/` | none reliable | text/code only; MiMo vision hit anti-abuse (`risk_control 441`) |

### Embedding models that WORK
| Provider | Prefix | Working models | Dim |
|---|---|---|---|
| Gemini API key | `gemini/` | `gemini-embedding-001`, `gemini-embedding-2-preview` | **3072** |

- `text-embedding-004`, `text-embedding-005`, `embedding-001` → **404** on this key/API version.
- **Gemini CLI and Kiro do NOT provide embeddings** — embeddings ride only on Gemini API keys.

### Verified quotas / capacity
- **Gemini CLI: 1,000 requests/day PER model, resets daily** (confirmed in Quota Tracker). 4 usable
  vision models → **4,000 vision/day per Google account**.
- **Kiro: 50 credits/month** (credit pool, ~monthly reset) — Claude/GLM/MiniMax incl. vision.
- **Gemini API key:** Google free-tier RPD, varies per model (not shown in tracker; only visible on 429).

### Capacity with 4 Gemini CLI accounts + 4 Gemini API keys
- **Vision ≈ 16,000/day** (4 models × 1,000 × 4 accounts, *verified floor*) + ~2–4k/day from keys.
- **Embeddings ≈ 4,000–8,000/day** (4 keys, `gemini-embedding-001`, *estimated*).
- **Bottleneck = embeddings, not vision.** For more embedding throughput, add Jina or Voyage
  (token-based free tiers, far higher than Gemini's per-day cap).

---

## 4. Design requirements (to be met — NOT yet designed)

The implementation is deferred until these are designed. Captured from the request:

1. **User chooses the vision model** by name, from the models 9Router actually exposes.
2. **User chooses the embedding model** by name, independently of vision.
3. **Only the chosen model is used** for its task (no silent auto-pick).
4. **Metadata is recorded per result** — which provider/model produced each caption and each
   embedding must be saved with the item (and surfaced in the UI).
5. Model selection should reflect **live availability / quota** where possible (9Router exposes
   `/v1/models`, `/v1/models/embedding`, and a Quota Tracker).
6. Health/status of the 9Router endpoint should be visible (health check).
7. Stay within the house rule: **free models only** (no paid tiers).

> Photo-vault ALREADY has most of the machinery for #3 and #4 — see Section 6. The design work is
> mainly: provider plumbing, model-list sourcing/filtering, and the selection UI + settings storage.

---

## 5. Known mismatches to handle in the design

| Photo-vault assumption | With 9Router | Implication for design |
|---|---|---|
| Auto-detects loaded model via LM Studio native `/api/v0/models` | 9Router has **no** `/api/v0` → auto-detect returns nothing, falls back to `/v1/models[0]` (arbitrary) | **Must always pass an explicit model id**; never rely on auto-detect |
| `client.models.list()` → few local models | Returns **~400+** models across all providers | Filter for the dropdown: use `/v1/models/embedding` for embed; curate a free vision set |
| Non-streaming single JSON | 9Router **defaults to streaming** | OpenAI SDK sends `stream:false` (vision OK); embeddings aren't streamed (OK). Verify any raw HTTP path sends `stream:false` |
| Default embed model `text-embedding-004` (768-d) | **404s**; working model is `gemini-embedding-001` (**3072-d**) | Change model id; **pin the model** so each ChromaDB collection keeps one dimension |
| #425 "vision not forwarded to Claude" concern | **Gemini vision forwards fine** (verified); only Claude-via-9Router was the reported bug | Prefer Gemini models for vision through 9Router; treat Kiro/Claude vision as best-effort |

---

## 6. Existing photo-vault mechanisms the design can lean on

- `constants.py` — single source for endpoints/ports (add `NINEROUTER_URL` + read port from `ports.json`).
- `vision.get_image_caption(force_provider=…, model=…, with_model=True)` — already forces one model
  and returns `(text, model_label)`; indexer stores `caption_model` + `caption_history`. (**covers #3, #4**)
- `embeddings.py` registry — each embedding model gets its **own ChromaDB collection** keyed by
  `(model_name, dimension)`, with `active_model` + `models/active` endpoints; refuses dimension changes.
  (**covers #2, #4, and the dimension-pinning safety**)
- `validator.py` — provider health checks (add a 9Router check: `GET /v1/models` → 200). (**covers #6**)
- API `models` / `models/active` endpoints + Svelte "Index & Manage" tab — natural home for the selector.

Likely shape (for later): add a first-class `"9router"` provider branch to the existing vision/embeddings
provider chains (thin wrapper over the current OpenAI-compatible code with `base_url=NINEROUTER_URL`),
plus model-list sourcing/filtering and a settings-backed model selector. **Not to be built until the
Section 4 design is done.**

---

## 7. Operational notes

- Change the 9Router dashboard password (Settings) before enabling any Tunnel/Tailscale exposure.
- 9Router has native Tailscale/Tunnel toggles (aligns with the phone→PC remote-access setup) — but
  keep photo-vault itself on loopback; reach it via the existing `rover`/`tailscale serve` front door.
- Scaling free capacity = add more Google accounts (CLI, round-robin) + more API keys (embeddings),
  all pooled automatically by 9Router. Quota Tracker shows exact per-account numbers once connected.
