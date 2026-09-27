# Synthetic search preview fixture

`docs/testing/search_preview_fixture.py` creates a small, isolated library for
manual browser checks. It generates 40 geometric PNGs and seeds their records
through `catalog_db`; it does not launch the app or run inference.

Choose a new temporary directory or an existing empty directory, then pass it
explicitly. For example, in PowerShell:

```powershell
$FixtureDir = Join-Path $env:TEMP 'photo-vault-search-preview'
.venv\Scripts\python.exe docs/testing/search_preview_fixture.py --target $FixtureDir
```

The script refuses the repository or its `data/` subtree as a target, and
refuses any existing target that is not an empty directory. It sets
`PHOTO_VAULT_DATA_DIR` to the target and `PHOTO_VAULT_ENV_FILE=-` before
importing Photo Vault modules, removes inherited provider credentials and
endpoints, and uses synthetic records only. The fixture writes a canonical
`person_map.json` entry for **Preview Person** with a synthetic zero vector so
the Smart name suggestion can be checked. It creates no face files or face
index; use the name field only and do not run a face/person search against this
placeholder. It prints a JSON manifest with stable IDs, result counts, today's
month/day, and the generated anniversary years. Keep the directory intact
while the primary starts the preview runtime.

## Browser checks

Treat each row as a Browse request/filter and compare the returned cards against
the listed IDs. The fixture deliberately uses punctuation-free unique terms so
the visible results are easy to inspect.

| Check | Query or filter | Expected IDs / count |
|---|---|---|
| Quoted phrase | `"birthday cake"` | `sv-bday-cake-01` through `sv-bday-cake-04` (4); the birthday card is excluded. |
| Exclusion token boundary | `vacation -cat` | `sv-vacation-01` through `sv-vacation-04` and `sv-vacation-catalog` (5). `cat` is a substring in “vacation” and “catalog,” so neither creates a false negative. |
| Exact cat word | `cat` | `sv-cat-01` through `sv-cat-03` (3); vacation results stay out. |
| OCR/text field | query `receipt`, text scope `text` | `sv-receipt-text` (1); invoice and screenshot records have separate OCR strings. |
| Screenshot shortcut/facet | photo type `screenshot` | `sv-screenshot-text`, `sv-screenshot-clean` (2). Only the first has recognized text. |
| Animal/details search | `golden retriever`, details scope | `sv-animal-dog` (1). Also try `bear` → `sv-animal-bear`. |
| Camera scope | `Canon EOS`, camera scope | `sv-camera-canon` (1); make/model is stored as `Canon` + `Canon EOS R5` to check prefix de-duplication. |
| Place scope | `Monument Valley`, place scope | `sv-place-monument` (1). |
| Valid GPS | Has location `yes` | `sv-place-monument`, `sv-gps-zero`, `sv-gps-coordinate` (3); zero latitude/longitude is valid. |
| Missing GPS | Has location `no` | 37 records, including `sv-gps-incomplete`; incomplete coordinates count as missing. |
| Recognized text | Has recognized text `yes` | `sv-receipt-text`, `sv-invoice-text`, `sv-screenshot-text` (3). |
| Missing caption | Has caption `no` | `sv-history-only`, `sv-no-caption-empty`, `sv-no-caption-missing` (3). |
| Caption history isolation | query `retiredmarkerq7` | 0; this marker exists only in an old caption-history entry. |
| Unicode | query `café` | `sv-unicode-cafe` (1). |
| Same month/day across years | month/day from the printed manifest | `sv-onthisday-1` through `sv-onthisday-4` (4 generated years; the manifest lists them). |
| Date range | 2019-01-01 through 2021-12-31 | `sv-date-early`, `sv-date-middle` (2). |

The full catalog contains 40 generated images. The other 37 rows without valid
GPS include every fixture except the three named positive GPS examples. Smart
search and provider-backed behavior are outside this preview; no searches here
should require inference. The primary owns server startup and browser
validation.
