# Retrieval implementation report

Date: 2026-09-26. Scope: `src/embeddings.py`, `src/search.py`, `src/db.py`,
`src/vision.py`, and synthetic retrieval tests. The primary owns indexer/jobs
integration. No user media or `.env` contents were read.

## Implemented retrieval contract

- New registry entries now record a canonical immutable profile (provider,
  provider model ID, embedding-text schema, task strategy, dimension, cosine
  metric), a digest, and a collision-resistant collection name. `db.collection()`
  creates profiled collections with `hnsw:space=cosine` and profile metadata,
  then rejects metadata/metric mismatches. Existing records with no profile
  continue using their stored collection name and raw input strategy; the code
  does not move or rewrite their vectors.
- Gemini's dead `text-embedding-004` default is replaced by
  `gemini-embedding-001`. New Gemini profiles send
  `RETRIEVAL_DOCUMENT` at index time and `RETRIEVAL_QUERY` at search time. The
  provider uses synchronous `batchEmbedContents` for batches, with response
  cardinality validation and Google's documented request order. New local Nomic
  profiles use `search_document:` and `search_query:` prefixes. These strategies
  are never retroactively applied to legacy records.
- Provider identity, vector count, response indexes, numeric values, finiteness,
  non-zero vectors, and consistent dimensions are checked before results are
  returned for storage. Existing registry corruption now fails closed instead
  of appearing to be an empty registry.
- A search snapshots the active registry record once and passes it into the
  matching collection lookup and query embedding. It rejects a missing active
  profile and a returned provider/model identity mismatch. The query-vector LRU
  holds at most 128 vectors, keyed by profile identity and whitespace-normalized
  query.
- Person-constrained semantic search passes matching media IDs to Chroma before
  `n_results`, so top-k is selected from that person's candidates. Chroma 1.5.9
  is installed in the project virtual environment and its query API includes
  `ids`; a fresh temporary PersistentClient integration test exercises the
  actual call. Blank browse now uses bounded collection reads and never embeds
  the synthetic word `photo`.
- Vision output validation rejects a non-object response, missing required
  fields, a blank/non-string caption, out-of-vocabulary categorical values,
  invalid person counts, and malformed list fields before indexing accepts the
  caption.

## Integration and upgrade behavior

Indexer single-item writes, batch job writes, counts, and freshness checks now
resolve collections through `db.collection`. Explicit registry snapshots are
passed where already available; otherwise the helper loads the registry entry.
The content fingerprint covers caption/source/text inputs; the immutable profile
is enforced by the enclosing collection name and profile metadata, so vectors
from different retrieval recipes never share a freshness namespace.

Existing legacy registry records intentionally keep their collection and raw
preprocessing. This delivery does not reset or re-embed the personal library.
A future profile migration for an already registered model needs an explicit
new collection and rebuild; changing a registry profile in place is unsupported.
New model registrations use the new cosine/task-aware profiles. No automatic
collection cleanup or destructive cutover was performed.

The app currently has no relevance set for the household photo corpus, so no
claim is made that Gemini 001 or Nomic is more accurate. Query caching and
provider batching reduce repeated latency and round trips, but quality should
be measured with synthetic/manual relevance labels before any ranking change.
The backend's separate SQL lexical Library path is owned by the primary; hybrid
rank fusion remains a follow-up after lexical candidates and dense candidates
share bounded person/filter semantics.

## Validation

- Focused embedding/search/vision/profile tests passed: 161 passed in 61.16s
  under the suite-wide socket guard, with the inherited Gemini key blanked.
- The actual Chroma integration test passed against 1.5.9 using only synthetic
  vectors: it confirmed cosine/profile metadata and that `ids` constrains the
  top-1 result before ranking.
- `py_compile` passed for the four edited source modules and focused test files.
  The primary subsequently ran the integrated backend suite successfully.
- One test mock gap resulted in one unintended Gemini embedding request over
  synthetic test strings (no photo or private content); the test is now mocked
  at the batch function. Subsequent focused and integrated tests ran with
  inherited credentials cleared and all provider sockets blocked. The Windows
  asyncio internal control socketpair is the only permitted socket construction.

## Official references checked 2026-09-26

- [Gemini API deprecations](https://ai.google.dev/gemini-api/docs/deprecations):
  `text-embedding-004` shut down January 14, 2026; Gemini Embedding 001 is listed
  with a May 14, 2028 shutdown date.
- [Gemini embeddings guide](https://ai.google.dev/gemini-api/docs/embeddings):
  documents 001 retrieval task types and explains the newer Embedding 2 vector
  space is incompatible with 001 and requires re-embedding.
- [Gemini embeddings REST API](https://ai.google.dev/api/embeddings): documents
  synchronous `batchEmbedContents` and `RETRIEVAL_QUERY` /
  `RETRIEVAL_DOCUMENT` task types; batch output follows request order.
- [Google AI developer release notes](https://developers.googleblog.com/en/gemini-embedding-001-a-powerful-new-foundation-model-for-text-embedding/):
  describes Gemini Embedding 001 availability, including free-tier access. No
  quota amount or sustained free capacity is assumed here.
- [Nomic Embed Text v1.5 model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5):
  documents `search_document:` and `search_query:` prefixes.
- [Chroma collection API](https://docs.trychroma.com/reference/python/collection):
  documents `query` and `get`; installed Chroma 1.5.9's local signature and
  behavior were additionally checked through the synthetic integration test.
