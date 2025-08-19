### Updated plan tailored to your notebook (two loops; add CitationEntity objects, not strings)

Where to add: Use your existing placeholder cell (Cell 40: “add citation entities to chunk metadata”).

1) Citation matching helpers are already defined in the notebook
- No imports needed. Use the in-notebook `find_citation_matches` implementation you pasted.

2) Open DB helper and prepare caches
- `db = get_duckdb_helper(DEFAULT_DUCKDB)` (same variable used across the notebook).
- Prepare:
  - `chunks_cache: Dict[str, List[Chunk]] = {}` to avoid repeated DB reads per document.
  - `dirty_chunks: Dict[str, Chunk] = {}` to accumulate all updated chunks globally (keyed by `chunk_id`).

3) Two-loop annotation (no third nesting; add CitationEntity objects)
- Iterate over citations directly:
  - `for ce in db.get_all_citation_entities():`
    - Fetch chunks for `ce.document_id` using cache:
      - `chunks = chunks_cache.get(ce.document_id) or db.get_chunks_by_document_id(ce.document_id)`; store in cache if missing.
    - For each `chunk` in `chunks`:
      - If `find_citation_matches(ce.data_citation, chunk.text)` returns any matches:
        - Ensure `chunk.chunk_metadata.citation_entities` is a list (init to `[]` if `None`).
        - Append the `CitationEntity` object `ce` only if its `data_citation` is not already present for that chunk (dedupe by `data_citation` only).
-        - Record the updated chunk: `dirty_chunks[chunk.chunk_id] = chunk`.
-
- Validation before upsert (aim: 100% entity retention)
- After both loops complete, but before any upsert:
  - Build `orig_unique = {ce.data_citation for ce in citations}` where `citations = db.get_all_citation_entities()` was the list iterated.
  - Build `matched_unique = {ent.data_citation for ck in dirty_chunks.values() for ent in (ck.chunk_metadata.citation_entities or [])}`.
  - If `len(matched_unique) != len(orig_unique)` (or `matched_unique != orig_unique`), treat as validation failure: print a concise diff of missing/extra IDs and DO NOT upsert; stop with a clear error so you can inspect.

- If validation passes, upsert once:
  - If `dirty_chunks` is non-empty, call `db.bulk_insert_chunks(list(dirty_chunks.values()))`.
- Notes:
  - Add `CitationEntity` objects to `chunk.chunk_metadata.citation_entities` (not strings or match snippets). The notebook’s `Chunk.to_duckdb_row()` handles serialization to `VARCHAR[]`.
  - Do not alter `previous_chunk_id`, `next_chunk_id`, or `token_count`.

4) No document-level updates needed
- Skip updating the `documents` table. `citations` and `chunks` both carry `document_id`, which is sufficient for downstream steps.
- Close DB: `db.close()`.



5) Proceed with your existing Step 4 cells
- Run `construct_datasets_pipeline(embedder)` as in Cell 43; `get_query_texts` will now find target+neighbors via `chunk_metadata.citation_entities`.

Key cautions
- Use the notebook’s `DEFAULT_DUCKDB` and in-notebook models/classes.
- Deduplicate `CitationEntity` per chunk by `data_citation` only.
- Perform a single global `bulk_insert_chunks` with all updated chunks at the end.