## Kaggle Submission Plan (Siloed Scripts)

### Objective
Prepare Kaggle-friendly, siloed Python modules under `src/kaggle/` that:
- Only import from standard libraries and `src/kaggle/*` (no cross-package imports).
- Reuse existing project logic where specified, inlined where necessary.
- Persist intermediate state in a DuckDB file under `/kaggle/tmp/`.
- Prefer simplicity and reproducibility over avoiding duplication.

### Environment and paths (Kaggle)
- Trained RF model: `/kaggle/input/rf_model.pkl`
- Offline model cache (already in Kaggle): `/kaggle/input/baaibge-small-en-v1.5/transformers/default/1/offline_models/`
- Embedding model name: `BAAI/bge-small-en-v1.5` (384 dims)
 - DuckDB path: `/kaggle/tmp/mdc.duckdb` (created on first use; ephemeral)
 - Intermediate Outputs (CSVs/Parquet): `/kaggle/tmp/` (ephemeral)
 - Persistent Outputs (CSVs/Parquet): `/kaggle/working/` (preserved in versioned output)
- Test PDFs directory (already in Kaggle): `/kaggle/input/make-data-count-finding-data-references/test/PDF/`
- Test XML directory (already in Kaggle): `/kaggle/input/make-data-count-finding-data-references/test/XML/` --> (only if there are some documents that are only provided as XML but not pdf)
- BAML directory (already uploaded to Kaggle): `/kaggle/input/baml-folders/src/` --> this includes both `baml_src` and `baml_client` subdirectories that directly mirror the current versions of these directories in `src`

### What “idempotent” means (requested clarification)
In this plan, an idempotent function can be safely re-run without causing duplicates or inconsistent state (e.g., running schema initialization multiple times won’t re-create tables or break constraints; repeated upserts won’t create duplicates).

---

## Steps and Files

### 1) Document parsing (DONE)
- File: `src/kaggle/build_docs.py`
- Status: Completed. Uses unstructured (strategy="fast") with OCR fallback (Tesseract via PyMuPDF/pdf2image) to build `Document` objects.

### 2) Chunking (sliding window only) and context integration
- New file: `src/kaggle/chunking.py`
- Purpose: Centralize sliding-window chunk ops for downstream scripts.
- Implementation details:
  - Use sliding window from `src/kaggle/helpers.py` (no semantic splitter here).
  - Functions:
    - `load_input_from_duckdb(conn) -> list[Document]`: fetch only documents that need chunking, or all.
    - `create_chunks_from_documents(docs, chunk_size, overlap) -> list[Chunk]`: build chunks from `Document.full_text` using helpers’ `sliding_window_chunks`.
    - `link_adjacent_chunks(chunks) -> list[Chunk]`: fill `previous_chunk_id`/`next_chunk_id`.
    - `save_chunks_to_duckdb(conn, chunks) -> None`.
    - `create_chunks_summary_csv(conn, out_path) -> Path`.
  - No embeddings here; keep it purely chunking.
- Edits to existing script: `src/kaggle/get_citation_context.py` should `import` the above functions and use them instead of maintaining its own chunking logic.
- Note: `src/kaggle/chunk_and_embed.py` becomes unnecessary; we will not use it in the Kaggle pipeline (acknowledging the earlier confusion).

### 3) DuckDB schema and utilities (no embeddings persistence)
- New file: `src/kaggle/duckdb.py`
- Purpose: Provide DB connection, schema init, and CRUD helpers, reusing existing schema and method patterns from `api/database` and `api/utils` (excluding embedding persistence).
- Implementation notes:
   - Reuse/port the schema definitions from `api/database/duckdb_schema.py` and utilities from `api/utils/duckdb_utils.py` where applicable.
  - Functions:
    - `get_conn(db_path: str | None = None)` → default to `/kaggle/tmp/mdc.duckdb`.
    - `init_schema(conn)` → idempotent table/index creation for: `documents`, `chunks`, `datasets`, `engineered_feature_values` (+ any needed helper tables). No embeddings table.
    - `upsert_documents(conn, docs: list[Document])`.
    - `bulk_insert_chunks(conn, chunks: list[Chunk])`.
    - `bulk_upsert_datasets(conn, datasets: list[Dataset])`.
    - `eav_helpers` for `engineered_feature_values` (bulk insert, fetch by dataset/document).
    - Fetch helpers as needed (e.g., pending documents, chunks by document).
    - `get_full_dataset_dataframe`for building rf model input from `datasets` and `engineered_feature_values` tables

### 4) Citation context
- File: `src/kaggle/get_citation_context.py`
- Changes:
  - Replace internal chunking with imports from `src/kaggle/chunking.py`:
    - `load_input_from_duckdb`, `create_chunks_from_documents`, `link_adjacent_chunks`, `save_chunks_to_duckdb`, `create_chunks_summary_csv`.
  - Connect via `src/kaggle/duckdb.py`.
  - Keep other logic as-is but ensure all imports are from `src/kaggle`.

### 5) Validation entities
- File: `src/kaggle/get_citation_entities_val.py`
- Changes:
  - Fix imports to only use `src/kaggle/*` and standard libs.
  - Use `src/kaggle/duckdb.py` for DB I/O.
  - Change baml import to the directory mentioned above. 

### 6) Construct datasets (Kaggle version)
- New file: `src/kaggle/construct_datasets_kaggle.py`
- Goal: Consolidate current multi-module logic into a single Kaggle-ready script with minimal dependencies.
- Source modules to reuse/adapt:
  - `src/construct_queries.py` (query string construction; `add_target_chunk_and_neighbors` with a small tweak to return a dict or a lightweight pydantic model instead of `BatchRetrievalResult`).
  - `src/kaggle/retrieval_module.py` (`hybrid_retrieve_with_boost`).
  - `api/services/dataset_construction_service.py` (`mask_dataset_ids_in_text`, `construct_datasets_from_retrieval_results`).
- Detailed flow:
  1. Build query texts from `CitationEntity` and `Chunk` (reuse `construct_queries` utilities). No `Document` loading here.
  2. For each `dataset_id`, restrict candidate chunks to the same `document_id` (both `CitationEntity` and `Chunk` include `document_id`).
  3. Prepare `id_to_text` from the document’s chunks; compute `id_to_dense` on the fly using `BAAI/bge-small-en-v1.5` (use the embedding helper in `src/kaggle/helpers.py`).
  4. Call `hybrid_retrieve_with_boost(query_text, id_to_text, id_to_dense, ...)` to get relevant chunks. Wrap this to associate `{dataset_id: [chunk_ids, target_chunk_id, neighbor_ids]`.
  5. Ensure target chunk and neighbors are included (minor tweak/adapter to `add_target_chunk_and_neighbors`).
  6. Apply `mask_dataset_ids_in_text` and `construct_datasets_from_retrieval_results` to produce dataset texts.
  7. Construct `Dataset` objects and bulk upsert into DuckDB.
  8. Embed dataset texts (bulk) with `BAAI/bge-small-en-v1.5` using the embedding function in `src/kaggle/helpers.py`.
   9. Create a DataFrame: rows = `dataset_id`, columns = 384 embedding dims; save to `/kaggle/tmp/dataset_embeddings.parquet` (or `.csv` if preferred).
- Notes:
  - Use batched embedding to control memory.
  - Keep the per-document constraint on retrieval strict to avoid cross-document leakage.

### 7) Clustering + dimensionality reduction (Leiden/igraph unchanged)
- Files: `src/kaggle/clustering_kaggle.py`, `src/kaggle/dimensionality_reduction_kaggle.py`
- Requirement: DO NOT change the Leiden + igraph approach; reuse existing logic from `src/clustering.py` and any relevant `dimensionality_reduction.py` utilities.
- Implementation:
  - Vendor (inline) only the minimum required functions into the Kaggle versions to keep them siloed, preserving algorithms and parameter defaults.
  - Write outputs to `engineered_feature_values` (e.g., `UMAP_1`, `UMAP_2`, `LEIDEN_1`, plus any extra Leiden dims/components) via `src/kaggle/duckdb.py`.

### 8) Global features: UMAP, PCA, neighborhood stats (Optional)
- File: `src/kaggle/global_features_kaggle.py`
- Purpose: Compute global feature summaries and neighborhood stats per dataset.
- Implementation:
  - Build per-dataset feature vectors (e.g., pooled chunk embeddings or dataset embeddings already computed in Step 6).
  - Run PCA/UMAP; compute kNN-based neighborhood metrics.
  - Store in `engineered_feature_values` (EAV).

### 9) Build inputs for trained classifier (in notebook)
- Use DuckDB helper method that constructs a full DataFrame from `Dataset` and `engineered_feature_values`.
- Reference: first 3 cells of `notebooks/training_input.ipynb`.
- Export features to `/kaggle/working/training_features.parquet`.

### 10) Inference → submission
- New file: `src/kaggle/inference_kaggle.py`
- Load trained model artifact (provided via Kaggle dataset or notebook cell).
- Predict dataset type per `(article_id, dataset_id)` and write `submission.csv` (columns: `article_id,dataset_id,type`) to `/kaggle/working/submission.csv`.

---

## Exports and artifacts
- `/kaggle/tmp/mdc.duckdb` (DuckDB file with documents/chunks/datasets/EAV)
- `/kaggle/working/chunks_summary.csv` (from `create_chunks_summary_csv`)
- `/kaggle/tmp/dataset_embeddings.parquet` (384-dim BGE-small v1.5 vectors)
- `/kaggle/working/training_features.parquet` (joined Dataset+features for classifier)
- `/kaggle/working/submission.csv` (final output)

---

## Implementation order
1) `src/kaggle/duckdb.py` (schema + helpers, reusing `api/database` + `api/utils` patterns; no embeddings storage)
2) `src/kaggle/chunking.py` (sliding-window only) and edit `src/kaggle/get_citation_context.py`
3) `src/kaggle/get_citation_entities_val.py` (imports)
4) `src/kaggle/construct_datasets_kaggle.py` (queries → retrieval → dataset build → dataset embeddings file)
5) `src/kaggle/clustering_kaggle.py` and `src/kaggle/dimensionality_reduction_kaggle.py` (Leiden/igraph logic intact)
6) `src/kaggle/global_features_kaggle.py` (optional, if required by pipeline)
7) `src/kaggle/inference_kaggle.py`

---

## Notes
- Embedding model throughout: `BAAI/bge-small-en-v1.5` (not MiniLM); ensure the embedding helper in `src/kaggle/helpers.py` uses this by default.
- We are not persisting embeddings in DuckDB. We persist only logical entities (documents, chunks, datasets, engineered features), and export embeddings to files when needed.
- All scripts will be siloed under `src/kaggle/` and callable from the Kaggle notebook in the stated order.


