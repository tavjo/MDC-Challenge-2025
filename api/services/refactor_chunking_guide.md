Below is a detailed, end-to-end refactoring plan.  We’ll:

1. Define a new per-document result model.  
2. Write two helper functions—one to prepare (CPU-bound) and one to commit (I/O-bound).  
3. Refactor your existing `run_semantic_chunking_pipeline` into a two-phase, parallel-prepare / serial-commit runner.  
4. Touch your FastAPI endpoint to pass through concurrency knobs.  

At the end is a concise checklist of file-by-file edits.

---

## 1) Pydantic model updates (src/models.py)

Edit `src/models.py`:

```diff
 class ChunkingResult(BaseModel):
     """Result of the chunking pipeline"""
     success: bool
     total_documents: int
     total_unique_datasets: int
     total_chunks: int
     total_tokens: int
     avg_tokens_per_chunk: float
     validation_passed: bool
     pipeline_completed_at: str
     entity_retention: float
     output_path: Optional[str]
     output_files: Optional[List[str]]
     lost_entities: Optional[Dict[str, Any]]
     error: Optional[str]

+# New: per-document result
+class DocumentChunkingResult(BaseModel):
+    document_id: str
+    success: bool
+    error: Optional[str] = None
+
+    # pipeline parameters
+    chunk_size: int
+    chunk_overlap: int
+    cfg_path: str
+    collection_name: str
+
+    # citation stats
+    pre_chunk_total_citations: int
+    post_chunk_total_citations: int
+    validation_passed: bool
+    entity_retention: float
+    lost_entities: Optional[Dict[str, Any]] = None
+
+    # chunk stats
+    total_chunks: int
+    total_tokens: int
+    avg_tokens_per_chunk: float
+
+    # outputs
+    output_path: Optional[str] = None
+    output_files: Optional[List[str]] = None
+
+    # timing
+    pipeline_started_at: str
+    pipeline_completed_at: str
```

No breaking changes to `ChunkingResult`—it remains your API-level summary.

---

## 2) Per-document “prepare” & “commit” helpers

In `api/services/chunking_and_embedding_services.py`, add above line 614 (or near other top-level functions):

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import duckdb
from datetime import datetime
from src.models import DocumentChunkingResult
```

### 2a) `prepare_document`
This runs entirely in memory (and can call Chroma embed APIs if you like).

```python
def prepare_document(
    document: Document,
    citation_entities: List[CitationEntity],
    chunk_size: int,
    chunk_overlap: int,
    cfg_path: str,
    collection_name: str,
) -> dict:
    """
    Phase 1: CPU/I-O-light work for one document.
    Returns a dict with:
      - document
      - chunks (List[Chunk])
      - pre/post citation counts & lost_entities
      - stats (total_tokens, avg_tokens…)
      - pipeline_started_at
    """
    start = datetime.now().isoformat()

    # 1) Pre-chunk inventory
    pre_df = create_pre_chunk_entity_inventory(document, citation_entities)
    pre_total = pre_df['count'].sum()

    # 2) Create & link chunks
    chunks = create_chunks_from_document(document, citation_entities,
                                         chunk_size, chunk_overlap)
    chunks = link_adjacent_chunks(chunks)

    # 3) Validate & repair
    post_passed, lost_df = validate_chunk_integrity(
        chunks, pre_df, citation_entities
    )
    if not post_passed:
        chunks, _ = repair_lost_citations_strict(document, chunks, lost_df)

    # recalc post stats
    post_total = sum(
      len(make_pattern(cid).findall(ck.text))
      for ck in chunks
      for cid in lost_df['citation_id'].unique()
    )
    total_tokens = sum(ck.chunk_metadata.token_count for ck in chunks)
    avg_tokens = total_tokens / len(chunks) if chunks else 0.0

    return {
        "document": document,
        "chunks": chunks,
        "pre_total_citations": int(pre_total),
        "post_total_citations": int(post_total),
        "validation_passed": post_passed,
        "lost_entities": lost_df.to_dict() if not post_passed else None,
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens,
        "pipeline_started_at": start,
        # carry through pipeline params
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "cfg_path": cfg_path,
        "collection_name": collection_name,
    }
```

### 2b) `commit_document`
Serializes writes to DuckDB, then ChromaDB, with rollback on failure:

```python
def commit_document(prep: dict, db_path: str) -> DocumentChunkingResult:
    """
    Phase 2: single-threaded persistence. Returns DocumentChunkingResult.
    Rolls back on DuckDB errors; cleans up on ChromaDB errors.
    """
    doc: Document = prep["document"]
    chunks: List[Chunk] = prep["chunks"]
    start = prep["pipeline_started_at"]
    now = datetime.now().isoformat()

    # Build base result
    base = dict(
        document_id=doc.doi,
        success=False,
        chunk_size=prep["chunk_size"],
        chunk_overlap=prep["chunk_overlap"],
        cfg_path=prep["cfg_path"],
        collection_name=prep["collection_name"],
        pre_chunk_total_citations=prep["pre_total_citations"],
        post_chunk_total_citations=prep["post_total_citations"],
        validation_passed=prep["validation_passed"],
        entity_retention=(
          prep["post_total_citations"] / prep["pre_total_citations"] * 100
          if prep["pre_total_citations"] else 100.0
        ),
        lost_entities=prep["lost_entities"],
        total_chunks=prep["total_chunks"],
        total_tokens=prep["total_tokens"],
        avg_tokens_per_chunk=prep["avg_tokens"],
        pipeline_started_at=start,
        pipeline_completed_at=now,
    )

    # 1) DuckDB transaction
    try:
        conn = duckdb.connect(db_path)
        conn.begin()
        save_chunks_to_duckdb(chunks, db_path)
        conn.commit()
    except Exception as e:
        conn.rollback()
        base.update(error=f"DuckDB save failed: {e}")
        return DocumentChunkingResult(**base)

    # 2) ChromaDB
    try:
        save_chunk_objs_to_chroma(chunks,
                                 collection_name=prep["collection_name"],
                                 cfg_path=prep["cfg_path"])
    except Exception as e:
        # best-effort cleanup by chunk_id
        cleanup_chroma_by_ids(
            [ck.chunk_id for ck in chunks],
            prep["collection_name"], prep["cfg_path"]
        )
        base.update(error=f"ChromaDB save failed: {e}")
        return DocumentChunkingResult(**base)

    # success
    base["success"] = True
    return DocumentChunkingResult(**base)
```

> You’ll need to write `cleanup_chroma_by_ids(chunk_ids, collection_name, cfg_path)` to remove any partial inserts.

---

## 3) Refactor `run_semantic_chunking_pipeline`

Replace the body at line 716 in `api/services/chunking_and_embedding_services.py` with:

```python
def run_semantic_chunking_pipeline(
    documents_path: str = …,
    citation_entities_path: str = …,
    output_dir: str = “Data”,
    chunk_size: int = 300,
    chunk_overlap: int = 2,
    collection_name: str = “semantic_chunks”,
    cfg_path: str = “configs/chunking.yaml”,
    use_duckdb: bool = True,
    db_path: str = “artifacts/mdc_challenge.db”,
    max_workers: int = 4,
) -> ChunkingResult:
    # 1) Load input data
    if use_duckdb:
        docs, cites = load_input_data_from_duckdb(db_path)
    else:
        docs, cites = load_input_data(documents_path, citation_entities_path)

    # 2) Phase 1: parallel preparation
    prepped = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {
          exe.submit(
            prepare_document, doc, cites,
            chunk_size, chunk_overlap,
            cfg_path, collection_name
          ): doc
          for doc in docs
        }
        for fut in as_completed(futures):
            try:
                prepped.append(fut.result())
            except Exception as e:
                # crash in prepare → record as failed DocumentChunkingResult
                doc = futures[fut]
                prepped.append({
                  "document": doc,
                  "chunks": [],
                  "pre_total_citations": 0,
                  "post_total_citations": 0,
                  "validation_passed": False,
                  "lost_entities": None,
                  "total_chunks": 0,
                  "total_tokens": 0,
                  "avg_tokens": 0,
                  "pipeline_started_at": datetime.now().isoformat(),
                  "chunk_size": chunk_size,
                  "chunk_overlap": chunk_overlap,
                  "cfg_path": cfg_path,
                  "collection_name": collection_name,
                  "prep_error": str(e)
                })

    # 3) Phase 2: serial commit
    doc_results: List[DocumentChunkingResult] = []
    for prep in prepped:
        if "prep_error" in prep:
            # build a failure DocumentChunkingResult directly
            doc_results.append(DocumentChunkingResult(
              document_id=prep["document"].doi,
              success=False,
              error=prep["prep_error"],
              chunk_size=chunk_size,
              chunk_overlap=chunk_overlap,
              cfg_path=cfg_path,
              collection_name=collection_name,
              pre_chunk_total_citations=0,
              post_chunk_total_citations=0,
              validation_passed=False,
              entity_retention=0.0,
              lost_entities=None,
              total_chunks=0,
              total_tokens=0,
              avg_tokens_per_chunk=0.0,
              pipeline_started_at=prep["pipeline_started_at"],
              pipeline_completed_at=datetime.now().isoformat()
            ))
        else:
            doc_results.append(commit_document(prep, db_path))

    # 4) Summarize into one ChunkingResult
    return summarize_run(doc_results)
```

You’ll need to add or import `summarize_run`, which should take `List[DocumentChunkingResult]` and produce your legacy `ChunkingResult` (as per our earlier pseudocode).

---

## 4) Update `ChunkingPipelinePayload` and CLI payload

- In `src/models.py`, update `ChunkingPipelinePayload` to include:

  ```python
  max_workers: Optional[int] = Field(1, description="Number of parallel worker threads")
  ```

- In the FastAPI `/run_semantic_chunking` handler (`api/chunk_and_embed_api.py`), add to `pipeline_params`:

  ```python
  pipeline_params["max_workers"] = payload.max_workers or 1
  ```

- In `src/run_semantic_chunking.py`:
  1. Modify `SemanticChunkingPipeline.__init__` signature to accept `max_workers: int = 1` and set `self.max_workers = max_workers`.
  2. In `_construct_payload()`, include:

     ```python
     max_workers=self.max_workers,
     ```

---

## 5) FastAPI endpoint tweak (api/chunk_and_embed_api.py)

In your `/run_semantic_chunking` handler, pull `max_workers: int = Query(4)` from the payload or query string, and pass it into `run_semantic_chunking_pipeline(...)`.  The endpoint still returns a single `ChunkingResult`.

---

## 5) (Optional) CLI script

If you drive via `src/run_semantic_chunking.py`, import the new runner directly instead of HTTP, accept a `--max-workers` flag, and print both the per-doc JSON and the consolidated summary.

---

## ✅ Final Checklist

1. **src/models.py**  
   – Add `DocumentChunkingResult`.  
   – Add `max_workers` to `ChunkingPipelinePayload`.  

2. **api/services/chunking_and_embedding_services.py**  
   – Implement `prepare_document` (phase 1).  
   – Implement `commit_document` (phase 2).  
   – Add `cleanup_chroma_by_ids`.  
   – Refactor `run_semantic_chunking_pipeline` into two phases + `summarize_run`.  

3. **api/chunk_and_embed_api.py**  
   – Accept and pass through `max_workers` in `/run_semantic_chunking`.  

4. **src/run_semantic_chunking.py**  
   – Add `max_workers` param and include it in `_construct_payload()`.  

5. **Write tests**  
   – Unit-test `prepare_document` happy & error paths.  
   – Unit-test `commit_document` rollback.  
   – Integration test: N docs with one forced failure.  

Once you’ve ticked off those steps, you’ll have a fully fault-tolerant, parallelized chunking pipeline with per-document visibility—and your `/run_semantic_chunking` remains a single, summary API call.