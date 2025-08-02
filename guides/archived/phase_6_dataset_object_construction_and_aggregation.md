## Checklist

- [ ] Create `api/services/dataset_construction_service.py` and implement `construct_datasets_from_retrieval_results`.
- [ ] Store `Dataset` instance in DuckDB using helper in `api/utils/duckdb_utils.py`
- [ ] Implement `mask_dataset_ids_in_text` in `api/services/dataset_construction_service.py`.
- [ ] Compute and store embeddings using existing embeddings services (`api/chunk_and_embed_api.py`)
- [ ] Create `scripts/run_dataset_construction.py` to orchestrate Phase 6: load retrieval results, call dataset construction service, export `dataset_embedding_stats.csv`, and validate Dataset objects.

## Phase 6 Completion: Dataset Object Construction & Aggregation

### 6.1 Create Dataset Construction Service
**Location**: `api/services/dataset_construction_service.py`

**Core Functions:**
```python

API_ENDPOINTS = {
    "base_api_url": "http://localhost:8000",
    "create_chunks": "/create_chunks",
    "batch_create_chunks": "/batch_create_chunks",
    "embed_chunks": "/embed_chunks",
    "run_semantic_chunking": "/run_semantic_chunking",
    "chunk_specific_documents": "/chunk/documents"
}

@timer_wrap
def construct_datasets_from_retrieval_results(
    retrieval_results_path: str = "reports/retrieval/retrieval_results.json",
    db_path: str = "artifacts/mdc_challenge.db",
    collection_name: str = "mdc_training_data",
    k_neighbors: int = 5,
    mask_token: str = "<DATASET_ID>"
) -> List[Dataset]:
    """
    For each dataset ID in retrieval results:
    1. Fetch all associated chunk IDs and retrieve full Chunk objects from DuckDB
    2. Concatenate chunk texts, mask dataset ID tokens
    3. Compute aggregation statistics (total_tokens, avg_tokens_per_chunk, etc.)
    4. Re-embed masked concatenated text
    5. Collect k-nearest neighbor embeddings and compute neighborhood stats
    6. Construct Dataset Pydantic objects
    """

# Save `Dataset` objects to DuckDB

@timer_wrap 
def mask_dataset_ids_in_text(text: str, dataset_ids: List[str], mask_token: str = "<DATASET_ID>") -> str:
    """
    Robust multi-ID masking with regex escaping:
    - Handles multiple dataset IDs per text
    - Case-insensitive matching with word boundaries  
    - Prevents leakage from compound citations
    """
    import re
    if not ids:
        return text
    pattern = r'(' + r'|'.join(re.escape(i) for i in ids) + r')'
    return re.sub(pattern, mask_token, text, flags=re.IGNORECASE)


@timer_wrap
def compute_dataset_embeddings(
    datasets: List[Dataset],
    dataset_metadata: Dict[str, Dict[str, str]],
    collection_name: str = "dataset-aggregates-train",
    cfg_path: str = "configs/chunking.yaml"

) -> Dict[str, np.ndarray]:
    """
    Compute dataset-level embeddings and store in ChromaDB:
    - Re-embed concatenated masked text 
    - Store in dedicated ChromaDB collection for dataset-level vectors
    """
    # url for embeddings services: 
    # construct payload for embeddings service

```

### 6.2 Integration Script
**Location**: `scripts/run_dataset_construction.py`

Orchestrates the complete Phase 6:
1. Load retrieval results JSON
2. Call dataset construction service
3. Bulk updload datasets to DuckDB using DBHelper
4. Mask Citations
5. Embeddings & storage to ChromaDB
6. Export `dataset_embedding_stats.csv`
---