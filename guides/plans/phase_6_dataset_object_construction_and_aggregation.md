## Checklist

- [ ] Create `api/services/dataset_construction_service.py` and implement `construct_datasets_from_retrieval_results`.
- [ ] Implement `mask_dataset_ids_in_text` in `api/services/dataset_construction_service.py`.
- [ ] Implement `compute_neighborhood_embedding_stats` in `api/services/dataset_construction_service.py`.
- [ ] Create `scripts/run_dataset_construction.py` to orchestrate Phase 6: load retrieval results, call dataset construction service, export `dataset_embedding_stats.csv`, and validate Dataset objects.
- [ ] Implement `store_dataset_embeddings_in_chroma`.
- [ ] Reuse `save_chunks_to_chroma` from `semantic_chunking.py`.

## Phase 6 Completion: Dataset Object Construction & Aggregation

### 6.1 Create Dataset Construction Service
**Location**: `api/services/dataset_construction_service.py`

**Core Functions:**
```python
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
    db_path: str,
    collection_name: str,
    dataset_collection_name: str = "dataset-aggregates-train"
) -> Dict[str, np.ndarray]:
    """
    Compute dataset-level embeddings and store in ChromaDB:
    - Re-embed concatenated masked text
    - Store in dedicated ChromaDB collection for dataset-level vectors
    """

# **Use existing  save_chunks_to_chroma function in `semantic_chunking.py`**
@timer_wrap  
def store_dataset_embeddings_in_chroma(
    dataset_embeddings: Dict[str, np.ndarray],
    dataset_metadata: Dict[str, Dict[str, str]],
    collection_name: str = "dataset-aggregates-train",
    cfg_path: str = "configs/chunking.yaml"
):
    """Store dataset-level embeddings in dedicated ChromaDB collection"""

# ** Store Dataset Objects in DuckDB**

```

### 6.2 Integration Script
**Location**: `scripts/run_dataset_construction.py`

Orchestrates the complete Phase 6:
1. Load retrieval results JSON
2. Call dataset construction service
3. Bulk updload datasets to DuckDB using DBHelper
4. Export `dataset_embedding_stats.csv`
---