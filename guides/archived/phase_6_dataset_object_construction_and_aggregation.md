## Checklist

- [✅] Create `api/services/dataset_construction_service.py` and implement `construct_datasets_from_retrieval_results`.
- [✅] Store `Dataset` instance in DuckDB using helper in `api/utils/duckdb_utils.py`
- [✅] Implement `mask_dataset_ids_in_text` in `api/services/dataset_construction_service.py`.
- [✅] Create `src/run_dataset_construction.py` to orchestrate Phase 6: load retrieval results, use dataset construction API, use embeddings API to re-embed.

## Phase 6 Completion: Dataset Object Construction & Aggregation

### 6.1 Create Dataset Construction Service
**Location**: `api/services/dataset_construction_service.py`

**Core Functions:**
```python
# Default database path
DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"
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
def construct_datasets_from_retrieval_results(
    retrieval_results_path: str = "reports/retrieval/retrieval_results.json",
    db_path: str = DEFAULT_DUCKDB_PATH,
    mask_token: str = "<DATASET_ID>"
) -> List[Dataset]:
    """
    For each dataset ID in retrieval results:
    1. Fetch all associated chunk IDs and retrieve full Chunk objects from DuckDB
    2. Concatenate chunk texts, mask dataset ID tokens
    3. Compute aggregation statistics (total_tokens, avg_tokens_per_chunk, etc.)
    4. Construct Dataset Pydantic objects
    5. Save `Dataset` Objects to DuckDB
    """


# `src/run_dataset_construction.py` 

API_ENDPOINTS = {
    "base_api_url": "http://localhost:8000",
    "embed_chunks": "/embed_chunks",
    "construct_datasets": "/get_datasets"
}
from src.models import DatasetConstructionPayload, EmbeddingResult, EmbeddingPayload
import requests

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
    # Build url for embeddings services: 
    # construct payload for embeddings service

def main():
    """
    1. Use requests to construct and save Dataset objects via API 
    2. Use requests to create embeddings from combined text and save to ChromaDB via API 
    """

```

### 6.2 Integration Script
**Location**: `src/run_dataset_construction.py`

Orchestrates the complete Phase 6:
1. Load retrieval results JSON
2. Call dataset construction service
3. Bulk updload datasets to DuckDB using DBHelper
4. Mask Citations
5. Embeddings & storage to ChromaDB
---