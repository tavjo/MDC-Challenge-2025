## Checklist

- [ ] Read relevant files thoroughly (`duckdb_schema.py`, `duckdb_utils.py`, `semantic_chunking.py`, `pyproject.toml`, `Dockerfile.api`, `docker-compose.yaml`)
- [ ] Add dependencies to `pyproject.toml` if needed: `leidenalg`, `python-igraph`, and `scikit-learn`. **Don't forget to activate virtual environment first!** Then use command `uv add leidenalg python-igraph scikit-learn`
- [ ] Create `src/clustering.py`.
- [ ] Implement `build_knn_similarity_graph`.
- [ ] Implement `determine_similarity_threshold`.
- [ ] Implement `run_leiden_clustering`.
- [ ] Update dataset records in DuckDB with cluster membership
- [ ] Add new feature values (aside from clusters) from neighborhood embeddings stats to `engineered_feature_values` table in DuckDB using functions that already exist in `duckdb_utils.py`.
- [ ] Export summary report in json to `reports/clustering`

## Phase 7: Network-Based Clustering

### 7.1 Create Clustering Module
**Location**: `src/clustering.py`

**Core Functions:**
```python

# ** Load Dataset embeddings from ChromaDB**

@timer_wrap
def compute_neighborhood_embedding_stats(
    dataset_embedding: np.ndarray,
    chroma_client,
    collection_name: str,
    k: int = 5
) -> Dict[str, float]:
    """
    Find k-nearest neighbors to dataset embedding and compute:
    - mean, max, variance of neighbor similarities
    - mean, max, variance of neighbor embedding norms
    """

@timer_wrap  
def build_knn_similarity_graph(
    dataset_embeddings: Dict[str, np.ndarray], 
    k_neighbors: int = 30,
    similarity_threshold: float = None,
    threshold_method: str = "degree_target"
) -> ig.Graph:
    """
    Memory-efficient k-NN similarity graph construction:
    1. Use sklearn.neighbors.NearestNeighbors for O(N*k) memory complexity
    2. Apply similarity threshold to filter edges
    3. Build igraph directly (no NetworkX conversion)
    4. Default to degree-target heuristic for deterministic thresholding
    """

# upsert engineered features to DuckDB

@timer_wrap
def determine_similarity_threshold(
    distances: np.ndarray, 
    method: str = "degree_target",
    target_degree: int = 15
) -> float:
    """
    Dynamically determine similarity threshold:
    - degree_target: Target average node degree (default, deterministic)
    - percentile_90: Use 90th percentile of similarities
    - elbow_method: Find largest gap in sorted similarities
    """

@timer_wrap
def run_leiden_clustering(
    graph: ig.Graph, 
    resolution: float = 1.0, 
    min_cluster_size: int = 2,
    random_seed: int = 42
) -> Dict[str, str]:
    """
    Apply Leiden clustering with safeguards:
    1. Use igraph directly (no NetworkX conversion)
    2. Set random seed for reproducibility
    3. Filter out clusters smaller than min_cluster_size
    4. Return {dataset_id: cluster_label} mapping
    """

# Update `Dataset` objects in DuckDB by adding value for cluster
# **Add new features to DuckDB `engineered_feature_values` table** 
```


### 7.2 Clustering Dependencies
**New Dependencies** (add to `pyproject.toml`):
```toml
leidenalg = "^0.10.1"
python-igraph = "^0.11.0"  
scikit-learn = "^1.3.0"
```
---