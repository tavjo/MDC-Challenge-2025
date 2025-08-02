## Checklist

- [ ] Add `umap-learn` and `matplotlib` dependencies to `pyproject.toml`. **Don't forget to activate virtual environment first!** Then use command `uv add matplotlib umap-learn`
- [ ] Load dataset-level embeddings from ChromaDB collection named: "dataset-aggregates-train"
- [ ] Create `src/dimensionality_reduction.py` and implement `run_umap_reduction`, `create_cluster_visualization`, `create_ground_truth_visualization`.
- [ ] Implement `run_pca_reduction` and run on dataset embeddings within each cluster
- [ ] Construct `EngineeredFeatures` instances and either save to DuckDB or export as csv file

## Phase 8: Dimensionality Reduction & Visualization

### 8.1 Dimensionality Reduction Module  
**Location**: `src/dimensionality_reduction.py`

**Core Functions:**
```python
@timer_wrap  
def run_umap_reduction(
    embeddings: Dict[str, np.ndarray],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Apply UMAP dimensionality reduction with reproducibility:
    - Set random_state for deterministic embedding layout
    - Critical for consistent visualizations across pipeline runs
    """
# Upsert engineered features to DuckDB

@timer_wrap
def create_cluster_visualization(
    umap_embeddings: Dict[str, np.ndarray],
    cluster_labels: Dict[str, str],
    output_path: str = "reports/cluster_visualization.png"
):
    """Create scatter plot colored by Leiden cluster labels using matplotlib"""

@timer_wrap  
def create_ground_truth_visualization(
    umap_embeddings: Dict[str, np.ndarray], 
    datasets: List[Dataset],
    output_path: str = "reports/ground_truth_visualization.png"
) -> None:
    """Create scatter plot colored by dataset_type (PRIMARY/SECONDARY) labels using matplotlib"""

# Per Leiden cluster Dimensionality Reduction (i.e. run PCA on only the embeddings of the datasets that belong to a given Leiden cluster --> keep PC with 95% variance; reduces dimensionality from ~400 features to N clusters)
@timer_wrap
def run_pca_reduction(
    dataset_embeddings: Dict[str, np.ndarray], 
    n_components: int = None,
    variance_threshold: float = 0.95,
    random_seed: int = 42
) -> Tuple[Dict[str, np.ndarray], PCA]:
    """
    Apply PCA to dataset embeddings with reproducibility:
    - Auto-determine n_components to retain ≥95% variance if not specified
    - Set random seed for consistent results across runs
    - Return transformed embeddings and fitted PCA object
    """
# Upsert engineered features to DuckDB
```

### 8.2 Visualization Dependencies
**Dependencies** (scikit-learn already included from clustering):
```toml
umap-learn = "^0.5.4"
matplotlib = "^3.7.0"
# Note: Seaborn removed for lighter Docker builds - matplotlib sufficient
```
---