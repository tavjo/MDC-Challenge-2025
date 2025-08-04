## Checklist

- [ ] Add `umap-learn` and `matplotlib` dependencies to `pyproject.toml`. **Don't forget to activate virtual environment first!** Then use command `uv add matplotlib umap-learn`
- [ ] Create helper script to import `train_labels.csv` and update `Dataset` table with PRIMARY/SECONDARY labels
- [ ] Create `src/dimensionality_reduction.py` with `Reducer` class following similar API pattern as `run_clustering.py`
- [ ] Implement `Reducer.load_embeddings_from_api()` method using same API endpoints
- [ ] Implement `Reducer.load_datasets_from_duckdb()` method to persist Dataset objects for reuse
- [ ] Implement `run_umap_reduction()` method and store UMAP features in DuckDB via `upsert_engineered_features_batch`
- [ ] Implement `create_cluster_visualization()` and `create_ground_truth_visualization()` methods
- [ ] Implement per-cluster PCA: group datasets by cluster, run PCA on each cluster's embeddings, keep PC1 (highest explained variance)
- [ ] Store per-cluster PCA features in DuckDB via `upsert_engineered_features_batch` as `LEIDEN_1`, `LEIDEN_2`, etc.
- [ ] Implement `run_pipeline()` method: load embeddings/datasets → UMAP → save UMAP features → visualizations → per-cluster PCA → save PCA features
- [ ] Create `reports/dimensionality_reduction/` output directory structure
- [ ] Add graceful failure handling when cluster assignments are missing from datasets

## Phase 8: Dimensionality Reduction & Visualization

### 8.1 Dimensionality Reduction Module  
**Location**: `src/dimensionality_reduction.py`

**Core Functions:**
```python
class Reducer:
    """
    Dimensionality reduction pipeline following similar pattern to ClusteringPipeline.
    Loads embeddings via API, runs UMAP + per-cluster PCA, creates visualizations,
    and saves engineered features to DuckDB.
    """
    
    def load_embeddings_from_api(self) -> Optional[Dict[str, np.ndarray]]:
        """Load embeddings from ChromaDB via API call using same endpoints as clustering."""
    
    def load_datasets_from_duckdb(self) -> Optional[List[Dataset]]:
        """Load all Dataset objects from DuckDB and persist for reuse across methods."""
    
    @timer_wrap  
    def run_umap_reduction(
        self,
        embeddings: Dict[str, np.ndarray],
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        random_seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Apply UMAP dimensionality reduction with reproducibility:
        - Set random_state for deterministic embedding layout
        - Save UMAP_1, UMAP_2 features to DuckDB via upsert_engineered_features_batch
        """

    @timer_wrap
    def create_cluster_visualization(
        self,
        umap_embeddings: Dict[str, np.ndarray],
        output_path: str = "reports/dimensionality_reduction/cluster_visualization.png"
    ):
        """Create scatter plot colored by Leiden cluster labels using matplotlib"""

    @timer_wrap  
    def create_ground_truth_visualization(
        self,
        umap_embeddings: Dict[str, np.ndarray], 
        output_path: str = "reports/dimensionality_reduction/ground_truth_visualization.png"
    ) -> None:
        """Create scatter plot colored by dataset_type (PRIMARY/SECONDARY) labels using matplotlib"""

    @timer_wrap
    def run_per_cluster_pca(
        self,
        dataset_embeddings: Dict[str, np.ndarray], 
        random_seed: int = 42
    ) -> bool:
        """
        Group datasets by cluster, run PCA on each cluster's embeddings:
        - Keep PC1 (highest explained variance) for each cluster
        - Save as LEIDEN_1, LEIDEN_2, etc. features to DuckDB
        - Fail gracefully if no cluster assignments exist
        """
    
    def run_pipeline(self) -> dict:
        """
        Complete pipeline: load embeddings/datasets → UMAP → save UMAP features 
        → visualizations → per-cluster PCA → save PCA features
        """
```

### 8.2 Helper Script for Dataset Labels
**Location**: `scripts/import_dataset_labels.py`
```python
# Import train_labels.csv and update Dataset table with PRIMARY/SECONDARY labels
# Map article_id from labels to document_id in datasets table
# Update dataset_type field for each dataset
```

### 8.3 Pipeline Integration
**Similar to `src/run_clustering.py`:**
- Use same API endpoints (`http://localhost:8000/load_embeddings`)
- Follow same error handling and logging patterns
- Create `reports/dimensionality_reduction/` output directory
- Save pipeline results as JSON for tracking

### 8.4 Visualization Dependencies
**Dependencies** (scikit-learn already included from clustering):
```toml
umap-learn = "^0.5.4"
matplotlib = "^3.7.0"
# Note: Seaborn removed for lighter Docker builds - matplotlib sufficient
```
---