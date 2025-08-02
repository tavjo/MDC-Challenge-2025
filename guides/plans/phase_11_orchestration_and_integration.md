## Checklist

- [ ] Create `scripts/run_feature_engineering_pipeline.py` and implement the `main` function to orchestrate Phases 6-10.
- [ ] Update `docker-compose.yml` to add the feature engineering command to the preprocessing service.

## Phase 11: Orchestration & Integration

### 11.1 Master Pipeline Script
**Location**: `scripts/run_feature_engineering_pipeline.py`

```python
@timer_wrap
def main(
    retrieval_results_path: str = "reports/retrieval/retrieval_results.json",
    db_path: str = "artifacts/mdc_challenge.db",
    collection_name: str = "mdc_training_data",
    dataset_collection_name: str = "dataset-aggregates-train",
    output_dir: str = "reports/feature_engineering",
    similarity_threshold_method: str = "degree_target",
    clustering_resolution: float = 1.0,
    k_neighbors: int = 5,
    k_similarity_neighbors: int = 30,
    visualization: bool = True
):
    """
    Complete feature engineering pipeline with optimizations:
    1. Phase 6: Dataset construction from retrieval results
    2. Phase 7: Leiden clustering
    3. Phase 8: PCA & UMAP dimensionality reduction with reproducibility
    """
    # ... full function as in original guide ...
```

### 11.2 Docker Integration
**Location**: Update `docker-compose.yml`

```yaml
services:
  preprocess:
    # ... existing config ...
    command: >
      sh -c "
        python scripts/run_chunking_pipeline.py &&
        python src/construct_queries.py &&
        python scripts/run_feature_engineering_pipeline.py
      "
```
---