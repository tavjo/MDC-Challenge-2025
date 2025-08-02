## Checklist

- [ ] Create `docs/current/feature_engineering_guide.md` with usage examples.
- [ ] Create or update `configs/feature_engineering.yaml` with configuration settings for dataset construction, clustering, dimensionality reduction, storage, security, and output.

## Phase 13: Documentation & Maintenance

### 13.1 Update Documentation
- **Create** `docs/current/feature_engineering_guide.md` with usage examples

### 13.2 Configuration Management
**Location**: `configs/feature_engineering.yaml`

```yaml
dataset_construction:
  mask_token: "<DATASET_ID>"
  k_neighbors: 5
  re_embed_masked_text: true
  robust_multi_id_masking: true

clustering:
  similarity_threshold_method: "degree_target"  # deterministic default
  target_degree: 15  # for degree_target method
  k_similarity_neighbors: 30  # k-NN graph construction
  leiden_resolution: 1.0
  min_cluster_size: 2

dimensionality_reduction:
  pca:
    variance_threshold: 0.95
    random_seed: 42
  umap:
    n_neighbors: 15
    min_dist: 0.1
    n_components: 2
    random_seed: 42

storage:
  # ChromaDB collections
  chunk_collection: "mdc_training_data"
  dataset_collection: "dataset-aggregates-train"
  
  # DuckDB settings
  enable_foreign_keys: true
  enable_minimal_indexes: true
  parquet_bulk_upsert: true

security:
  validate_paths: true
  strip_control_characters: true
  sanitize_text_fields: true

output:
  export_visualizations: true
  export_intermediate_files: true
  temp_directory: "/tmp"
```
---