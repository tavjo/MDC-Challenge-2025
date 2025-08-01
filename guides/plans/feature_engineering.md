## Current State Summary

### Understanding of Current Information Flow

Based on my analysis of the codebase, here's the current end-to-end information flow:

**Phases 1-5 (Completed):**
1. **EDA Phase**: Analyzed `train_labels.csv` and PDF/XML corpus → produced inventory CSVs
2. **Document Parsing**: Used `unstructured.partition_pdf` → created `Document` Pydantic objects → stored in `documents.json` and DuckDB `documents` table
3. **Citation Extraction**: Dual-mode extraction (regex for training, LLM+BAML for inference) → produced `CitationEntity` objects → stored in `citations.json` and DuckDB `citations` table  
4. **Chunking**: Custom sliding-window chunker (~300 tokens, 30% overlap) with citation integrity validation → produced `Chunk` objects → stored in DuckDB `chunks` table
5. **Embeddings**: SentenceTransformers embedding of chunks → stored in ChromaDB (`local_chroma/` folder)

**Phase 6 (Partially Implemented):**
- ✅ **Retrieval Infrastructure**: Complete ChromaDB + DuckDB hybrid retrieval system in `api/services/retriever_services.py`
- ✅ **Query Construction**: Script in `src/construct_queries.py` builds dataset-specific queries and calls batch retrieval API
- ✅ **Retrieval Results**: Batch retrieval produces `reports/retrieval/retrieval_results.json` with chunk IDs per dataset
- ❌ **Missing**: Dataset object construction from retrieved chunks, text masking, re-embedding, and aggregation statistics

**Phases 7-8 (Not Implemented):**
- No clustering infrastructure exists
- No dimensionality reduction implementation  
- No datasets table in DuckDB schema
- No bulk upsert mechanism for Dataset objects

---

# Comprehensive Feature Engineering Implementation Plan

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
def mask_dataset_ids_in_text(text: str, dataset_id: str, mask_token: str = "<DATASET_ID>") -> str:
    """Replace all instances of dataset_id with mask_token to prevent leakage"""

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
```

### 6.2 Integration Script
**Location**: `scripts/run_dataset_construction.py`

Orchestrates the complete Phase 6:
1. Load retrieval results JSON
2. Call dataset construction service
3. Export `datasets.json` and `dataset_embedding_stats.csv`
4. Validate Dataset objects against expected schema

---

## Phase 7: Network-Based Clustering

### 7.1 Create Clustering Module
**Location**: `src/clustering.py`

**Core Functions:**
```python
@timer_wrap
def compute_dataset_embeddings(
    datasets: List[Dataset],
    chroma_client,
    collection_name: str,
    method: Literal["mean", "re_embed"] = "re_embed"
) -> Dict[str, np.ndarray]:
    """
    Compute dataset-level embeddings using either:
    - Method A: Mean of constituent chunk embeddings (faster)
    - Method B: Re-embed concatenated masked text (potentially richer)
    """

@timer_wrap  
def build_similarity_matrix(dataset_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    """Create N×N cosine similarity matrix from dataset embeddings"""

@timer_wrap
def determine_similarity_threshold(similarity_matrix: np.ndarray, method: str = "percentile_90") -> float:
    """
    Dynamically determine similarity threshold based on network density:
    - percentile_90: Use 90th percentile of similarities
    - elbow_method: Find largest gap in sorted similarities  
    - density_target: Target specific edge density (e.g., 10% of possible edges)
    """

@timer_wrap
def build_similarity_network(similarity_matrix: np.ndarray, threshold: float) -> networkx.Graph:
    """Convert similarity matrix to NetworkX graph using threshold for edge existence"""

@timer_wrap
def run_leiden_clustering(graph: networkx.Graph, resolution: float = 1.0) -> Dict[str, str]:
    """
    Apply Leiden clustering using leidenalg library:
    1. Convert NetworkX → igraph  
    2. Apply leidenalg.find_partition()
    3. Return {dataset_id: cluster_label} mapping
    """

@timer_wrap
def encode_cluster_features(datasets: List[Dataset], cluster_mapping: Dict[str, str]) -> List[Dataset]:
    """Add cluster field to Dataset objects and return binary cluster feature encodings"""
```

### 7.2 Clustering Dependencies
**New Dependencies** (add to `pyproject.toml`):
```toml
leidenalg = "^0.10.1"
python-igraph = "^0.11.0"  
networkx = "^3.1"
```

---

## Phase 8: Dimensionality Reduction & Visualization

### 8.1 Dimensionality Reduction Module  
**Location**: `src/dimensionality_reduction.py`

**Core Functions:**
```python
@timer_wrap
def run_pca_reduction(
    dataset_embeddings: Dict[str, np.ndarray], 
    n_components: int = None,
    variance_threshold: float = 0.95
) -> Tuple[Dict[str, np.ndarray], PCA]:
    """
    Apply PCA to dataset embeddings:
    - Auto-determine n_components to retain ≥95% variance if not specified
    - Return transformed embeddings and fitted PCA object
    """

@timer_wrap  
def run_umap_reduction(
    embeddings: Dict[str, np.ndarray],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2
) -> Dict[str, np.ndarray]:
    """Apply UMAP dimensionality reduction for 2D visualization"""

@timer_wrap
def create_cluster_visualization(
    umap_embeddings: Dict[str, np.ndarray],
    cluster_labels: Dict[str, str],
    output_path: str = "reports/cluster_visualization.png"
):
    """Create scatter plot colored by Leiden cluster labels"""

@timer_wrap  
def create_ground_truth_visualization(
    umap_embeddings: Dict[str, np.ndarray], 
    datasets: List[Dataset],
    output_path: str = "reports/ground_truth_visualization.png"
):
    """Create scatter plot colored by dataset_type (PRIMARY/SECONDARY) labels"""
```

### 8.2 Visualization Dependencies
**New Dependencies**:
```toml
umap-learn = "^0.5.4"
scikit-learn = "^1.3.0"  
matplotlib = "^3.7.0"
seaborn = "^0.12.0"
```

---

## Phase 9: Database Schema Extension  

### 9.1 Add Datasets Table
**Location**: Update `api/database/duckdb_schema.py`

```python
def create_datasets_table(self):
    """Create the datasets table matching the Dataset model."""
    logger.info("Creating datasets table...")
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id VARCHAR NOT NULL,
        doc_id VARCHAR NOT NULL,
        total_tokens INTEGER NOT NULL,
        avg_tokens_per_chunk REAL NOT NULL,
        total_char_length INTEGER NOT NULL, 
        clean_text_length INTEGER NOT NULL,
        cluster VARCHAR,
        dataset_type VARCHAR CHECK (dataset_type IN ('PRIMARY', 'SECONDARY')),
        text TEXT NOT NULL,
        -- Dimensionality reduction features
        umap_1 REAL,
        umap_2 REAL, 
        pc_1 REAL,
        pc_2 REAL,
        -- Neighborhood embedding statistics  
        neighbor_mean_similarity REAL,
        neighbor_max_similarity REAL,
        neighbor_var_similarity REAL,
        neighbor_mean_norm REAL,
        neighbor_max_norm REAL,
        neighbor_var_norm REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (dataset_id, doc_id),
        FOREIGN KEY (doc_id) REFERENCES documents(doi)
    );
    """
    
    self.conn.execute(create_table_sql)
    logger.info("Datasets table created successfully")

def create_datasets_indexes(self):
    """Create indexes for datasets table performance."""
    # Index on cluster for clustering analysis
    self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_datasets_cluster ON datasets(cluster);
    """)
    
    # Index on dataset_type for classification analysis  
    self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_datasets_type ON datasets(dataset_type);
    """)
    
    # Composite index for dataset_id lookups
    self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_datasets_id ON datasets(dataset_id);
    """)
```

### 9.2 Update Dataset Model
**Location**: Update `src/models.py`

```python
class Dataset(BaseModel):
    """Dataset Citation Extracted from Document text"""
    dataset_id: str = Field(..., description="Dataset ID")
    doc_id: str = Field(..., description="DOI in which the dataset citation was found")
    total_tokens: int = Field(..., description="Total number of tokens in all chunks.")
    avg_tokens_per_chunk: float = Field(..., description="Average number of tokens per chunk.")
    total_char_length: int = Field(..., description="Total number of characters")
    clean_text_length: int = Field(..., description="Total number of characters after cleaning")
    cluster: Optional[str] = Field(None, description="Leiden cluster assignment")
    dataset_type: Optional[Literal["PRIMARY", "SECONDARY"]] = Field(None, description="Dataset Type: main target of the classification task")
    text: str = Field(..., description="Text in the document where the dataset citation is found (masked)")
    
    # Dimensionality reduction features
    umap_1: Optional[float] = Field(None, description="UMAP dimension 1")
    umap_2: Optional[float] = Field(None, description="UMAP dimension 2") 
    pc_1: Optional[float] = Field(None, description="First principal component")
    pc_2: Optional[float] = Field(None, description="Second principal component")
    
    # Neighborhood embedding statistics
    neighbor_mean_similarity: Optional[float] = Field(None, description="Mean similarity to k-nearest neighbors")
    neighbor_max_similarity: Optional[float] = Field(None, description="Max similarity to k-nearest neighbors")
    neighbor_var_similarity: Optional[float] = Field(None, description="Variance of similarities to k-nearest neighbors")
    neighbor_mean_norm: Optional[float] = Field(None, description="Mean norm of k-nearest neighbor embeddings")
    neighbor_max_norm: Optional[float] = Field(None, description="Max norm of k-nearest neighbor embeddings") 
    neighbor_var_norm: Optional[float] = Field(None, description="Variance of norms of k-nearest neighbor embeddings")
    
    def to_duckdb_row(self) -> Dict[str, Any]:
        """Convert Dataset to DuckDB row dictionary for bulk upsert."""
        return {
            "dataset_id": self.dataset_id,
            "doc_id": self.doc_id,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_chunk": self.avg_tokens_per_chunk,
            "total_char_length": self.total_char_length,
            "clean_text_length": self.clean_text_length,
            "cluster": self.cluster,
            "dataset_type": self.dataset_type,
            "text": self.text,
            "umap_1": self.umap_1,
            "umap_2": self.umap_2, 
            "pc_1": self.pc_1,
            "pc_2": self.pc_2,
            "neighbor_mean_similarity": self.neighbor_mean_similarity,
            "neighbor_max_similarity": self.neighbor_max_similarity,
            "neighbor_var_similarity": self.neighbor_var_similarity,
            "neighbor_mean_norm": self.neighbor_mean_norm,
            "neighbor_max_norm": self.neighbor_max_norm,
            "neighbor_var_norm": self.neighbor_var_norm,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    @classmethod
    def from_duckdb_row(cls, row: Dict[str, Any]) -> "Dataset":
        """Rehydrate Dataset from DuckDB row."""
        return cls(**{k: v for k, v in row.items() if k not in ["created_at", "updated_at"]})
```

---

## Phase 10: Bulk Upsert Implementation

### 10.1 Bulk Upsert Service
**Location**: `api/services/dataset_upsert_service.py`

```python
@timer_wrap
def bulk_upsert_datasets(
    datasets: List[Dataset],
    db_path: str = "artifacts/mdc_challenge.db",
    batch_size: int = 1000
) -> Dict[str, Any]:
    """
    Efficiently upsert Dataset objects using DuckDB's MERGE INTO statement.
    Uses parameterized batching to avoid memory issues with large datasets.
    """
    
    conn = duckdb.connect(db_path)
    
    # Prepare batch data
    upsert_data = [ds.to_duckdb_row() for ds in datasets]
    
    total_batches = (len(upsert_data) + batch_size - 1) // batch_size
    logger.info(f"Upserting {len(datasets)} datasets in {total_batches} batches")
    
    for i in range(0, len(upsert_data), batch_size):
        batch = upsert_data[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} records)")
        
        # Create temporary table for this batch
        conn.execute("DROP TABLE IF EXISTS temp_datasets_batch")
        conn.execute("""
            CREATE TEMPORARY TABLE temp_datasets_batch AS 
            SELECT * FROM datasets WHERE 1=0
        """)
        
        # Insert batch data into temporary table
        conn.executemany("""
            INSERT INTO temp_datasets_batch VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, [tuple(row.values()) for row in batch])
        
        # Execute MERGE statement for this batch
        merge_sql = """
        MERGE INTO datasets AS target
        USING temp_datasets_batch AS source
        ON target.dataset_id = source.dataset_id AND target.doc_id = source.doc_id
        WHEN MATCHED THEN UPDATE SET
            total_tokens = source.total_tokens,
            avg_tokens_per_chunk = source.avg_tokens_per_chunk,
            total_char_length = source.total_char_length,
            clean_text_length = source.clean_text_length,
            cluster = source.cluster,
            dataset_type = source.dataset_type,
            text = source.text,
            umap_1 = source.umap_1,
            umap_2 = source.umap_2,
            pc_1 = source.pc_1,
            pc_2 = source.pc_2,
            neighbor_mean_similarity = source.neighbor_mean_similarity,
            neighbor_max_similarity = source.neighbor_max_similarity,
            neighbor_var_similarity = source.neighbor_var_similarity,
            neighbor_mean_norm = source.neighbor_mean_norm,
            neighbor_max_norm = source.neighbor_max_norm,
            neighbor_var_norm = source.neighbor_var_norm,
            updated_at = CURRENT_TIMESTAMP
        WHEN NOT MATCHED THEN INSERT (
            dataset_id, doc_id, total_tokens, avg_tokens_per_chunk,
            total_char_length, clean_text_length, cluster, dataset_type, text,
            umap_1, umap_2, pc_1, pc_2,
            neighbor_mean_similarity, neighbor_max_similarity, neighbor_var_similarity,
            neighbor_mean_norm, neighbor_max_norm, neighbor_var_norm,
            created_at, updated_at
        ) VALUES (
            source.dataset_id, source.doc_id, source.total_tokens, source.avg_tokens_per_chunk,
            source.total_char_length, source.clean_text_length, source.cluster, source.dataset_type, source.text,
            source.umap_1, source.umap_2, source.pc_1, source.pc_2,
            source.neighbor_mean_similarity, source.neighbor_max_similarity, source.neighbor_var_similarity,
            source.neighbor_mean_norm, source.neighbor_max_norm, source.neighbor_var_norm,
            source.created_at, source.updated_at
        )
        """
        
        conn.execute(merge_sql)
        logger.info(f"Batch {batch_num} upserted successfully")
    
    # Return summary statistics
    total_count = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
    conn.close()
    
    return {
        "success": True,
        "total_datasets_upserted": len(datasets),
        "total_datasets_in_db": total_count,
        "batches_processed": total_batches
    }
```

---

## Phase 11: Orchestration & Integration

### 11.1 Master Pipeline Script
**Location**: `scripts/run_feature_engineering_pipeline.py`

```python
@timer_wrap
def main(
    retrieval_results_path: str = "reports/retrieval/retrieval_results.json",
    db_path: str = "artifacts/mdc_challenge.db",
    collection_name: str = "mdc_training_data",
    output_dir: str = "reports/feature_engineering",
    similarity_threshold_method: str = "percentile_90",
    clustering_resolution: float = 1.0,
    k_neighbors: int = 5,
    visualization: bool = True
):
    """
    Complete feature engineering pipeline:
    1. Phase 6: Dataset construction from retrieval results
    2. Phase 7: Network-based Leiden clustering  
    3. Phase 8: PCA & UMAP dimensionality reduction + visualizations
    4. Phase 9: DuckDB datasets table creation
    5. Phase 10: Bulk upsert of enriched Dataset objects
    """
    
    logger.info("🚀 Starting Feature Engineering Pipeline")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase 6: Dataset Construction
    logger.info("Phase 6: Constructing Dataset objects from retrieval results")
    datasets = construct_datasets_from_retrieval_results(
        retrieval_results_path=retrieval_results_path,
        db_path=db_path,
        collection_name=collection_name,
        k_neighbors=k_neighbors
    )
    logger.info(f"✅ Constructed {len(datasets)} Dataset objects")
    
    # Save intermediate results
    datasets_json_path = os.path.join(output_dir, "datasets.json")
    with open(datasets_json_path, 'w') as f:
        json.dump([ds.model_dump() for ds in datasets], f, indent=2)
    
    # Phase 7: Clustering
    logger.info("Phase 7: Network-based Leiden clustering")
    dataset_embeddings = compute_dataset_embeddings(datasets, chroma_client, collection_name)
    similarity_matrix = build_similarity_matrix(dataset_embeddings)
    threshold = determine_similarity_threshold(similarity_matrix, similarity_threshold_method)
    network = build_similarity_network(similarity_matrix, threshold)
    cluster_mapping = run_leiden_clustering(network, clustering_resolution)
    
    # Add cluster labels to datasets
    for dataset in datasets:
        dataset.cluster = cluster_mapping.get(dataset.dataset_id, "unknown")
    
    logger.info(f"✅ Clustering complete: {len(set(cluster_mapping.values()))} clusters found")
    
    # Phase 8: Dimensionality Reduction
    logger.info("Phase 8: PCA & UMAP dimensionality reduction")
    pca_embeddings, pca_model = run_pca_reduction(dataset_embeddings)
    umap_embeddings = run_umap_reduction(pca_embeddings)
    
    # Add DR features to datasets
    for dataset in datasets:
        if dataset.dataset_id in umap_embeddings:
            dataset.umap_1, dataset.umap_2 = umap_embeddings[dataset.dataset_id]
        if dataset.dataset_id in pca_embeddings:
            dataset.pc_1, dataset.pc_2 = pca_embeddings[dataset.dataset_id][:2]
    
    logger.info("✅ Dimensionality reduction complete")
    
    # Create visualizations if requested
    if visualization:
        logger.info("Creating visualizations")
        create_cluster_visualization(umap_embeddings, cluster_mapping, 
                                   os.path.join(output_dir, "cluster_visualization.png"))
        create_ground_truth_visualization(umap_embeddings, datasets,
                                        os.path.join(output_dir, "ground_truth_visualization.png"))
        logger.info("✅ Visualizations saved")
    
    # Phase 9: Database Schema Update
    logger.info("Phase 9: Updating DuckDB schema")
    schema_initializer = DuckDBSchemaInitializer(db_path)
    schema_initializer.create_connection()
    schema_initializer.create_datasets_table()
    schema_initializer.create_datasets_indexes()
    schema_initializer.close()
    logger.info("✅ Database schema updated")
    
    # Phase 10: Bulk Upsert
    logger.info("Phase 10: Bulk upserting Dataset objects")
    upsert_result = bulk_upsert_datasets(datasets, db_path)
    logger.info(f"✅ Upserted {upsert_result['total_datasets_upserted']} datasets")
    
    # Export final enriched datasets
    enriched_datasets_path = os.path.join(output_dir, "enriched_datasets.json")
    with open(enriched_datasets_path, 'w') as f:
        json.dump([ds.model_dump() for ds in datasets], f, indent=2)
    
    # Export summary statistics
    summary = {
        "pipeline_completed_at": datetime.now().isoformat(),
        "total_datasets": len(datasets),
        "total_clusters": len(set(cluster_mapping.values())),
        "similarity_threshold": float(threshold),
        "pca_explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
        "upsert_result": upsert_result,
        "output_files": {
            "datasets": datasets_json_path,
            "enriched_datasets": enriched_datasets_path,
            "cluster_visualization": os.path.join(output_dir, "cluster_visualization.png"),
            "ground_truth_visualization": os.path.join(output_dir, "ground_truth_visualization.png")
        }
    }
    
    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"🎉 Feature Engineering Pipeline Complete! Summary: {summary_path}")
    return summary
```

### 11.2 Docker Integration
**Location**: Update `docker-compose.yml`

Add the feature engineering step to the preprocessing service:

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

## Phase 12: Testing & Validation

### 12.1 Unit Tests
**Location**: `tests/test_feature_engineering.py`

```python
class TestFeatureEngineering:
    def test_dataset_construction(self):
        """Test Dataset object construction from retrieval results"""
        
    def test_similarity_matrix_construction(self):
        """Test similarity matrix creation and properties"""
        
    def test_leiden_clustering(self):
        """Test Leiden clustering with known graph structure"""
        
    def test_dimensionality_reduction(self):
        """Test PCA and UMAP reduction preserve relative distances"""
        
    def test_bulk_upsert(self):
        """Test bulk upsert creates and updates records correctly"""
        
    def test_dataset_masking(self):
        """Test dataset ID masking preserves context while removing identifiers"""
```

### 12.2 Integration Tests
**Location**: `tests/test_pipeline_integration.py`

```python  
class TestPipelineIntegration:
    def test_end_to_end_feature_engineering(self):
        """Test complete pipeline from retrieval results to enriched datasets"""
        
    def test_docker_pipeline_execution(self):
        """Test pipeline execution within Docker environment"""
```

---

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

clustering:
  similarity_threshold_method: "percentile_90"  # or "elbow_method", "density_target"
  leiden_resolution: 1.0
  min_cluster_size: 2

dimensionality_reduction:
  pca:
    variance_threshold: 0.95
  umap:
    n_neighbors: 15
    min_dist: 0.1
    n_components: 2

database:
  batch_size: 1000
  enable_indexes: true

output:
  export_visualizations: true
  export_intermediate_files: true
```

---

## Expected Outputs & Benefits

After implementation, this pipeline will produce:

1. **Enriched Dataset Objects** with:
   - Masked text to prevent overfitting
   - Leiden cluster assignments as binary features  
   - PCA/UMAP coordinates for visualization
   - Neighborhood embedding statistics
   
2. **Rich Visualizations**:
   - UMAP scatter plots colored by cluster membership
   - UMAP scatter plots colored by ground truth labels (PRIMARY/SECONDARY)
   - Cluster quality assessment plots

3. **Production-Ready Database**:
   - Indexed `datasets` table with all engineered features
   - Bulk upsert capability for efficient updates
   - Full integration with existing DuckDB infrastructure

4. **Improved Classifier Performance**:
   - Network-based cluster features capture dataset relationships
   - Dimensionality reduction features provide additional signal
   - Masked embeddings prevent label leakage

This comprehensive plan maintains consistency with existing codebase architecture while implementing state-of-the-art feature engineering techniques optimized for your biomedical dataset citation classification task.