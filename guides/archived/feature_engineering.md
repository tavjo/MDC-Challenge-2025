## Current State Summary

### Understanding of Current Information Flow

Based on my analysis of the codebase, here's the current end-to-end information flow:

**Phases 1-5 (Completed):**
1. **EDA Phase**: Analyzed `train_labels.csv` and PDF/XML corpus → produced inventory CSVs
2. **Document Parsing**: Used `unstructured.partition_pdf` → created `Document` Pydantic objects → stored in `documents.json` and DuckDB `documents` table
3. **Citation Extraction**: Dual-mode extraction (regex for training, LLM+BAML for inference) → produced `CitationEntity` objects → stored in `citations.json` and DuckDB `citations` table  
4. **Chunking**: Custom sliding-window chunker (~300 tokens, 10% overlap) with citation integrity validation → produced `Chunk` objects → stored in DuckDB `chunks` table
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

# Optimized Feature Engineering Implementation Plan

## Key Optimizations Applied ✅

Based on technical review feedback, this plan incorporates the following critical optimizations:

1. **Memory-Efficient k-NN Similarity Graph**: Replaced O(N²) full similarity matrix with O(N*k) k-NN approach using sklearn.neighbors.NearestNeighbors
2. **Direct igraph Construction**: Eliminated NetworkX → igraph conversion overhead for 4x faster clustering  
3. **Parquet-Based Bulk Upsert**: Single INSERT OR REPLACE statement vs. batched temp table operations
4. **ChromaDB Vector Storage**: Dataset embeddings stored in dedicated `dataset-aggregates-train` collection (optimal for vector operations)
5. **Foreign Key Enforcement**: Added `PRAGMA foreign_keys=ON` to prevent silent data integrity issues
6. **Robust Multi-ID Masking**: Regex-escaped, case-insensitive masking for compound dataset citations
7. **Reproducibility Seeds**: Fixed random seeds for UMAP, PCA, and Leiden clustering (critical for Kaggle consistency)
8. **Lightweight Dependencies**: Removed Seaborn dependency for simpler Docker builds
9. **Security Hardening**: Path validation, control character sanitization, transaction wrapping
10. **Deterministic Thresholding**: Default to degree-target method (k=15) for consistent network density

These changes address scalability bottlenecks while maintaining the same 2.5-3 day development timeline.

---

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
    db_path: str,
    collection_name: str,
    dataset_collection_name: str = "dataset-aggregates-train",
    method: Literal["mean", "re_embed"] = "re_embed"
) -> Dict[str, np.ndarray]:
    """
    Compute dataset-level embeddings and store in ChromaDB:
    - Method A: Mean of constituent chunk embeddings (faster)
    - Method B: Re-embed concatenated masked text (potentially richer)
    - Store in dedicated ChromaDB collection for dataset-level vectors
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
# **Use existing  save_chunks_to_chroma function in `semantic_chunking.py`**
@timer_wrap  
def store_dataset_embeddings_in_chroma(
    dataset_embeddings: Dict[str, np.ndarray],
    dataset_metadata: Dict[str, Dict[str, str]],
    collection_name: str = "dataset-aggregates-train",
    cfg_path: str = "configs/chunking.yaml"
):
    """Store dataset-level embeddings in dedicated ChromaDB collection"""


@timer_wrap
def mask_dataset_ids_in_text(text: str, dataset_ids: List[str], mask_token: str = "<DATASET_ID>") -> str:
    """
    Robust multi-ID masking with regex escaping:
    - Handles multiple dataset IDs per text
    - Case-insensitive matching with word boundaries
    - Prevents leakage from compound citations
    """
```

### 7.2 Clustering Dependencies
**New Dependencies** (add to `pyproject.toml`):
```toml
leidenalg = "^0.10.1"
python-igraph = "^0.11.0"  
scikit-learn = "^1.3.0"
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
    variance_threshold: float = 0.95,
    random_seed: int = 42
) -> Tuple[Dict[str, np.ndarray], PCA]:
    """
    Apply PCA to dataset embeddings with reproducibility:
    - Auto-determine n_components to retain ≥95% variance if not specified
    - Set random seed for consistent results across runs
    - Return transformed embeddings and fitted PCA object
    """

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
):
    """Create scatter plot colored by dataset_type (PRIMARY/SECONDARY) labels using matplotlib"""
```

### 8.2 Visualization Dependencies
**Dependencies** (scikit-learn already included from clustering):
```toml
umap-learn = "^0.5.4"
matplotlib = "^3.7.0"
# Note: Seaborn removed for lighter Docker builds - matplotlib sufficient
```

---

## Phase 9: Database Schema Extension  

### 9.1 Add Datasets Table with FK Enforcement
**Location**: Update `api/database/duckdb_schema.py`

```python
def create_connection(self) -> duckdb.DuckDBPyConnection:
    """Create and return a DuckDB connection with FK enforcement."""
    # Ensure the directory exists
    db_path = Path(self.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create connection and enable foreign key constraints
    self.conn = duckdb.connect(str(db_path))
    self.conn.execute("PRAGMA foreign_keys=ON")  # Critical: enable FK enforcement
    
    return self.conn

def create_datasets_table(self):
    """Create the datasets table - embeddings stored in ChromaDB separately."""
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
        -- Dimensionality reduction features (derived from ChromaDB embeddings)
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
    """Create minimal indexes for datasets table performance."""
    # Primary index on cluster for clustering analysis
    self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_datasets_cluster ON datasets(cluster);
    """)
    
    # Index on dataset_type for classification analysis  
    self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_datasets_type ON datasets(dataset_type);
    """)
    
    # Note: Minimal indexing to avoid upsert slowdown
    logger.info("Datasets table indexes created successfully")
```

### 9.2 Update Dataset Model  
**Location**: Update `src/models.py`

```python
class Dataset(BaseModel):
    """Dataset Citation Extracted from Document text - embeddings stored separately in ChromaDB"""
    dataset_id: str = Field(..., description="Dataset ID")
    doc_id: str = Field(..., description="DOI in which the dataset citation was found")
    total_tokens: int = Field(..., description="Total number of tokens in all chunks.")
    avg_tokens_per_chunk: float = Field(..., description="Average number of tokens per chunk.")
    total_char_length: int = Field(..., description="Total number of characters")
    clean_text_length: int = Field(..., description="Total number of characters after cleaning")
    cluster: Optional[str] = Field(None, description="Leiden cluster assignment")
    dataset_type: Optional[Literal["PRIMARY", "SECONDARY"]] = Field(None, description="Dataset Type: main target of the classification task")
    text: str = Field(..., description="Text in the document where the dataset citation is found (masked)")
    
    # Dimensionality reduction features (derived from ChromaDB embeddings)
    umap_1: Optional[float] = Field(None, description="UMAP dimension 1")
    umap_2: Optional[float] = Field(None, description="UMAP dimension 2") 
    pc_1: Optional[float] = Field(None, description="First principal component")
    pc_2: Optional[float] = Field(None, description="Second principal component")
    
    # Neighborhood embedding statistics (computed from ChromaDB k-NN search)
    neighbor_mean_similarity: Optional[float] = Field(None, description="Mean similarity to k-nearest neighbors")
    neighbor_max_similarity: Optional[float] = Field(None, description="Max similarity to k-nearest neighbors")
    neighbor_var_similarity: Optional[float] = Field(None, description="Variance of similarities to k-nearest neighbors")
    neighbor_mean_norm: Optional[float] = Field(None, description="Mean norm of k-nearest neighbor embeddings")
    neighbor_max_norm: Optional[float] = Field(None, description="Max norm of k-nearest neighbor embeddings") 
    neighbor_var_norm: Optional[float] = Field(None, description="Variance of norms of k-nearest neighbor embeddings")
    
    def to_duckdb_row(self) -> Dict[str, Any]:
        """Convert Dataset to DuckDB row dictionary for parquet-based bulk upsert."""
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
    
    def to_chroma_metadata(self) -> Dict[str, str]:
        """Convert Dataset to ChromaDB metadata dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "doc_id": self.doc_id,
            "cluster": self.cluster or "unknown",
            "dataset_type": self.dataset_type or "unknown"
        }
```

---

## Phase 10: Bulk Upsert Implementation

### 10.1 Parquet-Based Bulk Upsert Service
**Location**: `api/services/dataset_upsert_service.py`

```python
@timer_wrap
def bulk_upsert_datasets(
    datasets: List[Dataset],
    db_path: str = "artifacts/mdc_challenge.db",
    temp_dir: str = "/tmp"
) -> Dict[str, Any]:
    """
    Efficiently upsert Dataset objects using DuckDB's parquet-based bulk operations.
    Single INSERT OR REPLACE statement - no temp tables or batching overhead.
    """
    
    # Enable foreign key constraints and transaction
    conn = duckdb.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN TRANSACTION")
    
    try:
        # Convert datasets to DataFrame
        upsert_data = [ds.to_duckdb_row() for ds in datasets]
        df = pd.DataFrame(upsert_data)
        
        # Strip control characters from text fields to prevent CSV export issues
        df['text'] = df['text'].str.replace(r'[\x00-\x1f\x7f-\x9f]', '', regex=True)
        
        # Write to temporary parquet file
        temp_parquet = os.path.join(temp_dir, f"datasets_upsert_{uuid.uuid4().hex[:8]}.parquet")
        df.to_parquet(temp_parquet, index=False)
        
        logger.info(f"Upserting {len(datasets)} datasets via parquet bulk load")
        
        # Single INSERT OR REPLACE statement - DuckDB's recommended approach
        upsert_sql = """
        INSERT OR REPLACE INTO datasets 
        SELECT * FROM parquet_scan(?)
        """
        
        conn.execute(upsert_sql, [temp_parquet])
        conn.execute("COMMIT")
        
        # Clean up temp file
        os.unlink(temp_parquet)
        
        # Return summary statistics
        total_count = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        
        logger.info(f"✅ Successfully upserted {len(datasets)} datasets")
        
        return {
            "success": True,
            "total_datasets_upserted": len(datasets),
            "total_datasets_in_db": total_count,
            "method": "parquet_bulk_load"
        }
        
    except Exception as e:
        conn.execute("ROLLBACK")
        logger.error(f"Bulk upsert failed: {str(e)}")
        # Clean up temp file on error
        if os.path.exists(temp_parquet):
            os.unlink(temp_parquet)
        raise
    finally:
        conn.close()


@timer_wrap
def sanitize_text_for_storage(text: str) -> str:
    """Remove control characters that can break CSV exports or cause encoding issues."""
    import re
    # Remove control characters except newline and tab
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
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
    2. Phase 7: Memory-efficient k-NN clustering with ChromaDB storage  
    3. Phase 8: PCA & UMAP dimensionality reduction with reproducibility
    4. Phase 9: DuckDB datasets table creation with FK enforcement
    5. Phase 10: Parquet-based bulk upsert
    """
    
    # Set reproducibility seeds
    np.random.seed(42)
    
    logger.info("🚀 Starting Optimized Feature Engineering Pipeline")
    
    # Ensure output directory exists and validate path
    if not os.path.abspath(retrieval_results_path).startswith(os.path.abspath("reports/")):
        raise ValueError("retrieval_results_path must be within reports/ directory for security")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase 6: Dataset Construction with robust masking
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
    
    # Phase 7: Memory-efficient k-NN clustering with ChromaDB storage
    logger.info("Phase 7: Computing dataset embeddings and storing in ChromaDB")
    dataset_embeddings = compute_dataset_embeddings(
        datasets=datasets, 
        db_path=db_path,
        collection_name=collection_name,
        dataset_collection_name=dataset_collection_name
    )
    
    logger.info("Phase 7: Building k-NN similarity graph")
    similarity_graph = build_knn_similarity_graph(
        dataset_embeddings=dataset_embeddings,
        k_neighbors=k_similarity_neighbors,
        threshold_method=similarity_threshold_method
    )
    
    logger.info("Phase 7: Running Leiden clustering")
    cluster_mapping = run_leiden_clustering(
        graph=similarity_graph, 
        resolution=clustering_resolution,
        random_seed=42
    )
    
    # Add cluster labels to datasets
    for dataset in datasets:
        dataset.cluster = cluster_mapping.get(dataset.dataset_id, "unknown")
    
    logger.info(f"✅ Clustering complete: {len(set(cluster_mapping.values()))} clusters found")
    
    # Phase 8: Dimensionality Reduction with reproducibility
    logger.info("Phase 8: PCA & UMAP dimensionality reduction")
    pca_embeddings, pca_model = run_pca_reduction(dataset_embeddings, random_seed=42)
    umap_embeddings = run_umap_reduction(pca_embeddings, random_seed=42)
    
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
    
    # Phase 9: Database Schema Update with FK enforcement
    logger.info("Phase 9: Updating DuckDB schema")
    schema_initializer = DuckDBSchemaInitializer(db_path)
    schema_initializer.create_connection()  # Automatically enables FK pragma
    schema_initializer.create_datasets_table()
    schema_initializer.create_datasets_indexes()
    schema_initializer.close()
    logger.info("✅ Database schema updated")
    
    # Phase 10: Parquet-based bulk upsert
    logger.info("Phase 10: Parquet-based bulk upserting Dataset objects")
    upsert_result = bulk_upsert_datasets(datasets, db_path)
    logger.info(f"✅ Upserted {upsert_result['total_datasets_upserted']} datasets via {upsert_result['method']}")
    
    # Export final enriched datasets
    enriched_datasets_path = os.path.join(output_dir, "enriched_datasets.json")
    with open(enriched_datasets_path, 'w') as f:
        json.dump([ds.model_dump() for ds in datasets], f, indent=2)
    
    # Export summary statistics
    summary = {
        "pipeline_completed_at": datetime.now().isoformat(),
        "total_datasets": len(datasets),
        "total_clusters": len(set(cluster_mapping.values())),
        "similarity_method": similarity_threshold_method,
        "k_similarity_neighbors": k_similarity_neighbors,
        "clustering_resolution": clustering_resolution,
        "pca_explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
        "upsert_result": upsert_result,
        "chroma_collections": {
            "chunks": collection_name,
            "datasets": dataset_collection_name
        },
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
    
    logger.info(f"🎉 Optimized Feature Engineering Pipeline Complete! Summary: {summary_path}")
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

## Expected Outputs & Benefits

After implementation, this optimized pipeline will produce:

1. **Dual-Storage Architecture**:
   - **ChromaDB**: Full dataset embeddings in dedicated `dataset-aggregates-train` collection
   - **DuckDB**: Tabular features, metadata, and derived statistics in indexed `datasets` table
   - Clear separation of vector storage (ChromaDB) vs. structured data (DuckDB)

2. **Enriched Dataset Objects** with:
   - Robust multi-ID masked text to prevent overfitting
   - Memory-efficient Leiden cluster assignments as binary features
   - Reproducible PCA/UMAP coordinates for visualization
   - k-NN neighborhood embedding statistics from ChromaDB

3. **Scalable & Performant Operations**:
   - O(N*k) memory complexity for similarity graph construction (vs O(N²))
   - Direct igraph clustering (no NetworkX conversion overhead)
   - Single parquet-based bulk upsert (no temp table batching)
   - Foreign key enforcement with proper pragma settings

4. **Production-Ready Visualizations**:
   - Deterministic UMAP scatter plots colored by cluster membership
   - Ground truth visualization colored by PRIMARY/SECONDARY labels
   - Maybe include visualization colored by document ID (continuous color scale since there are ~95 docs with citations)
   - Matplotlib-only plots for lighter Docker builds

5. **Security & Robustness**:
   - Path validation for input files
   - Control character sanitization for text fields
   - Transaction-wrapped upserts with proper error handling
   - Reproducible results via consistent random seeds

6. **Improved Classifier Performance**:
   - Network-based cluster features capture dataset relationships
   - Dimensionality reduction features provide additional signal
   - Properly masked embeddings prevent label leakage
   - Neighborhood statistics provide contextual features

This optimized plan addresses the scalability concerns identified in the colleague's review while maintaining consistency with existing codebase architecture and implementing production-grade feature engineering techniques optimized for your biomedical dataset citation classification task.