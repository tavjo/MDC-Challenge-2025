## Checklist

- [ ] Create or update `tests/test_feature_engineering.py` with unit tests for dataset construction, similarity matrix creation, Leiden clustering, dimensionality reduction, bulk upsert, and dataset masking.
- [ ] Create or update `tests/test_pipeline_integration.py` with integration tests for end-to-end feature engineering and Docker pipeline execution.

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