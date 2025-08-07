# MDC Challenge 2025 - Script Mapping Document (as of 2025-08-07)

## Overview
This document maps the key scripts involved in the MDC (Make Data Count) Challenge 2025 pipeline, which processes scientific documents to identify and classify dataset citations as Primary or Secondary.

## Core Pipeline

### 1. **Entry Points & Orchestration**
- **`docker-compose.yml`** - Container orchestration for microservices
- **`Makefile`** - Docker management automation
- **`Dockerfile.api`** - Chunking & embedding API container
- **`dockerfile`** - Document parsing API container

### 2. **Document Processing Pipeline**
- **`src/get_document_objects.py`** - **Main document parsing orchestration** (actual file that parses documents)
- **`api/parse_doc_api.py`** - FastAPI service for document parsing
- **`api/services/document_parsing_service.py`** - Core document parsing logic
- **`src/extract_pdf_text_unstructured.py`** - PDF text extraction using Unstructured

### 3. **Citation Extraction**
- **`src/get_citation_entities.py`** - Citation entity extraction (regex + LLM-driven)
- **`src/baml_src/`** - BAML configuration for LLM-based citation extraction

### 4. **Chunking & Embedding Pipeline**
- **`api/chunk_and_embed_api.py`** - FastAPI service for chunking and embedding
- **`api/services/chunking_and_embedding_services.py`** - Core chunking logic
- **`api/services/embeddings_services.py`** - Embedding generation services
- **`src/semantic_chunking.py`** - Semantic chunking implementation
- **`src/run_semantic_chunking.py`** - Chunking pipeline orchestration

### 5. **Retrieval & Dataset Construction**
- **`src/construct_queries.py`** - Query construction for retrieval
- **`src/construct_datasets.py`** - Dataset object construction from retrieval results
- **`api/services/retriever_services.py`** - Retrieval services
- **`api/services/dataset_construction_service.py`** - Dataset construction services

### 6. **Feature Engineering & Analysis**
- **`src/clustering.py`** - Leiden clustering implementation
- **`src/run_clustering.py`** - Clustering pipeline orchestration
- **`src/umap.py`** - UMAP implementation (preprocessing for dimensionality reduction)
- **`src/dimensionality_reduction.py`** - PCA/UMAP dimensionality reduction
- **`src/run_dimensionality_reduction.py`** - Dimensionality reduction orchestration
- **`src/run_neighborhood_stats.py`** - Neighborhood statistics computation
- **`api/services/neighborhood_stats.py`** - Neighborhood stats services

### 7. **Training & Model Building**
- **`src/training.py`** - Random Forest training implementation
- **`run_training.sh`** - Full training script with parameter handling
- **`quick_train.sh`** - Simple training script with defaults

### 8. **Database & Utilities**
- **`api/utils/duckdb_utils.py`** - DuckDB operations and utilities
- **`api/database/duckdb_schema.py`** - Database schema definitions
- **`src/helpers.py`** - Common utility functions
- **`src/models.py`** - Pydantic data models

## Supporting & Analysis

### 9. **Essential Analysis**
- **`notebooks/training_input.ipynb`** - Training data analysis and visualization (only essential notebook)

### 10. **Essential Configuration**
- **`configs/chunking.yaml`** - Chunking and embedding configuration
- **`pyproject.toml`** - Project dependencies and configuration

### 11. **Supporting Scripts (Transitional)**
- **`scripts/import_dataset_labels.py`** - Dataset label import (used once, can be retired)

## Complete Pipeline Flow

### **Phase-by-Phase Flow:**
```
1. Document Parsing: PDF → Document Objects
   PDF files → src/get_document_objects.py → api/parse_doc_api.py → api/services/document_parsing_service.py

2. Citation Extraction: Documents → Citation Entities
   Document Objects → src/get_citation_entities.py

3. Chunking: Documents + Citations → Chunks + Embeddings
   Documents + Citations → src/run_semantic_chunking.py → api/chunk_and_embed_api.py

4. Retrieval & Dataset Construction: Chunks → Dataset Objects
   Chunks → src/construct_queries.py → src/construct_datasets.py

5. Feature Engineering: Dataset Objects → Features
   Dataset Objects → src/run_clustering.py (feature-based clustering) → src/run_neighborhood_stats.py → src/umap.py (global feature UMAP) → src/run_dimensionality_reduction.py (per cluster feature PCA)

6. Training: Features → Model
   Features → src/training.py (via run_training.sh or quick_train.sh)
```

### **API Dependencies:**
- **Document Parsing API** (port 3000): Handles PDF processing
- **Chunking & Embedding API** (port 8000): Handles chunking, embedding, retrieval, and feature engineering
- **Database**: DuckDB for structured data storage
- **Vector Store**: ChromaDB for embeddings

### **Complete Data Flow:**
```
PDFs → Document Objects → Citation Entities → Chunks → Embeddings → 
Retrieval Results → Dataset Objects → Clustered Features → UMAP Features → 
Reduced Dimensions → Neighborhood Stats → Training Data → Random Forest Model
```

### **Detailed Feature Engineering Flow:**
```
Dataset Objects 
    ↓
src/run_clustering.py (Leiden clustering)
    ↓
src/umap.py (UMAP preprocessing)
    ↓
src/run_dimensionality_reduction.py (PCA/UMAP dimensionality reduction)
    ↓
src/run_neighborhood_stats.py (neighborhood statistics)
    ↓
Training Features
```

This represents the complete, essential script mapping for the MDC Challenge 2025 pipeline with the correct flow and dependencies.