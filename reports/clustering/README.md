# Network-Based Clustering Implementation

This directory contains clustering analysis reports and results from the Phase 7 network-based clustering pipeline.

## Overview

The clustering pipeline implements k-NN similarity graph construction and Leiden clustering for dataset embeddings stored in ChromaDB.

## Core Components

### 1. Clustering Module (`src/clustering.py`)
- `compute_neighborhood_embedding_stats()`: Computes neighborhood statistics for dataset embeddings
- `build_knn_similarity_graph()`: Memory-efficient k-NN similarity graph construction
- `determine_similarity_threshold()`: Dynamic threshold determination using multiple methods
- `run_leiden_clustering()`: Leiden clustering with safeguards and reproducibility
- `update_datasets_with_clusters()`: Updates DuckDB with cluster assignments
- `export_clustering_report()`: Exports comprehensive JSON reports
- `run_full_clustering_pipeline()`: Orchestrates the complete clustering workflow

### 2. Pipeline Script (`scripts/run_clustering_pipeline.py`)
Command-line interface for running the clustering pipeline with configurable parameters.

Usage:
```bash
python scripts/run_clustering_pipeline.py --help
```

## Key Features

1. **Memory-Efficient Graph Construction**: Uses sklearn.neighbors.NearestNeighbors for O(N*k) memory complexity
2. **Dynamic Similarity Thresholding**: Multiple methods including degree-target, percentile, and elbow methods
3. **Reproducible Clustering**: Fixed random seeds for consistent results
4. **Database Integration**: Direct updates to DuckDB dataset records
5. **Comprehensive Reporting**: Detailed JSON reports with clustering statistics

## Dependencies

- `leidenalg>=0.10.1`: Leiden clustering algorithm
- `python-igraph>=0.11.0`: Graph data structures and algorithms
- `scikit-learn>=1.3.0`: k-NN graph construction
- `chromadb`: Vector database for embeddings
- `numpy`, `pandas`: Data manipulation

## Database Schema

The clustering results are stored in the `datasets` table with the `cluster` field:

```sql
CREATE TABLE datasets (
    dataset_id VARCHAR PRIMARY KEY,
    -- ... other fields ...
    cluster VARCHAR,
    -- ... other fields ...
);
```

## Report Structure

Clustering reports include:
- Clustering summary (total datasets, clusters, size distribution)
- Graph construction statistics (vertices, edges, density)
- Neighborhood statistics (individual and aggregated)
- Complete cluster assignments

## Parameters

Key configurable parameters:
- `k_neighbors`: Number of neighbors for graph construction (default: 30)
- `similarity_threshold`: Edge creation threshold (auto-determined if not provided)
- `resolution`: Leiden clustering resolution (default: 1.0)
- `min_cluster_size`: Minimum valid cluster size (default: 2)
- `collection_name`: ChromaDB collection name (default: "dataset-aggregates-train")