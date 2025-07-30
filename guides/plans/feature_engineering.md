## Phase 1: Build the Dataset Similarity Graph & Leiden Clustering

1. **Create a new module**  
   File: `src/feature_engineering.py`  
   This will house all of the functions below.

2. **Load Retrieval Results → map dataset IDs → chunk IDs**  
   • Read your existing JSON at `reports/retrieval/retrieval_results.json` (written by `src/construct_queries.py`)  
   • Produce a `Dict[dataset_id, List[chunk_id]]`.

3. **Fetch & Aggregate Chunk Texts**  
   • Check whether `reports/retrieval/retrieval_results.json` exists; if so, load it and extract the mapping of `dataset_id` → retrieved `chunk_ids`.  
   • Initialize your DuckDB helper:
     ```python
     from api.utils.duckdb_utils import get_duckdb_helper
     db_helper = get_duckdb_helper(db_path)
     ```
   • For each `dataset_id`:
     1. **Find the target chunk** (the one containing the dataset citation) and its neighbor IDs:
        ```python
        all_chunks = db_helper.get_chunks_by_document_id(doc_id)
        target = next(
            c for c in all_chunks
            if any(ent.data_citation == dataset_id for ent in c.chunk_metadata.citation_entities)
        )
        neighbor_ids = [cid for cid in (target.chunk_metadata.previous_chunk_id,
                                        target.chunk_metadata.next_chunk_id) if cid]
        ```
     2. **Retrieve all text chunks**:
        ```python
        retrieved_ids = retrieval_map[dataset_id]
        chunks = db_helper.get_chunks_by_chunk_ids([target.chunk_id] + neighbor_ids + retrieved_ids)
        ```
     3. **Concatenate** `chunk.text` from that list into a single `agg_text`.
4. **Compute “Dataset” Embeddings**
   • First, update `embed_chunk` in `api/services/embeddings_services.py` to mirror your simplified logic in `retriever_services.py` (i.e. call `_embed_text` under the hood).  
   • Then call:
     ```python
     from api.services.embeddings_services import embed_chunk
     result = embed_chunk(agg_text, collection_name, cfg_path, local_model=True)
     embedding = result.embeddings
     ```
   • Store a mapping `dataset_id → embedding`.
5. **Build Cosine-Similarity Matrix & Threshold**  
   • Stack your vectors into an array `E`; compute  
     ```python
     from sklearn.metrics.pairwise import cosine_similarity
     sim = cosine_similarity(E)
     ```  
   • Extract the off-diagonal scores, pick a dynamic cutoff (e.g. 90th percentile via `np.percentile`). Tune if necessary.

6. **Construct Network & Run Leiden**  
   • Using `networkx` or `igraph` + `leidenalg`:  
     ```python
     G = nx.Graph()
     G.add_nodes_from(dataset_ids)
     for i,j in combinations: 
         if sim[i,j] >= threshold: G.add_edge(id_i, id_j)
     ```  
   • Convert to an igraph object (or use the igraph API directly) and run Leiden  
     ```python
     import leidenalg as la
     import igraph as ig
     partition = la.find_partition(ig_graph, la.RBConfigurationVertexPartition, resolution_parameter=1.0)
     cluster_map = {dataset_id: part for dataset_id, part in zip(dataset_ids, partition.membership)}
     ```

7. **Record Integer Cluster IDs**
   • After Leiden clustering, collect a map `dataset_id → cluster_id` (integer). One‐hot encoding will occur later in the training‐set assembly.

---

## Phase 2: Dimensionality Reduction & Visualization

1. **PCA → retain ≥ 95 % variance**  
   • In `src/feature_engineering.py`, write  
     ```python
     from sklearn.decomposition import PCA
     pca = PCA(n_components=0.95)
     pcs = pca.fit_transform(np.stack(list(embeddings.values())))
     df_pca = pd.DataFrame(pcs[:, :2], columns=["PC1", "PC2"], index=dataset_ids)
     ```

2. **UMAP → 2D layout**  
   • Then  
     ```python
     import umap
     reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
     umap_coords = reducer.fit_transform(pcs)
     df_umap = pd.DataFrame(umap_coords, columns=["UMAP1", "UMAP2"], index=dataset_ids)
     ```

3. **Load `dataset_type` Labels**
   • Directly read your CSV with pandas:
     ```python
     import pandas as pd
     labels_df = pd.read_csv("Data/train_labels.csv")
     type_map = dict(zip(labels_df["dataset_id"], labels_df["type"]))
     ```
   • Merge `type_map` into your DataFrame by `dataset_id` (values `"PRIMARY"`/`"SECONDARY"`).

4. **Create & Save Plots**  
   • Two scatter‐plots:  
     1. Color by Leiden cluster membership  
     2. Color by `type` label  
   • Save figures under `reports/feature_engineering/`.

---

## Phase 3: DuckDB Schema & Data Model Updates

1. **Extend `api/database/duckdb_schema.py`**  
   • Add a method `create_datasets_table()` after your existing tables:  
     ```sql
     CREATE TABLE IF NOT EXISTS datasets (
       dataset_id VARCHAR PRIMARY KEY,
       doc_id VARCHAR,
       total_tokens INTEGER,
       avg_tokens_per_chunk REAL,
       total_char_length INTEGER,
       clean_text_length INTEGER,
       leiden_cluster INTEGER,
       -- one column per PC, e.g., pc1 REAL, pc2 REAL
       umap1 DOUBLE,
       umap2 DOUBLE,
       FOREIGN KEY (doc_id) REFERENCES documents(doi)
     );
     ```  
   • Invoke it in `create_schema()` so it’s automatically created.

2. **Extend `DuckDBHelper` in `duckdb_utils.py`**  
   • Add a `store_datasets(self, datasets: List[Dataset]) -> bool` method:  
     - Convert each `Dataset` to a dict via `dataset.model_dump()` or a new `to_duckdb_row()`.  
     - Bulk‐upsert using a pandas DataFrame and `self.engine.execute(… USING df…)`.  
   • This will let the pipeline script simply call `db_helper.store_datasets(dataset_list)`.

---

## Phase 4: Orchestration Script

1. **New CLI script**  
   Path: `scripts/run_feature_engineering.py`.

2. **Workflow in `main()`**  
   ```python
   # 1. Read retrieval JSON → chunk IDs
   # 2. Call functions in src/feature_engineering:
   #      aggregate → embed → build_graph → cluster → reduce_dimensionality → visualize
   # 3. Build List[Dataset] Pydantic objects:
   #      for each dataset_id: Dataset(dataset_id=…, doc_id=…, total_tokens=…, …, cluster=…, dataset_type=…, text=…)
   # 4. Connect to DuckDB:
       db_helper = get_duckdb_helper(db_path)
       db_helper.store_datasets(dataset_list)
   ```

3. **Invocation**  
   ```bash
   python scripts/run_feature_engineering.py \
     --db-path artifacts/mdc_challenge.db \
     --retrieval-json reports/retrieval/retrieval_results.json \
     --output-dir reports/feature_engineering
   ```
---

### Clarifying Questions

1. **Retrieval vs. Live API**  
   • Do you want the feature pipeline to read your saved `retrieval_results.json`, or call the batch‐retrieve API endpoint on the fly?

2. **Embedding Source**  
   • The plan uses `embed_chunk` on the aggregated text. Would you prefer pulling raw chunk embeddings out of ChromaDB instead, then averaging or otherwise aggregating?

3. **Label Storage**  
   • I see `LabelMapper` expects `Data/train_labels.csv`. Are your `dataset_type` labels already loaded into DuckDB, or should we continue reading CSV via `LabelMapper`?

4. **Cluster Feature Representation**  
   • You asked for “binary engineered features” from clusters. Should we store one‐hot columns in DuckDB (e.g. `cluster_0`, `cluster_1`, …), or just store the integer cluster ID?


Other points to flag/confirm:
- Retrieving the “target chunk” by scanning every chunk’s `citation_entities` in DuckDB can be slow at scale; would you prefer persisting a direct `dataset_id → target_chunk_id` mapping during chunking?
- We’re importing the private `_embed_text` function—would you instead lean on the public `embed_chunk` API for clarity?
- Please confirm that `retrieval_results.json` is structured with a top‐level `"chunk_ids"` dict mapping each `dataset_id` to its retrieved chunk ID list.


Questions/concerns:

- Once you’ve updated `embed_chunk`, I can draft the exact signature and helper‐method edits for `embeddings_services.py`.  
- Let me know if the structure of `retrieval_results.json` ever changes (it should have a top-level `chunk_ids` dict).  
- Otherwise, the plan aligns with your flow—no further gaps identified.

