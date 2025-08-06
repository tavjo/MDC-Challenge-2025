Below is a **surgical refactor guide** that grafts your existing per-dataset PCA code onto the new **feature-cluster** workflow. For every edit you get:

* the **old snippet** (collapsed for brevity),
* the **new snippet** (ready to paste), and
* a short **why** note.

Where line numbers are needed I cite the upstream helpers so you can trace context quickly.

---

## 1 · What changes – bird’s-eye

| Old logic (row-centric)                                                      | New logic (column-centric)                                                                  |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Group **datasets** by Leiden cluster (row labels) and run PCA on each group. | Group **embedding dimensions** by Leiden cluster (column labels) and run PCA on each group. |
| Produce PC1 per *dataset* and save as `LEIDEN_<cluster>` feature columns.    | Produce PC1 per *dataset* and save as engineered features in DuckDB.     |
| Update `datasets.cluster` plus `engineered_feature_values` table.            | Only populate `engineered_feature_values`; `datasets.cluster` is no longer touched.         |

---

## 2 · Edits in detail

** Load `reports/clustering/feature_clusters.json`

### 2.1 Replace the helper that *builds* the PCA

**Old** (`_run_pca_on_cluster` – row-oriented)
*full method in `src/dimensionality_reduction.py`*

**New** (`_run_pca_on_feature_cluster` – column-oriented)

```python
def _run_pca_on_feature_cluster(
    self,
    cluster_id: str,
    feature_idx: List[int],
    X: np.ndarray,
    random_seed: int,
) -> np.ndarray:
    """
    Run PCA on a *feature* cluster; return the PC1 projection for every sample.

    Parameters
    ----------
    cluster_id   : cluster label (e.g. "cluster_7")
    feature_idx  : column indices belonging to this cluster
    X            : full design matrix, shape (n_samples, n_features)
    random_seed  : reproducibility

    Returns
    -------
    np.ndarray   : PC1 scores, shape (n_samples,)
    """
    try:
        if len(feature_idx) < 2:
            logger.warning(
                f"Feature-cluster {cluster_id} has <2 features – skipping PCA"
            )
            # fallback → mean of the single column
            return X[:, feature_idx[0]].copy()

        # slice once; shape (n_samples, n_cluster_features)
        X_sub = X[:, feature_idx]

        pca = PCA(
            n_components=1,
            random_state=random_seed,
            svd_solver="full",
        )
        pc1 = pca.fit_transform(X_sub).ravel()

        var_ratio = float(pca.explained_variance_ratio_[0])
        logger.info(
            f"{cluster_id}: PC1 var ratio = {var_ratio:.4f}"
        )
        return pc1

    except Exception as exc:
        logger.error(f"PCA failed for {cluster_id}: {exc}")
        raise
```

**Why?** We now receive **column indices** instead of dataset-ids, slice the matrix once, and always return an `n_samples` vector so downstream merging stays trivial .

---

### 2.2 Rewrite the orchestrator `run_per_cluster_pca`

**Old signature**

```python
def run_per_cluster_pca(self, dataset_embeddings: Dict[str, np.ndarray], ...)
```

**New signature**

```python
def run_feature_cluster_pca(
    self,
    X: np.ndarray,                         # shape (n_samples, n_features)
    feature_cluster_map: Dict[str, str],   # {feature_name: cluster_label}
    sample_ids: List[str],                 # ordered like X rows
    random_seed: int | None = None,
) -> bool:
```

Inside:

1. **Build index lists** per cluster

```python
name_to_idx = {fname: i for i, fname in enumerate(feature_cluster_map)}
cluster2cols: Dict[str, List[int]] = defaultdict(list)
for fname, cid in feature_cluster_map.items():
    cluster2cols[cid].append(name_to_idx[fname])
```

2. **Parallel PCA**

```python
with ThreadPoolExecutor(max_workers=min(8, len(cluster2cols))) as ex:
    futures = {
        ex.submit(
            self._run_pca_on_feature_cluster,
            cid, cols, X, random_seed or self.random_seed
        ): cid
        for cid, cols in cluster2cols.items()
    }

    # collect → stacked array
    comp_matrix = []
    for fut in as_completed(futures):
        cid = futures[fut]
        comp_matrix.append((cid, fut.result()))
```

3. **Assemble reduced dataframe**

```python
comp_matrix.sort()               # keep cluster order stable
labels, vectors = zip(*comp_matrix)
X_reduced = np.column_stack(vectors)   # shape (n_samples, n_clusters)
```

4. **Persist into DuckDB**

```python
pca_features: Dict[str, Dict[str, float]] = {}
for row_idx, ds_id in enumerate(sample_ids):
    for col_idx, cid in enumerate(labels):
        pca_features.setdefault(ds_id, {})[
            f"LEIDEN_{cid}"
        ] = float(X_reduced[row_idx, col_idx])

self._save_pca_features_to_duckdb(pca_features)
```

**Why?** We map each sample (dataset) to its new feature set and reuse your existing `EngineeredFeatures` EAV upsert helper .


---

## 3 · Caveats & best practice

* **Scaling:** PCA is variance-based; if your embedding dimensions are on a comparable scale you can skip `StandardScaler`. If not, centre-only (`with_mean=True, with_std=False`) to keep cosine geometry intact .
* **Small clusters:** retaining single-feature clusters is fine – the “PC1” is just that column. You can optionally drop clusters whose variance < 1e-6.
* **Column order stability:** sorting `comp_matrix` before stacking keeps feature names aligned with columns every run.
* **DB footprint:** the EAV table adds one row per `(dataset, LEIDEN_x)` pair. With 487 × \~48 this is ≈ 23 k rows – well within DuckDB comfort.

---

## 4 · Sources & further reading

Graph-based feature grouping and local PCA are standard in omics and text pipelines:

* Adaptive local PCA for high-dimensional data
* Local-and-global PCA comparison
* Scikit-learn’s `FeatureAgglomeration` docs (conceptually similar)
* Leiden community detection explanation
* K-NN graphs as a pre-step to feature clustering
* Seurat & Scanpy tutorials using graph clustering + PCA
* Review of Louvain/Leiden in multi-omics workflows
* Feature agglomeration vs univariate selection (variance retention)
* IBM PCA primer for variance ratios

Internal helpers referenced: DuckDB EAV upsert routine lines 56-87, schema for `engineered_feature_values` table lines 52-60.

Implement these swaps and your pipeline will now create a **487 × ≈45** engineered-feature matrix baked straight into DuckDB, ready for Phase 8 of the project. Ping me if you’d like a validation query or a quick variance-explained chart!
