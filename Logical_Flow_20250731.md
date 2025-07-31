## **Project Overview**

The MDC Challenge asks competitors to find every dataset citation in a scientific article and classify the citation as **Primary** (data generated for the study) or **Secondary** (reused data) ([makedatacount.org](https://makedatacount.org/read-our-blog/announcing-make-data-counts-kaggle-competition/?utm_source=chatgpt.com), [Kaggle](https://www.kaggle.com/competitions/make-data-count-finding-data-references?utm_source=chatgpt.com)).

 Main strategy is to:

1. **Parse PDFs → plain-text pages.**  
2. **Detect citations \+ page locations.**  
3. **Chunk pages around each citation, and embed the chunks.**  
4. **Retrieve and Aggregate per-dataset context with semantic search, mask citations, and embed full context.**  
5. **Engineer per-dataset features**  
6. **Train Random-Forest classifier.**  
7. **Optional: validate on small corpus of manually annotated papers with dataset citations**

All steps must run **completely offline** inside Docker, using local models and a temporary DuckDB file to avoid brittle path handling.

---

## **Phase 1 — Exploratory Data Analysis *(completed)***

| Element | Description |
| ----- | ----- |
| **Inputs** | `train_labels.csv`; 524 PDF files; 400 XML files |
| **Key Tasks** | Establish corpus coverage; verify label distribution; inventory PDFs lacking XML; create reusable Pandas profiling notebook. |
| **Outputs** | `conversion_candidates.csv` (PDFs missing XML); `document_inventory.csv` with article-level metadata. |

*Rationale.* An audit baseline prevents silent data loss in later phases and flags “edge-case” articles early ([DataCamp](https://www.datacamp.com/tutorial/duckdb-to-speed-up-data-pipelines?utm_source=chatgpt.com)).

---

## **Phase 2 — Parse Documents *(new, completed)***

| Element | Description |
| ----- | ----- |
| **Inputs** | `conversion_candidates.csv`; original PDFs |
| **Process** | Use *Unstructured*’s `partition_pdf` to extract ordered pages and high-level metadata inside a container (Intel-friendly dependency isolation). |
| **Outputs** | `documents.json` — list of `Document` Pydantic objects (page texts \+ PDF metadata). |

*Performance note.* 20 PDFs parse in ≈ 2 min; full 524-article run expected in ≈ 2 h (parallelisation with Python `multiprocessing` will cut this by \~75 %) ([unstructured.io](https://unstructured.io/blog/optimizing-unstructured-data-retrieval)).

---

## **Phase 3 — Extract Citations *(completed)***

| Element | Description |
| ----- | ----- |
| **Inputs** | `train_labels.csv`; `documents.json` |
| **Dual Extraction Logic** | • **Training mode — regex-driven.** Known dataset IDs anchor high-precision matches.  • **Inference mode — LLM‐driven.** Ollama \+ BAML prompt returns candidate citations when labels are unknown. |
| **Outputs** | `citations.json` — list of `CitationEntity` objects (dataset\_id, doc\_id, pages\[\]). Planned: `citation_stats.csv`. |

*Challenges & mitigation.*  
 LLMs sometimes hallucinate strings like “DRYAD” with no accession pattern. Constrain the output schema and post-filter against a curated regex library (GEO, Zenodo, DOIs, etc.) to raise precision ([Pinecone](https://www.pinecone.io/learn/chunking-strategies/?utm_source=chatgpt.com), [Milvus](https://milvus.io/ai-quick-reference/what-techniques-support-anonymization-in-legal-text-embeddings?utm_source=chatgpt.com)).

---

## **Phase 4 — Chunking *(completed)***

| Element | Description |
| ----- | ----- |
| **Inputs** | `documents.json`; `citations.json` |
| **Process** | 1\) Determine optimal window (e.g., 300 tokens with 30 overlap) based on a custom sliding window chunking method ([Pinecone](https://www.pinecone.io/learn/chunking-strategies/?utm_source=chatgpt.com), [Reddit](https://www.reddit.com/r/LangChain/comments/15mq21r/what_are_the_text_chunkingsplitting_and_embedding/?utm_source=chatgpt.com)). 2 ) Count citations pre-/post-chunk to detect truncation; repair if counts differ.  |
| **Outputs** | `chunks.json` — list of `Chunk` objects (chunk\_text, citation\_ids\[\], page\_span, token\_count). Planned: `chunk_stats.csv`. |

---

## **Phase 5 — Create Embeddings *(completed)***

| Element | Description |
| ----- | ----- |
| **Inputs** | `chunks.json`; `chunk_stats.csv` |
| **Embedding Engine** | OpenAI embedding model |
| **Vector Store** | **ChromaDB** running in-process, persisted to an on-disk folder inside the container so it can be zipped for Kaggle submission ([Chroma Docs](https://docs.trychroma.com/getting-started?utm_source=chatgpt.com), [Chroma Docs](https://docs.trychroma.com/?utm_source=chatgpt.com)). |
| **Outputs** | `chroma/` folder; `embedding_summary.csv` (per-article counts, mean norms, etc.). |

---

## **Phase 6 — Retrieve context & Construct `Dataset` Objects with similarity search (in progress)**

| Element | Description |
| ----- | ----- |
| **Inputs** | `chunks.json`; `chunk_stats.csv`; `embedding_summary.csv` |
| **Process** | Retrieve top K chunks associated with each dataset ID;  Group all chunks that reference the same dataset\_id;  Mask every dataset-ID token with a placeholder (`<DATASET_ID>`) before re-embedding chunk aggregates reduce leakage ([arXiv](https://arxiv.org/html/2504.16609v1?utm_source=chatgpt.com), [Milvus](https://milvus.io/ai-quick-reference/what-techniques-support-anonymization-in-legal-text-embeddings?utm_source=chatgpt.com)).  collect neighbourhood embeddings (N \= k nearest), aggregate statistics (mean, max, variance). |
| **Outputs** | `datasets.json` — list of `Dataset` Pydantic objects; `dataset_embedding_stats.csv`. |

*Critical constraint.* Do **not** modify this phase’s deliverables; downstream phases expect exactly this schema.

---

## **Phase 7 — Clustering & Dimensionality Reduction (UMAP / PCA)**

| Element | Description |
| ----- | ----- |
| **Inputs** | `datasets.json`; `dataset_embedding_stats.csv` |
| **Process** | Create similarity matrix  → use threshold to determine whether an edge exists between 2 nodes which in this case are chunks → apply Leiden clustering on networks. These will become additional engineered features to help our model learn.  Run PCA (retain ≥ 95 % variance) → feed PCs to UMAP (n\_neighbors ≈ 15, min\_dist ≈ 0.1) for 2-D manifold suited to Random-Forest feature importance visuals ([Medium](https://medium.com/%40aastha.code/dimensionality-reduction-pca-t-sne-and-umap-41d499da2df2?utm_source=chatgpt.com), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2211124721008597?utm_source=chatgpt.com)). Visualize UMAP by Leiden clusters |
| **Outputs** | `dataset_dr.csv` (dataset\_id, UMAP1, UMAP2, PC1, PC2, LeidenClusters, label). |

---

## **Phase 8 — Build Classifier Input**

| Element | Description |
| ----- | ----- |
| **Inputs** | `dataset_dr.csv`; `datasets.json`; `dataset_embedding_stats.csv` |
| **Process** | Merge dimensionality-reduced features with any hand-engineered variables (e.g., citation frequency, section entropy). |
| **Outputs** | `model_input.json` — list of ClassifierInput objects (ready for training). |

---

## **Phase 9 — Train Random-Forest Model**

| Element | Description |
| ----- | ----- |
| **Inputs** | `model_input.json` |
| **Model Choice** | Classic Random Forest (n ≈ 300, class-balanced) performs well on tabular embeddings and is easy to interpret ([Kaggle](https://www.kaggle.com/code/arunava21/word2vec-and-random-forest-classification?utm_source=chatgpt.com), [Stack Overflow](https://stackoverflow.com/questions/67381956/how-do-we-use-a-random-forest-for-sentence-classification-using-word-embedding?utm_source=chatgpt.com)). |
| **Outputs** | `rf_model.pkl` (saved weights \+ fitted scaler metadata). |

---

## **Phase 10 — Inference on Test Set**

| Element | Description |
| ----- | ----- |
| **Inputs** | `rf_model.pkl`; Docker preprocessing pipeline; unseen PDFs. |
| **Process** | Run Phases 2 → 8 inside the inference container, then predict labels with the stored forest. |
| **Outputs** | `submission.csv` for Kaggle; validation notebook capturing runtime and memory. |

---

## **Cross-Cutting Infrastructure**

### **DuckDB “Landing Zone”**

All intermediate tables (`documents`, `citations`, `chunks`, `datasets`, `dr_features`) are cached as DuckDB relations inside a single file, enabling SQL-style QA checks and eliminating brittle path juggling ([Medium](https://medium.com/%40RaajasSode/building-end-to-end-data-pipelines-locally-with-duckdb-e5a6913f210b?utm_source=chatgpt.com), [DataCamp](https://www.datacamp.com/tutorial/duckdb-to-speed-up-data-pipelines?utm_source=chatgpt.com)).

### **Container & Offline Requirements**

* **Base images.** Alpine \+ `uv` for Python deps; CUDA-free.

* **Two services in `docker-compose.yml`.** `preprocess` (Phases 2–8) and `train_infer` (Phases 9–10).

* **Volume mount** for DuckDB & Chroma directories so both services share state.

---

## **Recommendations & Best-Practice Notes**

| Area | Suggestion | Reason |
| ----- | ----- | ----- |
| Citation extraction | Combine BAML JSON schema with a *second-pass* regex whitelist; reject any candidate lacking a DOI/Accession-pattern. | Hybrid boosts precision without sacrificing recall. |
| Masking strategy | Replace each unique dataset\_id with the same `<DATASET_ID>` token so the model cannot memorize literal IDs yet still sees positional context ([arXiv](https://arxiv.org/html/2504.16609v1?utm_source=chatgpt.com)). | Reduces embedding leakage and over-fitting. |
| Chunk size tuning | Start at 400–600 tokens with 20 % overlap; use semantic-shift detection to split where cosine distance spikes ([Pinecone](https://www.pinecone.io/learn/chunking-strategies/)). | Balances context richness vs. vector count. |
| Vector store | Keep Chroma in **persistent-client** mode; during Kaggle submission zip the `chroma/` folder (tiny footprint). | Ensures reproducibility without re-embedding. |
| Parallel PDF parsing | Use Python `concurrent.futures.ProcessPoolExecutor`; `unstructured.partition_pdf` is GIL-bound and scales linearly until disk I/O saturates ([unstructured.io](https://unstructured.io/blog/optimizing-unstructured-data-retrieval)). | Cuts Phase 2 wall-time on 8-core runners. |
| Feature engineering | Add binary flags: *appears-in-Methods*, *cited-\>1 time*, *within-Results*; Random Forests exploit heterogeneous features well. | Provides interpretability and potential accuracy lift. |
| Evaluation | 5-fold stratified CV over articles (not citations) prevents leakage; monitor F1 plus per-class recall. | Aligns with Kaggle metric and mitigates skew. |

---

### **Where to Go Next**

* Experiment with **semantic‐aware chunking** libraries (e.g., *tiktoken* recursive splitter) to minimise broken sentences.

* Investigate **entity-replacement differential privacy** if you need stronger leakage guarantees ([Milvus](https://milvus.io/ai-quick-reference/what-techniques-support-anonymization-in-legal-text-embeddings?utm_source=chatgpt.com)).

* Profile DuckDB vs. Parquet read/write speed in your container to confirm the I/O payoff.

* Keep an eye on *Make Data Count* forum posts for any dataset updates or clarifications before final submission ([makedatacount.org](https://makedatacount.org/read-our-blog/announcing-make-data-counts-kaggle-competition/?utm_source=chatgpt.com)).

This refined workflow is longer, more explicit, and maintains the exact logical dependencies and outputs you defined. Let me know if you’d like deeper dives into any individual phase or further implementation tactics\!


