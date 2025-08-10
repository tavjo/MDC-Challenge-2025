# Expanded Inference Plan for Kaggle MDC Challenge Submission

This plan fleshes out your current inference pipeline without changing its logical flow or the embedding models. All steps are described so someone unfamiliar with the codebase can follow them. I inspected your mdc\_retrieval\_module.py for the hybrid retrieval logic and src/clustering.py for clustering/feature‑engineering functions and cite relevant parts below. Only the inference portion is described (the model is already trained). If anything is unclear or missing you should provide clarification before running the notebook.

## 1 Environment Setup

1. **Dataset & model files**

2. Add the Data directory (with train/, test/, train\_labels.csv and sample\_submission.csv) to your Kaggle notebook as an input dataset. The test PDFs are under Data/test/PDF; XMLs under Data/test/XML. A pre‑trained random‑forest model is provided as rf\_model.pkl – copy this into /kaggle/working so it is persisted during execution.

3. **Install dependencies** – run the following once in the notebook cell:

\!pip install unstructured\[all\] sentence-transformers chromadb duckdb igraph leidenalg \\  
      umap-learn baml==0.3.2 pandas numpy scikit-learn==1.4.2 \--quiet

* unstructured is used for PDF parsing and page cleaning. The extract\_pdf\_text\_unstructured.py script in your repo shows how partition\_pdf can be called with options to group elements into page texts[\[1\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/extract_pdf_text_unstructured.py#L111-L177).

* sentence-transformers loads the **BAAI/bge‑small‑en‑v1.5** model for dense embeddings. This model outputs 384‑dimensional vectors.

* chromadb, duckdb, igraph, leidenalg, umap-learn are required for building similarity graphs and clustering features.

* baml (v0.3.2) is the interface used to call the Qwen3 model for citation extraction.

* scikit‑learn 1.4.2 is pinned because your trained model expects the same version of the Random Forest implementation.

* **Cache models**

* The **Qwen3** model (5.2 GB) should be stored in a persistent directory such as /kaggle/working/qwen3/. If it isn’t already present, download it from HuggingFace before internet is disabled. Qwen3 is used through the BAML client.

* The **bge‑small‑en‑v1.5** embedding model is downloaded automatically by SentenceTransformer and cached in the notebook; no manual download is required.

* **Imports & global setup**

import os, json, re, numpy as np, pandas as pd  
from pathlib import Path  
from concurrent.futures import ThreadPoolExecutor  
from sentence\_transformers import SentenceTransformer  
import tiktoken  
import umap  
import igraph as ig  
import leidenalg  
import joblib  
import duckdb  
from unstructured.partition.pdf import partition\_pdf  
from baml import ModelClient

\# retrieval utilities from your repo  
from mdc\_retrieval\_module import hybrid\_retrieve\_with\_boost, build\_regex\_index, guess\_section

* Set random seeds (np.random.seed, Python random.seed, and scikit‑learn’s random\_state parameters) for reproducibility.

* Establish a thread‑pool for parallel parsing of PDFs to speed up document loading.

## 2 Document Parsing

1. **PDF parsing**

2. Use the helper in extract\_pdf\_text\_unstructured.py or call partition\_pdf() directly. The script shows how each PDF is parsed into a list of Element objects and then grouped by page number; each page’s text is cleaned (hyphens removed, newlines collapsed)[\[1\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/extract_pdf_text_unstructured.py#L111-L177). For inference, write a function parse\_pdf(path) that returns:

   * page\_texts: a list of page strings in order.

   * Metadata such as number of pages, total characters and total tokens. Token counts should be computed using tiktoken.get\_encoding("cl100k\_base") so they match the tokeniser used during training.

3. **XML parsing**

4. For each XML in Data/test/XML, parse with lxml.etree or BeautifulSoup. Extract textual content either as a single page or by splitting on paragraphs. Record the total token count and number of paragraphs (treated as pages).

5. **Construct Document objects**

6. For each file (PDF or XML), build a record similar to the Pydantic Document used during training:

| field | description |
| :---- | :---- |
| doi | derived from the filename (remove extension). |
| full\_text | list of cleaned page texts. |
| n\_pages | number of pages in full\_text. |
| total\_char\_length | sum of character lengths across pages. |
| total\_tokens | token count using the same tokenizer as training. |
| file\_path | original path to the PDF/XML. |

7. Assemble these into a list docs for downstream processing.

## 3 Chunking & Embedding

1. **Tokenisation & sliding windows**

2. Concatenate a document’s page texts into one string. Tokenise with tiktoken and create overlapping chunks of approximately **300 tokens** with **30‑token** overlap, just like the sliding\_window\_chunk\_text function used during training. Each chunk receives a unique chunk\_id (e.g. doi\_pageStart\_index) and metadata fields previous\_chunk\_id and next\_chunk\_id to link contiguous chunks.

3. **Section heuristics**

4. Use the guess\_section() function from your retrieval module to heuristically label a chunk as “data availability” or “methods.” The function checks whether cue phrases appear in the first 800 characters[\[2\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/kaggle/retrieval_module.py#L111-L120). Store the resulting section (or None) in a dictionary id\_to\_section keyed by chunk\_id.

5. **Compute dense embeddings**

6. Load the bge‑small model: embedder \= SentenceTransformer('BAAI/bge-small-en-v1.5'). Batch‑encode each chunk’s text into a 384‑dimensional NumPy array. Save two mappings:

   * id\_to\_text: maps chunk\_id → chunk text.

   * id\_to\_dense: maps chunk\_id → 384‑dimensional embedding.

## 4 Hybrid Retrieval of Candidate Chunks

The inference pipeline uses a hybrid retrieval stage before LLM citation extraction. The hybrid\_retrieve\_with\_boost() function fuses sparse BM25 and dense search, applies boosting based on regex matches and section priors, and then applies MMR re‑ranking. Its signature is shown in mdc\_retrieval\_module.py[\[3\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/kaggle/retrieval_module.py#L557-L591).

1. **Regex index**

2. Build a regex index to count DOI/accession matches in each chunk using regex\_index \= build\_regex\_index(id\_to\_text). The regex patterns are defined in ACCESSION\_REGEXES of the module[\[4\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/kaggle/retrieval_module.py#L111-L130) and include DOI URLs, numeric DOIs, GEO/SRA/GSE/PRJ accession formats, etc.

3. **Dense query vector**

4. Compute an overall dense representation of each document by averaging its chunk embeddings: dense\_query\_vec \= np.mean(np.vstack(list(id\_to\_dense.values())), axis=0).

5. **Run hybrid retrieval**

ranked\_chunk\_ids \= hybrid\_retrieve\_with\_boost(  
    query\_text="data availability deposited in GEO",  \# general DAS query  
    dense\_query\_vec=dense\_query\_vec,  
    id\_to\_dense=id\_to\_dense,  
    id\_to\_text=id\_to\_text,  
    id\_to\_section=id\_to\_section,  
    regex\_index=regex\_index,  
    \# optionally override BoostConfig parameters  
)

* Internally, the function computes a BM25 top‑k (sparse\_ids) and a dense top‑k (dense\_ids); these are combined by **reciprocal rank fusion (RRF)**. Scores are then boosted: chunks that contain regex matches receive a boost proportional to the number of matches; chunks adjacent to regex hits are boosted; chunks classified as “methods” or “data availability” get an additional boost[\[5\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/kaggle/retrieval_module.py#L557-L571). Finally, MMR (maximal marginal relevance) re‑ranking ensures diversity among the selected chunks[\[3\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/kaggle/retrieval_module.py#L557-L591).

* **Select top‑K candidate chunks**

* Choose a value of **K** (e.g. 30–50). A larger K increases recall but slows down the LLM stage. For each document:

* candidate\_ids  \= ranked\_chunk\_ids\[:K\]  
  candidate\_texts \= \[id\_to\_text\[cid\] for cid in candidate\_ids\]

* These candidate\_texts will be passed to the citation extraction model. Save the mapping from candidate chunk\_id to original document pages (page numbers can be found by storing the chunk’s token offsets).

## 5 Citation Extraction via Qwen3 & BAML

1. **Initialise BAML client**

qwen\_client \= ModelClient(  
    model\_name='qwen3',  
    model\_path='/kaggle/working/qwen3',  
    max\_tokens=512  
)

1. **Define extraction function**

2. Use your BAML specification ExtractCitation (likely defined in your repo) to prompt Qwen3. The prompt should instruct the model to:

   * Identify dataset citations (DOI or accession IDs) in the provided text.

   * Return a JSON list with fields such as data\_citation, evidence, and optionally citation\_type.

   * Ensure that returned identifiers match your regex patterns. Reject any hallucinated output by re‑matching with ACCESSION\_REGEXES[\[4\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/kaggle/retrieval_module.py#L111-L130).

3. **Call the model**

results \= qwen\_client.ExtractCitation(document=candidate\_texts)

* Iterate through the returned list and filter out entries whose data\_citation does not match a DOI or accession pattern. For each valid citation, store:

  * The identifier (data\_citation).

  * The document ID (the DOI derived from filename).

  * The list of pages where the citation appears (determine this by scanning the original page\_texts or by mapping chunk IDs back to page numbers).

  * The evidence snippet (text from the chunk containing the citation).

* **Deduplicate**

* Citations may be repeated across chunks. Consolidate identical data\_citation entries for a document and keep the union of their evidence pages.

## 6 Context Aggregation & Re‑Embedding

1. **Associate chunks to citations**

2. For each CitationEntity, find all chunks in the document that contain the citation (substring match or regex). These “positive” chunks contributed to training labels.

3. **Retrieve additional context**

4. To capture context beyond the exact mentions, optionally retrieve the nearest neighbours of each positive chunk in embedding space (using cosine similarity). For example, for each citation you might select the top 5 additional chunks that are closest in embedding space but do not contain the citation itself.

5. **Aggregate context**

6. Concatenate the texts of all positive and additional chunks into a single string. Before concatenation, **mask the citation identifiers** by replacing each occurrence of the DOI/accession with a placeholder such as \<DATASET\_ID\>. This leakage‑prevention step was used during training to avoid the classifier simply matching the presence of the ID.

7. **Re‑embed**

8. Encode the aggregated context using the same bge‑small model to produce a 384‑dimensional vector per citation. Collect these into a list citation\_vectors.

9. **Basic engineered features**

10. Compute simple scalar features for each citation:

    * n\_positive\_chunks: number of chunks that explicitly mention the citation.

    * mean\_positive\_tokens: mean token length of positive chunks.

    * has\_methods\_section: boolean indicating whether any positive chunk belonged to the “methods” section.

    * has\_data\_availability\_section: boolean indicating whether any positive chunk belonged to the “data availability” section.

    * Additional counts such as total characters or ratio of regex hits can be added if they were used during training.

## 7 Feature Engineering: Clustering, PCA & UMAP

The Random Forest classifier uses features derived from both clustering the embedding dimensions and global dimensionality reduction. The relevant functions are in src/clustering.py.

1. **Load or compute feature clusters**

2. If you have a saved file such as feature\_clusters.json from training, load it. Otherwise, compute clusters on training citation vectors using the provided functions:

   * **k‑NN similarity graph:** build\_knn\_similarity\_graph(dataset\_embeddings, k\_neighbors=12) constructs a memory‑efficient similarity graph over embedding dimensions[\[6\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/clustering.py#L61-L81). The function transposes the embedding matrix so that rows are features, computes nearest neighbours with cosine distance, applies a similarity threshold and creates an undirected igraph[\[7\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/clustering.py#L61-L112).

   * **Size‑balanced Leiden clustering:** pass the graph to run\_leiden\_clustering(graph, resolution=1.5, min\_cluster\_size=6, max\_cluster\_size=75, split\_factor=1.3, random\_seed=42) to assign each of the 384 dimensions to a cluster. The algorithm recursively splits oversized clusters and merges undersized ones[\[8\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/clustering.py#L176-L253).

3. **Per‑cluster PCA**

4. For each cluster, collect the corresponding dimensions from the citation vectors. Fit a 1‑component PCA on the training set and apply it to both training and test sets. This yields one scalar feature per cluster (so if there are n\_clusters clusters you get n\_clusters features).

5. **Global PCA & UMAP**

6. Standardise the citation vectors (subtract mean, divide by standard deviation) and fit a global PCA on training vectors, retaining enough components to explain at least 95 % of variance (e.g. 10–20 components). Transform the test vectors using this PCA. Keep the first two principal components as features PC1 and PC2.

7. Fit a UMAP model (e.g. n\_neighbors=15, min\_dist=0.1, random\_state=42) on the PCA‑reduced training vectors and transform the test vectors. Keep the first two UMAP dimensions (UMAP1, UMAP2).

8. **Neighbourhood statistics**

9. Optionally, compute k‑nearest neighbour distances in the embedding space for each test citation (k=10). Features can include mean distance, maximum distance and variance. This captures whether the citation vector lies in a dense or sparse region.

10. **Assemble the feature table**

11. Concatenate per‑cluster PCA features, global PCA components (PC1, PC2), UMAP components, neighbourhood statistics and the basic scalar features into a single pandas DataFrame. **Ordering matters** – the columns must match the order used during training. If you saved a list of feature names during training or can inspect rf\_model.feature\_names\_in\_, use that to reorder the columns.

## 8 Model Inference & Submission

1. **Load the Random Forest model**

rf\_model \= joblib.load('/kaggle/working/rf\_model.pkl')

* The model may include preprocessing steps in a Pipeline. Inspect rf\_model to see if there is a feature\_names\_in\_ attribute that lists the expected feature ordering. If not available, you should load the saved training feature schema (if provided) to ensure alignment.

* **Prepare the feature matrix**

* Convert the assembled feature dictionary for each citation into a pandas DataFrame. Reorder columns to match rf\_model.feature\_names\_in\_. Handle any missing features by filling with zeros or the training mean (depends on how missing values were treated during training).

* **Predict citation types**

y\_pred \= rf\_model.predict(feature\_matrix)  
labels \= \['Primary' if cls \== 1 else 'Secondary' for cls in y\_pred\]

* Check rf\_model.classes\_ to confirm which label corresponds to “Primary” vs “Secondary”. Some scikit‑learn models order classes alphabetically, so mapping may need to be inverted.

* **Build submission file**

* Combine the predictions with the CitationEntity records:

* submission \= pd.DataFrame({  
      'article\_id': \[ce.document\_id for ce in citation\_entities\],  
      'dataset\_id': \[ce.data\_citation for ce in citation\_entities\],  
      'type': labels  
  })  
  \# remove duplicates if the same citation appears multiple times  
  submission \= submission.drop\_duplicates()  
  submission.to\_csv('/kaggle/working/submission.csv', index=False)

* Compare the distribution of predicted labels to those in train\_labels.csv to ensure there are no glaring mismatches (e.g. 100 % of citations predicted as primary).

## 9 Runtime & Resource Considerations

* **Memory & storage** – With roughly 30 test articles, storing all chunk embeddings and aggregated citation vectors in memory should be feasible (\<2 GB). However, Qwen3 and BGE‑small models are large; keep them in /kaggle/working to avoid repeated downloads.

* **Parallel parsing** – Use a thread pool (e.g. ThreadPoolExecutor(max\_workers=4)) when parsing PDFs and computing embeddings. This can significantly reduce runtime.

* **Candidate chunk count** – Tuning K in the hybrid retrieval step controls Qwen3 usage. A small K (20–30) yields faster inference but might miss citations; a larger K (50+) improves recall at the cost of longer LLM calls.

* **Reproducibility** – Set global random seeds before all stochastic operations (clustering, PCA, UMAP) to ensure deterministic outputs. Provide random\_state values in scikit‑learn and UMAP functions.

* **Offline execution** – Since Kaggle competition kernels can disable internet access during scoring, download the Qwen3 model and any other external resources during the notebook’s first run and save them in /kaggle/working.

## Questions & Clarifications

1. **Training feature schema** – the model’s feature ordering must match training. If you have a file that records the order of features (e.g. feature\_names.json or rf\_model.feature\_names\_in\_), please provide it or confirm the ordering used during training. Without this, we can only guess the correct column order.

2. **Number of candidate chunks (K)** – what value of K did you use during training to select the top chunks for Qwen3? Using a different K at inference might slightly alter the distribution of evidence.

3. **Prototype priors** – the hybrid retrieval code includes optional prototype‑based boosts (centroid affinity). Were prototypes used during training? If yes, please provide the saved prototype vectors or confirm the parameters for BoostConfig (e.g. proto\_weight, proto\_min\_sim, mmr\_lambda) so inference can mirror training settings.

4. **Neighbourhood statistics** – did the Random Forest model include k‑NN distance features? If so, please specify the value of k and whether distances were computed within the training set or combined train+test set.

By addressing these questions and following the detailed steps above, you should be able to build a Kaggle submission notebook that mirrors your training pipeline and produces accurate citation type predictions.

---

[\[1\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/extract_pdf_text_unstructured.py#L111-L177) extract\_pdf\_text\_unstructured.py

[https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/extract\_pdf\_text\_unstructured.py](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/extract_pdf_text_unstructured.py)

[\[2\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/kaggle/retrieval_module.py#L111-L120) [\[3\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/kaggle/retrieval_module.py#L557-L591) [\[4\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/kaggle/retrieval_module.py#L111-L130) [\[5\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/kaggle/retrieval_module.py#L557-L571) retrieval\_module.py

[https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/kaggle/retrieval\_module.py](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/kaggle/retrieval_module.py)

[\[6\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/clustering.py#L61-L81) [\[7\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/clustering.py#L61-L112) [\[8\]](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/clustering.py#L176-L253) clustering.py

[https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/clustering.py](https://github.com/tavjo/MDC-Challenge-2025/blob/main/src/clustering.py)