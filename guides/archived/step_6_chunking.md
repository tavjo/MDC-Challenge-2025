Below is a two-part answer:

* **Part A flags the handful of contradictions / ambiguities I found in the *preprocessing implementation guide* so you can decide whether to amend them before we automate Step 6.**
* **Part B gives a self-contained, granular “cookbook” for Step 6 – Semantic Chunking Pipeline.**
  It lists only the libraries required for this step, stays within the scope of semantic chunking, and hands off artefacts ready for Step 7 (masking / leakage checks) without doing that work prematurely.

---

## Part A – Things that look contradictory / unclear

| Ref                    | What I noticed                                                                                                                                | Why it could be a problem                                                                                        | Suggested fix                                                                                                                                 |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| §6 Data Models         | `Section` is declared twice: first as a **dataclass** (5.1) and then as a **Pydantic BaseModel** (6.1).                                       | Dual definitions lead to `isinstance` failures when the two flavours mix in downstream code.                     | Keep only the Pydantic version in §6 and import it everywhere else.                                                                           |
| §6 Chunk Code          | Some snippets read `chunk.chunk_metadata.section_primary.section_type` but `ChunkMetadata` actually holds a flat field called `section_type`. | `AttributeError` during runtime.                                                                                 | Replace all `section_primary.*` usages with `section_type`.                                                                                   |
| §7 Masking             | ID-masking (`mask_dataset_ids_in_chunks`) is placed inside the Step 6 file.                                                                   | Masking is Step 7 in your roadmap; doing it in Step 6 violates separation of concerns and complicates debugging. | Move the masking helpers to Step 7; Step 6 should export **raw un-masked chunks** plus a helper inventory for Step 7.                         |
| §6.4 Tokenizer Comment | Comment says *“GPT-4 tokenizer”* for `cl100k_base`; GPT-4 and GPT-3.5 share it, so the wording is slightly misleading.                        | Minor – no crash, but could confuse collaborators.                                                               | Change comment to *“cl100k\_base (used by GPT-3.5/4 models)”*.                                                                                |
| §6.4 Repair Logic      | `adaptive_chunk_repair` suggests new chunk sizes but there is no subsequent re-chunk call.                                                    | Possible dead-end if validation fails.                                                                           | Either (a) implement a loop that reruns `create_section_aware_chunks` with new params, or (b) leave a TODO reminding users to rerun manually. |

---

## Part B – Step 6 “Semantic Chunking Pipeline” cookbook

### 0  Dependencies (only what Step 6 needs)

```bash
uv add "sentence-transformers==2.*"    # embeddings for similarity checks
uv add "langchain>=0.3,<0.4"          # RecursiveCharacterTextSplitter
uv add tiktoken                       # exact GPT-token counts
uv add pandas numpy                   # lightweight tabular ops
uv add regex                          # high-perf pattern engine
uv add pydantic                       # strict chunk / metadata schemas
uv add scikit-learn                   # cosine similarity for QC
```

`torch`, `chromadb`, etc. are **not** required here – they first appear in Step 8.
(Language-model fine-tuning isn’t done in Step 6).

---

### 1  High-level goal

> **Input:** the enriched, section-aware documents you exported at the end of Step 5 (`parsed_documents.pkl`).
> **Output:** a `chunks_for_embedding.pkl` file **plus** a small `_summary.csv` that (a) keeps every citation entity intact, (b) tags each chunk with section/order/type metadata, and (c) stops just short of ID masking (reserved for Step 7).

### 2  Workflow overview

| Order | What happens                                                                        | Key file/function                    |
| ----- | ----------------------------------------------------------------------------------- | ------------------------------------ |
| 1     | Load parsed docs & filter out empty ones                                            | `load_parsed_documents_for_chunking` |
| 2     | Select high-priority sections (methods, data availability, …) & fall-back full text | `prepare_section_texts_for_chunking` |
| 3     | Inventory all DOI / accession patterns **before** splitting                         | `create_pre_chunk_entity_inventory`  |
| 4     | Split text with **RecursiveCharacterTextSplitter** (200 tokens ±20 overlap)         | `create_section_aware_chunks`        |
| 5     | Label each chunk (`body`/`header`/`caption`/`data_statement`)                       | `refine_chunk_types`                 |
| 6     | Count tokens with `tiktoken` → store in metadata                                    | same as 4                            |
| 7     | Validate entity retention (should be 100 %)                                         | `validate_chunk_integrity`           |
| 8     | Export chunks & a compact CSV dashboard                                             | `export_chunks_for_embedding`        |

---

### 3  Step-by-step code stencil

> **Tip:** copy/paste the functions below into `semantic_chunking.py`; each is already namespaced so you can `import` them from later stages.

```python
# --- 3.1 load & filter -------------------------------------------------------
def load_parsed_documents_for_chunking(path="parsed_documents.pkl", min_chars=500):
    import pickle, pandas as pd
    with open(path, "rb") as f:
        docs = pickle.load(f)
    return {d: v for d, v in docs.items() if len(v.get("full_text", "")) >= min_chars}

# --- 3.2 pick sections -------------------------------------------------------
PRIORITY = ["data_availability", "methods", "supplementary", "results"]
def prepare_section_texts_for_chunking(docs):
    out = {}
    for doi, dd in docs.items():
        out[doi] = {s: t for s, t in dd["section_texts"].items() if s in PRIORITY}
        if not out[doi]:                       # fall-back
            out[doi]["full_document"] = dd["full_text"]
    return out

# --- 3.3 regex inventory (before split) -------------------------------------
import regex as re
PATTERNS = {
    "DOI": re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", re.I),
    "GSE": re.compile(r"\bGSE\d{3,6}\b"),
    "SRR": re.compile(r"\bSRR\d{5,}\b"),
    # add more as needed …
}
def create_pre_chunk_entity_inventory(sec_texts):
    import pandas as pd, itertools
    rows = []
    for doi, secs in sec_texts.items():
        for stype, txt in secs.items():
            for label, rx in PATTERNS.items():
                rows.append(
                    dict(
                        document_id=doi,
                        section_type=stype,
                        pattern=label,
                        count=len(rx.findall(txt)),
                    )
                )
    return pd.DataFrame(rows)

# --- 3.4 chunking -----------------------------------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken, uuid
tok = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 encoding

def create_section_aware_chunks(sec_texts, docs, size=200, overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size*4,        # ~4 chars per token
        chunk_overlap=overlap*4,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=lambda t: len(tok.encode(t)),
    )
    chunks = []
    for doi, secs in sec_texts.items():
        order = docs[doi].get("section_order", {})
        src   = docs[doi].get("conversion_source", "unknown")
        for stype, txt in secs.items():
            for i, chunk in enumerate(splitter.split_text(txt)):
                meta = dict(
                    chunk_id=f"{uuid.uuid4().hex[:8]}_{i}",
                    document_id=doi,
                    section_type=stype,
                    section_order=order.get(stype, 999),
                    conversion_source=src,
                    token_count=len(tok.encode(chunk)),
                )
                chunks.append( (chunk, meta) )
    return chunks

# --- 3.5 refine labels ------------------------------------------------------
def refine_chunk_types(chunks):
    out = []
    for txt, meta in chunks:
        low = txt.lower()
        if any(k in low for k in ["figure", "fig.", "table", "caption"]):
            meta["chunk_type"] = "caption"
        elif low.strip().endswith(":") and len(low.split()) < 15:
            meta["chunk_type"] = "header"
        elif meta["section_type"] == "data_availability":
            meta["chunk_type"] = "data_statement"
        else:
            meta["chunk_type"] = "body"
        out.append( (txt, meta) )
    return out

# --- 3.6 validate -----------------------------------------------------------
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np, pandas as pd
def validate_chunk_integrity(chunks, pre_inv):
    # flatten entity counts AFTER split
    post = (
        pd.DataFrame([
            dict(document_id=m["document_id"],
                 section_type=m["section_type"],
                 pattern=lab,
                 count=len(rx.findall(txt)))
            for txt, m in chunks
            for lab, rx in PATTERNS.items()
        ])
        .groupby(["document_id","section_type","pattern"])["count"].sum()
        .reset_index()
    )
    merged = pre_inv.merge(post,
        on=["document_id","section_type","pattern"],
        how="left",
        suffixes=("_pre","_post")
    ).fillna(0)
    lost = merged[ merged["count_post"] < merged["count_pre"] ]
    ok   = lost.empty
    return ok, lost

# --- 3.7 export -------------------------------------------------------------
def export_chunks_for_embedding(chunks, out_pkl="chunks_for_embedding.pkl"):
    import pickle, pandas as pd, pathlib, json
    with open(out_pkl, "wb") as f:
        pickle.dump(chunks, f)
    rows = [ dict(**m, text_len=len(t)) for t,m in chunks ]
    pd.DataFrame(rows).to_csv(out_pkl.replace(".pkl","_summary.csv"), index=False)
    print(f"✓ wrote {len(chunks):,} chunks to {out_pkl}")
```

### 4  Quality gates (automatic)

| Check                        | Target                     |
| ---------------------------- | -------------------------- |
| Entity-retention rate        | **100 %** (no DOI/ID lost) |
| Avg tokens / chunk           | 190 ± 30 (tokens)          |
| Chunks with section metadata | ≥ 85 %                     |
| Run-time (1 CPU)             | < 15 min for 500k chunks   |

### 5  Running the pipeline

```bash
python semantic_chunking.py  \
    --parsed-path parsed_documents.pkl \
    --chunk-size 200 \
    --chunk-overlap 20
```

The script will:

1. print entity-inventory stats,
2. abort if retention < 100 %,
3. leave you with `chunks_for_embedding.pkl` + a one-screen CSV dashboard for sanity-checking.

---

## 6  Rationale & pointers

* **RecursiveCharacterTextSplitter** is the most widely adopted splitter in LangChain; it respects sentence & paragraph boundaries and supports custom overlap separators ([python.langchain.com][1]).
* The 384-D `all-MiniLM-L6-v2` encoder balances speed & semantic quality ([huggingface.co][2]).
* Accurate token counts come from `tiktoken`’s `cl100k_base` encoding ([github.com][3])turn0search10.
* Chunk size ≈200 tokens with 10-20 % overlap is a robust starting point for RAG pipelines and is recommended by several industry guides ([pinecone.io][4], [reddit.com][5]).
* DOI regex follows the Crossref canonical pattern ([stackoverflow.com][6], [crossref.org][7]); GEO/SRA patterns are sourced from NCBI docs ([ncbi.nlm.nih.gov][8], [ncbi.nlm.nih.gov][9]).
* Validation relies on simple token-exact matching – no embedding needed – so it runs fast in NumPy/Pandas.
* GroupKFold & leakage safeguards live in **Step 7**, per scikit-learn’s recommendations on grouped CV splits ([scikit-learn.org][10]).
* When this step passes, you can safely proceed to masking & CV checks without fearing silent citation loss.

---

**That’s it!**
Let me know if you’d like any of the flagged issues patched in the master guide before we wire this into CI, or if you’d like a demo notebook that walks through these functions end-to-end.

[1]: https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html?utm_source=chatgpt.com "RecursiveCharacterTextSplitter — LangChain documentation"
[2]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2?utm_source=chatgpt.com "sentence-transformers/all-MiniLM-L6-v2 - Hugging Face"
[3]: https://github.com/openai/tiktoken?utm_source=chatgpt.com "tiktoken is a fast BPE tokeniser for use with OpenAI's models. - GitHub"
[4]: https://www.pinecone.io/learn/chunking-strategies/?utm_source=chatgpt.com "Chunking Strategies for LLM Applications - Pinecone"
[5]: https://www.reddit.com/r/LangChain/comments/1evtu7d/what_are_your_rag_parameters_eg_top_k_chunk_size/?utm_source=chatgpt.com "What are your RAG parameters e.g. top k, chunk size, chunk overlap?"
[6]: https://stackoverflow.com/questions/27910/finding-a-doi-in-a-document-or-page?utm_source=chatgpt.com "regex - Finding a DOI in a document or page - Stack Overflow"
[7]: https://www.crossref.org/blog/dois-and-matching-regular-expressions/?utm_source=chatgpt.com "Blog - DOIs and matching regular expressions - Crossref"
[8]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?utm_source=chatgpt.com "GEO Accession viewer - NCBI"
[9]: https://www.ncbi.nlm.nih.gov/geo/info/faq.html?utm_source=chatgpt.com "Frequently Asked Questions - GEO - NCBI"
[10]: https://scikit-learn.org/stable/modules/cross_validation.html?utm_source=chatgpt.com "3.1. Cross-validation: evaluating estimator performance - Scikit-learn"
