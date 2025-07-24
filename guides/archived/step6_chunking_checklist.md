## Implementation Checklist for Step 6 - Semantic Chunking Pipeline

### 🔧 **Phase 1: Environment Setup & Dependencies**

**1.1 Update Dependencies**
- [ ] Update `pyproject.toml` with required packages:
  ```toml
  dependencies = [
    # ... existing dependencies ...
    "langchain>=0.3.0,<0.4.0",              # RecursiveCharacterTextSplitter
    "tiktoken>=0.8.0,<0.9.0",                # exact GPT-token counts
    "pydantic>=2.0.0",                       # strict chunk/metadata schemas
    "scikit-learn>=1.3.0",                  # cosine similarity for QC
    # Note: regex package already in root pyproject.toml - used for entity patterns & Unicode strip
    # Note: sentence-transformers removed due to torch compatibility issues on macOS x86_64
  ]
  ```
- [ ] Run `uv sync` to install new dependencies # remember to activate virtual environment before running `uv sync`
- [ ] Test imports in Python environment

**1.2 Create Core Data Models**
- [x] Create `src/models.py` with Pydantic models:
  - [x] `ChunkMetadata` BaseModel with fields: chunk_id, document_id, section_type, section_order, conversion_source, token_count, chunk_type
  - [x] `Chunk` BaseModel with fields: text, metadata
  - [x] `ChunkingResult` BaseModel for pipeline output

Pydantic Data Models for Semantic Chunking:

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class Section(BaseModel):
    page_start: int
    page_end: Optional[int] = None
    section_type: Optional[str] = None  # "methods", "results", "data_availability", etc.
    subsections: Optional[List[str]] = []

class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str  # DOI from step 5
    section_type: Optional[str] = None        # Primary section containing this chunk
    section_order: Optional[int] = None       # Order within document sections
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    chunk_type: Optional[str] = None          # "body", "header", "caption"
    conversion_source: Optional[str] = None   # "grobid", "pdfplumber", "existing_xml"
    token_count: Optional[int] = None
    citation_entities: Optional[List[str]] = []  # Entities found in this chunk

class Chunk(BaseModel):
    chunk_id: str
    text: str
    score: Optional[float] = None       # similarity score (added later)
    chunk_metadata: ChunkMetadata
    
    def __str__(self):
        return f"Chunk({self.chunk_id[:8]}..., {len(self.text)} chars, {self.chunk_metadata.section_type})"
```

### 🏗️ **Phase 2: Address Contradictions (Part A Issues)**

**2.1 Fix Section Model Consistency**
- [x] Review `src/section_mapping.py` and ensure it uses Pydantic BaseModel consistently
- [x] Update any references to `section_primary.*` to use flat `section_type` field
- [x] Test compatibility with existing Step 5 output
- [x] Added `section_texts` and `section_order` fields to Step 5 output for Step 6 compatibility

**2.2 Separate Chunking from Masking**
- [x] Ensure Step 6 only handles chunking, not ID masking
- [x] Plan masking functionality for Step 7 (separate implementation)

### 📝 **Phase 3: Implement Core Chunking Pipeline**

**3.1 Create Main Chunking Module**
- [x] Create `src/semantic_chunking.py` with the following functions:

**3.2 Data Loading Functions**
- [x] `load_parsed_documents_for_chunking()`:
  - [x] Load from `Data/train/parsed/parsed_documents.pkl` (Step 5 output)
  - [x] Strip unicode control chars: `text = regex.sub(r'\p{C}', '', text)` (prevents tiktoken crashes)
  - [x] Skip docs with empty XML bodies; log count
  - [x] Filter documents with min_chars=500
  - [x] Return filtered document dictionary
  - [x] Add progress logging

**3.3 Section Preparation Functions**
- [x] `prepare_section_texts_for_chunking()`:
  - [x] Priority sections: `["data_availability", "methods", "supplementary", "results"]`
  - [x] Extract section_texts from parsed documents
  - [x] Fall back to full_text if no priority sections found
  - [x] For fallback cases, add `full_document` section type to ensure metadata coverage ≥ 85%
  - [x] Preserve section order information

**3.4 Entity Inventory Functions**
- [x] `create_pre_chunk_entity_inventory()`:
  - [x] Define regex patterns for DOI, GSE, SRR, etc.
  - [x] Count entities per document/section BEFORE chunking
  - [x] Return pandas DataFrame with counts
  - [x] Use patterns from chunk guide:
    ```python
    PATTERNS = {
    # ――― existing patterns ―――
    'DOI':          re.compile(r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b', re.IGNORECASE),
    'GEO_Series':   re.compile(r'\bGSE\d{3,6}\b'),      # GEO series   
    'GEO_Sample':   re.compile(r'\bGSM\d{3,6}\b'),      # GEO sample   
    'SRA_Run':      re.compile(r'\bSRR\d{5,}\b'),       # SRA run      
    'PDB_ID':       re.compile(r'\b[A-Za-z0-9]{4}\b'),
    'PDB_DOI':      re.compile(r'\b10\.2210/pdb[A-Za-z0-9]{4}/pdb\b', re.IGNORECASE),
    'ArrayExpress': re.compile(r'\bE-[A-Z]+-\d+\b'),    # E-MTAB, E-GEOD …  
    'dbGaP':        re.compile(r'\bphs\d{6}\b'),
    'TCGA':         re.compile(r'\bTCGA-[A-Z0-9-]+\b'),
    'ENA_Project':  re.compile(r'\bPRJ[EDN][A-Z]\d+\b'),
    'ENA_Study':    re.compile(r'\bERP\d{6,}\b'),       # already present
    'ENA_Sample':   re.compile(r'\bSAM[EDN][A-Z]?\d+\b'),

    # ――― new additions ―――
    # NCBI Sequence Read Archive (hierarchy)                     
    'SRA_Experiment': re.compile(r'\bSRX\d{5,}\b'),     # SRX   
    'SRA_Project':    re.compile(r'\bSRP\d{5,}\b'),     # SRP   
    'SRA_Sample':     re.compile(r'\bSRS\d{5,}\b'),     # SRS   
    'SRA_Study':      re.compile(r'\bSRA\d{5,}\b'),     # umbrella study

    # RefSeq & GenBank
    'RefSeq_Chromosome': re.compile(r'\bNC_\d{6,}(?:\.\d+)?\b'),  # e.g. NC_000913.3

    # EMBL-EBI / ENA prefixes
    'ENA_Run':        re.compile(r'\bERR\d{6,}\b'),     # ENA run
    'ENA_Experiment': re.compile(r'\bERX\d{6,}\b'),     # ENA experiment
    'ENA_Sample2':    re.compile(r'\bERS\d{6,}\b'),     # ENA sample

    # DDBJ (mirrors ENA formats)
    'DDBJ_Run':        re.compile(r'\bDRR\d{6,}\b'),    # DRR   
    'DDBJ_Experiment': re.compile(r'\bDRX\d{6,}\b'),    # DRX

    # ENCODE Project
    'ENCODE_Assay': re.compile(r'\bENCSR[0-9A-Z]{6}\b'),  # ENCSR…

    # PRIDE ProteomeXchange
    'PRIDE': re.compile(r'\bPXD\d{6,}\b'),               # PXD…
}
    ```

**3.5 Core Chunking Function**
- [x] `create_section_aware_chunks()`:
  - [x] Use `RecursiveCharacterTextSplitter` from langchain
  - [x] Parameters: chunk_size=200*4 chars, chunk_overlap=20*4 chars (note: chunk_size is characters, not tokens - LangChain semantics)
  - [x] Use `tiktoken.get_encoding("cl100k_base")` for token counting
  - [x] Wrap chunk loop with `tqdm` progress bar for developer feedback during long runs
  - [x] Generate unique chunk_id for each chunk
  - [x] Preserve section metadata (type, order, conversion_source)
  - [x] Return list of (text, metadata) tuples

**3.5-b Adjacent Chunk Linking**
- [x] `link_adjacent_chunks()`:
  - [x] Run immediately after `create_section_aware_chunks()`
  - [x] Populate `previous_chunk_id` and `next_chunk_id` for every chunk
  - [x] Skip first/last chunks → set pointer to `None`
  - [x] Optionally store `chunk_idx` in metadata (implicit index) for future experiments
  - [x] Enables neighbour-retrieval strategy for downstream processing

**3.6 Chunk Type Refinement**
- [x] `refine_chunk_types()`:
  - [x] Detect "caption" chunks (figure, fig., table, caption keywords)
  - [x] Detect "header" chunks (ends with ":", <15 words)
  - [x] Label data_availability sections as "data_statement"
  - [x] Default to "body" for other chunks

**3.7 Validation Functions**
- [x] `validate_chunk_integrity()`:
  - [x] Count entities AFTER chunking
  - [x] Compare with pre-chunking inventory
  - [x] Report any lost entities
  - [x] Return validation status and loss report
  - [x] Target: 100% entity retention
  - [x] On validation failure, add hint: "Rerun with `--repair` to auto-tune overlap"

#### Step-by-step code stencil

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
    # ――― existing patterns ―――
    'DOI':          re.compile(r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b', re.IGNORECASE),
    'GEO_Series':   re.compile(r'\bGSE\d{3,6}\b'),      # GEO series   
    'GEO_Sample':   re.compile(r'\bGSM\d{3,6}\b'),      # GEO sample   
    'SRA_Run':      re.compile(r'\bSRR\d{5,}\b'),       # SRA run      
    'PDB_ID':       re.compile(r'\b[A-Za-z0-9]{4}\b'),
    'PDB_DOI':      re.compile(r'\b10\.2210/pdb[A-Za-z0-9]{4}/pdb\b', re.IGNORECASE),
    'ArrayExpress': re.compile(r'\bE-[A-Z]+-\d+\b'),    # E-MTAB, E-GEOD …  
    'dbGaP':        re.compile(r'\bphs\d{6}\b'),
    'TCGA':         re.compile(r'\bTCGA-[A-Z0-9-]+\b'),
    'ENA_Project':  re.compile(r'\bPRJ[EDN][A-Z]\d+\b'),
    'ENA_Study':    re.compile(r'\bERP\d{6,}\b'),       # already present
    'ENA_Sample':   re.compile(r'\bSAM[EDN][A-Z]?\d+\b'),

    # ――― new additions ―――
    # NCBI Sequence Read Archive (hierarchy)                     
    'SRA_Experiment': re.compile(r'\bSRX\d{5,}\b'),     # SRX   
    'SRA_Project':    re.compile(r'\bSRP\d{5,}\b'),     # SRP   
    'SRA_Sample':     re.compile(r'\bSRS\d{5,}\b'),     # SRS   
    'SRA_Study':      re.compile(r'\bSRA\d{5,}\b'),     # umbrella study

    # RefSeq & GenBank
    'RefSeq_Chromosome': re.compile(r'\bNC_\d{6,}(?:\.\d+)?\b'),  # e.g. NC_000913.3

    # EMBL-EBI / ENA prefixes
    'ENA_Run':        re.compile(r'\bERR\d{6,}\b'),     # ENA run
    'ENA_Experiment': re.compile(r'\bERX\d{6,}\b'),     # ENA experiment
    'ENA_Sample2':    re.compile(r'\bERS\d{6,}\b'),     # ENA sample

    # DDBJ (mirrors ENA formats)
    'DDBJ_Run':        re.compile(r'\bDRR\d{6,}\b'),    # DRR   
    'DDBJ_Experiment': re.compile(r'\bDRX\d{6,}\b'),    # DRX

    # ENCODE Project
    'ENCODE_Assay': re.compile(r'\bENCSR[0-9A-Z]{6}\b'),  # ENCSR…

    # PRIDE ProteomeXchange
    'PRIDE': re.compile(r'\bPXD\d{6,}\b'),               # PXD…
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

def link_adjacent_chunks(chunks):
    # chunks is a list of (text, meta) in document-order
    by_doc = defaultdict(list)
    for txt, meta in chunks:
        by_doc[meta["document_id"]].append((txt, meta))

    linked = []
    for doc, items in by_doc.items():
        for i, (txt, meta) in enumerate(items):
            prev_id = items[i-1][1]["chunk_id"] if i > 0 else None
            next_id = items[i+1][1]["chunk_id"] if i < len(items)-1 else None
            meta["previous_chunk_id"] = prev_id
            meta["next_chunk_id"] = next_id
            linked.append((txt, meta))
    return linked

# Call `link_adjacent_chunks` right before export_chunks_for_embedding() so the pointers are written to disk. The first and last chunk in each document will simply carry None on one side.

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

### 🎯 **Phase 4: Pipeline Integration & Export**

**4.1 Export Functions**
- [x] `export_chunks_for_embedding()`:
  - [x] Save chunks to `chunks_for_embedding.pkl`
  - [x] Generate summary CSV with chunk statistics
  - [x] Include metadata: chunk_id, document_id, section_type, token_count, chunk_type, text_length
  - [x] Ensure both PKL and summary CSV include `previous_chunk_id` and `next_chunk_id` fields

**4.2 Main Pipeline Function**
- [x] `run_semantic_chunking_pipeline()`:
  - [x] Orchestrate all steps in sequence
  - [x] Handle error cases and logging
  - [x] Validate quality gates
  - [x] Export results

### 📊 **Phase 5: Quality Assurance**

**5.1 Implement Quality Gates**
- [x] Entity retention rate: **100%** (abort if < 100%)
- [x] Average tokens per chunk: 190 ± 30
- [x] Chunks with section metadata: ≥ 85%
- [x] Pointer integrity: ≥ 99.9% non-null for interior chunks (log count of first/last chunks excluded)
- [x] Runtime performance: < 15 min for 500k chunks

**5.2 Add Comprehensive Logging**
- [x] Progress indicators for each step
- [x] Statistics reporting (chunk counts, token distributions)
- [x] Error handling and recovery
- [x] Performance metrics

### 🔧 **Phase 6: Testing & Integration**

**6.1 Unit Tests**
- [x] Create `tests/test_semantic_chunking.py`:
  - [ ] Test each function with sample data
  - [ ] Test entity retention validation
  - [ ] Test chunk size distributions
  - [ ] Test error handling
  - [ ] Test `link_adjacent_chunks()` sets correct pointer chain on toy doc of 3 chunks

**6.2 Integration Testing**
- [ ] Test with actual Step 5 output (`parsed_documents.pkl`)
- [ ] Verify compatibility with existing data structures
- [ ] Test performance with full dataset

**6.3 CLI Interface**
- [x] Create command-line script `scripts/run_chunking_pipeline.py`:
  ```python
  python scripts/run_chunking_pipeline.py \
      --input-path Data/train/parsed/parsed_documents.pkl \
      --output-path chunks_for_embedding.pkl \
      --chunk-size 200 \
      --chunk-overlap 20
  ```

**6.4 Configuration Management (Optional)**
*Enables cleaner parameter management and experimentation*
- [ ] Consider centralizing tunables in `chunking_config.yaml`:
  - [ ] chunk_size, chunk_overlap, min_chars thresholds
  - [ ] Priority section lists and entity patterns
  - [ ] Quality gate thresholds and validation settings

### 📋 **Phase 7: Documentation & Validation**

**7.1 Update Documentation**
- [ ] Create `docs/step6_chunking_guide.md`
- [ ] Document parameter tuning guidelines
- [ ] Add troubleshooting section
- [ ] Add mini-section: *"Neighbour-retrieval rationale & usage pattern"* with code snippet

**7.2 Final Validation**
- [ ] Run full pipeline on training data
- [ ] Verify output format matches Step 7 requirements
- [ ] Confirm all quality gates pass
- [ ] Generate pipeline summary report

### 🏃 **Phase 8: Execution Priority**

**Immediate Next Steps (Priority 1):**
1. Update `pyproject.toml` with new dependencies
2. ✅ Create `src/models.py` with Pydantic models
3. ✅ Implement `load_parsed_documents_for_chunking()` function
4. Test with existing Step 5 output

**Critical Path Items:**
- ✅ Entity inventory and validation (failure = abort)
- ✅ Chunk size and token counting accuracy
- ✅ Section metadata preservation
- ✅ Integration with existing document structure

### 🔍 **Key Integration Points**

Based on existing code structure:
- **Input**: `Data/train/parsed/parsed_documents.pkl` (from Step 5)
- **Output**: `chunks_for_embedding.pkl` + `chunks_for_embedding_summary.csv`
- **Dependencies**: Existing document parsing and section mapping modules
- **Data Flow**: parsed_documents → section_texts → entity_inventory → chunks → validation → export

This checklist builds directly on existing Step 5 implementation and maintains compatibility with current data structures. The chunking pipeline will consume the validated documents from Step 5 and produce the chunk format needed for Step 7.

## 🎯 **IMPLEMENTATION In-Progress** 

All phases have been successfully implemented:

(Incomplete) **Phase 1**: Environment Setup & Dependencies (torch issue documented)
✅ **Phase 2**: Address Contradictions & Compatibility  
✅ **Phase 3**: Core Chunking Pipeline Implementation
✅ **Phase 4**: Pipeline Integration & Export
✅ **Phase 5**: Quality Assurance & Logging
(Incomplete) **Phase 6**: Testing & Integration (tests created but not run)
**Phase 7**: Documentation & Validation (not started; checklist =/= documentation)

**🚀 NOT Ready to Execute (need to first resolve dependency issue & re-run step 5):**
```bash
python scripts/run_chunking_pipeline.py --help
```
### 🚨 **TROUBLESHOOTING SECTION**

**Issue 1: Torch Dependency Platform Compatibility**
- **Problem**: `torch` dependency doesn't have compatible wheels for macOS x86_64 platform
- **Error**: `Distribution torch==2.7.1 @ registry+https://pypi.org/simple can't be installed because it doesn't have a source distribution or wheel for the current platform`
- **Platform**: macOS (`macosx_10_16_x86_64`)
- **Attempts Made**:
  1. **First Attempt**: Constrained `sentence-transformers` version to `>=2.0.0,<2.8.0` to avoid pulling in incompatible torch versions
  2. **Second Attempt**: Added explicit `torch` constraint `>=2.0.0,<2.7.0` to force a compatible version
- **Status**: Both attempts failed - torch 2.6.0 also lacks wheels for this platform
- **Next Steps**: May need to use CPU-only torch version or find alternative approach for embeddings
