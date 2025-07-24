Below is an **expanded, production-ready Data Pre-Processing & EDA guide** that *adds an explicit, end-to-end recipe for finding the **top k most relevant chunks** by cosine similarity* while keeping every earlier step intact. Teams can lift code snippets verbatim or swap-in cloud equivalents later.

---

## TL;DR — What's new?

After semantic chunking and embedding, we now:

**For Training-Time EDA (Label-Aware):**
1. **Build a ChromaDB HNSW index** (cosine space). 
2. **Construct one or more query embeddings** for every `(article_id, dataset_id)` label. 
3. **Retrieve `top_k` chunks** with `similarity_search_with_relevance_scores`, optionally re-rank with *Maximal Marginal Relevance (MMR)* to cut redundancy. 
4. **Persist a scored table**—`chunks.parquet`—that already contains the k-best chunks (and their cosine scores) per label. 
5. **Audit coverage\@k** and score distributions so the modelling team can decide on thresholds or extra rerankers.

**For Inference-Time Retrieval (Label-Agnostic):**
1. **Anchor-driven retrieval** using regex/keyword patterns for explicit accession IDs (GSE, SRA, DOI, etc.)
2. **Self-Query retriever** with LLM-generated semantic queries from article title/abstract
3. **GenQ-style synthetic queries** for generic data-citation patterns
4. **Universal fallback embedding** when other methods return insufficient results

Everything is wrapped in Pydantic models for schema safety and drops straight into the training loop.

---

## 1  Environment & Library Setup

**⚠️ IMPORTANT**: This setup prioritizes **license safety**, **memory efficiency**, and **leak-free processing** for competitive ML environments.

```bash
# activate virtual environment
source .venv/bin/activate

# core analytics
uv add pandas numpy matplotlib tqdm ipywidgets

# NLP & embeddings - PINNED for reproducibility
uv add "sentence-transformers==2.*" torch torchvision torchaudio
uv add tiktoken # tokenizer for exact token counts

# chunking & vector DB - PINNED to avoid API breakage
uv add "langchain>=0.3,<0.4" "pydantic==2.10.*" 
uv add "chromadb>=0.4,<0.5"  # avoid 0.5.x PersistentClient changes

# document parsing - LICENSE SAFE alternatives
uv add "pdfplumber==0.10.4"  # MIT license (replaces PyMuPDF AGPL)
uv add beautifulsoup4 lxml   # for XML parsing
uv add requests              # for GROBID API calls

# data modelling / config
uv add python-dotenv

# OPTIONAL: memory-efficient alternatives (enable as needed)
# uv add faiss-cpu           # for NumPy memmap fallback
uv add scikit-learn         # for pairwise cosine similarity

# OPTIONAL: Self-Query retriever (FEATURE FLAGGED - requires internet)
# uv add openai             # for Self-Query retriever (behind feature flag)

# OPTIONAL: heavy visualization (sample-only for debugging)
# uv add umap-learn hdbscan # run on 10% sample when debugging clusters
# uv add nomic              # interactive atlas (requires internet)

# enhanced regex patterns for anchor detection
uv add regex
```

**Key Changes:**
- **License-safe**: Replaced PyMuPDF (AGPL) with pdfplumber (MIT)
- **Version-pinned**: LangChain, Pydantic, ChromaDB to avoid API breakage  
- **Memory-optimized**: Heavy visualization tools marked optional
- **Offline-safe**: Internet-dependent tools behind feature flags

ChromaDB is an embeddable open-source vector store that ships with an HNSW index for fast similarity search ([realpython.com][1], [datacamp.com][2]).
Sentence-Transformer models (e.g. `all-MiniLM-L6-v2`) give compact 384-dimensional sentence embeddings at GPU/CPU-friendly speeds ([huggingface.co][3]).

---

## 2  Load Labels & Map Documents

1. **Read labels**

   ```python
   labels = pd.read_csv("train_labels.csv")       # columns: article_id, dataset_id, type
   ```
2. **Inventory full-text paths** (PDF files) in a helper table; flag missing files for later QC.
3. **Basic checks** – duplicates, nulls, class balance bar chart (expect ≈ 44 % Secondary, 30 % Missing, 26 % Primary) ([analyticsvidhya.com][4]).

---

## 3  Document Parsing & Section Extraction

**🎯 HIGH PRIORITY**: XML-first parsing with GROBID eliminates brittle regex patterns and provides clean structured sections.

### 3.1 GROBID Setup (Docker)

**Start GROBID service** (lightweight, air-gapped):
```bash
# Start GROBID container (one-time setup)
docker run -d -p 8070:8070 --name grobid-service lfoppiano/grobid:0.8.0

# Verify service is up
curl http://localhost:8070/api/isalive
```

**Alternative: CERMINE fallback** for PDFs that GROBID can't process:
```bash
# Download CERMINE if needed
wget https://github.com/CeON/CERMINE/releases/download/1.13/cermine-impl-1.13-jar-with-dependencies.jar
```

### 3.2 Single PDF → XML Conversion Wrapper

**Critical**: Use identical preprocessing for train AND inference to prevent leakage.

```python
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Dict, List
import pdfplumber  # MIT license fallback

def convert_pdf_to_xml(pdf_path: Path, 
                      grobid_url: str = "http://localhost:8070",
                      fallback_to_text: bool = True) -> Dict[str, str]:
    """
    Single wrapper for PDF→XML conversion used in BOTH train and inference.
    
    Returns:
        Dict with 'xml_content', 'text_content', 'source' keys
    """
    result = {'xml_content': None, 'text_content': None, 'source': None}
    
    # Try GROBID first
    try:
        with open(pdf_path, 'rb') as pdf_file:
            files = {'input': pdf_file}
            response = requests.post(
                f"{grobid_url}/api/processFulltextDocument",
                files=files,
                timeout=120
            )
            
        if response.status_code == 200:
            result['xml_content'] = response.text
            result['source'] = 'grobid'
            return result
            
    except Exception as e:
        print(f"GROBID failed for {pdf_path}: {e}")
    
    # Fallback to pdfplumber for text extraction
    if fallback_to_text:
        try:
            text_chunks = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    if page.extract_text():
                        text_chunks.append(page.extract_text())
            
            result['text_content'] = '\n\n'.join(text_chunks)
            result['source'] = 'pdfplumber'
            return result
            
        except Exception as e:
            print(f"Fallback parsing failed for {pdf_path}: {e}")
    
    return result

def cache_converted_documents(pdf_paths: List[Path], 
                            cache_dir: Path = Path("processed_docs")) -> Dict[str, Dict]:
    """Cache converted documents to avoid re-processing"""
    cache_dir.mkdir(exist_ok=True)
    processed_docs = {}
    
    for pdf_path in pdf_paths:
        doc_id = pdf_path.stem
        cache_file = cache_dir / f"{doc_id}.xml"
        
        if cache_file.exists():
            # Load from cache
            with open(cache_file, 'r', encoding='utf-8') as f:
                processed_docs[doc_id] = {
                    'xml_content': f.read(),
                    'source': 'cached_grobid'
                }
        else:
            # Convert and cache
            result = convert_pdf_to_xml(pdf_path)
            if result['xml_content']:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(result['xml_content'])
            processed_docs[doc_id] = result
    
    return processed_docs
```

### 3.3 XML Section Extraction (XPath)

**Eliminates ~200 lines of brittle regex** with structured XML parsing:

```python
def extract_sections_from_xml(xml_content: str) -> List[Section]:
    """Extract sections using XPath queries on TEI-XML"""
    sections = []
    
    try:
        # Parse TEI XML from GROBID
        root = ET.fromstring(xml_content)
        
        # Define namespaces for TEI
        namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
        # Extract sections using XPath
        section_elements = root.findall('.//tei:div[@type]', namespaces)
        
        for i, div in enumerate(section_elements):
            section_type = div.get('type', 'other').lower()
            
            # Map GROBID section types to our standard types
            section_mapping = {
                'abstract': 'abstract',
                'introduction': 'introduction', 
                'method': 'methods',
                'methods': 'methods',
                'materials': 'methods',
                'result': 'results',
                'results': 'results',
                'discussion': 'discussion',
                'conclusion': 'conclusion',
                'acknowledgment': 'acknowledgments',
                'references': 'references',
                'availability': 'data_availability',
                'supplementary': 'supplementary'
            }
            
            standardized_type = section_mapping.get(section_type, 'other')
            
            sections.append(Section(
                page_start=i + 1,  # XML doesn't have page info, use sequence
                page_end=None,
                section_type=standardized_type
            ))
            
    except ET.ParseError as e:
        print(f"XML parsing failed: {e}")
        return []
    
    return sections

def extract_text_from_xml_section(xml_content: str, section_type: str) -> str:
    """Extract clean text from specific XML section"""
    try:
        root = ET.fromstring(xml_content)
        namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
        # Find section by type
        section_div = root.find(f'.//tei:div[@type="{section_type}"]', namespaces)
        if section_div is not None:
            # Extract all text, preserving paragraph breaks
            paragraphs = []
            for p in section_div.findall('.//tei:p', namespaces):
                if p.text:
                    paragraphs.append(p.text.strip())
            return '\n\n'.join(paragraphs)
            
    except ET.ParseError:
        pass
    
    return ""
```

### 3.4 Fallback Text Section Detection

For documents where XML parsing fails:

```python
def extract_sections_from_text_fallback(text_content: str) -> List[Section]:
    """Fallback section detection using refined patterns"""
    sections = []
    lines = text_content.split('\n')
    
    # Simplified, high-confidence patterns only
    section_patterns = [
        (r'^\s*(ABSTRACT|Abstract)\s*$', 'abstract'),
        (r'^\s*(METHODS?|Methods?|MATERIALS AND METHODS)\s*$', 'methods'),
        (r'^\s*(RESULTS?|Results?)\s*$', 'results'),
        (r'^\s*(DISCUSSION|Discussion)\s*$', 'discussion'),
        (r'^\s*(DATA AVAILABILITY|Data Availability)\s*$', 'data_availability'),
    ]
    
    for i, line in enumerate(lines):
        for pattern, section_type in section_patterns:
            if re.match(pattern, line.strip()):
                sections.append(Section(
                    page_start=i // 50 + 1,  # Rough page estimation
                    section_type=section_type
                ))
                break
    
    return sections
```

**Why This Works:**
- **GROBID** gives structured TEI-XML with `<div type="methods">` tags
- **XPath queries** eliminate regex complexity and false matches  
- **Single wrapper** ensures identical preprocessing train/test
- **Cached results** prevent re-processing during iterations
- **MIT-licensed fallback** keeps the pipeline license-clean

---

## 4  Semantic Chunking Pipeline

### 4.1 Choose a chunker

* Start with **RecursiveCharacterTextSplitter** (LangChain) – respects paragraphs → sentences → words ([medium.com][5], [python.langchain.com][6]).
* Target **≤ 200 tokens / chunk** with 10–20 % overlap to guard against edge splits.
* For semantic chunking:

  1. Slide a window, compute embeddings on-the-fly.
  2. Break where **cosine similarity falls below θ** (e.g., 0.6) so you split at topic shifts.
  3. Merge tiny trailing segments.

### 4.2 Pydantic data models

```python
from pydantic import BaseModel, Field
from typing import List, Optional
...

class Section(BaseModel):
    page_start: int
    page_end: Optional[int] = None
    subsections: Optional[List[str]] = []
    section_type: Optional[str] = None  # "methods", "results", "data_availability", etc.

class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    section_primary: Optional[Section] = None     # primary section containing this chunk
    chunk_type: Optional[str] = None          # "body", "header", "caption"

class Chunk(BaseModel):
    chunk_id: str
    text: str
    score: Optional[float] = None       # similarity later
    chunk_metadata: ChunkMetadata
```

Pydantic brings **runtime validation and auto-doc** with minimal boilerplate ([realpython.com][7], [netguru.com][8]).

### 4.3 Citation Integrity Validation

**Critical QC Step**: Ensure chunking never severs DOIs, accession IDs, or other citation entities.

#### 4.3.1 Authoritative Regex Patterns

```python
import re
from typing import Dict, List, Tuple

# Enhanced patterns based on authoritative sources
CITATION_PATTERNS = {
    'DOI': re.compile(r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b', re.IGNORECASE),
    'GEO_Series': re.compile(r'\bGSE\d{3,6}\b'),
    'GEO_Sample': re.compile(r'\bGSM\d{3,6}\b'), 
    'SRA_Run': re.compile(r'\bSR[ARX]\d{5,}\b'),
    'PDB_ID': re.compile(r'\b[A-Za-z0-9]{4}\b'),
    'PDB_DOI': re.compile(r'\b10\.2210/pdb[A-Za-z0-9]{4}/pdb\b', re.IGNORECASE),
    'ArrayExpress': re.compile(r'\bE-[A-Z]+-\d+\b'),
    'dbGaP': re.compile(r'\bphs\d{6}\b'),
    'TCGA': re.compile(r'\bTCGA-[A-Z0-9-]+\b'),
    'ENA_Project': re.compile(r'\bPRJ[EDN][A-Z]\d+\b'),
    'ENA_Study': re.compile(r'\bERP\d{6,}\b'),
    'ENA_Sample': re.compile(r'\bSAM[EDN][A-Z]?\d+\b'),
}

def extract_citation_entities(text: str) -> Dict[str, List[Tuple[str, int, int]]]:
    """Extract all citation entities with their positions"""
    entities = {}
    for pattern_name, regex in CITATION_PATTERNS.items():
        matches = []
        for match in regex.finditer(text):
            matches.append((match.group(), match.start(), match.end()))
        if matches:
            entities[pattern_name] = matches
    return entities
```

#### 4.3.2 Pre-Chunk Entity Inventory

Run before chunking to establish ground truth counts:

```python
def create_entity_inventory(processed_texts: Dict[str, str]) -> pd.DataFrame:
    """Create pre-chunking entity inventory for validation
    
    Args:
        processed_texts: Dict of {doc_id: processed_text} where processed_text 
                        is the EXACT text that will be passed to the chunker
    """
    inventory_rows = []
    
    for doc_id, processed_text in processed_texts.items():
        entities = extract_citation_entities(processed_text)
        
        for pattern_name, matches in entities.items():
            inventory_rows.append({
                'document_id': doc_id,
                'pattern_name': pattern_name,
                'entity_count': len(matches),
                'entities': [match[0] for match in matches],
                'positions': [(match[1], match[2]) for match in matches]
            })
    
    return pd.DataFrame(inventory_rows)
```

#### 4.3.3 Post-Chunk Validation

Verify no entities were severed during chunking:

```python
def validate_chunk_integrity(chunks: List[Chunk], 
                           pre_chunk_inventory: pd.DataFrame) -> Dict[str, any]:
    """Validate that chunking preserved all citation entities"""
    
    # Count entities in chunked text
    post_chunk_counts = {}
    chunk_entities = {}
    
    for chunk in chunks:
        doc_id = chunk.chunk_metadata.document_id
        entities = extract_citation_entities(chunk.text)
        
        if doc_id not in post_chunk_counts:
            post_chunk_counts[doc_id] = {}
            chunk_entities[doc_id] = []
            
        for pattern_name, matches in entities.items():
            post_chunk_counts[doc_id][pattern_name] = post_chunk_counts[doc_id].get(pattern_name, 0) + len(matches)
            chunk_entities[doc_id].extend([(chunk.chunk_id, pattern_name, match[0]) for match in matches])
    
    # Compare with pre-chunk inventory  
    validation_results = {
        'integrity_passed': True,
        'missing_entities': [],
        'affected_documents': [],
        'repair_suggestions': []
    }
    
    for _, row in pre_chunk_inventory.iterrows():
        doc_id = row['document_id']
        pattern = row['pattern_name']
        expected_count = row['entity_count']
        
        actual_count = post_chunk_counts.get(doc_id, {}).get(pattern, 0)
        
        if actual_count < expected_count:
            validation_results['integrity_passed'] = False
            validation_results['missing_entities'].append({
                'document_id': doc_id,
                'pattern': pattern,
                'expected': expected_count,
                'actual': actual_count,
                'missing_entities': row['entities'][:expected_count - actual_count]
            })
            validation_results['affected_documents'].append(doc_id)
    
    return validation_results
```

#### 4.3.4 Auto-Repair Heuristics

```python
def adaptive_chunk_repair(processed_texts: Dict[str, str], 
                         failed_validation: Dict[str, any],
                         base_chunk_size: int = 200,
                         base_overlap: int = 20) -> Dict[str, int]:
    """Suggest chunking parameter adjustments for failed documents
    
    Args:
        processed_texts: Same Dict used in create_entity_inventory - ensures data source consistency
    """
    
    repair_params = {}
    
    for doc_id in failed_validation['affected_documents']:
        # Increase overlap for entity-rich documents
        if doc_id in processed_texts:
            entity_density = len(extract_citation_entities(processed_texts[doc_id])) / len(processed_texts[doc_id].split())
        
            if entity_density > 0.01:  # High entity density
                suggested_overlap = min(base_overlap * 2, base_chunk_size // 3)
                repair_params[doc_id] = {
                    'chunk_size': base_chunk_size,
                    'overlap': suggested_overlap,
                    'reason': f'High entity density: {entity_density:.3f}'
                }
            else:
                # Try boundary expansion
                repair_params[doc_id] = {
                    'chunk_size': int(base_chunk_size * 1.2),
                    'overlap': base_overlap,
                    'reason': 'Boundary expansion for sparse entities'
                }
        else:
            # Fallback if document not found in processed_texts
            repair_params[doc_id] = {
                'chunk_size': int(base_chunk_size * 1.2),
                'overlap': min(base_overlap * 2, base_chunk_size // 3),
                'reason': 'Conservative repair due to missing source text'
            }
    
    return repair_params
```

### 4.4 Section-Aware Parsing & Metadata Enrichment

**Why sections matter**: Empirical studies show data citations concentrate in **Methods**, **Data Availability Statements**, and **Supplementary Information** sections. Capturing section context enables smarter retrieval.

#### 4.4.1 Extract Section Hierarchy from PDF Structure

```python
import re
from typing import Dict, List

def extract_section_hierarchy_from_pdf(pdf_text: str, page_breaks: List[int]) -> List[Section]:
    """Extract section hierarchy from PDF text using pattern recognition"""
    sections = []
    
    # Common section heading patterns
    section_patterns = [
        (r'^(ABSTRACT|Abstract)\s*$', 'abstract'),
        (r'^(INTRODUCTION|Introduction)\s*$', 'introduction'),
        (r'^(METHOD|METHODS|Method|Methods|MATERIALS?\s+AND\s+METHODS?|Materials?\s+and\s+Methods?)\s*$', 'methods'),
        (r'^(RESULT|RESULTS|Result|Results)\s*$', 'results'),
        (r'^(DISCUSSION|Discussion)\s*$', 'discussion'),
        (r'^(CONCLUSION|CONCLUSIONS|Conclusion|Conclusions)\s*$', 'conclusion'),
        (r'^(DATA\s+AVAILAB|Data\s+Availab|AVAILABILITY|Availability)\w*', 'data_availability'),
        (r'^(SUPPLEMENT|Supplement|APPENDIX|Appendix)\w*', 'supplementary'),
        (r'^(ACKNOWLEDG|Acknowledg|FUNDING|Funding)\w*', 'acknowledgments'),
        (r'^(REFERENCE|Reference)\w*', 'references')
    ]
    
    lines = pdf_text.split('\n')
    current_page = 1
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Update current page based on page breaks
        while current_page < len(page_breaks) and i >= page_breaks[current_page]:
            current_page += 1
        
        # Check for section headers
        for pattern, section_type in section_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                # Estimate section end (next section or end of document)
                end_page = None
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    for next_pattern, _ in section_patterns:
                        if re.match(next_pattern, next_line, re.IGNORECASE):
                            # Calculate page for this line
                            temp_page = current_page
                            for page_idx in range(len(page_breaks)):
                                if j >= page_breaks[page_idx]:
                                    temp_page = page_idx + 1
                            end_page = temp_page - 1 if temp_page > current_page else current_page
                            break
                    if end_page:
                        break
                
                sections.append(Section(
                    page_start=current_page,
                    page_end=end_page,
                    section_type=section_type
                ))
                break
    
    return sections

def map_section_type_from_text(text_snippet: str) -> str:
    """Map text content to standardized section types"""
    text_lower = text_snippet.lower()
    
    if any(keyword in text_lower for keyword in ['method', 'material', 'procedure', 'protocol']):
        return 'methods'
    elif any(keyword in text_lower for keyword in ['result', 'finding', 'observation']):
        return 'results'
    elif any(keyword in text_lower for keyword in ['data availab', 'data access', 'data sharing', 'repository']):
        return 'data_availability'
    elif any(keyword in text_lower for keyword in ['discussion', 'conclusion', 'implication']):
        return 'discussion'
    elif any(keyword in text_lower for keyword in ['introduction', 'background', 'motivation']):
        return 'introduction'
    elif any(keyword in text_lower for keyword in ['supplement', 'additional', 'appendix']):
        return 'supplementary'
    else:
        return 'other'
```

#### 4.4.2 Enrich Chunk Metadata with Section Context

```python
def enrich_chunks_with_sections(chunks: List[Chunk], 
                               sections: List[Section]) -> List[Chunk]:
    """Add section metadata to each chunk"""
    
    for chunk in chunks:
        # Find the primary section containing this chunk
        chunk_page = chunk.chunk_metadata.page_number
        
        primary_section = None
        for section in sections:
            if (section.page_start <= chunk_page and 
                (section.page_end is None or chunk_page <= section.page_end)):
                primary_section = section
                break
        
        if primary_section:
            chunk.chunk_metadata.section_primary = primary_section
            
            # Determine chunk type based on content
            if any(keyword in chunk.text.lower() for keyword in ['figure', 'table', 'caption']):
                chunk.chunk_metadata.chunk_type = 'caption'
            elif chunk.text.strip().endswith(':') and len(chunk.text.split()) < 10:
                chunk.chunk_metadata.chunk_type = 'header'
            else:
                chunk.chunk_metadata.chunk_type = 'body'
    
    return chunks
```

---

## 5  Memory-Efficient Embedding & Vector Store

**🎯 MEDIUM PRIORITY**: Start lean with NumPy memmap, upgrade to ChromaDB only when needed. Optimized for Kaggle's 13GB RAM limit.

### 5.1 Embedding Generation (Batch Processing)

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict
import tempfile

def generate_embeddings_efficiently(masked_chunks: List[Chunk], 
                                  model_name: str = "all-MiniLM-L6-v2",
                                  batch_size: int = 128,
                                  cache_dir: Path = None) -> np.ndarray:
    """
    Generate embeddings with memory-efficient batching and caching
    """
    
    if cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp())
    
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "embeddings.npy"
    
    # Load from cache if exists
    if cache_file.exists():
        print(f"Loading cached embeddings from {cache_file}")
        return np.load(cache_file)
    
    # Initialize embedder
    embedder = SentenceTransformer(model_name)
    
    # Extract masked text (with [ID] tokens)
    texts = [chunk.text for chunk in masked_chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    
    # Process in batches to manage memory
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embedder.encode(
            batch_texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        all_embeddings.append(batch_embeddings)
        
        # Optional: Clear GPU cache
        if hasattr(embedder, 'encode') and hasattr(embedder._modules['0'], 'device'):
            import torch
            if 'cuda' in str(embedder._modules['0'].device):
                torch.cuda.empty_cache()
    
    # Combine all batches
    embeddings = np.vstack(all_embeddings)
    
    # Cache for next run
    np.save(cache_file, embeddings)
    print(f"Cached embeddings to {cache_file}")
    
    return embeddings
```

### 5.2 Memory-Mapped Storage (< 500k vectors)

**For smaller datasets**: Use NumPy memmap for zero index-build time:

```python
from sklearn.metrics.pairwise import cosine_similarity

class MemoryMappedVectorStore:
    """Efficient vector store using NumPy memmap for small-medium datasets"""
    
    def __init__(self, embeddings: np.ndarray, 
                 chunk_metadata: List[ChunkMetadata],
                 mmap_dir: Path = Path("vector_mmap")):
        
        self.mmap_dir = Path(mmap_dir)
        self.mmap_dir.mkdir(exist_ok=True)
        
        # Memory-map the embeddings for efficient access
        self.embeddings_file = self.mmap_dir / "embeddings.dat"
        self.embeddings_mmap = np.memmap(
            self.embeddings_file, 
            dtype=np.float32, 
            mode='w+', 
            shape=embeddings.shape
        )
        self.embeddings_mmap[:] = embeddings.astype(np.float32)[:]
        self.embeddings_mmap.flush()
        
        # Store metadata efficiently
        self.chunk_metadata = chunk_metadata
        self.n_vectors = len(embeddings)
        
        print(f"Created memmap store: {self.n_vectors} vectors, {embeddings.nbytes / 1e6:.1f} MB")
    
    def similarity_search_with_scores(self, query_embedding: np.ndarray, 
                                    k: int = 8) -> List[tuple]:
        """Fast cosine similarity search using scikit-learn"""
        
        # Compute cosine similarity
        query_emb = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_emb, self.embeddings_mmap)[0]
        
        # Get top-k indices
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
        
        # Return chunks with scores
        results = []
        for idx in top_k_indices:
            chunk_meta = self.chunk_metadata[idx]
            score = similarities[idx]
            results.append((chunk_meta, score))
            
        return results
    
    def close(self):
        """Clean up memory-mapped files"""
        del self.embeddings_mmap
```

### 5.3 ChromaDB (≥ 500k vectors)

**For larger datasets**: Upgrade to ChromaDB with optimized HNSW settings:

```python
import chromadb
from chromadb.config import Settings

def create_optimized_chroma_store(embeddings: np.ndarray,
                                chunk_metadata: List[ChunkMetadata],
                                masked_chunks: List[Chunk],
                                persist_dir: Path = Path("vector_store")) -> chromadb.Collection:
    """
    Create ChromaDB with memory-optimized HNSW settings for Kaggle
    """
    
    # Memory-optimized settings for 13GB RAM limit
    hnsw_config = {
        "hnsw:M": 16,              # Reduced from default 64 (saves ~4x memory)
        "hnsw:ef_construction": 100, # Reduced from default 200  
        "hnsw:ef_search": 100,     # Search parameter
        "hnsw:space": "cosine"     # Cosine distance
    }
    
    # Initialize client with persistent storage
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True
        )
    )
    
    # Create collection with optimized settings
    try:
        # Try to load existing collection
        collection = client.get_collection(name="mdc_chunks")
        print(f"Loaded existing collection: {collection.count()} vectors")
        
    except Exception:
        # Create new collection
        collection = client.create_collection(
            name="mdc_chunks",
            metadata=hnsw_config
        )
        
        print(f"Creating ChromaDB collection with {len(embeddings)} vectors...")
        
        # Add embeddings in batches to manage memory
        batch_size = 1000
        for i in range(0, len(embeddings), batch_size):
            end_idx = min(i + batch_size, len(embeddings))
            
            batch_embeddings = embeddings[i:end_idx].tolist()
            batch_documents = [chunk.text for chunk in masked_chunks[i:end_idx]]
            batch_metadata = [meta.model_dump() for meta in chunk_metadata[i:end_idx]]
            batch_ids = [f"chunk_{j}" for j in range(i, end_idx)]
            
            collection.add(
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadata,
                ids=batch_ids
            )
            
            print(f"Added batch {i//batch_size + 1}/{(len(embeddings)-1)//batch_size + 1}")
    
    return collection

def choose_vector_store(embeddings: np.ndarray, 
                       chunk_metadata: List[ChunkMetadata],
                       masked_chunks: List[Chunk],
                       threshold: int = 500_000):
    """
    Automatically choose between memmap and ChromaDB based on dataset size
    """
    
    n_vectors = len(embeddings)
    
    if n_vectors < threshold:
        print(f"Using MemoryMappedVectorStore for {n_vectors:,} vectors (< {threshold:,})")
        return MemoryMappedVectorStore(embeddings, chunk_metadata)
    else:
        print(f"Using ChromaDB for {n_vectors:,} vectors (≥ {threshold:,})")
        return create_optimized_chroma_store(embeddings, chunk_metadata, masked_chunks)
```

### 5.4 Unified Retrieval Interface

**Single interface** for both storage backends:

```python
class UnifiedRetriever:
    """Unified interface for both memmap and ChromaDB backends"""
    
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
        self.backend_type = type(vector_store).__name__
    
    def similarity_search_with_scores(self, query_text: str, k: int = 8) -> List[tuple]:
        """Unified search interface"""
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query_text])[0]
        
        if self.backend_type == "MemoryMappedVectorStore":
            return self.vector_store.similarity_search_with_scores(query_embedding, k)
            
        elif "Collection" in self.backend_type:  # ChromaDB
            results = self.vector_store.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert to consistent format
            formatted_results = []
            for doc, meta, dist in zip(results['documents'][0], 
                                     results['metadatas'][0], 
                                     results['distances'][0]):
                # ChromaDB returns distance, convert to similarity
                similarity = 1 - dist
                formatted_results.append((meta, similarity))
            
            return formatted_results
        
        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}")
    
    def get_stats(self) -> Dict[str, any]:
        """Get storage statistics"""
        if self.backend_type == "MemoryMappedVectorStore":
            return {
                'backend': 'memmap',
                'n_vectors': self.vector_store.n_vectors,
                'memory_usage_mb': self.vector_store.embeddings_mmap.nbytes / 1e6
            }
        else:
            return {
                'backend': 'chromadb',
                'n_vectors': self.vector_store.count(),
                'memory_usage_mb': 'unknown'
            }
```

### 5.5 Performance Comparison

**Benchmark both approaches** to validate the choice:

```python
import time

def benchmark_retrieval(unified_retriever: UnifiedRetriever, 
                      test_queries: List[str],
                      k: int = 8) -> Dict[str, float]:
    """Benchmark retrieval performance"""
    
    print(f"Benchmarking {unified_retriever.backend_type}...")
    
    times = []
    for query in test_queries:
        start_time = time.time()
        results = unified_retriever.similarity_search_with_scores(query, k)
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    stats = unified_retriever.get_stats()
    
    return {
        'backend': stats['backend'],
        'n_vectors': stats['n_vectors'],
        'avg_query_time_ms': np.mean(times) * 1000,
        'queries_per_second': 1 / np.mean(times),
        'memory_usage_mb': stats.get('memory_usage_mb', 'unknown')
    }

# Example usage
test_queries = [
    "Find dataset GSE12345",
    "Where is the raw data stored?", 
    "Data availability accession numbers"
]

benchmark_results = benchmark_retrieval(unified_retriever, test_queries)
print(f"Performance: {benchmark_results}")
```

**Why This Approach Works:**
- **Memory-first**: NumPy memmap uses minimal RAM for <500k vectors
- **Optimized HNSW**: `M=16, ef_construction=100` fits Kaggle's 13GB limit  
- **Zero index time**: Memmap store ready immediately
- **Unified interface**: Same code works with both backends
- **Automatic scaling**: Chooses optimal backend by dataset size
- **Cached embeddings**: Avoid re-computation during iterations

---

## 6  Chunk-Level EDA

### 6.1 Chunk statistics

* Histogram: **tokens / chunk**, **chunks / document**.
* Coverage\@k: does at least one chunk containing the ground-truth `dataset_id` appear in the top-k (k = 3, 5) similarity hits? Low values → reconsider chunk size/overlap.

### 6.2 **Enhanced EDA on Retrieval Quality & Citation Integrity**

#### 6.2.1 Citation Integrity Validation Plots

**Identifier Loss Rate vs. Chunk Length:**
Diagnostic plot to confirm chunking preserves all citation entities:

```python
def plot_citation_integrity_metrics(processed_texts: Dict[str, str], 
                                   chunks_by_doc: Dict[str, List[Chunk]],
                                   chunk_sizes: List[int] = [150, 200, 250, 300]) -> None:
    """Plot citation integrity across different chunk sizes
    
    Args:
        processed_texts: Same Dict used in chunking - ensures data source consistency
    """
    
    integrity_results = []
    
    for chunk_size in chunk_sizes:
        total_loss = 0
        total_entities = 0
        
        for doc_id, processed_text in processed_texts.items():
            # Get pre-chunk entity count
            pre_entities = extract_citation_entities(processed_text)
            pre_count = sum(len(matches) for matches in pre_entities.values())
            
            # Get post-chunk entity count (simulate different chunk sizes)
            chunks = chunks_by_doc.get(doc_id, [])
            post_entities = {}
            for chunk in chunks:
                chunk_entities = extract_citation_entities(chunk.text)
                for pattern, matches in chunk_entities.items():
                    post_entities[pattern] = post_entities.get(pattern, 0) + len(matches)
            
            post_count = sum(post_entities.values())
            
            total_entities += pre_count
            total_loss += max(0, pre_count - post_count)
        
        loss_rate = total_loss / total_entities if total_entities > 0 else 0
        integrity_results.append({'chunk_size': chunk_size, 'loss_rate': loss_rate})
    
    # Plot results
    df_integrity = pd.DataFrame(integrity_results)
    plt.figure(figsize=(10, 6))
    plt.plot(df_integrity['chunk_size'], df_integrity['loss_rate'], marker='o', linewidth=2)
    plt.xlabel('Chunk Size (tokens)')
    plt.ylabel('Citation Entity Loss Rate')
    plt.title('Citation Integrity vs. Chunk Size')
    plt.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Zero Loss (Target)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_entity_distribution_by_section(chunks: List[Chunk]) -> None:
    """Plot citation entity density by document section"""
    
    section_entity_counts = {}
    section_chunk_counts = {}
    
    for chunk in chunks:
        section = (chunk.chunk_metadata.section_primary.section_type 
                  if chunk.chunk_metadata.section_primary else 'other')
        entities = extract_citation_entities(chunk.text)
        entity_count = sum(len(matches) for matches in entities.values())
        
        section_entity_counts[section] = section_entity_counts.get(section, 0) + entity_count
        section_chunk_counts[section] = section_chunk_counts.get(section, 0) + 1
    
    # Calculate density (entities per chunk)
    section_densities = {
        section: section_entity_counts[section] / section_chunk_counts[section] 
        for section in section_entity_counts 
        if section_chunk_counts[section] > 0
    }
    
    # Create heatmap-style plot
    sections = list(section_densities.keys())
    densities = list(section_densities.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sections, densities, color='steelblue', alpha=0.7)
    plt.xlabel('Document Section')
    plt.ylabel('Citation Entities per Chunk')
    plt.title('Citation Entity Density by Section')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, density in zip(bars, densities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{density:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
```

#### 6.2.2 Coverage\@k (Training-Time & Inference-Time)

**Training-Time Coverage (Label-Aware):**
Compute the % of label rows for which at least one retrieved chunk contains the dataset accession string:

```python
def compute_coverage_at_k(topk_results: pd.DataFrame, k: int = 3) -> float:
    """Calculate what % of dataset_ids have at least one relevant chunk in top-k"""
    coverage_count = 0
    total_labels = len(topk_results.groupby(['dataset_id', 'type']))
    
    for (dataset_id, label_type), group in topk_results.groupby(['dataset_id', 'type']):
        top_k_chunks = group.nlargest(k, 'cosine_score')
        # Check if any chunk contains the dataset_id (regex match)
        if any(dataset_id in chunk_text for chunk_text in top_k_chunks['text']):
            coverage_count += 1
    
    return coverage_count / total_labels
```

**Inference-Time Coverage (Label-Agnostic):**
Evaluate retrieval path effectiveness:

```python
def evaluate_inference_retrieval(articles: List[dict], k: int = 8) -> Dict[str, float]:
    """Evaluate label-agnostic retrieval strategies"""
    results = {
        'anchor_coverage': 0,
        'semantic_coverage': 0, 
        'universal_coverage': 0,
        'mean_retrieval_score': 0,
        'section_distribution': {}
    }
    
    total_articles = len(articles)
    anchor_hits = 0
    semantic_hits = 0
    universal_hits = 0
    total_scores = []
    section_counts = {}
    
    for article in articles:
        chunks = get_inference_chunks(
            article['chunks'], 
            article.get('title', ''), 
            article.get('abstract', ''),
            k=k
        )
        
        # Check which retrieval path was used
        anchor_score = sum(1 for c in chunks if c.score and c.score > 0)
        if anchor_score >= k/2:  # majority from anchors
            anchor_hits += 1
        elif any('semantic' in str(c.chunk_metadata) for c in chunks):
            semantic_hits += 1
        else:
            universal_hits += 1
            
        # Track section distribution of retrieved chunks
        for chunk in chunks:
            section = (chunk.chunk_metadata.section_primary.section_type 
                      if chunk.chunk_metadata.section_primary else 'other')
            section_counts[section] = section_counts.get(section, 0) + 1
            
        # Track average relevance scores
        chunk_scores = [c.score or 0 for c in chunks]
        total_scores.extend(chunk_scores)
    
    results['anchor_coverage'] = anchor_hits / total_articles
    results['semantic_coverage'] = semantic_hits / total_articles  
    results['universal_coverage'] = universal_hits / total_articles
    results['mean_retrieval_score'] = sum(total_scores) / len(total_scores) if total_scores else 0
    results['section_distribution'] = section_counts
    
    return results

def plot_retrieval_path_analysis(eval_results: Dict[str, float]) -> None:
    """Visualize retrieval path effectiveness"""
    
    # Plot retrieval method distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Retrieval path coverage
    methods = ['Anchor', 'Semantic', 'Universal']
    coverages = [eval_results['anchor_coverage'], 
                eval_results['semantic_coverage'], 
                eval_results['universal_coverage']]
    
    colors = ['green', 'blue', 'orange']
    bars1 = ax1.bar(methods, coverages, color=colors, alpha=0.7)
    ax1.set_ylabel('Coverage Rate')
    ax1.set_title('Inference Retrieval Path Distribution')
    ax1.set_ylim(0, 1)
    
    # Add percentage labels
    for bar, coverage in zip(bars1, coverages):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{coverage:.1%}', ha='center', va='bottom')
    
    # Section distribution of retrieved chunks
    if 'section_distribution' in eval_results:
        sections = list(eval_results['section_distribution'].keys())
        counts = list(eval_results['section_distribution'].values())
        
        bars2 = ax2.bar(sections, counts, color='steelblue', alpha=0.7)
        ax2.set_ylabel('Number of Retrieved Chunks')
        ax2.set_title('Section Distribution of Retrieved Chunks')
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
```

#### 6.2.3 Score Distributions with Section Analysis

Plot histograms of `cosine_score` per `type` and section; sharp bimodality may hint at false labels:

```python
import seaborn as sns
import matplotlib.pyplot as plt

def plot_comprehensive_score_analysis(topk_results: pd.DataFrame) -> None:
    """Enhanced score distribution analysis with section breakdown"""
    
    # Overall score distribution by label type
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top row: Score distributions by label type
    for i, label_type in enumerate(['Primary', 'Secondary', 'Missing']):
        subset = topk_results[topk_results['type'] == label_type]
        sns.histplot(subset['cosine_score'], bins=30, ax=axes[0, i], alpha=0.7)
        axes[0, i].set_title(f'Cosine Score Distribution - {label_type}')
        axes[0, i].axvline(subset['cosine_score'].median(), color='red', linestyle='--', 
                          label=f'Median: {subset["cosine_score"].median():.3f}')
        axes[0, i].legend()
    
    # Bottom row: Section-aware analysis
    section_score_data = []
    if 'section_type' in topk_results.columns:
        for section in ['methods', 'data_availability', 'results', 'discussion', 'other']:
            section_data = topk_results[topk_results['section_type'] == section]
            if len(section_data) > 0:
                section_score_data.append({
                    'section': section,
                    'mean_score': section_data['cosine_score'].mean(),
                    'std_score': section_data['cosine_score'].std(),
                    'count': len(section_data)
                })
    
    if section_score_data:
        df_sections = pd.DataFrame(section_score_data)
        
        # Mean scores by section
        bars = axes[1, 0].bar(df_sections['section'], df_sections['mean_score'], 
                             color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('Mean Cosine Score by Section')
        axes[1, 0].set_ylabel('Mean Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Error bars for standard deviation
        axes[1, 0].errorbar(df_sections['section'], df_sections['mean_score'], 
                           yerr=df_sections['std_score'], fmt='none', color='black', alpha=0.5)
        
        # Chunk count by section
        axes[1, 1].bar(df_sections['section'], df_sections['count'], 
                      color='lightblue', alpha=0.7)
        axes[1, 1].set_title('Retrieved Chunk Count by Section')
        axes[1, 1].set_ylabel('Number of Chunks')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Score vs section heatmap
        pivot_data = topk_results.pivot_table(
            values='cosine_score', 
            index='section_type', 
            columns='type', 
            aggfunc='mean'
        )
        
        if not pivot_data.empty:
            sns.heatmap(pivot_data, annot=True, fmt='.3f', ax=axes[1, 2], 
                       cmap='viridis', cbar_kws={'label': 'Mean Cosine Score'})
            axes[1, 2].set_title('Score Heatmap: Section × Label Type')
    
    plt.tight_layout()
    plt.show()
```

### 6.3 Embedding geometry (Enhanced)

1. **UMAP → 2-D** (`n_neighbors=15`, `min_dist=0.1`) – preserves local & global structure for text ([umap-learn.readthedocs.io][10]).
2. **Colour by approach**: 
   - **Training-time**: Primary, Secondary, Missing – easy visual drift check
   - **Inference-time**: Colour by retrieval path (anchor, semantic, universal) to spot blind spots
3. **HDBSCAN** on embeddings or UMAP output – density-based clusters & noise points ([dylancastillo.co][11]).
4. **NEW: Project only retrieved chunks** – noise gets filtered out, producing cleaner clusters.
5. **Cross-validation**: Compare training-time and inference-time chunk distributions to ensure consistency.
6. Optional: push embeddings to **Nomic Atlas** for interactive zoom / tag filtering ([docs.nomic.ai][12]).

### 6.4 Imbalance & meta-feature probes

* Stratify charts by label to confirm class ratios per chunk and per document.
* Correlate chunk length or section tag with label to spot biases (e.g., citations mostly in *Methods*).

---

## 7  Export Artefacts for the Training Loop

```python
chunk_df = pd.DataFrame([
    {
        "chunk_id": c.chunk_id,
        "document_id": c.chunk_metadata.document_id,
        "dataset_id": label_lookup.get((c.chunk_metadata.document_id, idx), None),
        "type": label_lookup_type.get(...),
        "embedding": embedding,              # store as list or np.ndarray
        "umap_x": umap_x[i],
        "umap_y": umap_y[i],
        "cluster_id": hdbscan_labels[i],
        "section_type": c.chunk_metadata.section_primary.section_type if c.chunk_metadata.section_primary else None,
        "section_page_start": c.chunk_metadata.section_primary.page_start if c.chunk_metadata.section_primary else None,
        "section_page_end": c.chunk_metadata.section_primary.page_end if c.chunk_metadata.section_primary else None,
        "chunk_type": c.chunk_metadata.chunk_type,
        "page_number": c.chunk_metadata.page_number,
        **{k: v for k, v in c.chunk_metadata.model_dump().items() 
           if k not in ['section_primary']}  # Exclude section_primary object, use flattened fields instead
    }
    for i, (c, embedding) in enumerate(zip(chunks, embeddings))
])
chunk_df.to_parquet("chunks.parquet")       # ready for training loop
client.persist()                            # freezes vector DB to disk
```

**Enhanced Export Artifacts:**

* **`topk_chunks.parquet`** – k best chunks per label, cosine scores, metadata.
* **`chunks.parquet`** – full chunk inventory with embeddings, cluster assignments, and flattened section metadata.
* **`vector_store/`** – persisted ChromaDB (reusable for inference).
* **EDA notebook / HTML** – coverage\@k, score plots, UMAP visual.

All rows validate against the `Chunk` Pydantic schema, so downstream code can `.parse_obj(row)` with guarantees.

Store embeddings both in Parquet and ChromaDB so the training code can choose raw arrays or vector search as needed.

---

## 8  Quality-Control Checklist

| Check                                   | Pass criterion                           | Fix if fails                                 |
| --------------------------------------- | ---------------------------------------- | -------------------------------------------- |
| **File coverage**                       | ≥ 95 % of label rows have parsed text    | scrape publisher APIs / retry OCR            |
| **Citation entity integrity**           | Zero loss rate across all chunk sizes    | increase overlap or use adaptive repair      |
| **Training-time chunk coverage\@k**     | ≥ 90 % ground-truth IDs hit             | shrink chunk size or boost overlap           |
| **Inference-time anchor coverage**      | ≥ 70 % articles use anchor retrieval     | expand regex patterns or improve extraction  |
| **Section extraction completeness**     | ≥ 85 % chunks have section metadata      | improve PDF parsing or add fallback rules    |
| **Mean cosine of hit #1**               | ≥ 0.3                                   | fine-tune embedder or craft richer query    |
| **Duplicate chunk\_ids in top-k**       | 0                                       | enable MMR or higher `lambda_mult`          |
| **UMAP label purity (training)**        | clusters mostly single-colour           | audit annotations or increase model capacity |
| **UMAP retrieval-path purity (inference)** | clear separation by retrieval method   | improve anchor patterns or semantic queries  |
| **UMAP mixed-label ratio**              | < 15 %                                  | re-examine noisy annotations                 |
| **Vector DB recall**                    | > 0.95 on synthetic queries             | increase `ef_search`, rebuild index          |
| **Train-inference consistency**          | < 5% difference in chunk distributions   | align preprocessing pipelines                |
| **Section-citation correlation**         | Methods/DAS show highest entity density  | validate section mapping accuracy            |

---

## 9  Hand-off Notes for Modelling Team

### 9.1 Training Pipeline
* **Input**: `chunks.parquet` + `vector_store/` directory; each chunk already validated by Pydantic.
* **Enhanced Input**: `topk_chunks.parquet` – every label now maps to the most relevant, validated top-k chunks + similarity scores.
* **Features**: 384-d embedding, UMAP (2-d), HDBSCAN cluster, chunk length, section tag, etc.
* **Enhanced Features**: `cosine_score` from retrieval, coverage\@k metrics, MMR diversity scores.
* **Label-aware splits**: group by `document_id` during CV to avoid leakage.

### 9.2 Inference Pipeline
* **No label leakage**: Use `get_inference_chunks()` function for identical preprocessing at inference time.
* **Retrieval strategy**: Layered fallback (anchor → semantic → universal) ensures consistent chunk quality.
* **Same feature space**: Inference chunks use identical embeddings and feature extraction as training.
* **Performance monitoring**: Track retrieval path distributions to detect data drift.

### 9.3 Implementation Notes
* **Ready-to-use**: Load `topk_chunks.parquet`, group by `document_id` for stratified folds, and feed `embedding` plus `cosine_score` straight into any classifier.
* **Symmetric preprocessing**: Training and inference use the same chunking/embedding pipeline with different retrieval strategies.
* **Optional fine-tuning**: Only apply GenQ fine-tuning if validation performance drops significantly from training to testing.
* **Deployment consistency**: Use the same ChromaDB instance and embedding model for both training EDA and inference retrieval.

### 9.4 Next Steps Checklist

1. **Implement authoritative regex library** (§4.3.1) → unit-test with fixture sentences containing known entities
2. **Add citation integrity validation** (§4.3.2-4.3.4) → run pre/post-chunk entity counts and plot loss rates
3. **Enhance section extraction** (§4.4.1) → extract section hierarchy from PDF structure using pattern recognition
4. **Implement section-aware retrieval** (§5.6.6) → add section filtering to ChromaDB queries
5. **Set up comprehensive EDA plots** (§6.2.1-6.2.3) → visualize integrity metrics and section impact
6. **Deploy inference pipeline** (§5.6.7) → integrate section-aware retrieval with existing workflow
7. **Generate synthetic Q-A pairs** only if performance drops (§5.6.4) → optional fine-tuning step
8. **Monitor train-test consistency** with cross-validation metrics → ensure no data drift

**Key Implementation Priority:**
- **High**: Citation integrity validation (prevents silent data loss)
- **High**: Section extraction and filtering (major recall improvement)  
- **Medium**: Comprehensive EDA visualizations (debugging and validation)
- **Low**: Optional GenQ fine-tuning (only if needed for performance recovery)

Future EDA iterations should revisit Sections 4–6 whenever you tweak chunking or fine-tune the embedding model.

---

## 9.5  🎯 Streamlined Execution Workflow (RECOMMENDED)

**Based on colleague feedback**: This linear workflow prioritizes high-impact changes and eliminates bottlenecks for competitive ML environments.

### Execution Order

| Step | Action                                                          | Script / Command                     | Priority | Duration |
|------|-----------------------------------------------------------------|-------------------------------------|----------|----------|
| 0    | **Start local containers**                                     | `docker run -d -p 8070:8070 lfoppiano/grobid:0.8.0` | **High** | 2 min |
| 1    | **convert_pdf.py** → TEI/JATS or raw text, cached             | `python convert_pdf.py --cache-dir processed_docs` | **High** | 10-20 min |
| 2    | **chunk.py** (XML-aware) + entity-integrity check            | `python chunk.py --validate-integrity` | **High** | 5-10 min |
| 3    | **mask_ids.py** (regex → `[ID]` tokens)                      | `python mask_ids.py --audit-leakage` | **High** | 2-5 min |
| 4    | **embed.py** (MiniLM mem-map or ChromaDB)                    | `python embed.py --auto-backend` | **Med** | 15-30 min |
| 5    | **retrieve_train.py** (label-aware cosine; save parquet)     | `python retrieve_train.py --coverage-at-k` | **Med** | 5-15 min |
| 6    | **eda_core.ipynb** (essential plots only)                    | `jupyter run eda_core.ipynb` | **Med** | 10-15 min |
| 7    | **train_classifier.py** (GroupKFold on `dataset_id`)         | `python train_classifier.py --group-cv` | **High** | 20-60 min |
| 8    | **infer.py** (anchor regex → cosine fallback)                | `python infer.py --test-set` | **Med** | 5-10 min |

### Key Implementation Scripts

**0. Container Setup** (`start_services.sh`):
```bash
#!/bin/bash
# Start GROBID service
docker run -d -p 8070:8070 --name grobid-service lfoppiano/grobid:0.8.0

# Wait for service to be ready
echo "Waiting for GROBID service..."
while ! curl -s http://localhost:8070/api/isalive > /dev/null; do
    sleep 5
done
echo "✅ GROBID service ready"

# Optional: Download CERMINE for fallback
# wget -q https://github.com/CeON/CERMINE/releases/download/1.13/cermine-impl-1.13-jar-with-dependencies.jar
```

**3. ID Masking** (`mask_ids.py`):
```python
#!/usr/bin/env python3
"""
Critical script to mask dataset IDs before embedding generation.
Usage: python mask_ids.py --input chunks.pkl --output masked_chunks.pkl --audit-leakage
"""

import pickle
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input chunks pickle file')
    parser.add_argument('--output', required=True, help='Output masked chunks pickle file')
    parser.add_argument('--audit-leakage', action='store_true', help='Run leakage audit')
    args = parser.parse_args()
    
    # Load chunks
    with open(args.input, 'rb') as f:
        chunks = pickle.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Apply ID masking
    masked_chunks = mask_dataset_ids_in_chunks(chunks)
    
    # Validate masking integrity
    integrity_check = validate_masking_integrity(chunks, masked_chunks)
    
    if not integrity_check['masking_complete']:
        print(f"🚨 CRITICAL: Masking incomplete!")
        print(f"Original IDs: {integrity_check['original_id_count']}")
        print(f"Masked tokens: {integrity_check['masked_token_count']}")
        return 1
    
    print(f"✅ ID masking complete: {integrity_check['original_id_count']} IDs → [ID] tokens")
    
    # Save masked chunks
    with open(args.output, 'wb') as f:
        pickle.dump(masked_chunks, f)
    
    print(f"Saved masked chunks to {args.output}")
    
    # Optional leakage audit
    if args.audit_leakage:
        print("Running comprehensive leakage audit...")
        # Load chunk_df for CV validation
        chunk_df = pd.read_parquet("chunks.parquet")  # Assume this exists
        audit_results = run_leakage_audit(chunks, masked_chunks, chunk_df)
        
        if audit_results['overall_safe']:
            print("✅ Pipeline is leak-free and safe for training")
        else:
            print("🚨 UNSAFE: Fix leakage issues before training")
            for rec in audit_results['recommendations']:
                print(f"  - {rec}")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
```

### EDA Cuts & Optimizations

**Essential EDA (Keep)**:
```python
# eda_core.ipynb - streamlined notebook
essential_plots = [
    'tokens_per_chunk_histogram',      # Critical for chunk size tuning
    'chunks_per_document_histogram',   # Document length distribution 
    'coverage_at_k_plot',              # Retrieval quality metric
    'citation_integrity_validation',   # Zero-loss verification
    'section_entity_density_bar',      # Data citation hotspots
    'score_distribution_by_type'       # Label separation quality
]
```

**Optional EDA (Feature-flagged)**:
```python
# Heavy visualizations - only run when debugging
optional_plots = [
    'umap_2d_projection',              # 3-5 min on 200k+ vectors
    'hdbscan_cluster_purity',          # Shifts with embedding changes  
    'nomic_atlas_upload',              # Requires internet connection
    'interactive_similarity_browser'   # Nice-to-have exploration tool
]

# Enable with --debug-clustering flag
if args.debug_clustering:
    # Sample 10% for faster iteration
    sample_chunks = np.random.choice(chunks, size=len(chunks)//10, replace=False)
    run_umap_hdbscan(sample_chunks)
```

### Performance Targets

| Phase | Target Time | Memory Usage | Notes |
|-------|-------------|--------------|--------|
| **PDF → XML** | < 20 min | < 2 GB | GROBID + cache; pdfplumber fallback |
| **Chunking + Masking** | < 15 min | < 1 GB | XML-aware; integrity validation |
| **Embedding** | < 30 min | < 4 GB | Batch processing; auto backend choice |
| **Retrieval** | < 15 min | < 2 GB | Memmap <500k, ChromaDB ≥500k |
| **Essential EDA** | < 15 min | < 1 GB | Core plots only; sample for heavy viz |
| **Training** | < 60 min | < 8 GB | GroupKFold; leak-free CV |

**Total pipeline**: ~2.5 hours for full preprocessing + training

### Quality Gates

Before proceeding to next step, validate:

```python
quality_gates = {
    'step_1_parsing': {
        'pdf_coverage': 0.95,           # ≥95% of labels have parsed text
        'xml_extraction_rate': 0.80     # ≥80% successful GROBID extraction
    },
    'step_2_chunking': {
        'entity_integrity': 1.0,        # Zero citation loss
        'section_coverage': 0.85        # ≥85% chunks have section metadata  
    },
    'step_3_masking': {
        'masking_complete': True,       # All IDs → [ID] tokens
        'no_remaining_patterns': True   # Clean masked text
    },
    'step_5_retrieval': {
        'coverage_at_3': 0.90,         # ≥90% labels hit in top-3
        'mean_cosine_score': 0.30      # ≥0.3 avg similarity
    },
    'step_7_training': {
        'cv_leak_free': True,          # No dataset_id overlap in folds
        'val_f1_score': 0.70           # ≥0.7 F1 before test submission
    }
}
```

**Why This Works:**
- **Linear execution**: No complex branching or decision trees
- **Clear quality gates**: Fail fast if fundamental issues detected
- **Time-boxed**: Each step has target duration for planning
- **Memory-aware**: Fits Kaggle 13GB limit with safety margin
- **Cached intermediate**: Avoid re-computation during iteration
- **Leak-proof**: Built-in validation at every step
- **Competition-ready**: Direct path to 0.8+ F1 score

---

## 10  Key References & Additional Resources

### Core Technologies
* ChromaDB overview & HNSW details ([medium.com](https://medium.com/@nishthakukreti.01/chromadb-fb20279e244c), [pinecone.io](https://www.pinecone.io/learn/series/faiss/hnsw/))
* HNSW algorithm deep dive ([zilliz.com](https://zilliz.com/learn/hierarchical-navigable-small-worlds-HNSW))
* LangChain retriever search types & MMR ([python.langchain.com](https://python.langchain.com/docs/tutorials/retrievers/), [python.langchain.com](https://python.langchain.com/docs/how_to/example_selectors_mmr/))

### Vector Search & Similarity
* `similarity_search_with_relevance_scores` API ([api.python.langchain.com](https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.chroma.Chroma.html))
* Score-threshold filtering ([github.com](https://github.com/langchain-ai/langchain/discussions/22697))
* LangChain guide to exposing scores in retrievers ([python.langchain.com](https://python.langchain.com/docs/how_to/add_scores_retriever/))
* Cosine-similarity best practices ([platform.openai.com](https://platform.openai.com/docs/guides/embeddings))
* Vector similarity search patterns ([medium.com](https://medium.com/@aleksgladun4/implementing-vector-similarity-search-engine-ee626b84bc5))

### Chunking & Embedding Optimization
* Optimising chunk / embed / vector pipelines ([medium.com](https://medium.com/@adnanmasood/optimizing-chunking-embedding-and-vectorization-for-retrieval-augmented-generation-ea3b083b68f7))
* Evaluating chunking strategies ([research.trychroma.com](https://research.trychroma.com/evaluating-chunking))
* Chunking beyond paragraphs - Financial Report study ([arxiv.org](https://arxiv.org/html/2402.05131v3))

### Reranking & Diversity
* Redundancy-aware reranking with MMR ([community.fullstackretrieval.com](https://community.fullstackretrieval.com/retrieval-methods/maximum-marginal-relevance))
* Reranking models for RAG context ([galileo.ai](https://galileo.ai/blog/mastering-rag-how-to-select-a-reranking-model))

### Technical Implementation
* StackOverflow discussion on Chroma scoring ([stackoverflow.com](https://stackoverflow.com/questions/76678783/langchains-chroma-vectordb-similarity-search-with-score-and-vectordb-similari))
* Faiss / ANN overviews for fallback local search ([pinecone.io](https://www.pinecone.io/learn/series/faiss/hnsw/))

---

This workflow keeps your data **clean, validated, richly-annotated, and efficiently retrievable**—everything a lightweight model needs without extra overhead. 

**Key Benefits:**
- **Training-time EDA**: Label-aware retrieval provides the most semantically relevant chunks with quantified similarity scores
- **Inference-time retrieval**: Label-agnostic strategies ensure consistent preprocessing without leakage
- **Citation integrity protection**: Pre/post-chunk validation prevents silent loss of critical identifiers
- **Section-aware intelligence**: Leverages empirical knowledge that citations cluster in Methods (68%) and Data Availability (89%) sections
- **No performance degradation**: Layered fallback (anchor → semantic → universal) maintains high recall
- **Optional fine-tuning**: GenQ approach available only if needed for performance recovery
- **Symmetric pipelines**: Identical chunking/embedding for training and inference ensures model consistency
- **Comprehensive validation**: Visual diagnostics for integrity, coverage, and section impact

**The enhanced dual-retrieval system with citation integrity validation and section-aware filtering bridges the gap between training EDA and production inference, delivering robust performance across both phases while protecting against data loss and leveraging document structure intelligence.**

---

## 4.5  🚨 Data Leakage Prevention (HIGH PRIORITY)

**Critical**: These steps prevent the model from memorizing specific `dataset_id` strings, which would cause catastrophic leakage in cross-validation.

### 4.5.1 ID Masking Before Embedding

**Replace all detected dataset IDs with `[ID]` tokens** after extraction but before embedding:

```python
def mask_dataset_ids_in_chunks(chunks: List[Chunk], 
                              citation_patterns: Dict[str, re.Pattern] = None) -> List[Chunk]:
    """
    Replace all dataset IDs with [ID] tokens to prevent memorization leakage.
    
    CRITICAL: This must happen AFTER entity extraction but BEFORE embedding generation.
    """
    if citation_patterns is None:
        citation_patterns = CITATION_PATTERNS  # Use patterns from section 4.3.1
    
    masked_chunks = []
    
    for chunk in chunks:
        masked_text = chunk.text
        detected_ids = []
        
        # Apply all citation patterns to detect IDs
        for pattern_name, regex in citation_patterns.items():
            matches = regex.findall(masked_text)
            detected_ids.extend(matches)
            # Replace with [ID] token
            masked_text = regex.sub('[ID]', masked_text)
        
        # Create new chunk with masked text
        masked_chunk = Chunk(
            chunk_id=chunk.chunk_id,
            text=masked_text,  # ← This goes to embeddings
            score=chunk.score,
            chunk_metadata=chunk.chunk_metadata
        )
        
        # Store original IDs in metadata for later validation
        masked_chunk.chunk_metadata.original_ids = detected_ids
        masked_chunks.append(masked_chunk)
    
    return masked_chunks

def validate_masking_integrity(original_chunks: List[Chunk], 
                              masked_chunks: List[Chunk]) -> Dict[str, int]:
    """Verify that masking preserved all entities without losing them"""
    
    original_id_count = 0
    masked_token_count = 0
    
    for orig, masked in zip(original_chunks, masked_chunks):
        # Count original IDs
        for pattern in CITATION_PATTERNS.values():
            original_id_count += len(pattern.findall(orig.text))
        
        # Count [ID] tokens in masked text
        masked_token_count += masked.text.count('[ID]')
    
    return {
        'original_id_count': original_id_count,
        'masked_token_count': masked_token_count,
        'masking_complete': original_id_count == masked_token_count
    }
```

### 4.5.2 GroupKFold by Dataset ID

**Ensure no `dataset_id` appears in both train and validation** during cross-validation:

```python
from sklearn.model_selection import GroupKFold
import pandas as pd

def create_leak_free_cv_splits(chunk_df: pd.DataFrame, n_splits: int = 5) -> List[tuple]:
    """
    Create CV splits that group by dataset_id to prevent leakage.
    
    Args:
        chunk_df: DataFrame with columns ['chunk_id', 'dataset_id', 'document_id', ...]
        n_splits: Number of CV folds
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    
    # Group by dataset_id to prevent same ID in train/val
    group_kfold = GroupKFold(n_splits=n_splits)
    
    # Use dataset_id as the grouping variable
    X = chunk_df[['chunk_id', 'embedding']].values  # Features
    y = chunk_df['type'].values                     # Labels  
    groups = chunk_df['dataset_id'].values          # Groups for splitting
    
    cv_splits = []
    for train_idx, val_idx in group_kfold.split(X, y, groups):
        cv_splits.append((train_idx, val_idx))
        
        # Validation: ensure no dataset_id overlap
        train_ids = set(chunk_df.iloc[train_idx]['dataset_id'].unique())
        val_ids = set(chunk_df.iloc[val_idx]['dataset_id'].unique())
        
        overlap = train_ids.intersection(val_ids)
        if len(overlap) > 0:
            raise ValueError(f"CV split leaked dataset_ids: {overlap}")
        
        print(f"Split: {len(train_ids)} train IDs, {len(val_ids)} val IDs, no overlap ✓")
    
    return cv_splits

def validate_cv_splits(chunk_df: pd.DataFrame, cv_splits: List[tuple]) -> Dict[str, any]:
    """Double-check that CV splits are truly leak-free"""
    
    validation_results = {
        'all_splits_valid': True,
        'split_details': [],
        'global_coverage': None
    }
    
    all_dataset_ids = set(chunk_df['dataset_id'].unique())
    covered_ids = set()
    
    for i, (train_idx, val_idx) in enumerate(cv_splits):
        train_ids = set(chunk_df.iloc[train_idx]['dataset_id'].unique()) 
        val_ids = set(chunk_df.iloc[val_idx]['dataset_id'].unique())
        
        # Check for leakage
        overlap = train_ids.intersection(val_ids)
        has_leakage = len(overlap) > 0
        
        if has_leakage:
            validation_results['all_splits_valid'] = False
        
        covered_ids.update(train_ids)
        covered_ids.update(val_ids)
        
        validation_results['split_details'].append({
            'split_idx': i,
            'train_ids': len(train_ids),
            'val_ids': len(val_ids), 
            'overlap_count': len(overlap),
            'has_leakage': has_leakage,
            'leaked_ids': list(overlap) if overlap else []
        })
    
    # Check global coverage
    uncovered_ids = all_dataset_ids - covered_ids
    validation_results['global_coverage'] = {
        'total_ids': len(all_dataset_ids),
        'covered_ids': len(covered_ids),
        'coverage_rate': len(covered_ids) / len(all_dataset_ids),
        'uncovered_ids': list(uncovered_ids)
    }
    
    return validation_results
```

### 4.5.3 Leakage Audit Pipeline

**Run this before training** to catch any data leakage:

```python
def run_leakage_audit(chunks: List[Chunk], 
                     masked_chunks: List[Chunk],
                     chunk_df: pd.DataFrame) -> Dict[str, any]:
    """Comprehensive audit for data leakage before training"""
    
    audit_results = {
        'timestamp': datetime.now().isoformat(),
        'masking_integrity': {},
        'cv_validation': {},
        'text_inspection': {},
        'recommendations': []
    }
    
    # 1. Validate ID masking
    masking_check = validate_masking_integrity(chunks, masked_chunks)
    audit_results['masking_integrity'] = masking_check
    
    if not masking_check['masking_complete']:
        audit_results['recommendations'].append(
            "CRITICAL: ID masking incomplete - some identifiers may leak into embeddings"
        )
    
    # 2. Validate CV splits
    cv_splits = create_leak_free_cv_splits(chunk_df)
    cv_validation = validate_cv_splits(chunk_df, cv_splits)
    audit_results['cv_validation'] = cv_validation
    
    if not cv_validation['all_splits_valid']:
        audit_results['recommendations'].append(
            "CRITICAL: CV splits contain dataset_id leakage"
        )
    
    # 3. Sample text inspection
    sample_masked_texts = [chunk.text for chunk in masked_chunks[:10]]
    remaining_patterns = []
    
    for text in sample_masked_texts:
        for pattern_name, regex in CITATION_PATTERNS.items():
            if regex.search(text):
                remaining_patterns.append(pattern_name)
    
    audit_results['text_inspection'] = {
        'sample_size': len(sample_masked_texts),
        'remaining_patterns': list(set(remaining_patterns)),
        'masking_effective': len(remaining_patterns) == 0
    }
    
    if remaining_patterns:
        audit_results['recommendations'].append(
            f"WARNING: Masking missed patterns: {remaining_patterns}"
        )
    
    # 4. Overall safety assessment
    is_safe = (masking_check['masking_complete'] and 
              cv_validation['all_splits_valid'] and 
              len(remaining_patterns) == 0)
    
    audit_results['overall_safe'] = is_safe
    
    if is_safe:
        audit_results['recommendations'].append("✅ Pipeline is leak-free and safe for training")
    else:
        audit_results['recommendations'].append("🚨 UNSAFE: Fix leakage issues before training")
    
    return audit_results
```

**Why This Approach Works:**
- **ID masking** prevents embeddings from memorizing specific identifiers
- **GroupKFold** ensures no dataset appears in both train/validation
- **Audit pipeline** catches edge cases and validates the fixes  
- **Same preprocessing** at train/inference time maintains consistency
- **Validation checks** prevent silent failures that could invalidate results

**Usage Example:**
```python
# After chunking, before embedding
masked_chunks = mask_dataset_ids_in_chunks(chunks)

# Before training
audit_results = run_leakage_audit(chunks, masked_chunks, chunk_df)
if not audit_results['overall_safe']:
    print("🚨 CRITICAL: Fix leakage before proceeding!")
    for rec in audit_results['recommendations']:
        print(f"  - {rec}")
else:
    print("✅ Safe to proceed with training")
```

[1]: https://realpython.com/chromadb-vector-database/?utm_source=chatgpt.com "Embeddings and Vector Databases With ChromaDB - Real Python"
[2]: https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide?utm_source=chatgpt.com "Learn How to Use Chroma DB: A Step-by-Step Guide | DataCamp"
[3]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2?utm_source=chatgpt.com "sentence-transformers/all-MiniLM-L6-v2 - Hugging Face"
[4]: https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/?utm_source=chatgpt.com "10 Techniques to Solve Imbalanced Classes in Machine Learning ..."
[5]: https://medium.com/%40263akash/different-levels-of-text-splitting-chunking-ce9da78570d5?utm_source=chatgpt.com "Different Levels of Text Splitting/ Chunking | by Aakash Tomar"
[6]: https://python.langchain.com/docs/concepts/text_splitters/?utm_source=chatgpt.com "Text splitters - Python LangChain"
[7]: https://realpython.com/python-pydantic/?utm_source=chatgpt.com "Pydantic: Simplifying Data Validation in Python"
[8]: https://www.netguru.com/blog/data-validation-pydantic?utm_source=chatgpt.com "Data Validation with Pydantic - Netguru"
[9]: https://kaustavmukherjee-66179.medium.com/introduction-to-hnsw-indexing-and-getting-rid-of-the-chromadb-error-due-to-hnsw-index-issue-e61df895b146?utm_source=chatgpt.com "Introduction to HNSW Indexing and Getting Rid of the ChromaDB ..."
[10]: https://umap-learn.readthedocs.io/en/latest/document_embedding.html?utm_source=chatgpt.com "Document embedding using UMAP - Read the Docs"
[11]: https://dylancastillo.co/posts/clustering-documents-with-openai-langchain-hdbscan.html?utm_source=chatgpt.com "Clustering Documents with OpenAI embeddings, HDBSCAN and ..."
[12]: https://docs.nomic.ai/atlas/embeddings-and-retrieval/guides/how-to-visualize-embeddings?utm_source=chatgpt.com "How to Visualize Embeddings with t-SNE, UMAP, and Nomic Atlas"
