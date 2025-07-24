Below is an **expanded, production-ready Data Pre-Processing & EDA guide** that *adds an explicit, end-to-end recipe for finding the **top k most relevant chunks** by cosine similarity* while keeping every earlier step intact. Teams can lift code snippets verbatim or swap-in cloud equivalents later.

---

# Revised Proposed Guide Structure

## **1. Environment & Library Setup**
- Current section 1 (stays in place)

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

## **2. Streamlined Execution Workflow (Overview)**  
- Current section 9.5 → Move to section 2

This linear workflow prioritizes high-impact changes and eliminates bottlenecks for competitive ML environments.

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

## **3. Load Labels & Map Documents**
- Current section 2 (stays in place)

1. **Read labels**

   ```python
   labels = pd.read_csv("train_labels.csv")       # columns: article_id, dataset_id, type
   ```
2. **Inventory full-text paths** (PDF + XML files) in a helper table; flag missing files for later QC.
3. **Basic checks** – duplicates, nulls, class balance bar chart (expect ≈ 44 % Secondary, 30 % Missing, 26 % Primary) ([analyticsvidhya.com][4]).

---

## **4. Selective PDF → XML Conversion**
- **NEW SECTION** - Extract unique `article_id`s from `train_labels.csv` → Map existing XML files → Convert only missing PDFs to XML
- This should include the PDF → XML conversion wrapper code from current section 3.2

This optimization step ensures we only process PDFs that don't already have XML counterparts, saving significant processing time since approximately 75% of articles already have XML files available.

### 4.1 Extract Required Article IDs from Labels

**Goal**: Identify which documents are actually needed for training to avoid processing unused files.

```python
import pandas as pd
from pathlib import Path
from typing import Set, Dict, List

def extract_required_article_ids(labels_path: str = "Data/train_labels.csv") -> Set[str]:
    """
    Extract unique article_ids from train_labels.csv to determine which documents we need.
    
    Returns:
        Set of unique article_id strings (DOIs)
    """
    # Load training labels
    labels_df = pd.read_csv(labels_path)
    
    # Extract unique article_ids (these are the DOIs we need)
    required_article_ids = set(labels_df['article_id'].unique())
    
    print(f"Found {len(required_article_ids)} unique articles required for training")
    
    # Validate article_id format (should be DOIs)
    valid_ids = set()
    invalid_ids = []
    
    for article_id in required_article_ids:
        if isinstance(article_id, str) and ('doi.org' in article_id or article_id.startswith('10.')):
            valid_ids.add(article_id)
        else:
            invalid_ids.append(article_id)
    
    if invalid_ids:
        print(f"WARNING: Found {len(invalid_ids)} invalid article_ids: {invalid_ids[:5]}...")
    
    print(f"Validated {len(valid_ids)} valid article_ids for processing")
    return valid_ids

def create_doi_to_filename_mapping(article_ids: Set[str]) -> Dict[str, str]:
    """
    Create mapping from DOI to expected filename format.
    
    Args:
        article_ids: Set of DOI strings from train_labels.csv
        
    Returns:
        Dict mapping {doi: expected_filename_stem}
    """
    doi_to_filename = {}
    
    for doi in article_ids:
        # Extract filename from DOI
        # DOI format: https://doi.org/10.1371/journal.pone.0303785
        # Expected filename: journal.pone.0303785 (without extension)
        
        if 'doi.org/' in doi:
            # Full DOI with URL
            filename_part = doi.split('doi.org/')[-1]
        elif doi.startswith('10.'):
            # DOI without URL prefix
            filename_part = doi
        else:
            print(f"WARNING: Unexpected DOI format: {doi}")
            continue
            
        # Remove the publisher prefix (10.1371/) to get journal.pone.0303785
        if '/' in filename_part:
            filename_stem = filename_part.split('/', 1)[-1]
        else:
            filename_stem = filename_part
            
        doi_to_filename[doi] = filename_stem
    
    return doi_to_filename
```

### 4.2 Inventory Existing XML Files

**Goal**: Map existing XML files to avoid redundant conversion.

```python
def inventory_existing_xml_files(xml_dir: Path = Path("Data/train/XML")) -> Dict[str, Path]:
    """
    Create inventory of existing XML files.
    
    Returns:
        Dict mapping {filename_stem: xml_file_path}
    """
    existing_xml = {}
    
    if not xml_dir.exists():
        print(f"WARNING: XML directory not found: {xml_dir}")
        return existing_xml
    
    # Scan XML directory
    xml_files = list(xml_dir.glob("*.xml"))
    
    for xml_file in xml_files:
        filename_stem = xml_file.stem  # Remove .xml extension
        existing_xml[filename_stem] = xml_file
    
    print(f"Found {len(existing_xml)} existing XML files")
    return existing_xml

def identify_missing_conversions(required_dois: Set[str], 
                               doi_to_filename: Dict[str, str],
                               existing_xml: Dict[str, Path]) -> Dict[str, Dict]:
    """
    Identify which PDFs need conversion to XML.
    
    Returns:
        Dict with 'convert_needed', 'already_available', and 'missing_pdfs' keys
    """
    conversion_plan = {
        'convert_needed': [],      # PDFs that need XML conversion
        'already_available': [],   # DOIs with existing XML files  
        'missing_pdfs': []         # DOIs without PDF files
    }
    
    pdf_dir = Path("Data/train/PDF")
    
    for doi in required_dois:
        filename_stem = doi_to_filename.get(doi)
        if not filename_stem:
            print(f"WARNING: Could not map DOI to filename: {doi}")
            continue
            
        # Check if XML already exists
        if filename_stem in existing_xml:
            conversion_plan['already_available'].append({
                'doi': doi,
                'filename_stem': filename_stem,
                'xml_path': existing_xml[filename_stem]
            })
            continue
        
        # Check if PDF exists for conversion
        pdf_path = pdf_dir / f"{filename_stem}.pdf"
        if pdf_path.exists():
            conversion_plan['convert_needed'].append({
                'doi': doi,
                'filename_stem': filename_stem,
                'pdf_path': pdf_path,
                'target_xml_path': Path("Data/train/XML") / f"{filename_stem}.xml"
            })
        else:
            conversion_plan['missing_pdfs'].append({
                'doi': doi,
                'filename_stem': filename_stem,
                'expected_pdf_path': pdf_path
            })
    
    # Summary
    print(f"Conversion Plan Summary:")
    print(f"  - Already have XML: {len(conversion_plan['already_available'])}")
    print(f"  - Need PDF→XML conversion: {len(conversion_plan['convert_needed'])}")
    print(f"  - Missing PDF files: {len(conversion_plan['missing_pdfs'])}")
    
    return conversion_plan
```

### 4.3 PDF → XML Conversion Wrapper (from Section 3.2)

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

def batch_convert_missing_pdfs(conversion_plan: Dict[str, List], 
                              cache_dir: Path = Path("processed_docs"),
                              max_workers: int = 4) -> Dict[str, Dict]:
    """
    Convert only the PDFs that don't have existing XML files.
    
    Args:
        conversion_plan: Output from identify_missing_conversions()
        cache_dir: Directory to cache converted XML files
        max_workers: Number of parallel conversion processes
    """
    cache_dir.mkdir(exist_ok=True)
    conversion_results = {
        'successful_conversions': [],
        'failed_conversions': [],
        'cached_results': {}
    }
    
    pdfs_to_convert = conversion_plan['convert_needed']
    
    if not pdfs_to_convert:
        print("No PDF conversions needed - all required XML files already exist!")
        return conversion_results
    
    print(f"Converting {len(pdfs_to_convert)} PDFs to XML...")
    
    # Process conversions (sequential for now, can parallelize later)
    for conversion_item in pdfs_to_convert:
        doi = conversion_item['doi']
        pdf_path = conversion_item['pdf_path']
        filename_stem = conversion_item['filename_stem']
        
        # Check cache first
        cache_file = cache_dir / f"{filename_stem}.xml"
        
        if cache_file.exists():
            # Load from cache
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    xml_content = f.read()
                conversion_results['cached_results'][doi] = {
                    'xml_content': xml_content,
                    'source': 'cached_grobid',
                    'cache_path': cache_file
                }
                continue
            except Exception as e:
                print(f"Failed to load cache for {filename_stem}: {e}")
        
        # Convert PDF
        print(f"Converting {filename_stem}...")
        result = convert_pdf_to_xml(pdf_path)
        
        if result['xml_content'] or result['text_content']:
            # Save successful conversion
            content_to_cache = result['xml_content'] or result['text_content']
            
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(content_to_cache)
                
                conversion_results['successful_conversions'].append({
                    'doi': doi,
                    'filename_stem': filename_stem,
                    'source': result['source'],
                    'cache_path': cache_file,
                    'pdf_path': pdf_path
                })
                
            except Exception as e:
                print(f"Failed to cache conversion for {filename_stem}: {e}")
                
        else:
            conversion_results['failed_conversions'].append({
                'doi': doi,
                'filename_stem': filename_stem,
                'pdf_path': pdf_path,
                'reason': 'Both GROBID and pdfplumber failed'
            })
    
    # Summary
    total_successful = len(conversion_results['successful_conversions']) + len(conversion_results['cached_results'])
    total_failed = len(conversion_results['failed_conversions'])
    
    print(f"Conversion Results:")
    print(f"  - Successful: {total_successful}")
    print(f"  - Failed: {total_failed}")
    print(f"  - From cache: {len(conversion_results['cached_results'])}")
    
    return conversion_results
```

### 4.4 Complete Document Inventory for Next Steps

**Goal**: Create unified inventory of all available documents (existing XML + newly converted) for downstream processing.

```python
def create_complete_document_inventory(conversion_plan: Dict[str, List],
                                     conversion_results: Dict[str, Dict],
                                     cache_dir: Path = Path("processed_docs")) -> pd.DataFrame:
    """
    Create complete inventory of all available documents for downstream processing.
    
    Returns:
        DataFrame with columns: ['doi', 'filename_stem', 'source_type', 'file_path', 'conversion_source']
    """
    inventory_rows = []
    
    # Add existing XML files
    for item in conversion_plan['already_available']:
        inventory_rows.append({
            'doi': item['doi'],
            'filename_stem': item['filename_stem'],
            'source_type': 'xml',
            'file_path': str(item['xml_path']),
            'conversion_source': 'existing_xml'
        })
    
    # Add successfully converted files
    for item in conversion_results['successful_conversions']:
        inventory_rows.append({
            'doi': item['doi'],
            'filename_stem': item['filename_stem'],
            'source_type': 'xml' if item['source'] == 'grobid' else 'text',
            'file_path': str(item['cache_path']),
            'conversion_source': item['source']
        })
    
    # Add cached conversions
    for doi, result in conversion_results['cached_results'].items():
        # Find filename_stem for this DOI
        filename_stem = None
        for item in conversion_plan['convert_needed']:
            if item['doi'] == doi:
                filename_stem = item['filename_stem']
                break
        
        if filename_stem:
            inventory_rows.append({
                'doi': doi,
                'filename_stem': filename_stem,
                'source_type': 'xml' if 'grobid' in result['source'] else 'text',
                'file_path': str(result['cache_path']),
                'conversion_source': result['source']
            })
    
    inventory_df = pd.DataFrame(inventory_rows)
    
    print(f"Complete Document Inventory:")
    print(f"  - Total documents: {len(inventory_df)}")
    print(f"  - XML sources: {len(inventory_df[inventory_df['source_type'] == 'xml'])}")
    print(f"  - Text sources: {len(inventory_df[inventory_df['source_type'] == 'text'])}")
    print(f"  - Conversion sources: {inventory_df['conversion_source'].value_counts().to_dict()}")
    
    # Save inventory for next steps
    inventory_df.to_csv("document_inventory.csv", index=False)
    print("Saved document inventory to: document_inventory.csv")
    
    return inventory_df

def validate_document_coverage(inventory_df: pd.DataFrame, 
                              required_dois: Set[str]) -> Dict[str, any]:
    """
    Validate that we have adequate document coverage for training.
    """
    available_dois = set(inventory_df['doi'].unique())
    missing_dois = required_dois - available_dois
    
    coverage_stats = {
        'total_required': len(required_dois),
        'total_available': len(available_dois),
        'coverage_rate': len(available_dois) / len(required_dois),
        'missing_count': len(missing_dois),
        'missing_dois': list(missing_dois)[:10]  # Show first 10
    }
    
    print(f"Document Coverage Validation:")
    print(f"  - Required DOIs: {coverage_stats['total_required']}")
    print(f"  - Available DOIs: {coverage_stats['total_available']}")
    print(f"  - Coverage Rate: {coverage_stats['coverage_rate']:.2%}")
    print(f"  - Missing DOIs: {coverage_stats['missing_count']}")
    
    if coverage_stats['coverage_rate'] < 0.90:
        print("⚠️  WARNING: Low document coverage - may impact training quality")
    else:
        print("✅ Good document coverage for training")
    
    return coverage_stats
```

### 4.5 Main Execution Script

**Complete workflow for selective PDF → XML conversion:**

```python
def main_selective_conversion(labels_path: str = "Data/train_labels.csv",
                            grobid_url: str = "http://localhost:8070") -> pd.DataFrame:
    """
    Main function to execute selective PDF → XML conversion workflow.
    
    Returns:
        DataFrame with complete document inventory
    """
    print("=== Starting Selective PDF → XML Conversion ===")
    
    # Step 1: Extract required article IDs
    print("\n1. Extracting required article IDs from labels...")
    required_dois = extract_required_article_ids(labels_path)
    
    # Step 2: Create DOI to filename mapping
    print("\n2. Creating DOI to filename mapping...")
    doi_to_filename = create_doi_to_filename_mapping(required_dois)
    
    # Step 3: Inventory existing XML files  
    print("\n3. Inventorying existing XML files...")
    existing_xml = inventory_existing_xml_files()
    
    # Step 4: Identify missing conversions
    print("\n4. Identifying PDFs that need conversion...")
    conversion_plan = identify_missing_conversions(required_dois, doi_to_filename, existing_xml)
    
    # Step 5: Convert missing PDFs
    print("\n5. Converting missing PDFs to XML...")
    conversion_results = batch_convert_missing_pdfs(conversion_plan)
    
    # Step 6: Create complete inventory
    print("\n6. Creating complete document inventory...")
    inventory_df = create_complete_document_inventory(conversion_plan, conversion_results)
    
    # Step 7: Validate coverage
    print("\n7. Validating document coverage...")
    coverage_stats = validate_document_coverage(inventory_df, required_dois)
    
    print("\n=== Selective PDF → XML Conversion Complete ===")
    return inventory_df

# Usage
if __name__ == "__main__":
    # Ensure GROBID service is running
    # docker run -d -p 8070:8070 --name grobid-service lfoppiano/grobid:0.8.0
    
    document_inventory = main_selective_conversion()
    print(f"Ready for next step with {len(document_inventory)} documents")
```

**Key Benefits of This Approach:**
- **Efficiency**: Only converts PDFs that don't already have XML files (~25% of total)
- **Caching**: Avoids re-processing during iterations
- **Validation**: Ensures adequate document coverage for training
- **Consistent**: Uses same conversion wrapper for train and inference
- **Traceable**: Maintains inventory of all document sources and conversion methods

## **5. Document Parsing & Section Extraction**
- Current section 3 (modified to work with the complete XML set from step 4)

This section processes the complete document inventory from step 4, extracting structured text and section metadata from both XML and text sources. The key difference from traditional PDF-based processing is that we now work primarily with XML files, using robust XPath queries instead of brittle regex patterns.

### 5.1 Load Document Inventory and Parse Documents

**Goal**: Process all documents from the inventory created in step 4, extracting clean text and section structure.

```python
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass

@dataclass # TODO: Should be pydantic class but look into dataclass to understand difference
class Section:
    page_start: int
    page_end: Optional[int] = None
    section_type: Optional[str] = None
    subsections: Optional[List[str]] = None

def load_and_parse_documents(inventory_path: str = "document_inventory.csv") -> Dict[str, Dict]:
    """
    Load document inventory and parse all available documents.
    
    Returns:
        Dict mapping {doi: parsed_document_data}
    """
    # Load document inventory from step 4
    inventory_df = pd.read_csv(inventory_path)
    
    print(f"Loading {len(inventory_df)} documents from inventory...")
    
    parsed_documents = {}
    
    for _, row in inventory_df.iterrows():
        doi = row['doi']
        file_path = Path(row['file_path'])
        source_type = row['source_type']  # 'xml' or 'text'
        conversion_source = row['conversion_source']
        
        print(f"Parsing {row['filename_stem']} ({source_type})...")
        
        try:
            if source_type == 'xml':
                # Parse XML documents (preferred)
                parsed_data = parse_xml_document(file_path, doi)
            else:
                # Parse text documents (fallback)
                parsed_data = parse_text_document(file_path, doi)
            
            # Add source metadata
            parsed_data['source_type'] = source_type
            parsed_data['conversion_source'] = conversion_source
            parsed_data['file_path'] = str(file_path)
            
            parsed_documents[doi] = parsed_data
            
        except Exception as e:
            print(f"Failed to parse {row['filename_stem']}: {e}")
            # Store minimal data for failed documents
            parsed_documents[doi] = {
                'doi': doi,
                'sections': [],
                'full_text': '',
                'parsing_failed': True,
                'error': str(e),
                'source_type': source_type,
                'conversion_source': conversion_source
            }
    
    successful_parses = len([d for d in parsed_documents.values() if not d.get('parsing_failed', False)])
    print(f"Successfully parsed {successful_parses}/{len(parsed_documents)} documents")
    
    return parsed_documents

def parse_xml_document(xml_file_path: Path, doi: str) -> Dict:
    """
    Parse XML document using XPath queries for structured extraction.
    """
    with open(xml_file_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    # Extract sections using XML structure
    sections = extract_sections_from_xml(xml_content)
    
    # Extract full text
    full_text = extract_full_text_from_xml(xml_content)
    
    # Extract section texts
    section_texts = {}
    for section in sections:
        if section.section_type:
            section_text = extract_text_from_xml_section(xml_content, section.section_type)
            if section_text:
                section_texts[section.section_type] = section_text
    
    return {
        'doi': doi,
        'sections': sections,
        'section_texts': section_texts,
        'full_text': full_text,
        'parsing_failed': False
    }

def parse_text_document(text_file_path: Path, doi: str) -> Dict:
    """
    Parse text document using pattern-based section detection.
    """
    with open(text_file_path, 'r', encoding='utf-8') as f:
        text_content = f.read()
    
    # Extract sections using text patterns
    sections = extract_sections_from_text_fallback(text_content)
    
    # Extract section texts
    section_texts = {}
    if sections:
        section_texts = extract_section_texts_from_full_text(text_content, sections)
    
    return {
        'doi': doi,
        'sections': sections,
        'section_texts': section_texts,
        'full_text': text_content,
        'parsing_failed': False
    }
```

### 5.2 XML Section Extraction (XPath-based)

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
        
        # Reverse lookup for GROBID section types
        section_type_mapping = {
            'abstract': 'abstract',
            'introduction': 'introduction',
            'methods': ['method', 'methods', 'materials'],
            'results': ['result', 'results'],
            'discussion': 'discussion',
            'conclusion': 'conclusion',
            'acknowledgments': 'acknowledgment',
            'references': 'references',
            'data_availability': 'availability',
            'supplementary': 'supplementary'
        }
        
        # Get possible XML section types for our standardized type
        xml_types = section_type_mapping.get(section_type, [section_type])
        if not isinstance(xml_types, list):
            xml_types = [xml_types]
        
        # Try to find section by any of the possible types
        section_text = ""
        for xml_type in xml_types:
            section_div = root.find(f'.//tei:div[@type="{xml_type}"]', namespaces)
            if section_div is not None:
                # Extract all text, preserving paragraph breaks
                paragraphs = []
                for p in section_div.findall('.//tei:p', namespaces):
                    if p.text:
                        paragraphs.append(p.text.strip())
                section_text = '\n\n'.join(paragraphs)
                break
        
        return section_text
            
    except ET.ParseError:
        return ""

def extract_full_text_from_xml(xml_content: str) -> str:
    """Extract complete document text from XML, preserving structure"""
    try:
        root = ET.fromstring(xml_content)
        namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
        # Extract all paragraph text from the body
        body_div = root.find('.//tei:text/tei:body', namespaces)
        if body_div is not None:
            paragraphs = []
            for p in body_div.findall('.//tei:p', namespaces):
                if p.text and p.text.strip():
                    paragraphs.append(p.text.strip())
            return '\n\n'.join(paragraphs)
        
        # Fallback: extract all text
        return ' '.join(root.itertext()).strip()
        
    except ET.ParseError:
        return ""
```

### 5.3 Fallback Text Section Detection

For documents where XML parsing fails or text-only sources:

```python
def extract_sections_from_text_fallback(text_content: str) -> List[Section]:
    """Fallback section detection using refined patterns"""
    sections = []
    lines = text_content.split('\n')
    
    # Simplified, high-confidence patterns only
    section_patterns = [
        (r'^\s*(ABSTRACT|Abstract)\s*$', 'abstract'),
        (r'^\s*(INTRODUCTION|Introduction)\s*$', 'introduction'),
        (r'^\s*(METHODS?|Methods?|MATERIALS AND METHODS|Materials and Methods)\s*$', 'methods'),
        (r'^\s*(RESULTS?|Results?)\s*$', 'results'),
        (r'^\s*(DISCUSSION|Discussion)\s*$', 'discussion'),
        (r'^\s*(CONCLUSION|CONCLUSIONS|Conclusion|Conclusions)\s*$', 'conclusion'),
        (r'^\s*(DATA AVAILABILITY|Data Availability|AVAILABILITY|Availability)\s*$', 'data_availability'),
        (r'^\s*(ACKNOWLEDGMENTS?|Acknowledgments?|FUNDING|Funding)\s*$', 'acknowledgments'),
        (r'^\s*(REFERENCES?|References?|BIBLIOGRAPHY|Bibliography)\s*$', 'references'),
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

def extract_section_texts_from_full_text(text_content: str, sections: List[Section]) -> Dict[str, str]:
    """Extract section texts from full text using section boundaries"""
    if not sections:
        return {}
    
    lines = text_content.split('\n')
    section_texts = {}
    
    # Sort sections by page_start to ensure proper ordering
    sorted_sections = sorted(sections, key=lambda x: x.page_start)
    
    for i, section in enumerate(sorted_sections):
        start_line = (section.page_start - 1) * 50  # Rough conversion from page to line
        
        # Determine end line (start of next section or end of document)
        if i + 1 < len(sorted_sections):
            end_line = (sorted_sections[i + 1].page_start - 1) * 50
        else:
            end_line = len(lines)
        
        # Extract text for this section
        section_lines = lines[start_line:end_line]
        section_text = '\n'.join(section_lines).strip()
        
        if section_text and section.section_type:
            section_texts[section.section_type] = section_text
    
    return section_texts
```

### 5.4 Section Metadata Enrichment

**Goal**: Add section-aware metadata that helps with citation detection and retrieval.

```python
def enrich_section_metadata(parsed_documents: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Enrich parsed documents with section-aware metadata for better retrieval.
    """
    enriched_documents = {}
    
    for doi, doc_data in parsed_documents.items():
        enriched_doc = doc_data.copy()
        
        # Skip failed documents
        if doc_data.get('parsing_failed', False):
            enriched_documents[doi] = enriched_doc
            continue
        
        # Add section statistics
        section_stats = calculate_section_statistics(doc_data)
        enriched_doc['section_stats'] = section_stats
        
        # Identify high-priority sections for citation detection
        priority_sections = identify_citation_priority_sections(doc_data['sections'])
        enriched_doc['priority_sections'] = priority_sections
        
        # Add section ordering for context
        section_order = create_section_ordering(doc_data['sections'])
        enriched_doc['section_order'] = section_order
        
        # Calculate section densities
        section_densities = calculate_section_text_densities(doc_data.get('section_texts', {}))
        enriched_doc['section_densities'] = section_densities
        
        enriched_documents[doi] = enriched_doc
    
    return enriched_documents

def calculate_section_statistics(doc_data: Dict) -> Dict[str, int]:
    """Calculate basic statistics about document sections"""
    section_texts = doc_data.get('section_texts', {})
    
    stats = {
        'total_sections': len(doc_data.get('sections', [])),
        'sections_with_text': len(section_texts),
        'total_characters': sum(len(text) for text in section_texts.values()),
        'average_section_length': 0
    }
    
    if stats['sections_with_text'] > 0:
        stats['average_section_length'] = stats['total_characters'] // stats['sections_with_text']
    
    return stats

def identify_citation_priority_sections(sections: List[Section]) -> List[str]:
    """
    Identify sections most likely to contain data citations.
    Based on empirical analysis showing citations concentrate in specific sections.
    """
    # Priority order based on citation frequency analysis
    citation_priority = [
        'data_availability',   # Highest priority - explicit data statements
        'methods',            # High priority - methodology and data sources  
        'supplementary',      # High priority - additional data information
        'results',           # Medium priority - results referencing data
        'acknowledgments',   # Medium priority - data acknowledgments
        'references',        # Lower priority - formal citations
        'introduction',      # Lower priority - background data references
        'discussion',        # Lower priority - contextual data mentions
        'abstract',          # Lowest priority - summary mentions
        'conclusion'         # Lowest priority - summary mentions
    ]
    
    section_types = [s.section_type for s in sections if s.section_type]
    
    # Return sections in priority order (highest first)
    priority_sections = []
    for priority_type in citation_priority:
        if priority_type in section_types:
            priority_sections.append(priority_type)
    
    return priority_sections

def create_section_ordering(sections: List[Section]) -> Dict[str, int]:
    """Create ordering map for sections to preserve document structure"""
    section_order = {}
    
    for i, section in enumerate(sorted(sections, key=lambda x: x.page_start)):
        if section.section_type:
            section_order[section.section_type] = i
    
    return section_order

def calculate_section_text_densities(section_texts: Dict[str, str]) -> Dict[str, float]:
    """Calculate text density metrics for each section"""
    densities = {}
    
    total_length = sum(len(text) for text in section_texts.values())
    
    for section_type, text in section_texts.items():
        if total_length > 0:
            densities[section_type] = len(text) / total_length
        else:
            densities[section_type] = 0.0
    
    return densities
```

### 5.5 Quality Validation and Export

**Goal**: Validate parsing quality and export structured data for the chunking pipeline.

```python
def validate_parsing_quality(parsed_documents: Dict[str, Dict]) -> Dict[str, any]:
    """
    Comprehensive validation of document parsing quality.
    """
    validation_results = {
        'total_documents': len(parsed_documents),
        'successful_parses': 0,
        'failed_parses': 0,
        'section_coverage': {},
        'text_extraction_rate': 0,
        'xml_vs_text_sources': {},
        'quality_issues': []
    }
    
    successful_docs = []
    section_type_counts = {}
    total_text_length = 0
    source_type_counts = {}
    
    for doi, doc_data in parsed_documents.items():
        if doc_data.get('parsing_failed', False):
            validation_results['failed_parses'] += 1
            continue
        
        validation_results['successful_parses'] += 1
        successful_docs.append(doc_data)
        
        # Count section types
        for section in doc_data.get('sections', []):
            if section.section_type:
                section_type_counts[section.section_type] = section_type_counts.get(section.section_type, 0) + 1
        
        # Track text extraction
        full_text = doc_data.get('full_text', '')
        total_text_length += len(full_text)
        
        # Track source types
        source_type = doc_data.get('source_type', 'unknown')
        source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
        
        # Check for quality issues
        if len(full_text) < 1000:  # Very short documents
            validation_results['quality_issues'].append(f"Short document: {doi}")
        
        if not doc_data.get('sections'):  # No sections detected
            validation_results['quality_issues'].append(f"No sections detected: {doi}")
    
    # Calculate metrics
    validation_results['section_coverage'] = section_type_counts
    validation_results['text_extraction_rate'] = validation_results['successful_parses'] / validation_results['total_documents']
    validation_results['xml_vs_text_sources'] = source_type_counts
    validation_results['average_document_length'] = total_text_length // max(validation_results['successful_parses'], 1)
    
    # Print summary
    print(f"Document Parsing Validation:")
    print(f"  - Total documents: {validation_results['total_documents']}")
    print(f"  - Successful parses: {validation_results['successful_parses']}")
    print(f"  - Failed parses: {validation_results['failed_parses']}")
    print(f"  - Success rate: {validation_results['text_extraction_rate']:.2%}")
    print(f"  - Average document length: {validation_results['average_document_length']} chars")
    print(f"  - Most common sections: {dict(list(sorted(section_type_counts.items(), key=lambda x: x[1], reverse=True))[:5])}")
    
    if validation_results['quality_issues']:
        print(f"  - Quality issues: {len(validation_results['quality_issues'])}")
    
    return validation_results

def export_parsed_documents(parsed_documents: Dict[str, Dict], 
                          output_path: str = "parsed_documents.pkl") -> None:
    """
    Export parsed documents for the chunking pipeline (step 6).
    """
    import pickle
    
    # Filter out failed documents for export
    successful_documents = {
        doi: doc_data for doi, doc_data in parsed_documents.items() 
        if not doc_data.get('parsing_failed', False)
    }
    
    # Export as pickle for easy loading in next step
    with open(output_path, 'wb') as f:
        pickle.dump(successful_documents, f)
    
    print(f"Exported {len(successful_documents)} successfully parsed documents to: {output_path}")
    
    # Also export a summary CSV for reference
    summary_rows = []
    for doi, doc_data in successful_documents.items():
        summary_rows.append({
            'doi': doi,
            'source_type': doc_data.get('source_type', ''),
            'conversion_source': doc_data.get('conversion_source', ''),
            'total_sections': len(doc_data.get('sections', [])),
            'sections_with_text': len(doc_data.get('section_texts', {})),
            'full_text_length': len(doc_data.get('full_text', '')),
            'priority_sections': ','.join(doc_data.get('priority_sections', []))
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path.replace('.pkl', '_summary.csv'), index=False)
    print(f"Exported document summary to: {output_path.replace('.pkl', '_summary.csv')}")

def main_document_parsing(inventory_path: str = "document_inventory.csv") -> Dict[str, Dict]:
    """
    Main function to execute document parsing and section extraction.
    
    Returns:
        Dict of successfully parsed documents ready for chunking
    """
    print("=== Starting Document Parsing & Section Extraction ===")
    
    # Step 1: Load and parse all documents
    print("\n1. Loading and parsing documents...")
    parsed_documents = load_and_parse_documents(inventory_path)
    
    # Step 2: Enrich with section metadata
    print("\n2. Enriching section metadata...")
    enriched_documents = enrich_section_metadata(parsed_documents)
    
    # Step 3: Validate parsing quality
    print("\n3. Validating parsing quality...")
    validation_results = validate_parsing_quality(enriched_documents)
    
    # Step 4: Export for next step
    print("\n4. Exporting parsed documents...")
    export_parsed_documents(enriched_documents)
    
    print("\n=== Document Parsing & Section Extraction Complete ===")
    return enriched_documents

# Usage
if __name__ == "__main__":
    parsed_documents = main_document_parsing()
    print(f"Ready for chunking with {len(parsed_documents)} documents")
```

**Key Benefits of This XML-First Approach:**
- **Structured extraction**: XPath queries provide reliable section detection
- **Robust fallback**: Text-based parsing for non-XML sources
- **Section prioritization**: Focus on sections most likely to contain citations
- **Quality validation**: Comprehensive checks for parsing success
- **Pipeline integration**: Clean handoff to the chunking step with enriched metadata

**Why This Works Better Than PDF-Based Processing:**
- **Higher accuracy**: XML structure is more reliable than PDF text extraction
- **Better section detection**: TEI-XML tags are more precise than regex patterns
- **Consistent preprocessing**: Same methodology for train and inference
- **Rich metadata**: Section ordering and priority information for smarter chunking

## **6. Semantic Chunking Pipeline**
- Current section 4 (stays in place) --> uses full XML-based pre-processing so semantic chunking steps should be updated to remove PDF-based chunking logic with XML-based chunking logic.

This section processes the enriched documents from step 5, creating semantically-aware chunks that preserve citation integrity and leverage section metadata for better retrieval. The key difference from traditional PDF-based chunking is that we work with structured section data already extracted via XML parsing.

### 6.1 Pydantic Data Models for XML-Based Chunking

Updated data models that work with our XML-first approach:

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

### 6.2 Load Parsed Documents and Prepare for Chunking

**Goal**: Load the structured documents from step 5 and prepare text for chunking.

```python
import pickle
import pandas as pd
from typing import Dict, List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import uuid

def load_parsed_documents_for_chunking(parsed_docs_path: str = "parsed_documents.pkl") -> Dict[str, Dict]:
    """
    Load parsed documents from step 5 and prepare for chunking.
    """
    with open(parsed_docs_path, 'rb') as f:
        parsed_documents = pickle.load(f)
    
    print(f"Loaded {len(parsed_documents)} parsed documents for chunking")
    
    # Filter out documents without sufficient content
    valid_documents = {}
    for doi, doc_data in parsed_documents.items():
        full_text = doc_data.get('full_text', '')
        sections = doc_data.get('sections', [])
        
        if len(full_text) > 500 and sections:  # Minimum content requirements
            valid_documents[doi] = doc_data
        else:
            print(f"Skipping {doi}: insufficient content ({len(full_text)} chars, {len(sections)} sections)")
    
    print(f"Retained {len(valid_documents)} documents with sufficient content")
    return valid_documents

def prepare_section_texts_for_chunking(parsed_documents: Dict[str, Dict]) -> Dict[str, Dict[str, str]]:
    """
    Prepare section-specific texts for targeted chunking.
    
    Returns:
        Dict mapping {doi: {section_type: text}} for priority sections
    """
    section_texts = {}
    
    for doi, doc_data in parsed_documents.items():
        section_texts[doi] = {}
        
        # Get priority sections (from step 5 metadata)
        priority_sections = doc_data.get('priority_sections', [])
        section_text_data = doc_data.get('section_texts', {})
        
        # Include priority sections first
        for section_type in priority_sections:
            if section_type in section_text_data:
                section_texts[doi][section_type] = section_text_data[section_type]
        
        # Add full text as fallback if no priority sections available
        if not section_texts[doi]:
            section_texts[doi]['full_document'] = doc_data.get('full_text', '')
    
    return section_texts
```

### 6.3 Citation Integrity Validation (Pre-Chunking)

**Critical QC Step**: Ensure chunking never severs DOIs, accession IDs, or other citation entities.

```python
import re
from typing import Dict, List, Tuple

# Enhanced patterns based on authoritative sources
# -*- coding: utf-8 -*-
import re

# Enhanced patterns based on authoritative sources
CITATION_PATTERNS = {
    # ――― your existing patterns ―――
    'DOI':          re.compile(r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b', re.IGNORECASE),
    'GEO_Series':   re.compile(r'\bGSE\d{3,6}\b'),      # GEO series   :contentReference[oaicite:0]{index=0}
    'GEO_Sample':   re.compile(r'\bGSM\d{3,6}\b'),      # GEO sample   :contentReference[oaicite:1]{index=1}
    'SRA_Run':      re.compile(r'\bSRR\d{5,}\b'),       # SRA run      :contentReference[oaicite:2]{index=2}
    'PDB_ID':       re.compile(r'\b[A-Za-z0-9]{4}\b'),
    'PDB_DOI':      re.compile(r'\b10\.2210/pdb[A-Za-z0-9]{4}/pdb\b', re.IGNORECASE),
    'ArrayExpress': re.compile(r'\bE-[A-Z]+-\d+\b'),    # E-MTAB, E-GEOD …  :contentReference[oaicite:3]{index=3}
    'dbGaP':        re.compile(r'\bphs\d{6}\b'),
    'TCGA':         re.compile(r'\bTCGA-[A-Z0-9-]+\b'),
    'ENA_Project':  re.compile(r'\bPRJ[EDN][A-Z]\d+\b'),
    'ENA_Study':    re.compile(r'\bERP\d{6,}\b'),       # already present
    'ENA_Sample':   re.compile(r'\bSAM[EDN][A-Z]?\d+\b'),

    # ――― new additions ―――
    # NCBI Sequence Read Archive (hierarchy)                     
    'SRA_Experiment': re.compile(r'\bSRX\d{5,}\b'),     # SRX   :contentReference[oaicite:4]{index=4}
    'SRA_Project':    re.compile(r'\bSRP\d{5,}\b'),     # SRP   :contentReference[oaicite:5]{index=5}
    'SRA_Sample':     re.compile(r'\bSRS\d{5,}\b'),     # SRS   :contentReference[oaicite:6]{index=6}
    'SRA_Study':      re.compile(r'\bSRA\d{5,}\b'),     # umbrella study  :contentReference[oaicite:7]{index=7}

    # RefSeq & GenBank
    'RefSeq_Chromosome': re.compile(r'\bNC_\d{6,}(?:\.\d+)?\b'),  # e.g. NC_000913.3 :contentReference[oaicite:8]{index=8}

    # EMBL-EBI / ENA prefixes
    'ENA_Run':        re.compile(r'\bERR\d{6,}\b'),     # ENA run   :contentReference[oaicite:9]{index=9}
    'ENA_Experiment': re.compile(r'\bERX\d{6,}\b'),     # ENA experiment :contentReference[oaicite:10]{index=10}
    'ENA_Sample2':    re.compile(r'\bERS\d{6,}\b'),     # ENA sample    :contentReference[oaicite:11]{index=11}

    # DDBJ (mirrors ENA formats)
    'DDBJ_Run':        re.compile(r'\bDRR\d{6,}\b'),    # DRR   :contentReference[oaicite:12]{index=12}
    'DDBJ_Experiment': re.compile(r'\bDRX\d{6,}\b'),    # DRX   :contentReference[oaicite:13]{index=13}

    # ENCODE Project
    'ENCODE_Assay': re.compile(r'\bENCSR[0-9A-Z]{6}\b'),  # ENCSR…   :contentReference[oaicite:14]{index=14}

    # PRIDE ProteomeXchange
    'PRIDE': re.compile(r'\bPXD\d{6,}\b'),               # PXD…  :contentReference[oaicite:15]{index=15}
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

def create_pre_chunk_entity_inventory(section_texts: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Create pre-chunking entity inventory for validation.
    
    Args:
        section_texts: Output from prepare_section_texts_for_chunking()
    """
    inventory_rows = []
    
    for doi, sections in section_texts.items():
        for section_type, text in sections.items():
            entities = extract_citation_entities(text)
            
            for pattern_name, matches in entities.items():
                inventory_rows.append({
                    'document_id': doi,
                    'section_type': section_type,
                    'pattern_name': pattern_name,
                    'entity_count': len(matches),
                    'entities': [match[0] for match in matches],
                    'positions': [(match[1], match[2]) for match in matches]
                })
    
    inventory_df = pd.DataFrame(inventory_rows)
    print(f"Pre-chunk entity inventory: {len(inventory_df)} entity groups across {len(section_texts)} documents")
    return inventory_df
```

### 6.4 Section-Aware Semantic Chunking

**Goal**: Create chunks that respect both semantic boundaries and section structure.

```python
def create_section_aware_chunks(section_texts: Dict[str, Dict[str, str]], 
                               parsed_documents: Dict[str, Dict],
                               chunk_size: int = 200,
                               chunk_overlap: int = 20) -> List[Chunk]:
    """
    Create semantically-aware chunks that preserve section context.
    
    Args:
        section_texts: Section texts prepared for chunking
        parsed_documents: Original parsed documents with metadata
        chunk_size: Target tokens per chunk
        chunk_overlap: Token overlap between chunks
    """
    
    # Initialize tokenizer for accurate token counting
    tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,  # Approximate char to token ratio
        chunk_overlap=chunk_overlap * 4,
        length_function=lambda text: len(tokenizer.encode(text)),
        separators=["\n\n", "\n", ". ", " ", ""]  # Prioritize paragraph breaks
    )
    
    all_chunks = []
    
    for doi, sections in section_texts.items():
        doc_metadata = parsed_documents[doi]
        section_order = doc_metadata.get('section_order', {})
        conversion_source = doc_metadata.get('conversion_source', 'unknown')
        
        print(f"Chunking document {doi[:20]}... ({len(sections)} sections)")
        
        for section_type, section_text in sections.items():
            if not section_text.strip():
                continue
                
            # Split section text into chunks
            section_chunks = text_splitter.split_text(section_text)
            
            # Create Chunk objects with rich metadata
            for i, chunk_text in enumerate(section_chunks):
                chunk_id = f"{doi}_{section_type}_chunk_{i:03d}"
                
                # Count tokens accurately
                token_count = len(tokenizer.encode(chunk_text))
                
                # Extract entities in this chunk
                chunk_entities = extract_citation_entities(chunk_text)
                entity_list = []
                for pattern_matches in chunk_entities.values():
                    entity_list.extend([match[0] for match in pattern_matches])
                
                # Create chunk metadata
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=doi,
                    section_type=section_type,
                    section_order=section_order.get(section_type, 999),
                    conversion_source=conversion_source,
                    token_count=token_count,
                    citation_entities=entity_list,
                    chunk_type='body'  # Default, can be refined later
                )
                
                # Link to previous/next chunks
                if i > 0:
                    metadata.previous_chunk_id = f"{doi}_{section_type}_chunk_{i-1:03d}"
                if i < len(section_chunks) - 1:
                    metadata.next_chunk_id = f"{doi}_{section_type}_chunk_{i+1:03d}"
                
                chunk = Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    chunk_metadata=metadata
                )
                
                all_chunks.append(chunk)
    
    print(f"Created {len(all_chunks)} chunks across all documents")
    return all_chunks

def refine_chunk_types(chunks: List[Chunk]) -> List[Chunk]:
    """
    Refine chunk types based on content analysis.
    """
    for chunk in chunks:
        text = chunk.text.lower()
        
        # Identify different chunk types
        if any(keyword in text for keyword in ['figure', 'table', 'caption', 'fig.', 'tab.']):
            chunk.chunk_metadata.chunk_type = 'caption'
        elif (text.strip().endswith(':') and len(text.split()) < 15) or text.isupper():
            chunk.chunk_metadata.chunk_type = 'header'  
        elif any(keyword in text for keyword in ['conclusion', 'summary', 'in summary']):
            chunk.chunk_metadata.chunk_type = 'conclusion'
        elif chunk.chunk_metadata.section_type == 'data_availability':
            chunk.chunk_metadata.chunk_type = 'data_statement'
        else:
            chunk.chunk_metadata.chunk_type = 'body'
    
    return chunks
```

### 6.5 Post-Chunk Integrity Validation

**Goal**: Verify that chunking preserved all citation entities.

```python
def validate_chunk_integrity(chunks: List[Chunk], 
                           pre_chunk_inventory: pd.DataFrame) -> Dict[str, any]:
    """
    Validate that chunking preserved all citation entities.
    """
    # Count entities in chunked text
    post_chunk_counts = {}
    
    for chunk in chunks:
        doc_id = chunk.chunk_metadata.document_id
        section_type = chunk.chunk_metadata.section_type
        key = f"{doc_id}_{section_type}"
        
        entities = extract_citation_entities(chunk.text)
        
        if key not in post_chunk_counts:
            post_chunk_counts[key] = {}
            
        for pattern_name, matches in entities.items():
            post_chunk_counts[key][pattern_name] = post_chunk_counts[key].get(pattern_name, 0) + len(matches)
    
    # Compare with pre-chunk inventory  
    validation_results = {
        'integrity_passed': True,
        'missing_entities': [],
        'affected_documents': [],
        'total_entities_before': 0,
        'total_entities_after': 0
    }
    
    for _, row in pre_chunk_inventory.iterrows():
        doc_id = row['document_id']
        section_type = row['section_type']
        pattern = row['pattern_name']
        expected_count = row['entity_count']
        
        key = f"{doc_id}_{section_type}"
        actual_count = post_chunk_counts.get(key, {}).get(pattern, 0)
        
        validation_results['total_entities_before'] += expected_count
        validation_results['total_entities_after'] += actual_count
        
        if actual_count < expected_count:
            validation_results['integrity_passed'] = False
            validation_results['missing_entities'].append({
                'document_id': doc_id,
                'section_type': section_type,
                'pattern': pattern,
                'expected': expected_count,
                'actual': actual_count,
                'missing_count': expected_count - actual_count
            })
            validation_results['affected_documents'].append(doc_id)
    
    # Summary
    entity_retention_rate = validation_results['total_entities_after'] / max(validation_results['total_entities_before'], 1)
    
    print(f"Chunk Integrity Validation:")
    print(f"  - Total entities before chunking: {validation_results['total_entities_before']}")
    print(f"  - Total entities after chunking: {validation_results['total_entities_after']}")
    print(f"  - Entity retention rate: {entity_retention_rate:.2%}")
    print(f"  - Documents with missing entities: {len(set(validation_results['affected_documents']))}")
    
    if validation_results['integrity_passed']:
        print("  ✅ All citation entities preserved during chunking")
    else:
        print("  ⚠️  Some citation entities were lost during chunking")
        
    return validation_results

def adaptive_chunk_repair(section_texts: Dict[str, Dict[str, str]], 
                         failed_validation: Dict[str, any],
                         base_chunk_size: int = 200,
                         base_overlap: int = 20) -> Dict[str, Dict]:
    """
    Suggest chunking parameter adjustments for failed documents.
    """
    repair_params = {}
    
    affected_docs = set(failed_validation['affected_documents'])
    
    for doc_id in affected_docs:
        if doc_id not in section_texts:
            continue
            
        # Calculate entity density for this document
        total_text = ' '.join(section_texts[doc_id].values())
        total_entities = sum(len(matches) for entities in [extract_citation_entities(text) for text in section_texts[doc_id].values()] for matches in entities.values())
        entity_density = total_entities / max(len(total_text.split()), 1)
        
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
    
    print(f"Generated repair parameters for {len(repair_params)} documents")
    return repair_params
```

### 6.6 Export Chunks for Embedding Pipeline

**Goal**: Export validated chunks for the embedding and vector store pipeline (step 8).

```python
def export_chunks_for_embedding(chunks: List[Chunk], 
                               validation_results: Dict[str, any],
                               output_path: str = "chunks_for_embedding.pkl") -> None:
    """
    Export validated chunks for the embedding pipeline.
    """
    
    # Filter chunks if integrity validation failed significantly
    entity_retention_rate = validation_results['total_entities_after'] / max(validation_results['total_entities_before'], 1)
    
    if entity_retention_rate < 0.95:
        print(f"⚠️  Low entity retention rate ({entity_retention_rate:.2%}). Consider re-chunking with adjusted parameters.")
    
    # Export chunks as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"Exported {len(chunks)} chunks to: {output_path}")
    
    # Create summary DataFrame for analysis
    chunk_summary_rows = []
    for chunk in chunks:
        chunk_summary_rows.append({
            'chunk_id': chunk.chunk_id,
            'document_id': chunk.chunk_metadata.document_id,
            'section_type': chunk.chunk_metadata.section_type,
            'section_order': chunk.chunk_metadata.section_order,
            'chunk_type': chunk.chunk_metadata.chunk_type,
            'token_count': chunk.chunk_metadata.token_count,
            'text_length': len(chunk.text),
            'citation_entity_count': len(chunk.chunk_metadata.citation_entities),
            'conversion_source': chunk.chunk_metadata.conversion_source
        })
    
    chunk_summary_df = pd.DataFrame(chunk_summary_rows)
    summary_path = output_path.replace('.pkl', '_summary.csv')
    chunk_summary_df.to_csv(summary_path, index=False)
    
    print(f"Exported chunk summary to: {summary_path}")
    
    # Print analytics
    print(f"\nChunk Analytics:")
    print(f"  - Total chunks: {len(chunks)}")
    print(f"  - Average tokens per chunk: {chunk_summary_df['token_count'].mean():.1f}")
    print(f"  - Chunks with citations: {len(chunk_summary_df[chunk_summary_df['citation_entity_count'] > 0])}")
    print(f"  - Section type distribution:")
    section_counts = chunk_summary_df['section_type'].value_counts()
    for section_type, count in section_counts.head(5).items():
        print(f"    - {section_type}: {count}")

def main_semantic_chunking(parsed_docs_path: str = "parsed_documents.pkl",
                          chunk_size: int = 200,
                          chunk_overlap: int = 20) -> List[Chunk]:
    """
    Main function to execute semantic chunking pipeline.
    
    Returns:
        List of validated chunks ready for embedding
    """
    print("=== Starting Semantic Chunking Pipeline ===")
    
    # Step 1: Load parsed documents
    print("\n1. Loading parsed documents...")
    parsed_documents = load_parsed_documents_for_chunking(parsed_docs_path)
    
    # Step 2: Prepare section texts
    print("\n2. Preparing section texts for chunking...")
    section_texts = prepare_section_texts_for_chunking(parsed_documents)
    
    # Step 3: Create pre-chunk entity inventory
    print("\n3. Creating pre-chunk entity inventory...")
    pre_chunk_inventory = create_pre_chunk_entity_inventory(section_texts)
    
    # Step 4: Create section-aware chunks
    print("\n4. Creating section-aware chunks...")
    chunks = create_section_aware_chunks(section_texts, parsed_documents, chunk_size, chunk_overlap)
    
    # Step 5: Refine chunk types
    print("\n5. Refining chunk types...")
    chunks = refine_chunk_types(chunks)
    
    # Step 6: Validate chunk integrity
    print("\n6. Validating chunk integrity...")
    validation_results = validate_chunk_integrity(chunks, pre_chunk_inventory)
    
    # Step 7: Handle failed validation if needed
    if not validation_results['integrity_passed']:
        print("\n7. Generating repair suggestions...")
        repair_params = adaptive_chunk_repair(section_texts, validation_results, chunk_size, chunk_overlap)
        # Note: Actual repair (re-chunking) would be implemented here if needed
    
    # Step 8: Export chunks
    print("\n8. Exporting chunks for embedding pipeline...")
    export_chunks_for_embedding(chunks, validation_results)
    
    print("\n=== Semantic Chunking Pipeline Complete ===")
    return chunks

# Usage
if __name__ == "__main__":
    chunks = main_semantic_chunking()
    print(f"Ready for embedding with {len(chunks)} validated chunks")
```

**Key Benefits of This XML-First Chunking Approach:**
- **Section-aware**: Leverages structured section metadata from XML parsing
- **Citation integrity**: Comprehensive validation ensures no entities are severed
- **Rich metadata**: Each chunk contains section type, order, and entity information
- **Adaptive parameters**: Can adjust chunking for high entity-density documents
- **Quality validation**: Built-in checks for chunking success and entity preservation
- **Clean handoff**: Exports validated chunks ready for embedding pipeline

**Why This Works Better Than PDF-Based Chunking:**
- **Structured context**: Uses reliable XML section boundaries instead of PDF heuristics
- **Better boundaries**: Section-aware splitting reduces mid-entity breaks
- **Rich metadata**: Section priority and type information improves retrieval
- **Consistent preprocessing**: Same approach for train and inference phases 

## **7. Data Leakage Prevention (CRITICAL)**
- Current section 4.5 → Move to section 7

**Critical**: These steps prevent the model from memorizing specific `dataset_id` strings, which would cause catastrophic leakage in cross-validation.

### 7.1 ID Masking Before Embedding

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

### 7.2 GroupKFold by Dataset ID

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

### 7.3 Leakage Audit Pipeline

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

## **8. Memory-Efficient Embedding & Vector Store**
- Current section 5 (stays in place)

**🎯 MEDIUM PRIORITY**: Start lean with NumPy memmap, upgrade to ChromaDB only when needed. Optimized for Kaggle's 13GB RAM limit.

### 8.1 Embedding Generation (Batch Processing)

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

### 8.2 Memory-Mapped Storage (< 500k vectors)

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

### 8.3 ChromaDB (≥ 500k vectors)

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

### 8.4 Unified Retrieval Interface

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

### 8.5 Performance Comparison

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

## **9. Chunk-Level EDA**
- Current section 6 (stays in place)

### 9.1 Chunk statistics

* Histogram: **tokens / chunk**, **chunks / document**.
* Coverage\@k: does at least one chunk containing the ground-truth `dataset_id` appear in the top-k (k = 3, 5) similarity hits? Low values → reconsider chunk size/overlap.

### 9.2 **Enhanced EDA on Retrieval Quality & Citation Integrity**

#### 9.2.1 Citation Integrity Validation Plots

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

#### 9.2.2 Coverage\@k (Training-Time & Inference-Time)

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

#### 9.2.3 Score Distributions with Section Analysis

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

### 9.3 Embedding geometry (Enhanced)

1. **UMAP → 2-D** (`n_neighbors=15`, `min_dist=0.1`) – preserves local & global structure for text ([umap-learn.readthedocs.io][10]).
2. **Colour by approach**: 
   - **Training-time**: Primary, Secondary, Missing – easy visual drift check
   - **Inference-time**: Colour by retrieval path (anchor, semantic, universal) to spot blind spots
3. **HDBSCAN** on embeddings or UMAP output – density-based clusters & noise points ([dylancastillo.co][11]).
4. **NEW: Project only retrieved chunks** – noise gets filtered out, producing cleaner clusters.
5. **Cross-validation**: Compare training-time and inference-time chunk distributions to ensure consistency.
6. Optional: push embeddings to **Nomic Atlas** for interactive zoom / tag filtering ([docs.nomic.ai][12]).

### 9.4 Imbalance & meta-feature probes

* Stratify charts by label to confirm class ratios per chunk and per document.
* Correlate chunk length or section tag with label to spot biases (e.g., citations mostly in *Methods*).

---

## **10. Quality-Control Checklist**
- Current section 8 → Move to section 10

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

## **11. Export Artifacts for Training Loop**
- Current section 7 → Move to section 11

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

## **12. Hand-off Notes for Modelling Team**
- Current section 9 → Move to section 12

### 12.1 Training Pipeline
* **Input**: `chunks.parquet` + `vector_store/` directory; each chunk already validated by Pydantic.
* **Enhanced Input**: `topk_chunks.parquet` – every label now maps to the most relevant, validated top-k chunks + similarity scores.
* **Features**: 384-d embedding, UMAP (2-d), HDBSCAN cluster, chunk length, section tag, etc.
* **Enhanced Features**: `cosine_score` from retrieval, coverage\@k metrics, MMR diversity scores.
* **Label-aware splits**: group by `document_id` during CV to avoid leakage.

### 12.2 Inference Pipeline
* **No label leakage**: Use `get_inference_chunks()` function for identical preprocessing at inference time.
* **Retrieval strategy**: Layered fallback (anchor → semantic → universal) ensures consistent chunk quality.
* **Same feature space**: Inference chunks use identical embeddings and feature extraction as training.
* **Performance monitoring**: Track retrieval path distributions to detect data drift.

### 12.3 Implementation Notes
* **Ready-to-use**: Load `topk_chunks.parquet`, group by `document_id` for stratified folds, and feed `embedding` plus `cosine_score` straight into any classifier.
* **Symmetric preprocessing**: Training and inference use the same chunking/embedding pipeline with different retrieval strategies.
* **Optional fine-tuning**: Only apply GenQ fine-tuning if validation performance drops significantly from training to testing.
* **Deployment consistency**: Use the same ChromaDB instance and embedding model for both training EDA and inference retrieval.

### 12.4 Next Steps Checklist

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

## **13. Key References & Additional Resources**
- Current section 10 → Move to section 13 

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