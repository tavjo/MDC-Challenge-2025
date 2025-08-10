#!/usr/bin/env python3
"""
Semantic Chunking Pipeline for MDC-Challenge-2025
Creates chunks from Document objects using semantic chunking + ChromaDB storage
"""

import json
import os
import sys
import uuid
import re
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Local imports
from src.models import Document, CitationEntity, Chunk, ChunkMetadata, ChunkingResult, DocumentChunkingResult
from src.helpers import initialize_logging, timer_wrap, load_docs, preprocess_text, sliding_window_chunks
from src.semantic_chunking import sliding_window_chunk_text, save_chunk_objs_to_chroma
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
# from src.get_citation_entities import CitationEntityExtractor
import duckdb
import time
from api.utils.duckdb_utils import get_duckdb_helper
import tiktoken

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

# ---------------------------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------------------------

@timer_wrap
def load_input_data(documents_path: str = "Data/train/documents_with_known_entities.json", 
                   citation_entities_path: str = "Data/citation_entities_known.json") -> Tuple[List[Document], List[CitationEntity]]:
    """
    Load Document and CitationEntity instances from JSON files.
    
    Args:
        documents_path: Path to documents JSON file
        citation_entities_path: Path to citation entities JSON file
        
    Returns:
        Tuple of (documents, citation_entities)
    """
    logger.info(f"Loading documents from: {documents_path}")
    documents_path = os.path.join(project_root, documents_path)
    documents = load_docs(documents_path)
    
    logger.info(f"Loading citation entities from: {citation_entities_path}")
    citation_entities_path = os.path.join(project_root, citation_entities_path)
    with open(citation_entities_path, "r") as f:
        raw_citations = json.load(f)
    citation_entities = [CitationEntity.model_validate(item) for item in raw_citations]
    
    logger.info(f"✅ Loaded {len(documents)} documents and {len(citation_entities)} citation entities")
    return documents, citation_entities

@timer_wrap
def load_input_data_from_duckdb(db_path: str = "artifacts/mdc_challenge.db") -> Tuple[List[Document], List[CitationEntity]]:
    """
    Load Document and CitationEntity instances from DuckDB database.
    
    Args:
        db_path: Path to DuckDB database file
        
    Returns:
        Tuple of (documents, citation_entities)
    """
    logger.info(f"Loading data from DuckDB: {db_path}")
    
    try:
        # Connect to DuckDB
        conn = duckdb.connect(db_path)
        
        # Load documents
        logger.info("Loading documents from DuckDB...")
        doc_query = "SELECT * FROM documents"
        doc_result = conn.execute(doc_query).fetchall()
        doc_columns = [desc[0] for desc in conn.description]
        
        documents = []
        for row in doc_result:
            row_dict = dict(zip(doc_columns, row))
            document = Document.from_duckdb_row(row_dict)
            documents.append(document)
        
        # Load citation entities
        logger.info("Loading citation entities from DuckDB...")
        citation_query = "SELECT * FROM citations"
        citation_result = conn.execute(citation_query).fetchall()
        citation_columns = [desc[0] for desc in conn.description]
        
        citation_entities = []
        for row in citation_result:
            row_dict = dict(zip(citation_columns, row))
            citation = CitationEntity.from_duckdb_row(row_dict)
            citation_entities.append(citation)
        
        conn.close()
        
        logger.info(f"✅ Loaded {len(documents)} documents and {len(citation_entities)} citation entities from DuckDB")
        return documents, citation_entities
        
    except Exception as e:
        logger.error(f"❌ Failed to load data from DuckDB: {str(e)}")
        raise

# ---------------------------------------------------------------------------
# Pre-Chunk Entity Inventory (Using Exact Citations)
# ---------------------------------------------------------------------------

def make_pattern(dataset_id: str) -> re.Pattern:
    """Create pattern for dataset citation: substring match anywhere, case-insensitive."""
    pat = re.escape(dataset_id)
    return re.compile(pat, flags=re.IGNORECASE)

def find_citation_in_text(citation: str, text: str) -> List[str]:
    """Robust dual matching: raw substring or preprocessed substring."""
    lower_cit = re.escape(citation.lower())
    # find all occurences of citation in the raw chunk text
    raw_matches = re.findall(lower_cit, text.lower())
    preprocessed_matches = re.findall(lower_cit, preprocess_text(text))
    return raw_matches + preprocessed_matches

@timer_wrap
def create_pre_chunk_entity_inventory(document: Document, citation_entities: List[CitationEntity]) -> pd.DataFrame:
    """
    Count exact dataset citations before chunking using CitationEntity objects.
    
    Args:
        document: Document object
        citation_entities: List of CitationEntity objects
        
    Returns:
        DataFrame with citation counts by document
    """
    logger.info(f"Creating pre-chunk entity inventory for document {document.doi}")
    
    # Filter citations for this document
    doc_citations = [citation for citation in citation_entities if citation.document_id == document.doi]
    
    if not doc_citations:
        logger.info(f"No citations found for document {document.doi}")
        return pd.DataFrame(columns=['document_id', 'citation_id', 'count'])
    
    logger.info(f"Found {len(doc_citations)} citations for document {document.doi}")
    
    # Standardize text format
    if isinstance(document.full_text, list):
        text = " ".join(document.full_text)
    else:
        text = document.full_text
    
    # Cache compiled patterns
    pattern_cache = {}
    rows = []
    
    for citation in doc_citations:
        # Cache pattern compilation
        if citation.data_citation not in pattern_cache:
            pattern_cache[citation.data_citation] = make_pattern(citation.data_citation)
        pattern = pattern_cache[citation.data_citation]
        # preprocess text before matching
        matches = pattern.findall(preprocess_text(text))
        if not matches:
            matches = find_citation_in_text(citation.data_citation, text)

        rows.append({
            'document_id': document.doi,
            'citation_id': citation.data_citation,
            'count': len(matches),
        })
    
    inventory_df = pd.DataFrame(rows)
    
    # Summary statistics
    total_citations = len(doc_citations)
    citations_found = len(inventory_df[inventory_df['count'] > 0])
    
    logger.info(f"✅ Created entity inventory:")
    logger.info(f"📊 Total citations in document: {total_citations}")
    logger.info(f"📊 Citations with matches: {citations_found}")
    
    return inventory_df

# ---------------------------------------------------------------------------
# Chunk Creation (Using Semantic Chunking)
# ---------------------------------------------------------------------------

@timer_wrap
def create_chunks_from_document(document: Document, citation_entities: List[CitationEntity], 
                                chunk_size: int = 300, chunk_overlap: int = 5) -> List[Chunk]:
    """
    Create chunks using semantic_chunking.py functions with citation assignment.
    
    Args:
        document: Document object
        citation_entities: List of CitationEntity objects
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        
    Returns:
        List of Chunk objects
    """
    logger.info(f"Creating semantic chunks for {document.doi}")
    
    # Group citation entities by document
    citations_by_doc = defaultdict(list)
    for citation in citation_entities:
        citations_by_doc[citation.document_id].append(citation)
    
    chunks = []
    
    doc = document
    # Standardize text format
    if isinstance(doc.full_text, list):
        text = " ".join(doc.full_text)
    else:
        text = doc.full_text

    # Use semantic chunking with overrides from pipeline
    raw_chunks = sliding_window_chunk_text(
        text
        )
    # Fallback: ensure no chunk exceeds the hard token limit
    # import tiktoken
    tok = tiktoken.get_encoding("cl100k_base")
    processed_chunks = []
    for chunk_text in raw_chunks:
        token_ids = tok.encode(chunk_text)
        if len(token_ids) <= 1500: 
            processed_chunks.append(chunk_text)
        else:
            logger.warning(f"Chunk too large ({len(token_ids)} tokens); splitting into smaller chunks using sliding window method.")
            processed_chunk = sliding_window_chunks(chunk_text, chunk_size)
            processed_chunks.extend(processed_chunk)
    # Preprocess text on all final chunks
    text_chunks = [pc for pc in processed_chunks]
    
    # Get citations for this document
    doc_citations = citations_by_doc.get(doc.doi, [])
    
    # Create Chunk objects with metadata
    for i, chunk_text in enumerate(text_chunks):
        chunk_id = f"{doc.doi}_{i}"
        
        # Find which citations appear in this chunk
        chunk_citations = []
        for citation in doc_citations:
            pattern = make_pattern(citation.data_citation)
            # match against preprocessed chunk text
            if pattern.search(preprocess_text(chunk_text)):
                chunk_citations.append(citation)
            elif find_citation_in_text(citation.data_citation, chunk_text):
                chunk_citations.append(citation)
            else:
                logger.debug(f"No match found for citation {citation.data_citation} in chunk {chunk_id}. Returning empty list.")
        
        # Use tiktoken for accurate token counting
        # import tiktoken
        tok = tiktoken.get_encoding("cl100k_base")
        token_count = len(tok.encode(chunk_text))
        
        chunk_meta = ChunkMetadata(
            chunk_id=chunk_id,
            token_count=token_count,
            citation_entities=chunk_citations,
            previous_chunk_id=None,  # Will be set by link_adjacent_chunks
            next_chunk_id=None
        )
        
        chunks.append(Chunk(
            chunk_id=chunk_id,
            document_id=doc.doi,
            text=chunk_text,
            chunk_metadata=chunk_meta
        ))
    
    logger.info(f"✅ Created {len(chunks)} chunks from {document.doi}")
    return chunks 

# ---------------------------------------------------------------------------
# Chunk Linking (Adapted from Deprecated Script)
# ---------------------------------------------------------------------------

@timer_wrap
def link_adjacent_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    Link adjacent chunks with previous/next chunk IDs.
    
    Args:
        chunks: List of Chunk objects
        
    Returns:
        List of Chunk objects with linked IDs
    """
    logger.info(f"Linking adjacent chunks for {len(chunks)} chunks")
    
    # Group chunks by document
    # by_document = defaultdict(list)
    # for i, chunk in enumerate(chunks):
    #     document_id = chunk.document_id
    #     by_document[document_id].append((i, chunk.text, chunk.chunk_metadata.model_dump()))

    doc_chunks = []
    for i, chunk in enumerate(chunks):
        doc_chunks.append((i, chunk.text, chunk.chunk_metadata.model_dump(), chunk.document_id))
    
    # Link chunks within each document
    linked_chunks = []
    # for document_id, doc_chunks in by_document.items():
    # Sort by chunk position (assuming order from semantic chunking)
    doc_chunks.sort(key=lambda x: x[0])
    
    for i, (original_idx, text, metadata, document_id) in enumerate(doc_chunks):
        # Set previous chunk ID
        if i > 0:
            prev_metadata = doc_chunks[i-1][2]
            metadata['previous_chunk_id'] = prev_metadata['chunk_id']
        
        # Set next chunk ID
        if i < len(doc_chunks) - 1:
            next_metadata = doc_chunks[i+1][2]
            metadata['next_chunk_id'] = next_metadata['chunk_id']
        
        new_chunk = Chunk(
            chunk_id=metadata['chunk_id'],
            document_id=document_id,
            text=text,
            chunk_metadata=ChunkMetadata.model_validate(metadata)
        )
        linked_chunks.append(new_chunk)
    
    logger.info(f"✅ Linked {len(linked_chunks)} chunks")
    return linked_chunks 

# ---------------------------------------------------------------------------
# Validation (Using Exact Citations)
# ---------------------------------------------------------------------------

@timer_wrap
def validate_chunk_integrity(chunks: List[Chunk], pre_chunk_inventory: pd.DataFrame, 
                           citation_entities: List[CitationEntity]) -> Tuple[bool, pd.DataFrame]:
    """
    Validate that exact citations are preserved after chunking.
    
    Args:
        chunks: List of Chunk objects
        pre_chunk_inventory: Entity inventory before chunking
        citation_entities: List of CitationEntity objects
        
    Returns:
        Tuple of (validation_passed, loss_report)
    """
    logger.info(f"Validating chunk integrity for {len(chunks)} chunks")
    
    # Group citation entities by document
    # citations_by_doc = defaultdict(list)
    # for citation in citation_entities:
    #     citations_by_doc[citation.document_id].append(citation)
    
    # Count citations AFTER chunking
    # before we do any of the things below, check if the pre_chunk_inventory is empty
    if pre_chunk_inventory.empty:
        logger.info(f"Pre-chunk inventory is empty, skipping validation")
        return True, pd.DataFrame(columns=['document_id', 'citation_id', 'count'])
    
    post_chunk_rows = []
    
    for chunk in chunks:
        text = chunk.text
        doc_id = chunk.document_id
        doc_citations = [citation for citation in citation_entities if citation.document_id == doc_id]
        
        for citation in doc_citations:
            pattern = make_pattern(citation.data_citation)
            # match against preprocessed chunk text
            matches = pattern.findall(preprocess_text(text))
            if not matches:
                matches = find_citation_in_text(citation.data_citation, text)
            
            post_chunk_rows.append({
                'document_id': doc_id,
                'citation_id': citation.data_citation,
                'count': len(matches),
            })
    
    # Aggregate post-chunk counts by document
    post_chunk_df = pd.DataFrame(post_chunk_rows)
    if post_chunk_df.empty:
        post_aggregated = pd.DataFrame(columns=['document_id', 'citation_id', 'count'])
    else:
        post_aggregated = (
            post_chunk_df
            .groupby(['document_id', 'citation_id'])['count']
            .sum()
            .reset_index()
        )
    
    # Merge with pre-chunk inventory
    merged = pre_chunk_inventory.merge(
        post_aggregated,
        on=['document_id', 'citation_id'],
        how='left',
        suffixes=('_pre', '_post')
    ).fillna(0)
    
    # Find lost citations
    lost_citations = merged[merged['count_post'] < merged['count_pre']]
    
    # Check if validation passed
    validation_passed = lost_citations.empty
    
    # Summary statistics
    total_citations_pre = pre_chunk_inventory['count'].sum()
    total_citations_post = post_aggregated['count'].sum()
    retention_rate = (total_citations_post / total_citations_pre) * 100 if total_citations_pre > 0 else 100
    
    logger.info(f"✅ Citation integrity validation:")
    logger.info(f"   Pre-chunk citations: {total_citations_pre}")
    logger.info(f"   Post-chunk citations: {total_citations_post}")
    logger.info(f"   Retention rate: {retention_rate:.1f}%")
    
    if validation_passed:
        logger.info(f"✅ SUCCESS: 100% citation retention achieved")
    else:
        logger.error(f"❌ FAILURE: {len(lost_citations)} citation losses detected")
    
    return validation_passed, lost_citations

# ---------------------------------------------------------------------------
# Post-chunk repair to guarantee 100 % citation retention
# ---------------------------------------------------------------------------
from collections import defaultdict
from typing import Dict, List, Tuple

@timer_wrap
@timer_wrap
def repair_lost_citations_strict(
    document: Document,
    chunks: List[Chunk],
    lost_citations: pd.DataFrame,
    chunk_size: int = 300,
    ctx_chars: int = 60,
) -> Tuple[List[Chunk], Dict[str, int]]:
    """
    Strictly re-inserts *every* missing copy of every citation ID.
    Keeps raw text (no preprocessing) so that validation works.
    """
    tok = tiktoken.get_encoding("cl100k_base")
    if lost_citations.empty:
        return chunks, {"inserted": 0, "merged": 0, "total_after": len(chunks)}

    # ----- 0)  helpers ------------------------------------------------------
    full_text = (
        document.full_text if isinstance(document.full_text, str)
        else " ".join(document.full_text)
    )
    # preprocess full text for matching
    normalized_full_text = preprocess_text(full_text)
    pattern_cache = {
        cid: make_pattern(cid) for cid in lost_citations["citation_id"].unique()
    }

    def snippet_around(match: re.Match) -> str:
        """Return a raw snippet centred on the match, < chunk_size tokens."""
        left = max(0, match.start() - ctx_chars)
        right = min(len(full_text), match.end() + ctx_chars)
        snippet = full_text[left:right]
        # hard-trim if token budget blown
        while len(tok.encode(snippet)) > chunk_size and ctx_chars:
            left += ctx_chars // 4
            right -= ctx_chars // 4
            snippet = full_text[left:right]
        return snippet

    # ----- 1)  current per-ID counts with normalized chunk text -----------
    current_counts = defaultdict(int)
    for ck in chunks:
        norm_chunk = preprocess_text(ck.text)
        for cid, pat in pattern_cache.items():
            current_counts[cid] += len(pat.findall(norm_chunk))

    # ----- 2)  process each missing ID -------------------------------------
    new_chunks, inserted, merged = [], 0, 0
    id2occ_snip_idx = defaultdict(int)  # how many snippets we already used

    for row in lost_citations.itertuples():
        cid, need = row.citation_id, row.count_pre - current_counts[row.citation_id]
        if need <= 0:
            continue
        pat = pattern_cache[cid]
        matches = list(pat.finditer(normalized_full_text))
        if len(matches) < need:
            logger.warning(f"⚠️  Only {len(matches)} occurrences of {cid} "
                           f"found in raw text, needed {need}.")
            need = len(matches)

        for k in range(need):
            m = matches[id2occ_snip_idx[cid] + k]
            snip = snippet_around(m)
            new_id = f"{document.doi}_fix_{cid}_{k+1}"
            meta = ChunkMetadata(
                chunk_id=new_id,
                token_count=len(tok.encode(snip)),
                citation_entities=[CitationEntity(document_id=document.doi, data_citation=cid)],
                previous_chunk_id=None,
                next_chunk_id=None,
            )
            new_chunks.append(
                Chunk(
                    chunk_id=new_id,
                    document_id=document.doi,
                    text=snip,             # RAW text – do *not* preprocess
                    chunk_metadata=meta,
                )
            )
            inserted += 1
        id2occ_snip_idx[cid] += need

    # ----- 3)  re-thread + return ------------------------------------------
    repaired = link_adjacent_chunks([ck for ck in chunks] + new_chunks)
    stats = {"inserted": inserted, "merged": merged, "total_after": len(repaired)}
    return repaired, stats



# ---------------------------------------------------------------------------
# DuckDB Storage Functions
# ---------------------------------------------------------------------------

@timer_wrap
def save_chunks_to_duckdb(chunks: List[Chunk], db_path: str = "artifacts/mdc_challenge.db") -> bool:
    """
    Save chunks to DuckDB database.
    
    Args:
        chunks: List of Chunk objects to save
        db_path: Path to DuckDB database file
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Saving {len(chunks)} chunks to DuckDB: {db_path}")
    
    try:
        db_helper = get_duckdb_helper(db_path)
        # Use batch upsert for better performance
        if db_helper.store_chunks_batch(chunks):
            logger.info(f"✅ Successfully batch-upserted {len(chunks)} chunks to DuckDB")
            db_helper.close()
            return True
        else:
            logger.error(f"❌ Failed to save chunks to DuckDB")
            return False
        
    except Exception as e:
        logger.error(f"❌ Failed to save chunks to DuckDB: {str(e)}")
        return False

# ---------------------------------------------------------------------------
# Export Functions
# ---------------------------------------------------------------------------

@timer_wrap
def export_chunks_to_json(chunks: List[Chunk], output_path: str = "chunks_for_embedding.json") -> str:
    """
    Export chunks to JSON for visual inspection using model_dump() approach.
    
    Args:
        chunks: List of Chunk objects
        output_path: Path for output JSON file
        
    Returns:
        Path to the exported JSON file
    """
    logger.info(f"Exporting {len(chunks)} chunks to JSON: {output_path}")
    
    # Use model_dump() to get JSON-serializable dicts
    to_export = [chunk.model_dump() for chunk in chunks]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(to_export, f, indent=4, ensure_ascii=False)
    
    logger.info(f"✅ Exported chunks to JSON: {output_path}")
    return output_path

@timer_wrap
def create_chunks_summary_csv(chunks: List[Chunk], export: bool = True, output_path: str = "chunks_for_embedding_summary.csv") -> str:
    """
    Export chunk summary statistics to CSV.
    
    Args:
        chunks: List of Chunk objects
        output_path: Path for output CSV file
        
    Returns:
        Path to the exported CSV file
    """
    logger.info(f"Exporting chunk summary to CSV: {output_path}")
    
    summary_rows = []
    for chunk in chunks:
        row = chunk.chunk_metadata.model_dump()
        row['text_length'] = len(chunk.text)
        row['text_preview'] = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    if export:
        summary_df.to_csv(output_path, index=False)
        logger.info(f"✅ Exported chunk summary to CSV: {output_path}")
    return summary_df 

# ---------------------------------------------------------------------------
# Main Pipeline Function
# ---------------------------------------------------------------------------

@timer_wrap
def prepare_document(
    document: Document,
    citation_entities: List[CitationEntity],
    chunk_size: int,
    chunk_overlap: int,
    cfg_path: str,
    collection_name: str,
) -> dict:
    """
    Phase 1: CPU/I-O-light work for one document.
    Returns a dict with document, chunks, stats, and params.
    """
    start = datetime.now().isoformat()

    # 1) Pre-chunk inventory
    pre_df = create_pre_chunk_entity_inventory(document, citation_entities)
    pre_total = pre_df['count'].sum()

    # 2) Create & link chunks
    chunks = create_chunks_from_document(document, citation_entities, chunk_size, chunk_overlap)
    chunks = link_adjacent_chunks(chunks)

    # 3) Validate & repair
    post_passed, lost_df = validate_chunk_integrity(chunks, pre_df, citation_entities)
    if not post_passed:
        chunks, _ = repair_lost_citations_strict(document, chunks, lost_df)

    # recalc post stats across all citations
    if 'citation_id' in pre_df.columns:
        citation_ids = pre_df['citation_id'].unique()
    else:
        citation_ids = [ce.data_citation for ce in citation_entities] if citation_entities else []
    post_total = sum(
        len(make_pattern(cid).findall(preprocess_text(ck.text)))
        for ck in chunks
        for cid in citation_ids
    )
    total_tokens = sum(ck.chunk_metadata.token_count for ck in chunks)
    avg_tokens = total_tokens / len(chunks) if chunks else 0.0

    return {
        "document": document,
        "chunks": chunks,
        "pre_total_citations": int(pre_total),
        "post_total_citations": int(post_total),
        "validation_passed": post_passed,
        "lost_entities": lost_df.to_dict() if not post_passed else None,
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens,
        "pipeline_started_at": start,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "cfg_path": cfg_path,
        "collection_name": collection_name,
    }

def build_document_result(prep: dict, db_path: str) -> Tuple[DocumentChunkingResult, pd.DataFrame]:
    """
    Phase 2: single-threaded persistence. Returns DocumentChunkingResult.
    Rolls back on DuckDB errors; cleans up on ChromaDB errors.
    """
    doc: Document = prep["document"]
    chunks: List[Chunk] = prep["chunks"]
    start = prep["pipeline_started_at"]
    now = datetime.now().isoformat()
    summary_df = create_chunks_summary_csv(chunks, export=False)
    # create dict of document_id to list of lost citation entities
    lost_entities = None
    if prep["lost_entities"]:
        lost_entities = {doc.doi: prep["lost_entities"]}
    else:
        lost_entities = None
    
    # calculate entity retention
    if lost_entities and doc.doi in lost_entities:
        diff = prep["pre_total_citations"] - len(lost_entities[doc.doi])
        entity_retention = diff / prep["pre_total_citations"] * 100
    else:
        entity_retention = 100.0

    base = dict(
        document_id=doc.doi,
        success=False,
        chunk_size=prep["chunk_size"],
        chunk_overlap=prep["chunk_overlap"],
        cfg_path=prep["cfg_path"],
        collection_name=prep["collection_name"],
        pre_chunk_total_citations=prep["pre_total_citations"],
        post_chunk_total_citations=prep["post_total_citations"],
        validation_passed=prep["validation_passed"],
        entity_retention=entity_retention,
        lost_entities=lost_entities,
        total_chunks=prep["total_chunks"],
        total_tokens=prep["total_tokens"],
        avg_tokens_per_chunk=prep["avg_tokens"],
        pipeline_started_at=start,
        pipeline_completed_at=now,
    )
    # Only consider validation for result; persistence happens in batch phase
    if not prep["validation_passed"]:
        base.update(error="Validation failed: entity retention less than 100%")
        return DocumentChunkingResult.model_validate(base), summary_df

    # Persistence of chunks is handled in batch Phase 3; no per-document DB or Chroma writes here
    base["success"] = True
    return DocumentChunkingResult.model_validate(base), summary_df


def summarize_run(doc_results: List[DocumentChunkingResult], total_unique_datasets: int) -> ChunkingResult:
    """
    Summarize list of DocumentChunkingResult into a single ChunkingResult.
    """
    total_documents = len(doc_results)
    total_unique_datasets = total_unique_datasets
    total_chunks = sum(r.total_chunks for r in doc_results)
    total_tokens = sum(r.total_tokens for r in doc_results)
    avg_tokens_per_chunk = total_tokens / total_chunks if total_chunks else 0.0
    validation_passed = all(r.validation_passed for r in doc_results)
    entity_retention = sum(r.entity_retention for r in doc_results) / total_documents if total_documents else 0.0
    # get lost entities across all documents
    lost_entities = {r.document_id: r.lost_entities for r in doc_results if r.lost_entities}
    error_messages = [r.error for r in doc_results if r.error]
    error = "; ".join(error_messages) if error_messages else None
    # Only successful overall if entity retention is 100% and all commits succeeded
    overall_success = validation_passed and all(r.success for r in doc_results)
    return ChunkingResult(
        success=overall_success,
        total_documents=total_documents,
        total_unique_datasets=total_unique_datasets,
        total_chunks=total_chunks,
        total_tokens=total_tokens,
        avg_tokens_per_chunk=avg_tokens_per_chunk,
        validation_passed=validation_passed,
        entity_retention=entity_retention,
        output_path=None,
        output_files=None,
        lost_entities=lost_entities,
        error=error,
        pipeline_completed_at=datetime.now().isoformat()
    )


@timer_wrap
def run_semantic_chunking_pipeline(documents_path: str = "Data/train/documents_with_known_entities.json",
                                 citation_entities_path: str = "Data/citation_entities_known.json",
                                 output_dir: str = "Data",
                                 chunk_size: int = 300,
                                 chunk_overlap: int = 2,
                                 collection_name: Optional[str] = None,
                                 cfg_path: str = "configs/chunking.yaml",
                                 subset: bool = False,
                                 subset_size: Optional[int] = None,
                                 use_duckdb: bool = True,
                                 db_path: str = "artifacts/mdc_challenge.db",
                                 max_workers: int = 4
                                 ) -> ChunkingResult:
    """
    Run the complete semantic chunking pipeline.
    
    Args:
        documents_path: Path to documents JSON file (used when use_duckdb=False)
        citation_entities_path: Path to citation entities JSON file (used when use_duckdb=False)
        output_dir: Directory for output files
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        collection_name: ChromaDB collection name
        cfg_path: Path to chunking configuration file
        subset: Whether to use a subset of documents
        subset_size: Size of subset to use
        use_duckdb: Whether to use DuckDB for data I/O (default: True)
        db_path: Path to DuckDB database file
        
    Returns:
        ChunkingResult object with pipeline results
    """
    # 1) Load input data
    if use_duckdb:
        docs, cites = load_input_data_from_duckdb(db_path)
    else:
        docs, cites = load_input_data(documents_path, citation_entities_path)
    
    if len(docs) == 0:
        raise ValueError("No documents found in the database.")
    if len(cites) == 0:
        raise ValueError("No citation entities found in the database.")
    
    # find documents with citations
    cite_ids = {c.document_id for c in cites}
    docs_with_citations = [doc for doc in docs if doc.doi in cite_ids]
    logger.info(f"Found {len(docs_with_citations)} documents with citations.")

    if subset:
        available = len(docs_with_citations)
        if subset_size is None:
            subset_size = min(20, available)
            logger.warning(f"No subset size provided; using {subset_size} based on availability ({available}).")
        elif subset_size > available:
            logger.warning(f"Requested {subset_size} docs but only {available} available; sampling all.")
        np.random.seed(42)
        # choose randomly from docs with citations
        logger.info(f"Choosing {subset_size} documents randomly from {available} documents with citations.")
        docs = np.random.choice(docs_with_citations, size=subset_size, replace=False).tolist()
        # filter cites to only include those with document_id in docs
        doc_ids = {doc.doi for doc in docs}
        cites = [cite for cite in cites if cite.document_id in doc_ids]
    else:
        docs = docs_with_citations

    # 2) Phase 1: document preparation
    phase1_start = time.time()
    logger.info(f"Phase 1: starting preparation of {len(docs)} docs (threshold {max_workers})")
    prepped = []
    summary_dfs = []
    if len(docs) < max_workers:
        logger.info(f"Phase 1: using ThreadPoolExecutor with {min(max_workers, len(docs))} threads for {len(docs)} docs")
        with ThreadPoolExecutor(max_workers=min(max_workers, len(docs))) as exe:
            futures = {
                exe.submit(prepare_document, doc, cites, chunk_size, chunk_overlap, cfg_path, collection_name): doc
                for doc in docs
            }
            for fut in as_completed(futures):
                try:
                    prepped.append(fut.result())
                except Exception as e:
                    doc = futures[fut]
                    prepped.append({
                        "document": doc,
                        "chunks": [],
                        "pre_total_citations": 0,
                        "post_total_citations": 0,
                        "validation_passed": False,
                        "lost_entities": None,
                        "total_chunks": 0,
                        "total_tokens": 0,
                        "avg_tokens": 0,
                        "pipeline_started_at": datetime.now().isoformat(),
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "cfg_path": cfg_path,
                        "collection_name": collection_name,
                        "prep_error": str(e)
                    })
    else:
        logger.info(f"Phase 1: using ProcessPoolExecutor with {max_workers} processes for {len(docs)} docs")
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {
                exe.submit(prepare_document, doc, cites, chunk_size, chunk_overlap, cfg_path, collection_name): doc
                for doc in docs
            }
            for fut in as_completed(futures):
                try:
                    prepped.append(fut.result())
                except Exception as e:
                    doc = futures[fut]
                    prepped.append({
                        "document": doc,
                        "chunks": [],
                        "pre_total_citations": 0,
                        "post_total_citations": 0,
                        "validation_passed": False,
                        "lost_entities": None,
                        "total_chunks": 0,
                        "total_tokens": 0,
                        "avg_tokens": 0,
                        "pipeline_started_at": datetime.now().isoformat(),
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "cfg_path": cfg_path,
                        "collection_name": collection_name,
                        "prep_error": str(e)
                    })
    logger.info(f"Phase 1 complete in {time.time() - phase1_start:.2f}s")

    # 3) Phase 2: build document results
    phase2_start = time.time()
    logger.info("Phase 2: building document results and summaries")
    doc_results: List[DocumentChunkingResult] = []
    for prep in prepped:
        if "prep_error" in prep:
            doc_results.append(DocumentChunkingResult(
                document_id=prep["document"].doi,
                success=False,
                error=prep["prep_error"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                cfg_path=cfg_path,
                collection_name=collection_name,
                pre_chunk_total_citations=0,
                post_chunk_total_citations=0,
                validation_passed=False,
                entity_retention=0.0,
                lost_entities=None,
                total_chunks=0,
                total_tokens=0,
                avg_tokens_per_chunk=0.0,
                pipeline_started_at=prep["pipeline_started_at"],
                pipeline_completed_at=datetime.now().isoformat()
            ))
        else:
            doc_to_commit, summary_df = build_document_result(prep, db_path)
            doc_results.append(doc_to_commit)
            summary_dfs.append(summary_df)
    logger.info(f"Phase 2 complete in {time.time() - phase2_start:.2f}s")
    if len(summary_dfs) > 0:
        summary = pd.concat(summary_dfs)
        summary.to_csv(Path(os.path.join(project_root, output_dir, "chunks_for_embedding_summary.csv")), index=False)
    else:
        logger.warning("No summary data to export.")

    # 4) Phase 3: batch persistence of validated chunks
    phase3_start = time.time()
    logger.info("Phase 3: batch persistence starting")
    valid_chunks: List[Chunk] = []
    for prep in prepped:
        if prep.get("validation_passed"):
            valid_chunks.extend(prep["chunks"])
    logger.info(f"Batch storing {len(valid_chunks)} validated chunks to DuckDB")
    save_chunks_to_duckdb(valid_chunks, db_path)
    logger.info(f"Batch storing {len(valid_chunks)} validated chunks to ChromaDB")
    if collection_name is None:
        collection_name = "mdc_training_data"
    save_chunk_objs_to_chroma(valid_chunks, collection_name=collection_name, cfg_path=cfg_path)
    logger.info(f"Phase 3 complete in {time.time() - phase3_start:.2f}s")
    # TODO: final validation: get list of citation entities from chunks and compare to number of unique datasets in pre-chunk inventory (NOT matches, but dataset IDs) --> can do groupby dataset ID and the filter for only those that have a non-zero count
    total_unique_datasets = 0
    for valid_chunk in valid_chunks:
        total_unique_datasets += len(valid_chunk.chunk_metadata.citation_entities)
    
    # make sure it matches the total number citation entities loaded from the database:
    initial_total_citations = len(cites)
    
    final_res = summarize_run(doc_results, total_unique_datasets)
    if total_unique_datasets != initial_total_citations:
        logger.warning(f"Total unique datasets {total_unique_datasets} found in validated chunks does not match initial total citations {initial_total_citations}")
        final_res.success = False
    final_res.entity_retention = total_unique_datasets / initial_total_citations * 100

    # 5) Summarize into one ChunkingResult
    return final_res

if __name__ == "__main__":
    # Run with default parameters
    results = run_semantic_chunking_pipeline(subset=False, use_duckdb=True, db_path=str(project_root / "artifacts" / "mdc_challenge.db"), cfg_path=str(project_root / "configs" / "chunking.yaml"))
    # Prepare output directory under project root
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "reports", "chunk_embed")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "chunking_results.json")
    if results.success:
        logger.info("✅ Semantic chunking completed successfully!")
        # save results to a json file in reports/chunk_embed
        with open(output_file, "w") as f:
            json.dump(results.model_dump(), f)
    else:
        logger.error(f"❌ Pipeline failed: {results.error}")
        # save results to a json file in reports/chunk_embed
        with open(output_file, "w") as f:
            json.dump(results.model_dump(), f) 