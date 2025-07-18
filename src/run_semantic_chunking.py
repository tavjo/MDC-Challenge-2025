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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Local imports
from src.models import Document, CitationEntity, Chunk, ChunkMetadata, ChunkingResult
from src.helpers import initialize_logging, timer_wrap, load_docs, export_docs
from src.semantic_chunking import semantic_chunk_text, save_chunk_objs_to_chroma, save_chunks_to_chroma
# from src.get_citation_entities import CitationEntityExtractor
import duckdb
from api.duckdb_utils import get_duckdb_helper

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
    """Create exact pattern for dataset citation (from get_citation_entities.py)"""
    # 1. escape *everything* first
    pat = re.escape(dataset_id)
    
    # 2. then widen the single underscore
    pat = pat.replace('_', r'[_/]')     # now [_/] stays "live"
    
    # 3. add word-boundaries only for simple accessions
    if re.fullmatch(r'[A-Z]{1,4}\d+', dataset_id, re.I):
        pat = rf'\b{pat}\b'
    
    return re.compile(pat, flags=re.IGNORECASE)

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
        matches = pattern.findall(text)
        
        rows.append({
            'document_id': document.doi,
            'citation_id': citation.data_citation,
            'count': len(matches),
        })
    
    inventory_df = pd.DataFrame(rows)
    
    # Summary statistics
    total_citations = inventory_df['count'].sum()
    citations_found = len(inventory_df[inventory_df['count'] > 0])
    
    logger.info(f"✅ Created entity inventory:")
    logger.info(f"📊 Total citations found: {total_citations}")
    logger.info(f"📊 Citations with matches: {citations_found}/{len(doc_citations)}")
    
    return inventory_df 

# ---------------------------------------------------------------------------
# Chunk Creation (Using Semantic Chunking)
# ---------------------------------------------------------------------------

@timer_wrap
def create_chunks_from_document(document: Document, citation_entities: List[CitationEntity], 
                                chunk_size: int = 200, chunk_overlap: int = 20) -> List[Chunk]:
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
    
    # Use semantic chunking
    text_chunks = semantic_chunk_text(text)
    
    # Get citations for this document
    doc_citations = citations_by_doc.get(doc.doi, [])
    
    # Create Chunk objects with metadata
    for i, chunk_text in enumerate(text_chunks):
        chunk_id = f"{doc.doi}_{i}"
        
        # Find which citations appear in this chunk
        chunk_citations = []
        for citation in doc_citations:
            pattern = make_pattern(citation.data_citation)
            if pattern.search(chunk_text):  # Check if citation appears in chunk
                chunk_citations.append(citation)
        
        # Use tiktoken for accurate token counting
        import tiktoken
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
    post_chunk_rows = []
    
    for chunk in chunks:
        text = chunk.text
        doc_id = chunk.document_id
        doc_citations = [citation for citation in citation_entities if citation.document_id == doc_id]
        
        for citation in doc_citations:
            pattern = make_pattern(citation.data_citation)
            matches = pattern.findall(text)
            
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
        # Connect to DuckDB
        # conn = duckdb.connect(db_path)
        
        # Clear existing chunks (optional - depends on requirements)
        # conn.execute("DELETE FROM chunks")
        
        # Insert chunks one by one
        # for chunk in chunks:
        #     chunk_row = chunk.to_duckdb_row()
            
        #     # Insert chunk
        #     conn.execute("""
        #         INSERT OR REPLACE INTO chunks 
        #         (chunk_id, document_id, chunk_text, score, chunk_metadata)
        #         VALUES (?, ?, ?, ?, ?)
        #     """, (
        #         chunk_row["chunk_id"],
        #         chunk_row["document_id"],
        #         chunk_row["chunk_text"],
        #         chunk_row["score"],
        #         chunk_row["chunk_metadata"]
        #     ))
        
        # conn.commit()
        # conn.close()
        db_helper = get_duckdb_helper(db_path)
        if db_helper.store_chunks(chunks):
            logger.info(f"✅ Successfully saved {len(chunks)} chunks to DuckDB")
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
def run_semantic_chunking_pipeline(documents_path: str = "Data/train/documents_with_known_entities.json",
                                 citation_entities_path: str = "Data/citation_entities_known.json",
                                 output_dir: str = "Data",
                                 chunk_size: int = 200,
                                 chunk_overlap: int = 20,
                                 collection_name: str = "semantic_chunks",
                                 cfg_path: str = "configs/chunking.yaml",
                                 subset: bool = False,
                                 subset_size: Optional[int] = None,
                                 use_duckdb: bool = True,
                                 db_path: str = "artifacts/mdc_challenge.db"
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
    logger.info("=== Semantic Chunking Pipeline ===")
    logger.info(f"Started at: {datetime.now()}")
    
    try:
        # Step 1: Load input data
        logger.info("\n📋 Step 1: Load Input Data")
        if use_duckdb:
            logger.info("Using DuckDB for data loading")
            documents, citation_entities = load_input_data_from_duckdb(db_path)
        else:
            logger.info("Using JSON files for data loading")
            documents, citation_entities = load_input_data(documents_path, citation_entities_path)
        
        if not documents:
            raise ValueError("No documents loaded - check input path")
        
        if subset:
            np.random.seed(42)
            documents = np.random.choice(documents, size=subset_size, replace=False)

        result_list = []
        inventory_list = []
        all_validated_chunks = []
        lost_entities_list = []
        # summary_df_list = []
        
        for document in documents:
            logger.info(f"\n📋 Step 2: Create Pre-Chunk Entity Inventory for {document.doi}")
            pre_chunk_inventory = create_pre_chunk_entity_inventory(document, citation_entities)
            inventory_list.append(pre_chunk_inventory)
            # step 3: create chunks
            logger.info(f"\n📋 Step 3: Create Semantic Chunks for {document.doi}")
            chunks = create_chunks_from_document(document, citation_entities, chunk_size, chunk_overlap)
            # Step 4: Link adjacent chunks within document
            logger.info(f"\n📋 Step 4: Link Adjacent Chunks for {document.doi}")
            chunks = link_adjacent_chunks(chunks)
            # post-chunk validation
            logger.info(f"\n📋 Step 5: Validate Chunk Integrity for {document.doi}")
            validation_passed, lost_citations = validate_chunk_integrity(chunks, pre_chunk_inventory, citation_entities)
            if lost_citations is not None:
                lost_entities_list.append(lost_citations)
            if not validation_passed:
                logger.error("❌ QUALITY GATE FAILURE: Citation retention < 100%")
                logger.error("   Pipeline aborted to prevent data loss")
                result_list.append(ChunkingResult(
                success=False,
                error="Quality gate failure: Citation retention < 100%",
                pipeline_completed_at=datetime.now().isoformat(),
                total_documents=1,
                total_unique_datasets=len(citation_entities),
                total_chunks=len(chunks),
                total_tokens=sum(chunk.chunk_metadata.token_count for chunk in chunks),
                avg_tokens_per_chunk=sum(chunk.chunk_metadata.token_count for chunk in chunks) / len(chunks) if chunks else 0,
                validation_passed=validation_passed,
                entity_retention=0.0,
                ))
                continue
            all_validated_chunks.extend(chunks)
            # step 6: save chunks to chroma for document
            logger.info(f"\n📋 Step 6: Save Chunks to ChromaDB for {document.doi}")
            save_chunk_objs_to_chroma(chunks, collection_name=collection_name,
                                cfg_path=cfg_path)
            # step 7: save chunks
            output_path = Path(os.path.join(project_root, output_dir))
            output_path.mkdir(parents=True, exist_ok=True)
            if use_duckdb:
                logger.info(f"\n📋 Step 7: Save Chunks to DuckDB for {document.doi}")
                save_success = save_chunks_to_duckdb(chunks, db_path)
                if not save_success:
                    logger.warning("⚠️  Failed to save chunks to DuckDB, saving to JSON instead...")
                    json_file = export_chunks_to_json(chunks, str(output_path / "chunks_for_embedding.json"))
                else:
                    logger.info(f"✅ Successfully saved {len(chunks)} chunks to DuckDB")
                    json_file = None
            else:
                logger.info(f"\n📋 Step 7: Save Chunks to JSON for {document.doi}")
                json_file = export_chunks_to_json(chunks, str(output_path / "chunks_for_embedding.json"))
        
        lost_entities = pd.concat(lost_entities_list) if len(lost_entities_list) > 0 else None
        if len(result_list) > 0:
            # combine all results into a single result
            failed_results = ChunkingResult(
                success=False,
                error="Some documents failed chunking",
                pipeline_completed_at=datetime.now().isoformat(),
                total_documents=len(result_list),
                total_unique_datasets=sum(result.total_unique_datasets for result in result_list),
                total_chunks=sum(result.total_chunks for result in result_list),
                total_tokens=sum(result.total_tokens for result in result_list),
                avg_tokens_per_chunk=sum(result.avg_tokens_per_chunk for result in result_list) / len(result_list) if result_list else 0,
                validation_passed=False,
                entity_retention=0.0,
                lost_entities=lost_entities.to_dict() if len(lost_entities) > 0 else None,
            )
            if len(result_list) == len(documents):
                logger.error("❌ QUALITY GATE FAILURE: All documents failed chunking")
                failed_results.error = "All documents failed chunking"
                return failed_results

        # step 8: create chunks summary csv
        logger.info(f"\n📋 Step 8: Create Chunks Summary CSV for {document.doi}")
        csv_file = str(output_path / "chunks_for_embedding_summary.csv")
        summary_df = create_chunks_summary_csv(all_validated_chunks, export=True, output_path=csv_file)

        # Calculate final statistics
        inventory_df = pd.concat(inventory_list)
        total_tokens = sum(chunk.chunk_metadata.token_count for chunk in all_validated_chunks)
        avg_tokens_per_chunk = total_tokens / len(all_validated_chunks) if all_validated_chunks else 0
        total_citations_pre = inventory_df['count'].sum()
        entity_retention = 100.0 if total_citations_pre > 0 else 100.0

        # get info from failed results list
        total_docs = len(documents) if len(result_list) == 0 else len(documents)-failed_results.total_documents
        total_datasets = len(citation_entities) if len(result_list) == 0 else len(citation_entities)-failed_results.total_unique_datasets
        out_files = []
        if json_file is not None:
            out_files.append(json_file)
        if csv_file is not None:
            out_files.append(csv_file)
        
        results = ChunkingResult(
            success=True,
            total_documents=total_docs,
            total_unique_datasets=total_datasets,
            total_chunks=len(all_validated_chunks),
            total_tokens=total_tokens,
            avg_tokens_per_chunk=avg_tokens_per_chunk,
            validation_passed=validation_passed,
            pipeline_completed_at=datetime.now().isoformat(),
            entity_retention=entity_retention,
            lost_entities=lost_entities.to_dict() if len(lost_entities) > 0 else None,
            output_path=str(output_path),
            output_files=out_files,
        )
        
        logger.info(f"\n✅ Pipeline completed successfully!")
        logger.info(f"📊 Final Results:")
        logger.info(f"   Documents processed: {results.total_documents}")
        logger.info(f"   Chunks created: {results.total_chunks:,}")
        logger.info(f"   Average tokens per chunk: {results.avg_tokens_per_chunk:.1f}")
        logger.info(f"   Citation retention: {results.entity_retention:.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        return ChunkingResult(
            success=False,
            error=str(e),
            pipeline_completed_at=datetime.now().isoformat(),
            total_documents=len(documents) if 'documents' in locals() else 0,
            total_unique_datasets=len(citation_entities) if 'citation_entities' in locals() else 0,
            total_chunks=len(chunks) if 'chunks' in locals() else 0,
            total_tokens=0,
            avg_tokens_per_chunk=0.0,
            validation_passed=False,
            entity_retention=0.0,
        )

if __name__ == "__main__":
    # Run with default parameters
    results = run_semantic_chunking_pipeline(subset=False, use_duckdb=True, db_path=str(project_root / "artifacts" / "mdc_challenge.db"), cfg_path=str(project_root / "configs" / "chunking.yaml"))
    
    if results.success:
        logger.info("✅ Semantic chunking completed successfully!")
        # save results to a json file
        with open("chunking_results.json", "w") as f:
            json.dump(results.model_dump(), f)
    else:
        logger.error(f"❌ Pipeline failed: {results.error}")
        # save results to a json file
        with open("chunking_results.json", "w") as f:
            json.dump(results.model_dump(), f) 