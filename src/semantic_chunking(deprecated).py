#!/usr/bin/env python3
"""
Semantic Chunking Module for Step 6
Implements intelligent document chunking with entity preservation
Following the MDC-Challenge-2025 Step 6 checklist
requires:
- parsed_documents.pkl


"""

import pickle
import pandas as pd
import uuid
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# Core dependencies
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import regex as re
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

import os, sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Local imports
from src.models import ChunkMetadata, Chunk, ChunkingResult, Document, Section
from src.helpers import initialize_logging, timer_wrap
from src.update_patterns import ENTITY_PATTERNS

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

# TODO: Make sure all mentions of Section and Document objects are aligned with updated version of these models (Done --> remove comments after testing). 

TEMP_SUFFIX = '.part'

# --- Constants ---
PRIORITY_SECTIONS = ["data_availability", "methods", "supplementary", "results"]

# --- Load & Filter Functions ---

@timer_wrap
def load_parsed_documents_for_chunking(path: str = "Data/train/parsed/parsed_documents.pkl", 
                                     min_chars: int = 500) -> List[Document]:
    """
    Load and filter parsed documents for chunking.
    
    Args:
        path: Path to parsed documents pickle file
        min_chars: Minimum character count for inclusion
        
    Returns:
        Dictionary of filtered documents
    """
    logger.info(f"Loading parsed documents from: {path}")
    
    try:
        with open(path, "rb") as f:
            docs = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"❌ Error: File not found at {path}")
        return []
    
    # convert to list of document objects
    docs = [Document.model_validate(doc) for doc in docs]
    
    # Filter by character count and strip unicode control characters
    filtered_docs = []
    skipped_count = 0
    
    for doc in docs:
        full_text = doc.full_text
        
        # Strip unicode control characters (prevents tiktoken crashes)
        full_text = re.sub(r'\p{C}', '', full_text)
        
        # Check if document has enough content
        if len(full_text) >= min_chars:
            # Update the document with cleaned text
            doc.full_text = full_text
            
            # Also clean section_texts if they exist
            if doc.sections:
                for section in doc.sections:
                    text = section.text
                    cleaned_section_text = re.sub(r'\p{C}', '', text)
                    section.text = cleaned_section_text            
            filtered_docs.append(doc)
        else:
            skipped_count += 1
    
    logger.info(f"✅ Loaded {len(filtered_docs)} documents")
    logger.warning(f"⚠️  Skipped {skipped_count} documents (< {min_chars} chars)")
    
    return filtered_docs


# --- Section Preparation Functions ---


@timer_wrap
def prepare_section_texts_for_chunking(docs: List[Document]) -> List[Tuple[str, Section]]:
    """
    Prepare section texts for chunking by filtering priority sections.
    
    Args:
        docs: List of parsed documents
        
    Returns:
        List of tuples of (doi, Section)
    """
    logger.info(f"Preparing section texts for {len(docs)} documents")
    # TODO: This is a mess. I need to spend some time fixing this. I don't like using generic dictionaries for this. Maybe I can export a list of tuples of (doi, Section). (Done)
    
    section_texts = []
    fallback_count = 0
    
    for doc_data in docs:
        doi = doc_data.doi
        sections_added_for_this_doc = 0
        
        # Try to get priority sections first
        if doc_data.sections:
            # Look through all sections to find priority ones
            for section in doc_data.sections:
                if section.section_type in PRIORITY_SECTIONS:
                    if section.text.strip():  # Only add non-empty sections
                        section_texts.append((doi, section))
                        sections_added_for_this_doc += 1
        
        # Fall back to full_document if no priority sections found for this document
        if sections_added_for_this_doc == 0:
            full_text = doc_data.full_text
            if full_text.strip():
                section = Section(
                    section_type="full_document",
                    text=full_text,
                    order=0,
                    char_length=len(full_text),
                    sec_level=0,
                    original_type="full_document",
                )
                section_texts.append((doi, section))
                fallback_count += 1
    
    logger.info(f"✅ Prepared section texts for {len(section_texts)} sections")
    logger.info(f"📊 Priority sections found: {len(section_texts) - fallback_count}")
    logger.warning(f"📊 Fallback to full document: {fallback_count}")
    
    return section_texts


# --- Entity Inventory Functions ---


@timer_wrap
def create_pre_chunk_entity_inventory(section_texts: List[Tuple[str, Section]]) -> pd.DataFrame:
    """
    Create entity inventory before chunking for validation.
    
    Args:
        section_texts: Dictionary of section texts by document and section type
        
    Returns:
        DataFrame with entity counts by document and section
    """
    # TODO: This is a mess. I need to spend some time fixing this. I don't like using generic dictionaries for this. Switched to a list of tuples of (doi, Section) but still need to fix this. (Done)
    logger.info(f"Creating pre-chunk entity inventory for {len(section_texts)} sections")
    
    if not section_texts:
        logger.warning("⚠️ No section texts provided - returning empty inventory")
        return pd.DataFrame(columns=['document_id', 'section_type', 'pattern', 'count'])
    
    rows = []
    
    for doi, section in section_texts:
        section_type = section.section_type
        text = section.text
        for entity_label, pattern in ENTITY_PATTERNS.items():
            try:
                matches = pattern.findall(text)
                count = len(matches)
                
                rows.append({
                    'document_id': doi,
                    'section_type': section_type,
                    'pattern': entity_label,
                    'count': count,
                })
            except Exception as e:
                logger.error(f"❌ Error processing pattern {entity_label} for {doi}: {e}")
                # Continue with count = 0
                rows.append({
                    'document_id': doi,
                    'section_type': section_type,
                    'pattern': entity_label,
                    'count': 0,
                })
    
    inventory_df = pd.DataFrame(rows)
    
    # Summary statistics
    total_entities = inventory_df['count'].sum()
    documents_with_entities = len(inventory_df[inventory_df['count'] > 0]['document_id'].unique())
    
    logger.info(f"✅ Created entity inventory:")
    logger.info(f"📊 Total entities found: {total_entities}")
    logger.info(f"📊 Documents with entities: {documents_with_entities}/{len(set(doi for doi, _ in section_texts))}")
    
    return inventory_df


# --- Core Chunking Functions ---

# Initialize tiktoken encoder
tok = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 encoding

@timer_wrap
def create_section_aware_chunks(section_texts: List[Tuple[str, Section]], 
                               docs: List[Document], 
                               chunk_size: int = 200, 
                               chunk_overlap: int = 20) -> List[Chunk]:
    """
    Create chunks from section texts with metadata preservation.
    
    Args:
        section_texts: Section texts prepared for chunking
        docs: Original document data for metadata
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        
    Returns:
        List of Chunk objects
    """
    logger.info(f"Creating section-aware chunks with size={chunk_size}, overlap={chunk_overlap}")
    # TODO: This is a mess. I need to spend some time fixing this. I don't like using generic dictionaries for this. Switched to a list of tuples of (doi, Section) but still need to fix this. (Done --> remove comments after testing)
    # Create a dictionary for faster document lookup
    docs_dict = {doc.doi: doc for doc in docs}
    
    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,  # ~4 chars per token
        chunk_overlap=chunk_overlap * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=lambda t: len(tok.encode(t)),
    )
    
    chunks = []
    
    # Process documents with progress bar
    for doi, section in tqdm(section_texts, desc="Chunking documents"):
        # Get document metadata using dictionary lookup
        doc = docs_dict.get(doi)
        if not doc:
            logger.error(f"❌ Document not found: {doi}")
            continue
            
        conversion_source = doc.conversion_source
        section_type = section.section_type
        text = section.text
        order = section.order
        
        # Split text into chunks
        section_chunks = splitter.split_text(text)
        
        # Create metadata for each chunk
        for i, chunk_text in enumerate(section_chunks):
            chunk_id = f"{uuid.uuid4().hex[:8]}_{i}"
            token_count = len(tok.encode(chunk_text))
            
            metadata = {
                'chunk_id': chunk_id,
                'document_id': doi,
                'section_type': section_type,
                'section_order': order,
                'format_type': doc.format_type,  # Add missing field
                'conversion_source': conversion_source,
                'token_count': token_count,
                'previous_chunk_id': None,
                'next_chunk_id': None,
                'chunk_type': None,
                'citation_entities': [],
            }
            chunk = Chunk(
                chunk_id=chunk_id,
                text = chunk_text,
                chunk_metadata = ChunkMetadata.model_validate(metadata)
            )
            chunks.append(chunk)
    
    logger.info(f"✅ Created {len(chunks)} chunks from {len(section_texts)} sections")
    
    return chunks


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
    by_document = defaultdict(list)
    for i, chunk in enumerate(chunks):
        document_id = chunk.chunk_metadata.document_id
        by_document[document_id].append((i, chunk.text, chunk.chunk_metadata.model_dump()))
    
    # Link chunks within each document
    linked_chunks = []
    for document_id, doc_chunks in by_document.items():
        # Sort by section order and chunk position
        doc_chunks.sort(key=lambda x: (x[2]['section_order'], x[0]))
        
        # Set up links
        for i, (original_idx, text, metadata) in enumerate(doc_chunks):
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
                text = text,
                chunk_metadata = ChunkMetadata.model_validate(metadata)
            )
            
            linked_chunks.append(new_chunk)
    
    logger.info(f"✅ Linked {len(linked_chunks)} chunks")
    
    return linked_chunks


@timer_wrap
def refine_chunk_types(chunks: List[Chunk]) -> List[Chunk]:
    """
    Refine chunk types based on content analysis.
    
    Args:
        chunks: List of Chunk objects
        
    Returns:
        List of Chunk objects with refined chunk types
    """
    logger.info(f"Refining chunk types for {len(chunks)} chunks")
    
    refined_chunks = []
    type_counts = defaultdict(int)
    
    for chunk in chunks:
        text_lower = chunk.text.lower()
        section_type = chunk.chunk_metadata.section_type
        metadata = chunk.chunk_metadata.model_dump()
        
        # Determine chunk type
        if any(keyword in text_lower for keyword in ["figure", "fig.", "table", "caption"]):
            chunk_type = "caption"
        elif text_lower.strip().endswith(":") and len(text_lower.split()) < 15:
            chunk_type = "header"
        elif section_type == "data_availability":
            chunk_type = "data_statement"
        else:
            chunk_type = "body"
        
        metadata['chunk_type'] = chunk_type
        type_counts[chunk_type] += 1
        refined_chunk = Chunk(
            chunk_id=metadata['chunk_id'],
            text = chunk.text,
            chunk_metadata = ChunkMetadata.model_validate(metadata)
        )
        
        refined_chunks.append(refined_chunk)
    
    logger.info(f"✅ Refined chunk types:")
    for chunk_type, count in type_counts.items():
        logger.info(f"   {chunk_type}: {count}")
    
    return refined_chunks


# --- Validation Functions ---

@timer_wrap
def validate_chunk_integrity(chunks: List[Chunk], 
                           pre_chunk_inventory: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
    """
    Validate that entities are preserved after chunking.
    
    Args:
        chunks: List of Chunk objects
        pre_chunk_inventory: Entity inventory before chunking
        
    Returns:
        Tuple of (validation_passed, loss_report)
    """
    logger.info(f"Validating chunk integrity for {len(chunks)} chunks")
    
    # Count entities AFTER chunking
    post_chunk_rows = []
    
    for chunk in chunks:
        document_id = chunk.chunk_metadata.document_id
        section_type = chunk.chunk_metadata.section_type
        text = chunk.text
        
        for entity_label, pattern in ENTITY_PATTERNS.items():
            matches = pattern.findall(text)
            count = len(matches)
            
            post_chunk_rows.append({
                'document_id': document_id,
                'section_type': section_type,
                'pattern': entity_label,
                'count': count,
            })
    
    # Aggregate post-chunk counts
    post_chunk_df = pd.DataFrame(post_chunk_rows)
    post_aggregated = (
        post_chunk_df
        .groupby(['document_id', 'section_type', 'pattern'])['count']
        .sum()
        .reset_index()
    )
    
    # Merge with pre-chunk inventory
    merged = pre_chunk_inventory.merge(
        post_aggregated,
        on=['document_id', 'section_type', 'pattern'],
        how='left',
        suffixes=('_pre', '_post')
    ).fillna(0)
    
    # Find lost entities
    lost_entities = merged[merged['count_post'] < merged['count_pre']]
    
    # Check if validation passed
    validation_passed = lost_entities.empty
    
    # Summary statistics
    total_entities_pre = pre_chunk_inventory['count'].sum()
    total_entities_post = post_aggregated['count'].sum()
    retention_rate = (total_entities_post / total_entities_pre) * 100 if total_entities_pre > 0 else 100
    
    logger.info(f"✅ Entity integrity validation:")
    logger.info(f"   Pre-chunk entities: {total_entities_pre}")
    logger.info(f"   Post-chunk entities: {total_entities_post}")
    logger.info(f"   Retention rate: {retention_rate:.1f}%")
    
    if validation_passed:
        logger.info(f"✅ SUCCESS: 100% entity retention achieved")
    else:
        logger.error(f"❌ FAILURE: {len(lost_entities)} entity losses detected")
        logger.error(f"   Rerun with --repair to auto-tune overlap")
    
    return validation_passed, lost_entities


# --- Export Functions ---

@timer_wrap
def export_chunks_for_embedding(chunks: List[Chunk], 
                               output_path: str = "chunks_for_embedding.pkl") -> List[str]:
    """
    Export chunks for embedding with summary statistics.
    
    Args:
        chunks: List of Chunk objects
        output_path: Path for output pickle file

    Returns:
        List of output file paths
    """
    logger.info(f"Exporting {len(chunks)} chunks to {output_path}")
    output_path = str(output_path) + TEMP_SUFFIX
    
    # Save main pickle file
    with open(output_path, "wb") as f:
        pickle.dump(chunks, f)
    
    # Create summary CSV
    summary_rows = []
    for chunk in chunks:
        row = chunk.chunk_metadata.model_dump()
        row['text_length'] = len(chunk.text)
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_path.replace(".pkl", f"_summary.csv{TEMP_SUFFIX}")
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"✅ Exported chunks:")
    logger.info(f"   Chunks file: {output_path}")
    logger.info(f"   Summary file: {summary_path}")
    logger.info(f"   Total chunks: {len(chunks):,}")

    return [output_path.replace(TEMP_SUFFIX, ""), summary_path.replace(TEMP_SUFFIX, "")]


# --- Main Pipeline Function ---

@timer_wrap
def run_semantic_chunking_pipeline(input_path: str = "Data/train/parsed/parsed_documents.pkl",
                                 output_path: str = "chunks_for_embedding.pkl",
                                 chunk_size: int = 200,
                                 chunk_overlap: int = 20,
                                 min_chars: int = 500) -> ChunkingResult:
    """
    Run the complete semantic chunking pipeline.
    
    Args:
        input_path: Path to parsed documents
        output_path: Path for output chunks
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        min_chars: Minimum character count for inclusion
        
    Returns:
        ChunkingResult object with pipeline results and statistics
    """
    logger.info("=== Semantic Chunking Pipeline (Step 6) ===")
    logger.info(f"Started at: {datetime.now()}")
    
    try:
        # Step 1: Load documents
        logger.info("\n📋 Step 1: Load & Filter Documents")
        docs = load_parsed_documents_for_chunking(input_path, min_chars)
        
        if not docs:
            raise ValueError("No documents loaded - check input path")
        
        # Step 2: Prepare section texts
        logger.info("\n📋 Step 2: Prepare Section Texts")
        section_texts = prepare_section_texts_for_chunking(docs)
        
        # Step 3: Create entity inventory
        logger.info("\n📋 Step 3: Create Entity Inventory")
        pre_chunk_inventory = create_pre_chunk_entity_inventory(section_texts)
        
        # Step 4: Create chunks
        logger.info("\n📋 Step 4: Create Section-Aware Chunks")
        chunks = create_section_aware_chunks(section_texts, docs, chunk_size, chunk_overlap)
        
        # Step 5: Link adjacent chunks
        logger.info("\n📋 Step 5: Link Adjacent Chunks")
        chunks = link_adjacent_chunks(chunks)
        
        # Step 6: Refine chunk types
        logger.info("\n📋 Step 6: Refine Chunk Types")
        chunks = refine_chunk_types(chunks)
        
        # Step 7: Validate integrity
        logger.info("\n📋 Step 7: Validate Chunk Integrity")
        validation_passed, lost_entities = validate_chunk_integrity(chunks, pre_chunk_inventory)
        
        # Quality gates check
        if not validation_passed:
            logger.error("❌ QUALITY GATE FAILURE: Entity retention < 100%")
            logger.error("   Pipeline aborted to prevent data loss")
            
            # Calculate required fields
            total_tokens = sum(chunk.chunk_metadata.token_count for chunk in chunks)
            avg_tokens_per_chunk = total_tokens / len(chunks) if chunks else 0
            total_unique_datasets = len(pre_chunk_inventory[pre_chunk_inventory['count'] > 0]['pattern'].unique()) if len(pre_chunk_inventory) > 0 else 0
            total_entities_pre = pre_chunk_inventory['count'].sum() if len(pre_chunk_inventory) > 0 else 0
            entity_retention = (total_entities_pre / total_entities_pre * 100) if total_entities_pre > 0 else 100
            
            return ChunkingResult(
                success=False,
                error="Quality gate failure: Entity retention < 100%",
                pipeline_completed_at=datetime.now().isoformat(),
                total_documents=len(docs),
                total_unique_datasets=total_unique_datasets,  # Add missing field
                total_chunks=len(chunks),
                total_tokens=total_tokens,
                avg_tokens_per_chunk=avg_tokens_per_chunk,
                validation_passed=validation_passed,
                entity_retention=entity_retention,
            )
        
        # Step 8: Export results
        logger.info("\n📋 Step 8: Export Chunks")
        output_files = export_chunks_for_embedding(chunks, output_path)
        
        # Calculate final statistics
        total_tokens = sum(chunk.chunk_metadata.token_count for chunk in chunks)
        avg_tokens_per_chunk = total_tokens / len(chunks) if chunks else 0
        
        # Calculate total unique datasets (entities) found
        total_unique_datasets = len(pre_chunk_inventory[pre_chunk_inventory['count'] > 0]['pattern'].unique()) if len(pre_chunk_inventory) > 0 else 0
        
        # Calculate entity retention properly
        total_entities_pre = pre_chunk_inventory['count'].sum() if len(pre_chunk_inventory) > 0 else 0
        entity_retention = (total_entities_pre / total_entities_pre * 100) if total_entities_pre > 0 else 100
        
        results = {
            'success': True,
            'total_documents': len(docs),
            'total_unique_datasets': total_unique_datasets,  # Add missing field
            'total_chunks': len(chunks),
            'total_tokens': total_tokens,
            'avg_tokens_per_chunk': avg_tokens_per_chunk,
            'validation_passed': validation_passed,
            'pipeline_completed_at': datetime.now().isoformat(),
            'entity_retention': entity_retention,
            'lost_entities': lost_entities.to_dict() if len(lost_entities) > 0 else None,
            'output_path': output_path,
            'output_files': output_files,
        }
        results = ChunkingResult.model_validate(results)
        
        logger.info(f"\n✅ Pipeline completed successfully!")
        logger.info(f"📊 Final Results:")
        logger.info(f"   Documents processed: {results.total_documents}")
        logger.info(f"   Chunks created: {results.total_chunks:,}")
        print(f"   Average tokens per chunk: {results.avg_tokens_per_chunk:.1f}")
        print(f"   Entity retention: {results.entity_retention:.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        
        # Initialize missing variables for error case
        total_tokens = 0
        avg_tokens_per_chunk = 0.0
        entity_retention = 0.0
        total_unique_datasets = 0
        
        return ChunkingResult(
            success=False,
            error=str(e),
            pipeline_completed_at=datetime.now().isoformat(),
            total_documents=len(docs) if 'docs' in locals() else 0,
            total_unique_datasets=total_unique_datasets,  # Add missing field
            total_chunks=len(chunks) if 'chunks' in locals() else 0,
            total_tokens=total_tokens,
            avg_tokens_per_chunk=avg_tokens_per_chunk,
            validation_passed=False,
            entity_retention=entity_retention,
        )


if __name__ == "__main__":
    # Run with default parameters
    results = run_semantic_chunking_pipeline()
    
    if results.success:
        logger.info("✅ Semantic chunking completed successfully!")
    else:
        logger.error(f"❌ Pipeline failed: {results.error}") 