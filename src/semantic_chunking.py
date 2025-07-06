#!/usr/bin/env python3
"""
Semantic Chunking Module for Step 6
Implements intelligent document chunking with entity preservation
Following the MDC-Challenge-2025 Step 6 checklist
"""

import pickle
import pandas as pd
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# Core dependencies
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import regex as re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Local imports
from .models import ChunkMetadata, Chunk, ChunkingResult


# --- 3.1 Load & Filter Functions ---

def load_parsed_documents_for_chunking(path: str = "Data/train/parsed/parsed_documents.pkl", 
                                     min_chars: int = 500) -> Dict[str, Dict[str, Any]]:
    """
    Load and filter parsed documents for chunking.
    
    Args:
        path: Path to parsed documents pickle file
        min_chars: Minimum character count for inclusion
        
    Returns:
        Dictionary of filtered documents
    """
    print(f"Loading parsed documents from: {path}")
    
    try:
        with open(path, "rb") as f:
            docs = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {path}")
        return {}
    
    # Filter by character count and strip unicode control characters
    filtered_docs = {}
    skipped_count = 0
    
    for doi, doc_data in docs.items():
        full_text = doc_data.get("full_text", "")
        
        # Strip unicode control characters (prevents tiktoken crashes)
        full_text = re.sub(r'\p{C}', '', full_text)
        
        # Check if document has enough content
        if len(full_text) >= min_chars:
            # Update the document with cleaned text
            doc_data["full_text"] = full_text
            
            # Also clean section_texts if they exist
            if "section_texts" in doc_data:
                cleaned_section_texts = {}
                for section_type, text in doc_data["section_texts"].items():
                    cleaned_section_texts[section_type] = re.sub(r'\p{C}', '', text)
                doc_data["section_texts"] = cleaned_section_texts
            
            filtered_docs[doi] = doc_data
        else:
            skipped_count += 1
    
    print(f"✅ Loaded {len(filtered_docs)} documents")
    print(f"⚠️  Skipped {skipped_count} documents (< {min_chars} chars)")
    
    return filtered_docs


# --- 3.2 Section Preparation Functions ---

PRIORITY_SECTIONS = ["data_availability", "methods", "supplementary", "results"]

def prepare_section_texts_for_chunking(docs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """
    Prepare section texts for chunking by filtering priority sections.
    
    Args:
        docs: Dictionary of parsed documents
        
    Returns:
        Dictionary mapping DOI to section texts
    """
    print(f"Preparing section texts for {len(docs)} documents")
    
    section_texts = {}
    fallback_count = 0
    
    for doi, doc_data in docs.items():
        section_texts[doi] = {}
        
        # Try to get priority sections first
        if "section_texts" in doc_data:
            doc_section_texts = doc_data["section_texts"]
            
            # Filter for priority sections
            for section_type in PRIORITY_SECTIONS:
                if section_type in doc_section_texts:
                    text = doc_section_texts[section_type]
                    if text.strip():  # Only add non-empty sections
                        section_texts[doi][section_type] = text
        
        # Fall back to full_document if no priority sections found
        if not section_texts[doi]:
            full_text = doc_data.get("full_text", "")
            if full_text.strip():
                section_texts[doi]["full_document"] = full_text
                fallback_count += 1
    
    print(f"✅ Prepared section texts for {len(section_texts)} documents")
    print(f"📊 Priority sections found: {len(section_texts) - fallback_count}")
    print(f"📊 Fallback to full document: {fallback_count}")
    
    return section_texts


# --- 3.3 Entity Inventory Functions ---

# Entity patterns for citation recognition
ENTITY_PATTERNS = {
    # Existing patterns
    'DOI': re.compile(r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b', re.IGNORECASE),
    'GEO_Series': re.compile(r'\bGSE\d{3,6}\b'),
    'GEO_Sample': re.compile(r'\bGSM\d{3,6}\b'),
    'SRA_Run': re.compile(r'\bSRR\d{5,}\b'),
    'PDB_ID': re.compile(r'\b[A-Za-z0-9]{4}\b'),
    'PDB_DOI': re.compile(r'\b10\.2210/pdb[A-Za-z0-9]{4}/pdb\b', re.IGNORECASE),
    'ArrayExpress': re.compile(r'\bE-[A-Z]+-\d+\b'),
    'dbGaP': re.compile(r'\bphs\d{6}\b'),
    'TCGA': re.compile(r'\bTCGA-[A-Z0-9-]+\b'),
    'ENA_Project': re.compile(r'\bPRJ[EDN][A-Z]\d+\b'),
    'ENA_Study': re.compile(r'\bERP\d{6,}\b'),
    'ENA_Sample': re.compile(r'\bSAM[EDN][A-Z]?\d+\b'),
    
    # New additions
    'SRA_Experiment': re.compile(r'\bSRX\d{5,}\b'),
    'SRA_Project': re.compile(r'\bSRP\d{5,}\b'),
    'SRA_Sample': re.compile(r'\bSRS\d{5,}\b'),
    'SRA_Study': re.compile(r'\bSRA\d{5,}\b'),
    'RefSeq_Chromosome': re.compile(r'\bNC_\d{6,}(?:\.\d+)?\b'),
    'ENA_Run': re.compile(r'\bERR\d{6,}\b'),
    'ENA_Experiment': re.compile(r'\bERX\d{6,}\b'),
    'ENA_Sample2': re.compile(r'\bERS\d{6,}\b'),
    'DDBJ_Run': re.compile(r'\bDRR\d{6,}\b'),
    'DDBJ_Experiment': re.compile(r'\bDRX\d{6,}\b'),
    'ENCODE_Assay': re.compile(r'\bENCSR[0-9A-Z]{6}\b'),
    'PRIDE': re.compile(r'\bPXD\d{6,}\b'),
}

def create_pre_chunk_entity_inventory(section_texts: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Create entity inventory before chunking for validation.
    
    Args:
        section_texts: Dictionary of section texts by document and section type
        
    Returns:
        DataFrame with entity counts by document and section
    """
    print(f"Creating pre-chunk entity inventory for {len(section_texts)} documents")
    
    rows = []
    
    for doi, sections in section_texts.items():
        for section_type, text in sections.items():
            for entity_label, pattern in ENTITY_PATTERNS.items():
                matches = pattern.findall(text)
                count = len(matches)
                
                rows.append({
                    'document_id': doi,
                    'section_type': section_type,
                    'pattern': entity_label,
                    'count': count,
                })
    
    inventory_df = pd.DataFrame(rows)
    
    # Summary statistics
    total_entities = inventory_df['count'].sum()
    documents_with_entities = len(inventory_df[inventory_df['count'] > 0]['document_id'].unique())
    
    print(f"✅ Created entity inventory:")
    print(f"📊 Total entities found: {total_entities}")
    print(f"📊 Documents with entities: {documents_with_entities}/{len(section_texts)}")
    
    return inventory_df


# --- 3.4 Core Chunking Functions ---

# Initialize tiktoken encoder
tok = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 encoding

def create_section_aware_chunks(section_texts: Dict[str, Dict[str, str]], 
                               docs: Dict[str, Dict[str, Any]], 
                               chunk_size: int = 200, 
                               chunk_overlap: int = 20) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Create chunks from section texts with metadata preservation.
    
    Args:
        section_texts: Section texts prepared for chunking
        docs: Original document data for metadata
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        
    Returns:
        List of (text, metadata) tuples
    """
    print(f"Creating section-aware chunks with size={chunk_size}, overlap={chunk_overlap}")
    
    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,  # ~4 chars per token
        chunk_overlap=chunk_overlap * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=lambda t: len(tok.encode(t)),
    )
    
    chunks = []
    total_docs = len(section_texts)
    
    # Process documents with progress bar
    for doi, sections in tqdm(section_texts.items(), desc="Chunking documents"):
        # Get document metadata
        doc_data = docs.get(doi, {})
        section_order = doc_data.get("section_order", {})
        conversion_source = doc_data.get("conversion_source", "unknown")
        
        # Process each section
        for section_type, text in sections.items():
            order = section_order.get(section_type, 999)
            
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
                    'conversion_source': conversion_source,
                    'token_count': token_count,
                    'previous_chunk_id': None,  # Will be set by link_adjacent_chunks
                    'next_chunk_id': None,      # Will be set by link_adjacent_chunks
                    'chunk_type': None,         # Will be set by refine_chunk_types
                    'citation_entities': [],    # Will be populated later if needed
                }
                
                chunks.append((chunk_text, metadata))
    
    print(f"✅ Created {len(chunks)} chunks from {total_docs} documents")
    
    return chunks


def link_adjacent_chunks(chunks: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Link adjacent chunks with previous/next chunk IDs.
    
    Args:
        chunks: List of (text, metadata) tuples
        
    Returns:
        List of (text, metadata) tuples with linked IDs
    """
    print(f"Linking adjacent chunks for {len(chunks)} chunks")
    
    # Group chunks by document
    by_document = defaultdict(list)
    for i, (text, metadata) in enumerate(chunks):
        document_id = metadata['document_id']
        by_document[document_id].append((i, text, metadata))
    
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
            
            linked_chunks.append((text, metadata))
    
    print(f"✅ Linked {len(linked_chunks)} chunks")
    
    return linked_chunks


def refine_chunk_types(chunks: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Refine chunk types based on content analysis.
    
    Args:
        chunks: List of (text, metadata) tuples
        
    Returns:
        List of (text, metadata) tuples with refined chunk types
    """
    print(f"Refining chunk types for {len(chunks)} chunks")
    
    refined_chunks = []
    type_counts = defaultdict(int)
    
    for text, metadata in chunks:
        text_lower = text.lower()
        section_type = metadata.get('section_type', '')
        
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
        
        refined_chunks.append((text, metadata))
    
    print(f"✅ Refined chunk types:")
    for chunk_type, count in type_counts.items():
        print(f"   {chunk_type}: {count}")
    
    return refined_chunks


# --- 3.5 Validation Functions ---

def validate_chunk_integrity(chunks: List[Tuple[str, Dict[str, Any]]], 
                           pre_chunk_inventory: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
    """
    Validate that entities are preserved after chunking.
    
    Args:
        chunks: List of chunked text and metadata
        pre_chunk_inventory: Entity inventory before chunking
        
    Returns:
        Tuple of (validation_passed, loss_report)
    """
    print(f"Validating chunk integrity for {len(chunks)} chunks")
    
    # Count entities AFTER chunking
    post_chunk_rows = []
    
    for text, metadata in chunks:
        document_id = metadata['document_id']
        section_type = metadata['section_type']
        
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
    
    print(f"✅ Entity integrity validation:")
    print(f"   Pre-chunk entities: {total_entities_pre}")
    print(f"   Post-chunk entities: {total_entities_post}")
    print(f"   Retention rate: {retention_rate:.1f}%")
    
    if validation_passed:
        print(f"✅ SUCCESS: 100% entity retention achieved")
    else:
        print(f"❌ FAILURE: {len(lost_entities)} entity losses detected")
        print(f"   Rerun with --repair to auto-tune overlap")
    
    return validation_passed, lost_entities


# --- 3.6 Export Functions ---

def export_chunks_for_embedding(chunks: List[Tuple[str, Dict[str, Any]]], 
                               output_path: str = "chunks_for_embedding.pkl") -> None:
    """
    Export chunks for embedding with summary statistics.
    
    Args:
        chunks: List of (text, metadata) tuples
        output_path: Path for output pickle file
    """
    print(f"Exporting {len(chunks)} chunks to {output_path}")
    
    # Save main pickle file
    with open(output_path, "wb") as f:
        pickle.dump(chunks, f)
    
    # Create summary CSV
    summary_rows = []
    for text, metadata in chunks:
        row = dict(metadata)
        row['text_length'] = len(text)
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_path.replace(".pkl", "_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"✅ Exported chunks:")
    print(f"   Chunks file: {output_path}")
    print(f"   Summary file: {summary_path}")
    print(f"   Total chunks: {len(chunks):,}")


# --- 3.7 Main Pipeline Function ---

def run_semantic_chunking_pipeline(input_path: str = "Data/train/parsed/parsed_documents.pkl",
                                 output_path: str = "chunks_for_embedding.pkl",
                                 chunk_size: int = 200,
                                 chunk_overlap: int = 20,
                                 min_chars: int = 500) -> Dict[str, Any]:
    """
    Run the complete semantic chunking pipeline.
    
    Args:
        input_path: Path to parsed documents
        output_path: Path for output chunks
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        min_chars: Minimum character count for inclusion
        
    Returns:
        Dictionary with pipeline results and statistics
    """
    print("=== Semantic Chunking Pipeline (Step 6) ===")
    print(f"Started at: {datetime.now()}")
    
    try:
        # Step 1: Load documents
        print("\n📋 Step 1: Load & Filter Documents")
        docs = load_parsed_documents_for_chunking(input_path, min_chars)
        
        if not docs:
            raise ValueError("No documents loaded - check input path")
        
        # Step 2: Prepare section texts
        print("\n📋 Step 2: Prepare Section Texts")
        section_texts = prepare_section_texts_for_chunking(docs)
        
        # Step 3: Create entity inventory
        print("\n📋 Step 3: Create Entity Inventory")
        pre_chunk_inventory = create_pre_chunk_entity_inventory(section_texts)
        
        # Step 4: Create chunks
        print("\n📋 Step 4: Create Section-Aware Chunks")
        chunks = create_section_aware_chunks(section_texts, docs, chunk_size, chunk_overlap)
        
        # Step 5: Link adjacent chunks
        print("\n📋 Step 5: Link Adjacent Chunks")
        chunks = link_adjacent_chunks(chunks)
        
        # Step 6: Refine chunk types
        print("\n📋 Step 6: Refine Chunk Types")
        chunks = refine_chunk_types(chunks)
        
        # Step 7: Validate integrity
        print("\n📋 Step 7: Validate Chunk Integrity")
        validation_passed, lost_entities = validate_chunk_integrity(chunks, pre_chunk_inventory)
        
        # Quality gates check
        if not validation_passed:
            print("❌ QUALITY GATE FAILURE: Entity retention < 100%")
            print("   Pipeline aborted to prevent data loss")
            return {
                'success': False,
                'error': 'Entity retention validation failed',
                'lost_entities': lost_entities,
                'chunks_created': len(chunks)
            }
        
        # Step 8: Export results
        print("\n📋 Step 8: Export Chunks")
        export_chunks_for_embedding(chunks, output_path)
        
        # Calculate final statistics
        total_tokens = sum(metadata['token_count'] for _, metadata in chunks)
        avg_tokens_per_chunk = total_tokens / len(chunks) if chunks else 0
        
        results = {
            'success': True,
            'total_documents': len(docs),
            'total_chunks': len(chunks),
            'total_tokens': total_tokens,
            'avg_tokens_per_chunk': avg_tokens_per_chunk,
            'validation_passed': validation_passed,
            'output_path': output_path,
            'pipeline_completed_at': datetime.now().isoformat()
        }
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Final Results:")
        print(f"   Documents processed: {results['total_documents']}")
        print(f"   Chunks created: {results['total_chunks']:,}")
        print(f"   Average tokens per chunk: {results['avg_tokens_per_chunk']:.1f}")
        print(f"   Entity retention: 100%")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'pipeline_failed_at': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Run with default parameters
    results = run_semantic_chunking_pipeline()
    
    if results['success']:
        print("✅ Semantic chunking completed successfully!")
    else:
        print(f"❌ Pipeline failed: {results['error']}") 