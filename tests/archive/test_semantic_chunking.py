#!/usr/bin/env python3
"""
Unit tests for semantic chunking functionality
"""

import pytest
import pandas as pd
import pickle
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Import the functions we want to test
from src.semantic_chunking import (
    load_parsed_documents_for_chunking,
    prepare_section_texts_for_chunking,
    create_pre_chunk_entity_inventory,
    create_section_aware_chunks,
    link_adjacent_chunks,
    refine_chunk_types,
    validate_chunk_integrity,
    export_chunks_for_embedding,
    ENTITY_PATTERNS,
    PRIORITY_SECTIONS
)


class TestSemanticChunking:
    """Test class for semantic chunking functions"""
    
    def setup_method(self):
        """Set up test data before each test method"""
        # Create sample document data
        self.sample_docs = {
            "10.1234/test.paper.1": {
                "doi": "10.1234/test.paper.1",
                "full_text": "This is a test paper with methods and results. It contains GSE12345 and SRR67890 datasets.",
                "section_texts": {
                    "methods": "This section describes methods using GSE12345 dataset from GEO.",
                    "results": "Results show significant findings with SRR67890 sequencing data.",
                    "data_availability": "Data is available at GSE12345 and SRR67890."
                },
                "section_order": {
                    "methods": 1,
                    "results": 2,
                    "data_availability": 3
                },
                "conversion_source": "grobid",
                "validation": {"validation_passed": True}
            },
            "10.1234/test.paper.2": {
                "doi": "10.1234/test.paper.2",
                "full_text": "Short test paper.",
                "section_texts": {},
                "section_order": {},
                "conversion_source": "existing_xml",
                "validation": {"validation_passed": True}
            }
        }
    
    def test_load_parsed_documents_for_chunking(self):
        """Test document loading and filtering"""
        # Create temporary pickle file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            pickle.dump(self.sample_docs, tmp)
            tmp_path = tmp.name
        
        try:
            # Test loading with default min_chars
            docs = load_parsed_documents_for_chunking(tmp_path, min_chars=50)
            assert len(docs) == 1  # Only the first doc should pass the min_chars filter
            assert "10.1234/test.paper.1" in docs
            
            # Test loading with very low min_chars
            docs = load_parsed_documents_for_chunking(tmp_path, min_chars=10)
            assert len(docs) == 2  # Both docs should pass
            
        finally:
            os.unlink(tmp_path)
    
    def test_prepare_section_texts_for_chunking(self):
        """Test section text preparation"""
        section_texts = prepare_section_texts_for_chunking(self.sample_docs)
        
        # Check that we get the expected structure
        assert len(section_texts) == 2
        assert "10.1234/test.paper.1" in section_texts
        assert "10.1234/test.paper.2" in section_texts
        
        # First document should have priority sections
        doc1_sections = section_texts["10.1234/test.paper.1"]
        assert "methods" in doc1_sections
        assert "results" in doc1_sections
        assert "data_availability" in doc1_sections
        
        # Second document should fall back to full_document
        doc2_sections = section_texts["10.1234/test.paper.2"]
        assert "full_document" in doc2_sections
    
    def test_create_pre_chunk_entity_inventory(self):
        """Test entity inventory creation"""
        section_texts = prepare_section_texts_for_chunking(self.sample_docs)
        inventory = create_pre_chunk_entity_inventory(section_texts)
        
        # Check that we get a DataFrame
        assert isinstance(inventory, pd.DataFrame)
        assert len(inventory) > 0
        
        # Check that we found some entities
        entity_counts = inventory[inventory['count'] > 0]
        assert len(entity_counts) > 0
        
        # Check for expected entities
        gse_entries = inventory[
            (inventory['pattern'] == 'GEO_Series') & 
            (inventory['count'] > 0)
        ]
        assert len(gse_entries) > 0
        
        srr_entries = inventory[
            (inventory['pattern'] == 'SRA_Run') & 
            (inventory['count'] > 0)
        ]
        assert len(srr_entries) > 0
    
    def test_create_section_aware_chunks(self):
        """Test chunk creation"""
        section_texts = prepare_section_texts_for_chunking(self.sample_docs)
        chunks = create_section_aware_chunks(section_texts, self.sample_docs, 
                                           chunk_size=50, chunk_overlap=10)
        
        # Check that we got chunks
        assert len(chunks) > 0
        
        # Check chunk structure
        for text, metadata in chunks:
            assert isinstance(text, str)
            assert isinstance(metadata, dict)
            assert 'chunk_id' in metadata
            assert 'document_id' in metadata
            assert 'section_type' in metadata
            assert 'token_count' in metadata
            assert metadata['token_count'] > 0
    
    def test_link_adjacent_chunks(self):
        """Test chunk linking"""
        section_texts = prepare_section_texts_for_chunking(self.sample_docs)
        chunks = create_section_aware_chunks(section_texts, self.sample_docs, 
                                           chunk_size=50, chunk_overlap=10)
        linked_chunks = link_adjacent_chunks(chunks)
        
        # Check that we got the same number of chunks
        assert len(linked_chunks) == len(chunks)
        
        # Group by document to check linking
        doc_chunks = {}
        for text, metadata in linked_chunks:
            doc_id = metadata['document_id']
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append((text, metadata))
        
        # Check linking for each document
        for doc_id, doc_chunk_list in doc_chunks.items():
            if len(doc_chunk_list) > 1:
                # Sort by section order to check proper linking
                doc_chunk_list.sort(key=lambda x: x[1]['section_order'])
                
                # Check that interior chunks have both prev and next
                for i in range(1, len(doc_chunk_list) - 1):
                    metadata = doc_chunk_list[i][1]
                    assert metadata['previous_chunk_id'] is not None
                    assert metadata['next_chunk_id'] is not None
                
                # Check that first chunk has no previous
                first_metadata = doc_chunk_list[0][1]
                assert first_metadata['previous_chunk_id'] is None
                
                # Check that last chunk has no next
                last_metadata = doc_chunk_list[-1][1]
                assert last_metadata['next_chunk_id'] is None
    
    def test_refine_chunk_types(self):
        """Test chunk type refinement"""
        section_texts = prepare_section_texts_for_chunking(self.sample_docs)
        chunks = create_section_aware_chunks(section_texts, self.sample_docs, 
                                           chunk_size=50, chunk_overlap=10)
        refined_chunks = refine_chunk_types(chunks)
        
        # Check that we got the same number of chunks
        assert len(refined_chunks) == len(chunks)
        
        # Check that all chunks have chunk_type
        for text, metadata in refined_chunks:
            assert 'chunk_type' in metadata
            assert metadata['chunk_type'] in ['body', 'caption', 'header', 'data_statement']
    
    def test_validate_chunk_integrity(self):
        """Test entity validation"""
        section_texts = prepare_section_texts_for_chunking(self.sample_docs)
        pre_inventory = create_pre_chunk_entity_inventory(section_texts)
        
        chunks = create_section_aware_chunks(section_texts, self.sample_docs, 
                                           chunk_size=50, chunk_overlap=10)
        
        validation_passed, lost_entities = validate_chunk_integrity(chunks, pre_inventory)
        
        # Check return types
        assert isinstance(validation_passed, bool)
        assert isinstance(lost_entities, pd.DataFrame)
        
        # For our test data, we should achieve 100% retention
        assert validation_passed, "Entity validation should pass for test data"
    
    def test_export_chunks_for_embedding(self):
        """Test chunk export functionality"""
        section_texts = prepare_section_texts_for_chunking(self.sample_docs)
        chunks = create_section_aware_chunks(section_texts, self.sample_docs, 
                                           chunk_size=50, chunk_overlap=10)
        chunks = link_adjacent_chunks(chunks)
        chunks = refine_chunk_types(chunks)
        
        # Export to temporary files
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "test_chunks.pkl")
            export_chunks_for_embedding(chunks, output_path)
            
            # Check that files were created
            assert os.path.exists(output_path)
            summary_path = output_path.replace(".pkl", "_summary.csv")
            assert os.path.exists(summary_path)
            
            # Check that we can load the pickle file
            with open(output_path, 'rb') as f:
                loaded_chunks = pickle.load(f)
            assert len(loaded_chunks) == len(chunks)
            
            # Check that summary CSV has expected columns
            summary_df = pd.read_csv(summary_path)
            expected_columns = ['chunk_id', 'document_id', 'section_type', 
                              'token_count', 'chunk_type', 'text_length']
            for col in expected_columns:
                assert col in summary_df.columns
    
    def test_entity_patterns(self):
        """Test that entity patterns work correctly"""
        test_text = """
        This paper uses data from GSE12345 and GSM67890 datasets.
        Sequences were submitted to SRR123456 and SRX789012.
        The DOI is 10.1234/example.paper.2023.
        PDB structure 1ABC was analyzed.
        """
        
        # Test specific patterns
        assert len(ENTITY_PATTERNS['GEO_Series'].findall(test_text)) == 1
        assert len(ENTITY_PATTERNS['GEO_Sample'].findall(test_text)) == 1
        assert len(ENTITY_PATTERNS['SRA_Run'].findall(test_text)) == 1
        assert len(ENTITY_PATTERNS['SRA_Experiment'].findall(test_text)) == 1
        assert len(ENTITY_PATTERNS['DOI'].findall(test_text)) == 1
        assert len(ENTITY_PATTERNS['PDB_ID'].findall(test_text)) == 1
    
    def test_priority_sections(self):
        """Test that priority sections are correctly defined"""
        expected_sections = ["data_availability", "methods", "supplementary", "results"]
        assert PRIORITY_SECTIONS == expected_sections


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"]) 