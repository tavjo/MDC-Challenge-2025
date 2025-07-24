#!/usr/bin/env python3
"""
Test script for DuckDB integration in run_semantic_chunking.py
Tests the modified pipeline functions with DuckDB I/O
"""

import sys
import os
import unittest
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.run_semantic_chunking import (
    load_input_data_from_duckdb,
    save_chunks_to_duckdb,
    run_semantic_chunking_pipeline
)
from src.semantic_chunking import OfflineEmbedder, download_offline_model
from src.models import Document, CitationEntity, Chunk
import duckdb

class TestDuckDBIntegration(unittest.TestCase):
    """Test DuckDB integration in semantic chunking pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db_path = "artifacts/test_mdc_challenge.db"
        self.main_db_path = "artifacts/mdc_challenge.db"
        
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def test_load_input_data_from_duckdb(self):
        """Test loading data from DuckDB"""
        print("\n🧪 Testing load_input_data_from_duckdb()...")
        
        # Check if main database exists
        if not os.path.exists(self.main_db_path):
            print(f"❌ Main database not found at {self.main_db_path}")
            print("   Please run the data migration first")
            return
            
        try:
            # Load data from DuckDB
            documents, citation_entities = load_input_data_from_duckdb(self.main_db_path)
            
            # Verify data was loaded
            self.assertIsInstance(documents, list)
            self.assertIsInstance(citation_entities, list)
            self.assertGreater(len(documents), 0, "No documents loaded")
            self.assertGreater(len(citation_entities), 0, "No citation entities loaded")
            
            # Verify data types
            self.assertIsInstance(documents[0], Document)
            self.assertIsInstance(citation_entities[0], CitationEntity)
            
            print(f"✅ Successfully loaded {len(documents)} documents and {len(citation_entities)} citations")
            
        except Exception as e:
            print(f"❌ Failed to load data from DuckDB: {str(e)}")
            raise
    
    def test_save_chunks_to_duckdb(self):
        """Test saving chunks to DuckDB"""
        print("\n🧪 Testing save_chunks_to_duckdb()...")
        
        # Create test database with schema
        conn = duckdb.connect(self.test_db_path)
        
        # Create chunks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id VARCHAR PRIMARY KEY,
                document_id VARCHAR NOT NULL,
                chunk_text TEXT NOT NULL,
                score REAL,
                chunk_metadata STRUCT(
                    created_at TIMESTAMP,
                    previous_chunk_id VARCHAR,
                    next_chunk_id VARCHAR,
                    token_count INTEGER,
                    citation_entities VARCHAR[]
                )
            )
        """)
        
        conn.close()
        
        # Create test chunks
        from src.models import ChunkMetadata
        test_chunks = [
            Chunk(
                chunk_id="test_chunk_1",
                document_id="test_doc_1",
                text="This is a test chunk with some content.",
                score=0.95,
                chunk_metadata=ChunkMetadata(
                    chunk_id="test_chunk_1",
                    token_count=10,
                    citation_entities=[]
                )
            ),
            Chunk(
                chunk_id="test_chunk_2",
                document_id="test_doc_1",
                text="This is another test chunk.",
                score=0.87,
                chunk_metadata=ChunkMetadata(
                    chunk_id="test_chunk_2",
                    token_count=8,
                    citation_entities=[]
                )
            )
        ]
        
        try:
            # Save chunks to DuckDB
            success = save_chunks_to_duckdb(test_chunks, self.test_db_path)
            self.assertTrue(success, "Failed to save chunks to DuckDB")
            
            # Verify chunks were saved
            conn = duckdb.connect(self.test_db_path)
            result = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            conn.close()
            
            self.assertEqual(result[0], 2, "Expected 2 chunks in database")
            
            print(f"✅ Successfully saved {len(test_chunks)} chunks to DuckDB")
            
        except Exception as e:
            print(f"❌ Failed to save chunks to DuckDB: {str(e)}")
            raise
    
    def test_run_semantic_chunking_pipeline_with_duckdb(self):
        """Test running the complete pipeline with DuckDB"""
        print("\n🧪 Testing full pipeline with DuckDB...")
        
        # Check if main database exists
        if not os.path.exists(self.main_db_path):
            print(f"❌ Main database not found at {self.main_db_path}")
            print("   Please run the data migration first")
            return
            
        try:
            # Run pipeline with DuckDB enabled, subset for testing
            result = run_semantic_chunking_pipeline(
                use_duckdb=True,
                db_path=self.main_db_path,
                subset=True,
                subset_size=5,  # Test with only 5 documents
                output_dir="test_output"
            )
            
            # Verify result
            self.assertIsNotNone(result)
            self.assertTrue(result.success, f"Pipeline failed: {result.error}")
            self.assertGreater(result.total_chunks, 0, "No chunks created")
            self.assertGreater(result.total_tokens, 0, "No tokens counted")
            
            print(f"✅ Pipeline completed successfully:")
            print(f"   Documents processed: {result.total_documents}")
            print(f"   Chunks created: {result.total_chunks}")
            print(f"   Total tokens: {result.total_tokens}")
            print(f"   Entity retention: {result.entity_retention}%")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            raise
    
    def test_database_connectivity(self):
        """Test basic DuckDB connectivity"""
        print("\n🧪 Testing DuckDB connectivity...")
        
        try:
            # Test database connection
            conn = duckdb.connect(self.main_db_path)
            
            # Check if required tables exist
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]
            
            required_tables = ['documents', 'citations', 'chunks']
            missing_tables = [table for table in required_tables if table not in table_names]
            
            if missing_tables:
                print(f"❌ Missing tables: {missing_tables}")
                print("   Please run the schema initialization first")
                return
            
            # Test data availability
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            citation_count = conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
            
            conn.close()
            
            print(f"✅ Database connectivity test passed:")
            print(f"   Tables found: {table_names}")
            print(f"   Documents: {doc_count}")
            print(f"   Citations: {citation_count}")
            
        except Exception as e:
            print(f"❌ Database connectivity test failed: {str(e)}")
            raise
    
    def test_offline_model_download(self):
        """Test downloading and caching offline models"""
        print("\n🧪 Testing offline model download...")
        
        try:
            # Test downloading the default model
            success = download_offline_model("bge-small-en-v1.5")
            self.assertTrue(success, "Failed to download offline model")
            
            # Check if model was cached
            import os
            cache_dir = "./offline_models"
            self.assertTrue(os.path.exists(cache_dir), "Cache directory not created")
            
            # Check if model info file was created
            info_file = os.path.join(cache_dir, "bge-small-en-v1.5_info.json")
            self.assertTrue(os.path.exists(info_file), "Model info file not created")
            
            print("✅ Offline model download test passed")
            
        except Exception as e:
            print(f"❌ Offline model download test failed: {str(e)}")
            raise
    
    def test_offline_embedder_initialization(self):
        """Test offline embedder initialization"""
        print("\n🧪 Testing offline embedder initialization...")
        
        try:
            # Test creating offline embedder
            embedder = OfflineEmbedder("bge-small-en-v1.5")
            self.assertIsNotNone(embedder, "Failed to create offline embedder")
            self.assertEqual(embedder.model_name, "bge-small-en-v1.5")
            
            # Test embedding generation
            test_text = "This is a test sentence for embedding."
            embedding = embedder.get_text_embedding(test_text)
            
            self.assertIsInstance(embedding, list, "Embedding should be a list")
            self.assertGreater(len(embedding), 0, "Embedding should not be empty")
            self.assertIsInstance(embedding[0], float, "Embedding elements should be floats")
            
            print(f"✅ Offline embedder test passed (embedding dim: {len(embedding)})")
            
        except Exception as e:
            print(f"❌ Offline embedder test failed: {str(e)}")
            raise
    
    def test_offline_embedder_batch_processing(self):
        """Test offline embedder batch processing"""
        print("\n🧪 Testing offline embedder batch processing...")
        
        try:
            embedder = OfflineEmbedder("bge-small-en-v1.5")
            
            # Test batch embedding generation
            test_texts = [
                "This is the first test sentence.",
                "This is the second test sentence.",
                "This is the third test sentence."
            ]
            
            embeddings = embedder.get_text_embeddings(test_texts)
            
            self.assertIsInstance(embeddings, list, "Batch embeddings should be a list")
            self.assertEqual(len(embeddings), len(test_texts), "Should have one embedding per text")
            
            for i, embedding in enumerate(embeddings):
                self.assertIsInstance(embedding, list, f"Embedding {i} should be a list")
                self.assertGreater(len(embedding), 0, f"Embedding {i} should not be empty")
                self.assertIsInstance(embedding[0], float, f"Embedding {i} elements should be floats")
            
            print(f"✅ Offline embedder batch test passed ({len(embeddings)} embeddings)")
            
        except Exception as e:
            print(f"❌ Offline embedder batch test failed: {str(e)}")
            raise
    
    def test_run_semantic_chunking_pipeline_with_offline_model(self):
        """Test running the complete pipeline with offline model"""
        print("\n🧪 Testing full pipeline with offline model...")
        
        # Check if main database exists
        if not os.path.exists(self.main_db_path):
            print(f"❌ Main database not found at {self.main_db_path}")
            print("   Please run the data migration first")
            return
            
        try:
            # First ensure the offline model is downloaded
            download_offline_model("bge-small-en-v1.5")
            
            # Create offline config file path
            offline_config = "configs/chunking_offline.yaml"
            
            # Run pipeline with offline model, subset for testing
            result = run_semantic_chunking_pipeline(
                use_duckdb=True,
                db_path=self.main_db_path,
                subset=True,
                subset_size=3,  # Test with only 3 documents
                output_dir="test_output_offline",
                cfg_path=offline_config
            )
            
            # Verify result
            self.assertIsNotNone(result)
            self.assertTrue(result.success, f"Pipeline failed: {result.error}")
            self.assertGreater(result.total_chunks, 0, "No chunks created")
            self.assertGreater(result.total_tokens, 0, "No tokens counted")
            
            print(f"✅ Offline pipeline completed successfully:")
            print(f"   Documents processed: {result.total_documents}")
            print(f"   Chunks created: {result.total_chunks}")
            print(f"   Total tokens: {result.total_tokens}")
            print(f"   Entity retention: {result.entity_retention}%")
            
        except Exception as e:
            print(f"❌ Offline pipeline test failed: {str(e)}")
            raise
    
    def test_model_selection_logic(self):
        """Test the model selection logic in _build_embedder"""
        print("\n🧪 Testing model selection logic...")
        
        try:
            from src.semantic_chunking import _build_embedder
            
            # Test offline model selection
            cfg = {"offline_model": {"cache_dir": "./offline_models"}}
            
            # Test explicit offline model
            embedder1 = _build_embedder("bge-small-en-v1.5", cfg)
            self.assertIsInstance(embedder1, OfflineEmbedder)
            
            # Test offline: prefix
            embedder2 = _build_embedder("offline:bge-small-en-v1.5", cfg)
            self.assertIsInstance(embedder2, OfflineEmbedder)
            
            # Test different offline models
            embedder3 = _build_embedder("all-MiniLM-L6-v2", cfg)
            self.assertIsInstance(embedder3, OfflineEmbedder)
            
            print("✅ Model selection logic test passed")
            
        except Exception as e:
            print(f"❌ Model selection logic test failed: {str(e)}")
            raise

def main():
    """Run all tests"""
    print("🧪 Running DuckDB Integration Tests for Semantic Chunking Pipeline")
    print("=" * 70)
    
    test_suite = unittest.TestSuite()
    
    # Add test methods
    test_suite.addTest(TestDuckDBIntegration('test_database_connectivity'))
    test_suite.addTest(TestDuckDBIntegration('test_load_input_data_from_duckdb'))
    test_suite.addTest(TestDuckDBIntegration('test_save_chunks_to_duckdb'))
    test_suite.addTest(TestDuckDBIntegration('test_run_semantic_chunking_pipeline_with_duckdb'))
    
    # Add offline model tests
    test_suite.addTest(TestDuckDBIntegration('test_offline_model_download'))
    test_suite.addTest(TestDuckDBIntegration('test_offline_embedder_initialization'))
    test_suite.addTest(TestDuckDBIntegration('test_offline_embedder_batch_processing'))
    test_suite.addTest(TestDuckDBIntegration('test_run_semantic_chunking_pipeline_with_offline_model'))
    test_suite.addTest(TestDuckDBIntegration('test_model_selection_logic'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} test(s) failed")
        
    return result.wasSuccessful()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 