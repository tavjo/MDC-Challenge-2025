"""
Test script for DuckDB migration

This script tests the data migration from JSON files to DuckDB tables.
It validates that the migration process works correctly and data integrity is maintained.
"""

import unittest
import tempfile
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add the parent directory to the path to import from src and api
sys.path.append(str(Path(__file__).parent.parent))

from src.models import Document, CitationEntity
from api.database.duckdb_schema import DuckDBSchemaInitializer  
from api.utils.migrate_data import DataMigrator

class TestDuckDBMigration(unittest.TestCase):
    """Test cases for DuckDB migration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name
        # Remove the temporary file so DuckDB can create it properly
        os.unlink(self.db_path)
        
        # Create test data
        self.test_documents = [
            {
                "doi": "10.1002/test1",
                "has_dataset_citation": True,
                "full_text": ["This is test document 1 page 1", "This is test document 1 page 2"],
                "total_char_length": 1000,
                "parsed_timestamp": "2024-01-01T00:00:00Z",
                "total_chunks": 5,
                "total_tokens": 200,
                "avg_tokens_per_chunk": 40.0,
                "file_hash": "hash123",
                "file_path": "/path/to/test1.pdf",
                "citation_entities": [],
                "n_pages": 2
            },
            {
                "doi": "10.1002/test2",
                "has_dataset_citation": False,
                "full_text": ["This is test document 2 page 1"],
                "total_char_length": 500,
                "parsed_timestamp": "2024-01-02T00:00:00Z",
                "total_chunks": 3,
                "total_tokens": 100,
                "avg_tokens_per_chunk": 33.3,
                "file_hash": "hash456",
                "file_path": "/path/to/test2.pdf",
                "citation_entities": [],
                "n_pages": 1
            }
        ]
        
        self.test_citations = [
            {
                "data_citation": "https://doi.org/10.1000/test1",
                "document_id": "10.1002/test1",
                "pages": [1, 2],
                "evidence": ["evidence text 1", "evidence text 2"]
            },
            {
                "data_citation": "https://doi.org/10.1000/test2",
                "document_id": "10.1002/test1",
                "pages": [2],
                "evidence": ["evidence text 3"]
            }
        ]
        
        # Create temporary JSON files
        self.temp_docs_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_documents, self.temp_docs_file)
        self.temp_docs_file.close()
        
        self.temp_citations_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_citations, self.temp_citations_file)
        self.temp_citations_file.close()
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files
        try:
            os.unlink(self.db_path)
            os.unlink(self.temp_docs_file.name)
            os.unlink(self.temp_citations_file.name)
        except:
            pass
    
    def test_schema_initialization(self):
        """Test DuckDB schema initialization."""
        schema_initializer = DuckDBSchemaInitializer(self.db_path)
        
        try:
            schema_initializer.create_schema()
            schema_initializer.validate_schema()
            
            # Check that tables exist
            tables = schema_initializer.conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]
            
            self.assertIn('documents', table_names)
            self.assertIn('citations', table_names)
            self.assertIn('chunks', table_names)
            
        finally:
            schema_initializer.close()
    
    def test_document_migration(self):
        """Test document migration from JSON to DuckDB."""
        migrator = DataMigrator(self.db_path)
        
        try:
            # Initialize schema
            schema_initializer = DuckDBSchemaInitializer(self.db_path)
            schema_initializer.create_schema()
            
            # Test document migration
            migrator.create_connection()
            docs_migrated = migrator.migrate_documents(self.temp_docs_file.name)
            
            # Verify migration
            self.assertEqual(docs_migrated, 2)
            
            # Check data in database
            doc_count = migrator.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            self.assertEqual(doc_count, 2)
            
            # Check specific document
            doc = migrator.conn.execute("SELECT * FROM documents WHERE doi = '10.1002/test1'").fetchone()
            self.assertIsNotNone(doc)
            self.assertEqual(doc[1], True)  # has_dataset_citation
            
        finally:
            migrator.close()
    
    def test_citation_migration(self):
        """Test citation migration from JSON to DuckDB."""
        migrator = DataMigrator(self.db_path)
        
        try:
            # Initialize schema
            schema_initializer = DuckDBSchemaInitializer(self.db_path)
            schema_initializer.create_schema()
            
            # Test citation migration
            migrator.create_connection()
            citations_migrated = migrator.migrate_citations(self.temp_citations_file.name)
            
            # Verify migration
            self.assertEqual(citations_migrated, 2)
            
            # Check data in database
            citation_count = migrator.conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
            self.assertEqual(citation_count, 2)
            
            # Check specific citation
            citation = migrator.conn.execute("""
                SELECT * FROM citations WHERE data_citation = 'https://doi.org/10.1000/test1'
            """).fetchone()
            self.assertIsNotNone(citation)
            self.assertEqual(citation[1], '10.1002/test1')  # document_id
            
        finally:
            migrator.close()
    
    def test_full_migration(self):
        """Test complete migration process."""
        migrator = DataMigrator(self.db_path)
        
        try:
            # Mock the file paths to use our test files
            with patch.object(migrator, 'migrate_documents') as mock_docs, \
                 patch.object(migrator, 'migrate_citations') as mock_citations:
                
                mock_docs.return_value = 2
                mock_citations.return_value = 2
                
                results = migrator.run_migration()
                
                self.assertTrue(results["migration_successful"])
                self.assertEqual(results["documents_migrated"], 2)
                self.assertEqual(results["citations_migrated"], 2)
                
        finally:
            migrator.close()
    
    def test_pydantic_model_validation(self):
        """Test that Pydantic models validate correctly."""
        # Test Document model
        doc_dict = self.test_documents[0]
        document = Document(**doc_dict)
        
        self.assertEqual(document.doi, "10.1002/test1")
        self.assertTrue(document.has_dataset_citation)
        self.assertEqual(document.total_char_length, 1000)
        
        # Test to_duckdb_row method
        row_data = document.to_duckdb_row()
        self.assertIsInstance(row_data, dict)
        self.assertIn("doi", row_data)
        self.assertIn("full_text", row_data)
        
        # Test CitationEntity model
        citation_dict = self.test_citations[0]
        citation = CitationEntity(**citation_dict)
        
        self.assertEqual(citation.data_citation, "https://doi.org/10.1000/test1")
        self.assertEqual(citation.document_id, "10.1002/test1")
        self.assertEqual(citation.pages, [1, 2])
        
        # Test to_duckdb_row method
        row_data = citation.to_duckdb_row()
        self.assertIsInstance(row_data, dict)
        self.assertIn("data_citation", row_data)
        self.assertIn("document_id", row_data)
    
    def test_error_handling(self):
        """Test error handling during migration."""
        migrator = DataMigrator(self.db_path)
        
        try:
            # Test with non-existent file
            docs_migrated = migrator.migrate_documents("nonexistent_file.json")
            self.assertEqual(docs_migrated, 0)  # Should handle gracefully
            
        except Exception as e:
            # It's okay if it raises an exception for non-existent file
            self.assertIn("No such file", str(e))
        
        finally:
            migrator.close()
    
    def test_data_integrity(self):
        """Test that data integrity is maintained after migration."""
        migrator = DataMigrator(self.db_path)
        
        try:
            # Initialize schema
            schema_initializer = DuckDBSchemaInitializer(self.db_path)
            schema_initializer.create_schema()
            
            # Migrate data
            migrator.create_connection()
            migrator.migrate_documents(self.temp_docs_file.name)
            migrator.migrate_citations(self.temp_citations_file.name)
            
            # Validate that we can reconstruct models from database
            doc_rows = migrator.conn.execute("SELECT * FROM documents").fetchall()
            citation_rows = migrator.conn.execute("SELECT * FROM citations").fetchall()
            
            self.assertEqual(len(doc_rows), 2)
            self.assertEqual(len(citation_rows), 2)
            
            # Check foreign key relationship
            joined_count = migrator.conn.execute("""
                SELECT COUNT(*) FROM documents d 
                JOIN citations c ON d.doi = c.document_id
            """).fetchone()[0]
            self.assertEqual(joined_count, 2)
            
        finally:
            migrator.close()


class TestMigrationIntegration(unittest.TestCase):
    """Integration tests for the migration process."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name
        # Remove the temporary file so DuckDB can create it properly
        os.unlink(self.db_path)
        
    def tearDown(self):
        """Clean up integration test fixtures."""
        try:
            os.unlink(self.db_path)
        except:
            pass
    
    def test_schema_and_migration_integration(self):
        """Test that schema creation and migration work together."""
        # Initialize schema
        schema_initializer = DuckDBSchemaInitializer(self.db_path)
        schema_initializer.create_schema()
        schema_initializer.validate_schema()
        schema_initializer.close()
        
        # Test that migrator can connect to existing schema
        migrator = DataMigrator(self.db_path)
        migrator.create_connection()
        
        # Verify tables exist
        tables = migrator.conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        
        self.assertIn('documents', table_names)
        self.assertIn('citations', table_names)
        self.assertIn('chunks', table_names)
        
        migrator.close()


def run_migration_test():
    """Run the migration test suite."""
    print("Running DuckDB Migration Tests...")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(loader.loadTestsFromTestCase(TestDuckDBMigration))
    suite.addTest(loader.loadTestsFromTestCase(TestMigrationIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("=" * 50)
    if result.wasSuccessful():
        print("✅ All migration tests passed!")
    else:
        print("❌ Some migration tests failed!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_migration_test()
    sys.exit(0 if success else 1) 