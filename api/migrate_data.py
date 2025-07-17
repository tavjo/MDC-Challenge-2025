"""
Data Migration Script

This script migrates existing JSON files to DuckDB tables:
- Data/train/documents_with_known_entities.json → documents table
- Data/citation_entities_known.json → citations table

It uses the existing Pydantic models to ensure data integrity.
"""

import json
import duckdb
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys, os

# Add the parent directory to the path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.models import Document, CitationEntity
from api.duckdb_schema import DuckDBSchemaInitializer
from src.helpers import timer_wrap, initialize_logging

# Configure logging
filename = os.path.basename(__file__)
logger = initialize_logging(log_file=filename)

class DataMigrator:
    def __init__(self, db_path: str = "artifacts/mdc_challenge.db"):
        """
        Initialize the data migrator.
        
        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = db_path
        self.conn = None
        
    def create_connection(self) -> duckdb.DuckDBPyConnection:
        """Create and return a DuckDB connection."""
        # Ensure the directory exists
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create connection
        self.conn = duckdb.connect(str(db_path))
        
        # Note: DuckDB supports foreign key constraints in table definitions
        # but doesn't require explicit enabling like other databases
        
        return self.conn
    
    def load_documents_from_json(self, file_path: str) -> List[Document]:
        """Load documents from JSON file and convert to Pydantic models."""
        logger.info(f"Loading documents from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            try:
                # Convert to Pydantic model for validation
                document = Document.model_validate(item)
                documents.append(document)
            except Exception as e:
                logger.error(f"Error processing document {item.get('doi', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def load_citations_from_json(self, file_path: str) -> List[CitationEntity]:
        """Load citation entities from JSON file and convert to Pydantic models."""
        logger.info(f"Loading citations from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        citations = []
        for item in data:
            try:
                # Convert to Pydantic model for validation
                citation = CitationEntity.model_validate(item)
                citations.append(citation)
            except Exception as e:
                logger.error(f"Error processing citation {item.get('data_citation', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded {len(citations)} citations")
        return citations
    
    def insert_documents(self, documents: List[Document]) -> int:
        """Insert documents into the DuckDB documents table."""
        logger.info(f"Inserting {len(documents)} documents into DuckDB...")
        
        inserted_count = 0
        for document in documents:
            try:
                # Use the to_duckdb_row method from the model
                row_data = document.to_duckdb_row()
                
                # Insert into documents table
                self.conn.execute("""
                    INSERT OR REPLACE INTO documents 
                    (doi, has_dataset_citation, full_text, total_char_length, 
                     parsed_timestamp, total_chunks, total_tokens, avg_tokens_per_chunk, 
                     file_hash, file_path, citation_entities, n_pages, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row_data["doi"],
                    row_data["has_dataset_citation"],
                    row_data["full_text"],
                    row_data["total_char_length"],
                    row_data["parsed_timestamp"],
                    row_data["total_chunks"],
                    row_data["total_tokens"],
                    row_data["avg_tokens_per_chunk"],
                    row_data["file_hash"],
                    row_data["file_path"],
                    row_data["citation_entities"],
                    row_data["n_pages"],
                    row_data["created_at"],
                    row_data["updated_at"]
                ))
                
                inserted_count += 1
                
            except Exception as e:
                logger.error(f"Error inserting document {document.doi}: {str(e)}")
                continue
        
        logger.info(f"Successfully inserted {inserted_count} documents")
        return inserted_count
    
    def insert_citations(self, citations: List[CitationEntity]) -> int:
        """Insert citations into the DuckDB citations table."""
        logger.info(f"Inserting {len(citations)} citations into DuckDB...")
        
        inserted_count = 0
        for citation in citations:
            try:
                # Use the to_duckdb_row method from the model
                row_data = citation.to_duckdb_row()
                
                # Insert into citations table
                self.conn.execute("""
                    INSERT OR REPLACE INTO citations 
                    (data_citation, document_id, pages, evidence, created_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    row_data["data_citation"],
                    row_data["document_id"],
                    row_data["pages"],
                    row_data["evidence"]
                ))
                
                inserted_count += 1
                
            except Exception as e:
                logger.error(f"Error inserting citation {citation.data_citation}: {str(e)}")
                continue
        
        logger.info(f"Successfully inserted {inserted_count} citations")
        return inserted_count
    
    def migrate_documents(self, json_file_path: str = "Data/train/documents_with_known_entities.json") -> int:
        """Migrate documents from JSON file to DuckDB table."""
        logger.info("Starting document migration...")
        
        # Load documents from JSON
        documents = self.load_documents_from_json(json_file_path)
        
        # Insert into DuckDB
        inserted_count = self.insert_documents(documents)
        
        logger.info(f"Document migration completed. Inserted {inserted_count} documents")
        return inserted_count
    
    def migrate_citations(self, json_file_path: str = "Data/citation_entities_known.json") -> int:
        """Migrate citations from JSON file to DuckDB table."""
        logger.info("Starting citation migration...")
        
        # Load citations from JSON
        citations = self.load_citations_from_json(json_file_path)
        
        # Insert into DuckDB
        inserted_count = self.insert_citations(citations)
        
        logger.info(f"Citation migration completed. Inserted {inserted_count} citations")
        return inserted_count
    
    def validate_migration(self) -> Dict[str, Any]:
        """Validate that the migration was successful."""
        logger.info("Validating migration...")
        
        validation_results = {}
        
        # Check document count
        doc_count = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        validation_results["documents_count"] = doc_count
        
        # Check citation count
        citation_count = self.conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
        validation_results["citations_count"] = citation_count
        
        # Check for documents with citations
        docs_with_citations = self.conn.execute("""
            SELECT COUNT(DISTINCT d.doi) 
            FROM documents d 
            JOIN citations c ON d.doi = c.document_id
        """).fetchone()[0]
        validation_results["documents_with_citations"] = docs_with_citations
        
        # Sample some records to verify structure
        sample_doc = self.conn.execute("SELECT * FROM documents LIMIT 1").fetchone()
        if sample_doc:
            validation_results["sample_document_structure"] = "OK"
        
        sample_citation = self.conn.execute("SELECT * FROM citations LIMIT 1").fetchone()
        if sample_citation:
            validation_results["sample_citation_structure"] = "OK"
        
        logger.info(f"Migration validation results: {validation_results}")
        return validation_results
    
    def run_migration(self) -> Dict[str, Any]:
        """Run the complete migration process."""
        logger.info("Starting data migration...")
        
        try:
            # Create connection
            if not self.conn:
                self.create_connection()
            
            # Initialize schema first
            schema_initializer = DuckDBSchemaInitializer(self.db_path)
            schema_initializer.conn = self.conn
            schema_initializer.create_schema()
            
            # Migrate documents
            docs_migrated = self.migrate_documents()
            
            # Migrate citations
            citations_migrated = self.migrate_citations()
            
            # Validate migration
            validation_results = self.validate_migration()
            
            # Prepare final results
            results = {
                "migration_successful": True,
                "documents_migrated": docs_migrated,
                "citations_migrated": citations_migrated,
                "validation": validation_results
            }
            
            logger.info("Data migration completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            return {
                "migration_successful": False,
                "error": str(e)
            }
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


@timer_wrap
def main():
    """Main function to run the data migration."""
    migrator = DataMigrator()
    
    try:
        results = migrator.run_migration()
        
        if results["migration_successful"]:
            logger.info("="*50)
            logger.info("MIGRATION SUMMARY")
            logger.info("="*50)
            logger.info(f"Documents migrated: {results['documents_migrated']}")
            logger.info(f"Citations migrated: {results['citations_migrated']}")
            logger.info(f"Documents in DB: {results['validation']['documents_count']}")
            logger.info(f"Citations in DB: {results['validation']['citations_count']}")
            logger.info(f"Documents with citations: {results['validation']['documents_with_citations']}")
            logger.info("="*50)
        else:
            logger.error(f"Migration failed: {results['error']}")
            
    except Exception as e:
        logger.error(f"Migration process failed: {str(e)}")
        raise
        
    finally:
        migrator.close()


if __name__ == "__main__":
    main() 