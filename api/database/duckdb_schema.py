"""
DuckDB Schema Initialization Script

This script creates the necessary DuckDB schema for the MDC Challenge 2025 project.
It aligns with the schema design from the chunk_and_embedding_api.md guide.
"""

import duckdb
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DuckDBSchemaInitializer:
    def __init__(self, db_path: str = "artifacts/mdc_challenge.db"):
        """
        Initialize the DuckDB schema initializer.
        
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
    
    def create_documents_table(self):
        """Create the documents table matching the Document model."""
        logger.info("Creating documents table...")
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS documents (
            doi VARCHAR PRIMARY KEY,
            has_dataset_citation BOOLEAN,
            full_text VARCHAR[], 
            total_char_length INTEGER,
            parsed_timestamp TIMESTAMP,
            total_chunks INTEGER,
            total_tokens INTEGER,
            avg_tokens_per_chunk REAL,
            file_hash VARCHAR,
            file_path VARCHAR,
            citation_entities VARCHAR[],
            n_pages INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        self.conn.execute(create_table_sql)
        logger.info("Documents table created successfully")
        
    def create_citations_table(self):
        """Create the citations table (normalized from citation_entities)."""
        logger.info("Creating citations table...")
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS citations (
            data_citation VARCHAR NOT NULL,
            document_id VARCHAR NOT NULL,
            pages INTEGER[],
            evidence VARCHAR[],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (data_citation, document_id),
            FOREIGN KEY (document_id) REFERENCES documents(doi)
        );
        """
        
        self.conn.execute(create_table_sql)
        logger.info("Citations table created successfully")
        
    def create_chunks_table(self):
        """Create the chunks table matching the Chunk model."""
        logger.info("Creating chunks table...")
        
        create_table_sql = """
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
            ),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(doi)
        );
        """
        
        self.conn.execute(create_table_sql)
        logger.info("Chunks table created successfully")
        
    def create_indexes(self):
        """Create indexes for better query performance."""
        logger.info("Creating indexes...")
        
        # Index on document_id for chunks table
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
            ON chunks(document_id);
        """)
        
        # Index on citation data_citation for faster lookups
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_citations_data_citation 
            ON citations(data_citation);
        """)
        
        # Index on citation document_id for faster lookups
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_citations_document_id 
            ON citations(document_id);
        """)
        
        logger.info("Indexes created successfully")
        
    def create_schema(self):
        """Create the complete database schema."""
        logger.info("Starting DuckDB schema creation...")
        
        try:
            # Create connection
            if not self.conn:
                self.create_connection()
            
            # Create tables
            self.create_documents_table()
            self.create_citations_table()
            self.create_chunks_table()
            
            # Create indexes
            self.create_indexes()
            
            logger.info("DuckDB schema creation completed successfully")
            
        except Exception as e:
            logger.error(f"Error creating schema: {str(e)}")
            raise
            
    def validate_schema(self):
        """Validate that the schema was created correctly."""
        logger.info("Validating schema...")
        
        # Check tables exist
        tables = self.conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        
        expected_tables = ['documents', 'citations', 'chunks']
        
        for table in expected_tables:
            if table not in table_names:
                raise ValueError(f"Table {table} was not created")
        
        # Check table structures
        for table in expected_tables:
            columns = self.conn.execute(f"DESCRIBE {table}").fetchall()
            logger.info(f"Table {table} has {len(columns)} columns")
            
        logger.info("Schema validation completed successfully")
        
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def main():
    """Main function to initialize the DuckDB schema."""
    schema_initializer = DuckDBSchemaInitializer()
    
    try:
        schema_initializer.create_schema()
        schema_initializer.validate_schema()
        logger.info("DuckDB schema initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Schema initialization failed: {str(e)}")
        raise
        
    finally:
        schema_initializer.close()


if __name__ == "__main__":
    main() 