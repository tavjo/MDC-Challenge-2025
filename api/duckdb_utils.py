"""
DuckDB utility functions for the MDC Challenge 2025 API

This module provides helper functions for database operations, following the 
specification from chunk_and_embedding_api.md.
"""

import sys
from pathlib import Path
from typing import List, Iterable, Union, Tuple

import duckdb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models import Document, CitationEntity, Chunk
from src.helpers import initialize_logging

# Initialize logging
logger = initialize_logging("duckdb_utils")

class DuckDBHelper:
    """
    Helper class for DuckDB operations.
    
    NOTE: DuckDB operates in single-writer mode. This means only one process
    can have a write transaction at a time. Concurrent calls to endpoints
    that write to the DB (e.g., /run_semantic_chunking) from multiple
    API workers will lead to a TransactionException.
    """
    
    def __init__(self, db_path: str = "artifacts/mdc_challenge.db"):
        """
        Initialize the DuckDB helper.
        
        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = db_path
        self.engine = self._create_engine()
        self._initialize_schema()
    
    def _create_engine(self) -> duckdb.DuckDBPyConnection:
        """Create DuckDB connection and engine."""
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(db_path))
    
    def _initialize_schema(self):
        """Initialize database schema if it doesn't exist."""
        # Create documents table
        self.engine.execute("""
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
            )
        """)
        
        # Create citation_entities table
        self.engine.execute("""
            CREATE TABLE IF NOT EXISTS citations (
                data_citation VARCHAR NOT NULL,
                document_id VARCHAR NOT NULL,
                pages INTEGER[],
                evidence VARCHAR[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (data_citation, document_id),
                FOREIGN KEY (document_id) REFERENCES documents(doi)
            )
        """)
        
        # Create chunks table
        self.engine.execute("""
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
            )
        """)
        
        # Create indexes for better performance
        self.engine.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
            ON chunks(document_id)
        """)
        
        self.engine.execute("""
            CREATE INDEX IF NOT EXISTS idx_citations_data_citation 
            ON citations(data_citation)
        """)
        
        self.engine.execute("""
            CREATE INDEX IF NOT EXISTS idx_citations_document_id 
            ON citations(document_id)
        """)
        
        logger.info("Database schema initialized successfully")
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents from the database."""
        try:
            result = self.engine.execute("SELECT * FROM documents")
            rows = result.fetchall()
            documents = []
            
            for row in rows:
                row_dict = dict(zip([desc[0] for desc in result.description], row))
                documents.append(Document.from_duckdb_row(row_dict))
            
            logger.info(f"Retrieved {len(documents)} documents from database")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}")
            raise ValueError(f"Database query failed: {str(e)}")
    
    def get_documents_by_query(self, query: str, limit: int = 100) -> List[Document]:
        """
        Get documents using a custom SQL query.
        
        Args:
            query: SQL query to execute
            limit: Maximum number of results to return
            
        Returns:
            List of Document objects
        """
        try:
            result = self.engine.execute(f"{query} LIMIT {limit}")
            rows = result.fetchall()
            documents = []
            
            for row in rows:
                row_dict = dict(zip([desc[0] for desc in result.description], row))
                documents.append(Document.from_duckdb_row(row_dict))
            
            logger.info(f"Retrieved {len(documents)} documents from custom query")
            return documents
            
        except Exception as e:
            logger.error(f"Custom query failed: {str(e)}")
            raise ValueError(f"Database query failed: {str(e)}")
    
    def get_citation_entities_by_doc_id(self, doc_id: str) -> List[CitationEntity]:
        """
        Get citation entities for a specific document.
        
        Args:
            doc_id: Document ID (DOI)
            
        Returns:
            List of CitationEntity objects
        """
        try:
            result = self.engine.execute(
                "SELECT * FROM citations WHERE document_id = ?",
                [doc_id]
            )
            rows = result.fetchall()
            entities = []
            
            for row in rows:
                row_dict = dict(zip([desc[0] for desc in result.description], row))
                entities.append(CitationEntity.from_duckdb_row(row_dict))
            
            logger.info(f"Retrieved {len(entities)} citation entities for document {doc_id}")
            return entities
            
        except Exception as e:
            logger.error(f"Failed to retrieve citation entities for {doc_id}: {str(e)}")
            raise ValueError(f"Database query failed: {str(e)}")
    
    def get_all_citation_entities(self) -> List[CitationEntity]:
        """Get all citation entities from the database."""
        try:
            result = self.engine.execute("SELECT * FROM citations")
            rows = result.fetchall()
            entities = []
            
            for row in rows:
                row_dict = dict(zip([desc[0] for desc in result.description], row))
                entities.append(CitationEntity.from_duckdb_row(row_dict))
            
            logger.info(f"Retrieved {len(entities)} citation entities from database")
            return entities
            
        except Exception as e:
            logger.error(f"Failed to retrieve citation entities: {str(e)}")
            raise ValueError(f"Database query failed: {str(e)}")
    
    def store_document(self, document: Document) -> bool:
        """
        Store a single document in the database.
        
        Args:
            document: Document object to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            doc_row = document.to_duckdb_row()
            self.engine.execute("""
                INSERT OR REPLACE INTO documents 
                (doi, has_dataset_citation, full_text, total_char_length, 
                 parsed_timestamp, total_chunks, total_tokens, avg_tokens_per_chunk,
                 file_hash, file_path, citation_entities, n_pages, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_row["doi"],
                doc_row["has_dataset_citation"],
                doc_row["full_text"],
                doc_row["total_char_length"],
                doc_row["parsed_timestamp"],
                doc_row["total_chunks"],
                doc_row["total_tokens"],
                doc_row["avg_tokens_per_chunk"],
                doc_row["file_hash"],
                doc_row["file_path"],
                doc_row["citation_entities"],
                doc_row["n_pages"],
                doc_row["created_at"],
                doc_row["updated_at"]
            ))
            
            logger.info(f"Stored document {document.doi} successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store document {document.doi}: {str(e)}")
            return False
    
    def store_documents(self, documents: List[Document]) -> bool:
        """
        Store multiple documents in the database.
        
        Args:
            documents: List of Document objects to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            for document in documents:
                doc_row = document.to_duckdb_row()
                self.engine.execute("""
                    INSERT OR REPLACE INTO documents 
                    (doi, has_dataset_citation, full_text, total_char_length, 
                     parsed_timestamp, total_chunks, total_tokens, avg_tokens_per_chunk,
                     file_hash, file_path, citation_entities, n_pages, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_row["doi"],
                    doc_row["has_dataset_citation"],
                    doc_row["full_text"],
                    doc_row["total_char_length"],
                    doc_row["parsed_timestamp"],
                    doc_row["total_chunks"],
                    doc_row["total_tokens"],
                    doc_row["avg_tokens_per_chunk"],
                    doc_row["file_hash"],
                    doc_row["file_path"],
                    doc_row["citation_entities"],
                    doc_row["n_pages"],
                    doc_row["created_at"],
                    doc_row["updated_at"]
                ))
            
            logger.info(f"Stored {len(documents)} documents successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store documents: {str(e)}")
            return False
    
    def store_chunks(self, chunks: List[Chunk]) -> bool:
        """
        Store chunks in the database.
        
        Args:
            chunks: List of Chunk objects to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            for chunk in chunks:
                chunk_row = chunk.to_duckdb_row()
                self.engine.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, document_id, chunk_text, score, chunk_metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    chunk_row["chunk_id"],
                    chunk_row["document_id"],
                    chunk_row["chunk_text"],
                    chunk_row["score"],
                    chunk_row["chunk_metadata"],
                ))
            
            logger.info(f"Stored {len(chunks)} chunks successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store chunks: {str(e)}")
            return False
    
    def get_chunks_by_document_id(self, document_id: str) -> List[Chunk]:
        """
        Get chunks for a specific document.
        
        Args:
            document_id: Document ID (DOI)
            
        Returns:
            List of Chunk objects
        """
        try:
            result = self.engine.execute(
                "SELECT * FROM chunks WHERE document_id = ?",
                [document_id]
            )
            rows = result.fetchall()
            chunks = []
            
            for row in rows:
                row_dict = dict(zip([desc[0] for desc in result.description], row))
                chunks.append(Chunk.from_duckdb_row(row_dict))
            
            logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks for {document_id}: {str(e)}")
            raise ValueError(f"Database query failed: {str(e)}")
    
    def get_chunks_by_chunk_ids(
        self,
        chunk_ids: Union[str, Iterable[str]],
    ) -> List[Chunk]:
        """
        Retrieve one or many chunks by their ``chunk_id``.

        Parameters
        ----------
        chunk_ids : str | Iterable[str]
            A single chunk ID or any iterable of IDs.

        Returns
        -------
        List[Chunk]
            The matching ``Chunk`` objects (empty list if none found).
        """
        # Normalise parameter(s) to a list
        if isinstance(chunk_ids, str):
            chunk_ids = [chunk_ids]
        chunk_ids = list(chunk_ids)  # ensure subscriptable/len()

        if not chunk_ids:  # graceful empty-input behaviour
            return []

        # Build a fully-parameterised IN (...) clause
        placeholders = ", ".join("?" * len(chunk_ids))
        sql = f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})"

        try:
            result = self.engine.execute(sql, chunk_ids)
            rows = result.fetchall()
            return [
                Chunk.from_duckdb_row(
                    dict(zip([d[0] for d in result.description], row))
                )
                for row in rows
            ]
        except Exception as exc:
            logger.error(
                f"Failed to fetch chunks for IDs {chunk_ids}: {exc}"
            )
            raise ValueError(str(exc)) from exc

    def get_citation_entities_by_keys(
        self,
        keys: Union[
            Tuple[str, str],  # (data_citation, document_id)
            Iterable[Tuple[str, str]],
        ],
    ) -> List[CitationEntity]:
        """
        Retrieve citation-entity rows by **composite primary key**.

        Parameters
        ----------
        keys : (str, str) | Iterable[(str, str)]
            A single key tuple or an iterable of the form
            ``(data_citation, document_id)``.

        Returns
        -------
        List[CitationEntity]
            Matching ``CitationEntity`` objects (empty list if none).
        """
        if isinstance(keys, tuple) and len(keys) == 2:
            keys = [keys]
        keys = list(keys)

        if not keys:
            return []

        # Flatten key tuples -> [dc1, doc1, dc2, doc2, ...]
        flat_params: List[str] = [
            elem for key in keys for elem in key  # type: ignore[misc]
        ]
        placeholders = ", ".join("(?, ?)" for _ in keys)
        sql = (
            "SELECT * FROM citations "
            f"WHERE (data_citation, document_id) IN ({placeholders})"
        )

        try:
            result = self.engine.execute(sql, flat_params)
            rows = result.fetchall()
            return [
                CitationEntity.from_duckdb_row(
                    dict(zip([d[0] for d in result.description], row))
                )
                for row in rows
            ]
        except Exception as exc:
            logger.error(
                f"Failed to fetch citation entities for keys {keys}: {exc}"
            )
            raise ValueError(str(exc)) from exc
    
    def get_database_stats(self) -> dict:
        """Get basic statistics about the database."""
        try:
            # Count documents
            doc_count = self.engine.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            
            # Count citations
            citation_count = self.engine.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
            
            # Count chunks
            chunk_count = self.engine.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            
            # Get documents with citations
            docs_with_citations = self.engine.execute(
                "SELECT COUNT(*) FROM documents WHERE has_dataset_citation = true"
            ).fetchone()[0]
            
            return {
                "total_documents": doc_count,
                "total_citations": citation_count,
                "total_chunks": chunk_count,
                "documents_with_citations": docs_with_citations
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {str(e)}")
            return {
                "total_documents": 0,
                "total_citations": 0,
                "total_chunks": 0,
                "documents_with_citations": 0,
                "error": str(e)
            }
    
    def close(self):
        """Close the database connection."""
        if self.engine:
            self.engine.close()
            logger.info("Database connection closed")

# Convenience functions for standalone use
def get_duckdb_helper(db_path: str = "artifacts/mdc_challenge.db") -> DuckDBHelper:
    """Get a DuckDB helper instance."""
    return DuckDBHelper(db_path)

def test_connection(db_path: str = "artifacts/mdc_challenge.db") -> bool:
    """Test database connection."""
    try:
        helper = DuckDBHelper(db_path)
        stats = helper.get_database_stats()
        helper.close()
        return "error" not in stats
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        return False 