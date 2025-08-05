"""
DuckDB utility functions for the MDC Challenge 2025 API

This module provides helper functions for database operations, following the 
specification from chunk_and_embedding_api.md.
"""

import sys, os
from pathlib import Path
from typing import List, Iterable, Union, Tuple, Dict, Any, Optional

import duckdb
import pandas as pd  # for bulk DataFrame-based upserts
from datetime import datetime, timezone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models import Document, CitationEntity, Chunk, Dataset, EngineeredFeatures
from src.helpers import initialize_logging
from api.database.duckdb_schema import DuckDBSchemaInitializer

# Initialize logging
filename = os.path.basename(__file__)
logger = initialize_logging(filename)

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
        schema = DuckDBSchemaInitializer()
        try:
            schema.create_schema()
            schema.validate_schema()
            logger.info("DuckDB schema initialization completed successfully")
        finally:
            schema.close()

    
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
            logger.error(f"Failed to store chunks: {e}")
            return False

    def store_chunks_batch(self, chunks: List[Chunk]) -> bool:
        """
        Batch upsert chunks using a single DataFrame-based SQL query.

        This avoids row-by-row INSERTs by registering a pandas DataFrame and
        executing one INSERT OR REPLACE ... SELECT statement.
        """
        try:
            # Convert chunk objects to a DataFrame
            records = [chunk.to_duckdb_row() for chunk in chunks]
            df = pd.DataFrame.from_records(records)
            # Register as a temporary table
            self.engine.register("chunks_buffer", df)
            # Bulk upsert with one SQL command
            self.engine.execute("""
                INSERT OR REPLACE INTO chunks 
                (chunk_id, document_id, chunk_text, score, chunk_metadata)
                SELECT chunk_id, document_id, chunk_text, score, chunk_metadata
                FROM chunks_buffer
            """ )
            logger.info(f"Batch upserted {len(chunks)} chunks via DataFrame.")
            # Clean up the temp view
            self.engine.unregister("chunks_buffer")
            return True
        except Exception as e:
            logger.error(f"Batch upsert failed: {e}")
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
    

    def bulk_upsert_datasets(
        self,
        datasets: List[Dataset]
    ) -> Dict[str, Any]:
        """
        Batch-upsert Dataset objects using a DataFrame buffer — identical pattern
        to `store_chunks_batch`, avoiding parquet/temp-file overhead.
        """
        if not datasets:
            logger.warning("No datasets supplied for upsert.")
            return {"success": True, "total_datasets_upserted": 0}

        # 1️⃣  Use existing connection, enforce FKs, start TX
        # self.engine.execute("PRAGMA foreign_keys=ON")
        self.engine.execute("BEGIN TRANSACTION")

        try:
            # 2️⃣  Convert to DataFrame
            records = [ds.to_duckdb_row() for ds in datasets]
            df = pd.DataFrame.from_records(records)

            # 3️⃣  Sanitize text fields (optional but safer)
            # if "text" in df.columns:
            #     df["text"] = _sanitize_series(df["text"])

            # 4️⃣  Register buffer table & bulk upsert
            self.engine.register("datasets_buffer", df)
            self.engine.execute("""
                INSERT OR REPLACE INTO datasets
                (dataset_id, document_id, total_tokens, avg_tokens_per_chunk,
                total_char_length, clean_text_length, cluster,
                dataset_type, text, created_at, updated_at)
                SELECT dataset_id, document_id, total_tokens, avg_tokens_per_chunk,
                    total_char_length, clean_text_length, cluster,
                    dataset_type, text, created_at, updated_at
                FROM datasets_buffer
            """)
            self.engine.unregister("datasets_buffer")

            self.engine.execute("COMMIT")

            total_in_db = self.engine.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
            logger.info(f"✅ Upserted {len(datasets)} datasets (total now {total_in_db}).")

            return {
                "success": True,
                "total_datasets_upserted": len(datasets),
                "total_datasets_in_db": total_in_db,
                "method": "dataframe_buffer"
            }

        except Exception as e:
            self.engine.execute("ROLLBACK")
            logger.error(f"Bulk upsert failed: {e}")
            raise
    
    def update_datasets(self, datasets: List[Dataset]) -> Dict[str, Any]:
        """
        Upsert datasets into the DuckDB 'datasets' table using INSERT ON CONFLICT.
        Updates existing rows (matched by dataset_id) or inserts new ones.
        Returns summary of operation.
        """
        if not datasets:
            logger.warning("No datasets supplied for upsert.")
            return {"success": True, "total_upserted": 0}

        # Convert Dataset instances to DuckDB-ready rows
        records = [ds.to_duckdb_row() for ds in datasets]
        df = pd.DataFrame.from_records(records)

        # Register in-memory buffer table
        self.engine.register("datasets_buffer", df)

        # Execute upsert via DuckDB ON CONFLICT clause
        self.engine.execute("""
            INSERT INTO datasets (
                dataset_id, document_id, total_tokens, avg_tokens_per_chunk,
                total_char_length, clean_text_length, cluster,
                dataset_type, text, created_at, updated_at
            )
            SELECT
                dataset_id, document_id, total_tokens, avg_tokens_per_chunk,
                total_char_length, clean_text_length, cluster,
                dataset_type, text, created_at, updated_at
            FROM datasets_buffer
            ON CONFLICT (dataset_id) DO UPDATE
            SET
                document_id = EXCLUDED.document_id,
                total_tokens = EXCLUDED.total_tokens,
                avg_tokens_per_chunk = EXCLUDED.avg_tokens_per_chunk,
                total_char_length = EXCLUDED.total_char_length,
                clean_text_length = EXCLUDED.clean_text_length,
                cluster = EXCLUDED.cluster,
                dataset_type = EXCLUDED.dataset_type,
                text = EXCLUDED.text,
                updated_at = EXCLUDED.updated_at;
        """)

        # Unregister buffer
        self.engine.unregister("datasets_buffer")

        # Fetch total rows count after upsert
        total_rows = self.engine.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        logger.info(f"✅ Upserted {len(datasets)} dataset rows (total in DB now: {total_rows})")

        return {
            "success": True,
            "total_upserted": len(datasets),
            "total_in_db": total_rows
        }
    
    # def get_all_datasets(self) -> List[Dataset]:
    #     """Get all datasets from the database."""
    #     try:
    #         result = self.engine.execute("SELECT * FROM datasets")
    #         rows = result.fetchall()
    #         return [Dataset.from_duckdb_row(row) for row in rows]
    #     except Exception as e:
    #         logger.error(f"Failed to get all datasets: {str(e)}")
    #         raise

    def get_all_datasets(self) -> List[Dataset]:
        """Get all datasets from the database."""
        try:
            result = self.engine.execute("SELECT * FROM datasets")
            rows = result.fetchall()
            datasets = []
            
            for row in rows:
                row_dict = dict(zip([desc[0] for desc in result.description], row))
                datasets.append(Dataset.from_duckdb_row(row_dict))
            
            logger.info(f"Retrieved {len(datasets)} datasets from database")
            return datasets
            
        except Exception as e:
            logger.error(f"Failed to retrieve datasets: {str(e)}")
            raise ValueError(f"Database query failed: {str(e)}")
    

    def upsert_engineered_features_batch(
        self,
        features: List[EngineeredFeatures],
    ) -> bool:
        """
        Batch-insert (or replace) rows into the `engineered_feature_values` table.
        Each `EngineeredFeatures` object is converted to its EAV rows
        via the model's `to_eav_rows()` helper.

        Returns
        -------
        bool
            True on success, False on error (see log).
        """
        if not features:
            logger.warning("No EngineeredFeatures supplied for upsert.")
            return True

        try:
            # 1️⃣  Flatten → list[dict]  (many rows per model!)
            eav_rows: List[Dict[str, Any]] = []
            for feat in features:
                # ensure fresh timestamps so OR REPLACE updates `updated_at`
                now_iso = datetime.now(timezone.utc).isoformat()
                for row in feat.to_eav_rows():
                    row["created_at"] = now_iso
                    row["updated_at"] = now_iso
                    eav_rows.append(row)

            # 2️⃣  Register DataFrame buffer (no copy → Arrow zero-copy)
            df = pd.DataFrame.from_records(eav_rows)
            self.engine.register("feat_buffer", df)

            # 3️⃣  One SQL statement = ✨ fast ✨
            self.engine.execute(
                """
                INSERT OR REPLACE INTO engineered_feature_values
                (dataset_id, document_id, feature_name, feature_value,
                created_at, updated_at)
                SELECT dataset_id, document_id, feature_name, feature_value,
                    created_at, updated_at
                FROM feat_buffer
                """
            )
            self.engine.unregister("feat_buffer")

            logger.info(f"✅ Upserted {len(eav_rows)} feature rows.")
            return True

        except Exception as exc:
            logger.error(f"Feature upsert failed: {exc}")
            return False
    
    def get_full_dataset_dataframe(
        self,
        dataset_ids: Optional[List[str]] = None,
        feature_filter: Optional[List[str]] = None,
        fill_na: bool = True,
    ) -> pd.DataFrame:
        """
        Return a wide DataFrame that merges fixed columns from `datasets`
        with *pivoted* engineered features.

        Parameters
        ----------
        dataset_ids : list[str] | None
            If provided, restrict to these dataset_id values.
        feature_filter : list[str] | None
            If provided, keep only these engineered feature names.
        fill_na : bool
            If True, fill missing feature columns with NaN; otherwise keep NaNs.

        Returns
        -------
        pd.DataFrame
            One row per dataset (plus `document_id`) with extra feature columns.
        """

        # 1️⃣  Load base datasets -------------------------------
        if dataset_ids:
            placeholders = ", ".join("?" * len(dataset_ids))
            ds_sql = f"SELECT * FROM datasets WHERE dataset_id IN ({placeholders})"
            df_ds = self.engine.execute(ds_sql, dataset_ids).df()
        else:
            df_ds = self.engine.execute("SELECT * FROM datasets").df()

        # 2️⃣  Load engineered features -------------------------
        if feature_filter:
            ph = ", ".join("?" * len(feature_filter))
            feat_sql = (
                "SELECT * FROM engineered_feature_values "
                f"WHERE feature_name IN ({ph})"
            )
            df_feat = self.engine.execute(feat_sql, feature_filter).df()
        else:
            df_feat = self.engine.execute(
                "SELECT * FROM engineered_feature_values"
            ).df()

        if df_feat.empty:
            logger.warning("No engineered features found; returning datasets only.")
            return df_ds

        # 3️⃣  Pivot long → wide  (Pandas = simpler than SQL PIVOT for unknown cols) --
        df_wide = (
            df_feat.pivot_table(
                index=["dataset_id", "document_id"],
                columns="feature_name",
                values="feature_value",
                aggfunc="first",
            )
            .reset_index()
        )  # same semantics as DuckDB PIVOT_WIDER:contentReference[oaicite:2]{index=2}

        if fill_na:
            df_wide = df_wide.fillna(pd.NA)

        # 4️⃣  Merge with datasets ------------------------------
        df_full = (
            df_ds.merge(
                df_wide,
                how="left",
                left_on=["dataset_id", "document_id"],
                right_on=["dataset_id", "document_id"],
                suffixes=("", "_drop"),
            )
            .drop(columns=[c for c in df_ds.columns if c.endswith("_drop")], errors="ignore")
        )

        logger.info(f"🔄 Built dataframe with shape {df_full.shape}.")
        return df_full
            
    
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

            # get datasets
            dataset_count = self.engine.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
            
            return {
                "total_documents": doc_count,
                "total_citations": citation_count,
                "total_chunks": chunk_count,
                "documents_with_citations": docs_with_citations,
                "total_datasets": dataset_count
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