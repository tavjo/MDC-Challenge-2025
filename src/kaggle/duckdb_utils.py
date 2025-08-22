# DuckDB Helpers
"""
Siloed DuckDB helper for Kaggle (class-based).

Reuses the schema shape from `api/database/duckdb_schema.py` and exposes
minimal methods used by the Kaggle pipeline. Embeddings are NOT persisted.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterable

import duckdb
from pathlib import Path
import sys, os

# Allow importing sibling kaggle helpers/models when used as a standalone script
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

# Local models
try:
    from src.kaggle.models import Document, Chunk, Dataset, EngineeredFeatures, CitationEntity
except Exception:
    from .models import Document, Chunk, Dataset, EngineeredFeatures, CitationEntity  # type: ignore

# temporary directories
base_tmp = "/kaggle/temp/"

artifacts = os.path.join(base_tmp, "artifacts")

DEFAULT_DUCKDB = os.path.join(artifacts, "mdc_challenge.db")

class KaggleDuckDBHelper:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or DEFAULT_DUCKDB
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine: duckdb.DuckDBPyConnection = duckdb.connect(str(path))
        self.init_schema()

    # --------------------------- Schema ---------------------------
    def init_schema(self) -> None:
        e = self.engine
        e.execute(
            """
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
        )

        e.execute(
            """
            CREATE TABLE IF NOT EXISTS citations (
                data_citation VARCHAR NOT NULL,
                document_id   VARCHAR NOT NULL,
                pages         INTEGER[],
                evidence      VARCHAR[],
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (data_citation, document_id)
            );
            """
        )

        e.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id   VARCHAR PRIMARY KEY,
                document_id VARCHAR NOT NULL,
                chunk_text  TEXT NOT NULL,
                score       REAL,
                chunk_metadata STRUCT(
                    created_at TIMESTAMP,
                    previous_chunk_id VARCHAR,
                    next_chunk_id VARCHAR,
                    token_count INTEGER,
                    citation_entities VARCHAR[]
                ),
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        e.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id      VARCHAR PRIMARY KEY,
                document_id     VARCHAR NOT NULL,
                total_tokens    INTEGER,
                avg_tokens_per_chunk REAL,
                total_char_length   INTEGER,
                clean_text_length   INTEGER,
                cluster         VARCHAR,
                dataset_type    VARCHAR,
                text            TEXT NOT NULL,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        e.execute(
            """
            CREATE TABLE IF NOT EXISTS engineered_feature_values (
                dataset_id    VARCHAR NOT NULL,
                document_id   VARCHAR NOT NULL,
                feature_name  VARCHAR NOT NULL,
                feature_value DOUBLE,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY   (dataset_id, document_id, feature_name)
            );
            """
        )

        # Indexes
        e.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);")
        e.execute("CREATE INDEX IF NOT EXISTS idx_citations_data_citation ON citations(data_citation);")
        e.execute("CREATE INDEX IF NOT EXISTS idx_citations_document_id ON citations(document_id);")
        e.execute("CREATE INDEX IF NOT EXISTS idx_datasets_document_id ON datasets(document_id);")
        e.execute("CREATE INDEX IF NOT EXISTS idx_datasets_cluster ON datasets(cluster);")
        e.execute("CREATE INDEX IF NOT EXISTS idx_feature_name ON engineered_feature_values(feature_name);")

    # ------------------------- Documents -------------------------
    def upsert_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        import pandas as pd
        df = pd.DataFrame.from_records([d.to_duckdb_row() for d in documents])
        e = self.engine
        e.execute("BEGIN TRANSACTION")
        try:
            e.register("documents_buffer", df)
            e.execute(
                """
                INSERT OR REPLACE INTO documents 
                (doi, has_dataset_citation, full_text, total_char_length, 
                 parsed_timestamp, total_chunks, total_tokens, avg_tokens_per_chunk,
                 file_hash, file_path, citation_entities, n_pages, created_at, updated_at)
                SELECT doi, has_dataset_citation, full_text, total_char_length, 
                       parsed_timestamp, total_chunks, total_tokens, avg_tokens_per_chunk,
                       file_hash, file_path, citation_entities, n_pages, created_at, updated_at
                FROM documents_buffer
                """
            )
            e.unregister("documents_buffer")
            e.execute("COMMIT")
        except Exception:
            e.execute("ROLLBACK")
            raise

    def get_all_documents(self) -> List[Document]:
        res = self.engine.execute("SELECT * FROM documents")
        rows = res.fetchall()
        cols = [d[0] for d in res.description]
        return [Document.from_duckdb_row(dict(zip(cols, r))) for r in rows]

    # --------------------------- Chunks --------------------------
    def bulk_insert_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        import pandas as pd
        df = pd.DataFrame.from_records([c.to_duckdb_row() for c in chunks])
        e = self.engine
        e.register("chunks_buffer", df)
        try:
            e.execute(
                """
                INSERT OR REPLACE INTO chunks 
                (chunk_id, document_id, chunk_text, score, chunk_metadata)
                SELECT chunk_id, document_id, chunk_text, score, chunk_metadata
                FROM chunks_buffer
                """
            )
        finally:
            e.unregister("chunks_buffer")

    def get_chunks_by_document(self, document_id: str) -> List[Chunk]:
        res = self.engine.execute("SELECT * FROM chunks WHERE document_id = ?", [document_id])
        rows = res.fetchall()
        cols = [d[0] for d in res.description]
        return [Chunk.from_duckdb_row(dict(zip(cols, r))) for r in rows]
    
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
            
            # logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            # logger.error(f"Failed to retrieve chunks for {document_id}: {str(e)}")
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
            # logger.error(
                # f"Failed to fetch chunks for IDs {chunk_ids}: {exc}"
            # )
            raise ValueError(str(exc)) from exc    

    # --------------------------- Datasets ------------------------
    def bulk_upsert_datasets(self, datasets: List[Dataset]) -> None:
        if not datasets:
            return
        import pandas as pd
        df = pd.DataFrame.from_records([d.to_duckdb_row() for d in datasets])
        e = self.engine
        e.execute("BEGIN TRANSACTION")
        try:
            e.register("datasets_buffer", df)
            e.execute(
                """
                INSERT OR REPLACE INTO datasets
                (dataset_id, document_id, total_tokens, avg_tokens_per_chunk,
                 total_char_length, clean_text_length, cluster,
                 dataset_type, text, created_at, updated_at)
                SELECT dataset_id, document_id, total_tokens, avg_tokens_per_chunk,
                       total_char_length, clean_text_length, cluster,
                       dataset_type, text, created_at, updated_at
                FROM datasets_buffer
                """
            )
            e.unregister("datasets_buffer")
            e.execute("COMMIT")
        except Exception:
            e.execute("ROLLBACK")
            raise

    # --------------------- Engineered Features -------------------
    def insert_engineered_features(self, features: EngineeredFeatures) -> None:
        rows = features.to_eav_rows()
        if not rows:
            return
        import pandas as pd
        df = pd.DataFrame.from_records(rows)
        e = self.engine
        e.register("feat_buffer", df)
        try:
            e.execute(
                """
                INSERT OR REPLACE INTO engineered_feature_values
                (dataset_id, document_id, feature_name, feature_value, created_at, updated_at)
                SELECT dataset_id, document_id, feature_name, feature_value, created_at, updated_at
                FROM feat_buffer
                """
            )
        finally:
            e.unregister("feat_buffer")

    # --------------------------- Citations -----------------------
    def get_all_citation_entities(self) -> List[CitationEntity]:
        res = self.engine.execute("SELECT * FROM citations")
        rows = res.fetchall()
        cols = [d[0] for d in res.description]
        return [CitationEntity.from_duckdb_row(dict(zip(cols, r))) for r in rows]

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
    

    def store_citations_batch(self, citation_entities: List[Any]) -> None:
        if not citation_entities:
            return
        import pandas as pd
        records: List[Dict[str, Any]] = []
        for ce in citation_entities:
            if hasattr(ce, "to_duckdb_row"):
                row = ce.to_duckdb_row()
            else:
                row = {
                    "data_citation": ce["data_citation"],
                    "document_id": ce["document_id"],
                    "pages": ce.get("pages"),
                    "evidence": ce.get("evidence"),
                }
            records.append(row)
        df = pd.DataFrame.from_records(records)
        e = self.engine
        e.register("citations_buffer", df)
        try:
            e.execute(
                """
                INSERT OR REPLACE INTO citations 
                (data_citation, document_id, pages, evidence, created_at)
                SELECT data_citation, document_id, pages, evidence, CURRENT_TIMESTAMP
                FROM citations_buffer
                """
            )
        finally:
            e.unregister("citations_buffer")

    # --------------------- Feature Assembly ----------------------
    def get_full_dataset_dataframe(
        self,
        dataset_ids: Optional[List[str]] = None,
        feature_filter: Optional[List[str]] = None,
        fill_na: bool = True,
    ):
        import pandas as pd
        # Datasets
        if dataset_ids:
            ph = ", ".join("?" * len(dataset_ids))
            df_ds = self.engine.execute(f"SELECT * FROM datasets WHERE dataset_id IN ({ph})", dataset_ids).df()
        else:
            df_ds = self.engine.execute("SELECT * FROM datasets").df()

        # Features
        if feature_filter:
            ph = ", ".join("?" * len(feature_filter))
            df_feat = self.engine.execute(
                f"SELECT * FROM engineered_feature_values WHERE feature_name IN ({ph})",
                feature_filter,
            ).df()
        else:
            df_feat = self.engine.execute("SELECT * FROM engineered_feature_values").df()

        if df_feat.empty:
            return df_ds

        df_pivot = df_feat.pivot_table(
            index=["dataset_id", "document_id"],
            columns="feature_name",
            values="feature_value",
            aggfunc="first",
        ).reset_index()
        merged = df_ds.merge(df_pivot, on=["dataset_id", "document_id"], how="left")
        if fill_na:
            merged = merged.fillna(value=float("nan"))
        return merged

    # ---------------------------- Close --------------------------
    def close(self) -> None:
        try:
            self.engine.close()
        except Exception:
            pass


def get_duckdb_helper(db_path: str = DEFAULT_DUCKDB) -> KaggleDuckDBHelper:
    return KaggleDuckDBHelper(db_path)


__all__ = [
    "KaggleDuckDBHelper",
    "get_duckdb_helper",
]