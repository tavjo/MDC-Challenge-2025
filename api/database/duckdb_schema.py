"""
DuckDB Schema Initialization Script – v2
Adds an EAV table for engineered features.
"""

import duckdb
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DuckDBSchemaInitializer:
    def __init__(self, db_path: str = "artifacts/mdc_challenge.db"):
        self.db_path = db_path
        self.conn = None

    # ───────────────────────────────────────────
    # Connection helper
    # ───────────────────────────────────────────
    def create_connection(self) -> duckdb.DuckDBPyConnection:
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(db_path))
        return self.conn

    # ───────────────────────────────────────────
    # Core tables (unchanged)
    # ───────────────────────────────────────────
    def create_documents_table(self):
        logger.info("Creating documents table...")
        self.conn.execute(
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
        logger.info("Documents table created successfully")

    def create_citations_table(self):
        logger.info("Creating citations table...")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS citations (
                data_citation VARCHAR NOT NULL,
                document_id   VARCHAR NOT NULL,
                pages         INTEGER[],
                evidence      VARCHAR[],
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (data_citation, document_id),
                FOREIGN KEY (document_id) REFERENCES documents(doi)
            );
            """
        )
        logger.info("Citations table created successfully")

    def create_chunks_table(self):
        logger.info("Creating chunks table...")
        self.conn.execute(
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
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(doi)
            );
            """
        )
        logger.info("Chunks table created successfully")

    def create_datasets_table(self):
        logger.info("Creating datasets table...")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id      VARCHAR PRIMARY KEY,
                document_id     VARCHAR NOT NULL,
                total_tokens    INTEGER,
                avg_tokens_per_chunk REAL,
                total_char_length   INTEGER,
                clean_text_length   INTEGER,
                cluster         VARCHAR,
                dataset_type    VARCHAR,        -- 'PRIMARY' | 'SECONDARY'
                text            TEXT NOT NULL,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(doi)
            );
            """
        )
        logger.info("Datasets table created successfully")

    # ───────────────────────────────────────────
    # NEW – engineered_feature_values (EAV table)
    # ───────────────────────────────────────────
    def create_engineered_feature_values_table(self):
        logger.info("Creating engineered_feature_values table...")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS engineered_feature_values (
                dataset_id    VARCHAR NOT NULL,
                document_id   VARCHAR NOT NULL REFERENCES documents(doi),
                feature_name  VARCHAR NOT NULL,
                feature_value DOUBLE,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY   (dataset_id, document_id, feature_name)
            );
            """
        )
        logger.info("engineered_feature_values table created successfully")

    # ───────────────────────────────────────────
    # Indexes (added one for feature_name)
    # ───────────────────────────────────────────
    def create_indexes(self):
        logger.info("Creating indexes...")

        # chunks
        self.conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_chunks_document_id
               ON chunks(document_id);"""
        )

        # citations
        self.conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_citations_data_citation
               ON citations(data_citation);"""
        )
        self.conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_citations_document_id
               ON citations(document_id);"""
        )

        # datasets
        self.conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_datasets_document_id
               ON datasets(document_id);"""
        )
        self.conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_datasets_cluster
               ON datasets(cluster);"""
        )

        # engineered_feature_values
        self.conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_feature_name
               ON engineered_feature_values(feature_name);"""
        )

        logger.info("Indexes created successfully")

    # ───────────────────────────────────────────
    # Orchestration
    # ───────────────────────────────────────────
    def create_schema(self):
        logger.info("Starting DuckDB schema creation...")
        try:
            if not self.conn:
                self.create_connection()

            # create tables
            self.create_documents_table()
            self.create_citations_table()
            self.create_chunks_table()
            self.create_datasets_table()
            self.create_engineered_feature_values_table()  # <── NEW

            # indexes
            self.create_indexes()

            logger.info("DuckDB schema creation completed successfully")
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            raise

    def validate_schema(self):
        logger.info("Validating schema...")

        expected_tables = [
            "documents",
            "citations",
            "chunks",
            "datasets",
            "engineered_feature_values",  # <── NEW
        ]

        tables = {
            t[0] for t in self.conn.execute("SHOW TABLES").fetchall()
        }  # SHOW TABLES is DuckDB-native:contentReference[oaicite:0]{index=0}

        missing = set(expected_tables) - tables
        if missing:
            raise ValueError(f"Missing tables: {', '.join(missing)}")

        for tbl in expected_tables:
            cols = self.conn.execute(f"DESCRIBE {tbl}").fetchall()  # DESCRIBE shows schema:contentReference[oaicite:1]{index=1}
            logger.info(f"Table {tbl} has {len(cols)} columns")

        logger.info("Schema validation completed successfully")

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def main():
    schema = DuckDBSchemaInitializer()
    try:
        schema.create_schema()
        schema.validate_schema()
        logger.info("DuckDB schema initialization completed successfully")
    finally:
        schema.close()


if __name__ == "__main__":
    main()
