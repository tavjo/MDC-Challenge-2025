# src/preprocess_train.py

"""
Run preprocessing steps to get training data. 
0. EDA on train data files (optional): scripts/run_prechunking_eda.py
1. Get document objects: src/get_document_objects.py
2. Get citation objects: src/get_citation_entities.py
3. Chunk and embed documents: src/run_semantic_chunking.py
4. Build Dataset objects: src/construct_datasets.py
5. Run neighborhood stats (optional): src/run_neighborhood_stats.py
6. Run global UMAP and PCA (optional): src/globals.py
7. Leiden Clustering: src/run_clustering.py
8. Dimensionality reduction: src/dimensionality_reduction.py
9. Save training data
"""

import os, sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Project root: {project_root}")
sys.path.append(project_root)

from typing import List, Optional, Tuple
from pathlib import Path

from src.helpers import timer_wrap, initialize_logging
from api.utils.duckdb_utils import DuckDBHelper

# Pipeline steps
from src.get_document_objects import get_document_objects
from src.get_citation_entities import CitationEntityExtractor
from src.run_semantic_chunking import SemanticChunkingPipeline
from src.construct_datasets import DatasetConstructionPipeline
from src.run_clustering import ClusteringPipeline
from src.dimensionality_reduction import Reducer as DimReducer
from src.globals import GlobalFeatures


filename = os.path.basename(__file__)
logger = initialize_logging(filename)

DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"
DEFAULT_CHROMA_CONFIG = "configs/chunking.yaml"
DOC_CHUNKS_COLLECTION = "mdc_training_data"
DATASET_AGGS_COLLECTION = "dataset-aggregates-train"


class PreprocessTrainingData:
    def __init__(
        self,
        pdf_paths: List[str],
        db_path: str = DEFAULT_DUCKDB_PATH,
        cfg_path: str = DEFAULT_CHROMA_CONFIG,
        doc_collection_name: str = DOC_CHUNKS_COLLECTION,
        dataset_collection_name: str = DATASET_AGGS_COLLECTION,
        output_dir: Optional[str] = None,
        subset: bool = False,
        subset_size: Optional[int] = None,
        force_docs: bool = False,
        force_citations: bool = False,
        force_chunking: bool = False,
        force_datasets: bool = False,
        force_clustering: bool = False,
        force_dimred: bool = False,
        force_globals: bool = False,
    ) -> None:
        self.pdf_paths = pdf_paths
        self.db_path = db_path
        self.cfg_path = cfg_path
        self.doc_collection_name = doc_collection_name
        self.dataset_collection_name = dataset_collection_name
        self.subset = subset
        self.subset_size = subset_size
        self.force_docs = force_docs
        self.force_citations = force_citations
        self.force_chunking = force_chunking
        self.force_datasets = force_datasets
        self.force_clustering = force_clustering
        self.force_dimred = force_dimred
        self.force_globals = force_globals

        self.output_dir = output_dir or os.path.join(project_root, "Data", "train")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _table_has_rows(self, table_name: str) -> bool:
        helper = DuckDBHelper(self.db_path)
        try:
            counts = helper.get_counts(table_name)
            n = (counts or {}).get(table_name, 0)
            logger.info(f"Table `{table_name}` row count: {n}")
            return (n or 0) > 0
        finally:
            helper.close()

    # def _list_pdf_paths(self, pdf_dir: Optional[str] = None) -> List[str]:
    #     base_dir = pdf_dir or os.path.join(project_root, "Data", "train", "PDF")
    #     if not os.path.isdir(base_dir):
    #         logger.error(f"PDF directory not found: {base_dir}")
    #         return []
    #     pdf_paths = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.lower().endswith(".pdf")]
    #     logger.info(f"Found {len(pdf_paths)} PDFs under {base_dir}")
    #     return pdf_paths

    @timer_wrap
    def run_documents(self, max_workers: int = 8) -> bool:
        if not self.force_docs and self._table_has_rows("documents"):
            logger.info("Skipping document parsing (reuse enabled and documents exist)")
            return True

        if not self.pdf_paths:
            logger.error("No PDF files found; cannot build document objects")
            return False
        pdf_paths = self.pdf_paths

        try:
            response = get_document_objects(
                pdf_paths=pdf_paths,
                subset=self.subset,
                subset_size=self.subset_size or 20,
                export_file=None,
                export_path=self.output_dir,
                max_workers=max_workers,
            )
            logger.info(f"Document API response: {response}")
            return True
        except Exception as exc:
            logger.error(f"Document parsing failed: {exc}")
            return False

    @timer_wrap
    def run_citations(
        self,
        data_dir: str = "Data",
        known_entities: bool = True,
        labels_file: Optional[str] = "train_labels.csv",
    ) -> bool:
        if not self.force_citations and self._table_has_rows("citations"):
            logger.info("Skipping citation extraction (reuse enabled and citations exist)")
            return True
        try:
            extractor = CitationEntityExtractor(
                data_dir=data_dir,
                known_entities=known_entities,
                labels_file=labels_file,
                draw_subset=self.subset,
                subset_size=self.subset_size,
                db_path=self.db_path,
            )
            extractor.extract_entities()
            extractor.save_entities()
            return True
        except Exception as exc:
            logger.error(f"Citation extraction failed: {exc}")
            return False

    @timer_wrap
    def run_chunk_and_embed(self, max_workers: Optional[int] = None) -> bool:
        if not self.force_chunking and self._table_has_rows("chunks"):
            logger.info("Skipping chunking+embedding (reuse enabled and chunks exist)")
            return True

        try:
            cpu_count = os.cpu_count() or 1
            default_workers = max(1, (cpu_count // 2))
            workers = max_workers or default_workers
            logger.info(f"Using {workers} workers for chunking")

            pipeline = SemanticChunkingPipeline(
                subset=self.subset,
                subset_size=self.subset_size,
                cfg_path=self.cfg_path,
                db_path=self.db_path,
                collection_name=self.doc_collection_name,
                max_workers=workers,
            )
            results = pipeline.run_pipeline()
            ok = bool(results and getattr(results, "success", False))
            if not ok and isinstance(results, dict):
                ok = results.get("success", False)
            logger.info(f"Chunking pipeline success: {ok}")
            return ok
        except Exception as exc:
            logger.error(f"Chunking+embedding failed: {exc}")
            return False

    @timer_wrap
    def run_construct_datasets(self, local_model: bool = True) -> bool:
        if not self.force_datasets and self._table_has_rows("datasets"):
            logger.info("Skipping dataset construction (reuse enabled and datasets exist)")
            return True
        try:
            pipeline = DatasetConstructionPipeline(
                db_path=self.db_path,
                collection_name=self.dataset_collection_name,
                cfg_path=self.cfg_path,
                local_model=local_model,
            )
            results = pipeline.run_pipeline()
            ok = bool(results and results.get("overall_success", False))
            logger.info(f"Dataset construction pipeline success: {ok}")
            return ok
        except Exception as exc:
            logger.error(f"Dataset construction failed: {exc}")
            return False

    @timer_wrap
    def run_clustering(self) -> bool:
        try:
            pipeline = ClusteringPipeline(
                collection_name=self.dataset_collection_name,
                cfg_path=self.cfg_path,
                db_path=self.db_path,
                target_n=70,
            )
            results = pipeline.run_pipeline()
            ok = bool(results and results.get("overall_success", False))
            logger.info(f"Clustering pipeline success: {ok}")
            return ok
        except Exception as exc:
            logger.error(f"Clustering failed: {exc}")
            return False

    @timer_wrap
    def run_dimensionality_reduction(self) -> bool:
        if not self.force_dimred and self._table_has_rows("engineered_feature_values"):
            logger.info("Skipping dimensionality reduction (reuse enabled and engineered features exist)")
            return True
        try:
            reducer = DimReducer(
                collection_name=self.dataset_collection_name,
                cfg_path=self.cfg_path,
                db_path=self.db_path,
            )
            results = reducer.run_pipeline()
            ok = bool(results and results.get("overall_success", False))
            logger.info(f"Dimensionality reduction pipeline success: {ok}")
            return ok
        except Exception as exc:
            logger.error(f"Dimensionality reduction failed: {exc}")
            return False

    # -----------------------------
    # Optional: Global UMAP and PCA (sample-level)
    # -----------------------------
    @timer_wrap
    def run_global_umap_pca(self) -> bool:
        try:
            gf = GlobalFeatures(
                collection_name=self.dataset_collection_name,
                cfg_path=self.cfg_path,
                db_path=self.db_path,
            )
            # Use globals convenience to run both
            results = gf.run_sample_umap_and_pca()
            ok = results is not None
            logger.info(f"Global UMAP/PCA success: {ok}")
            return ok
        except Exception as exc:
            logger.error(f"Global UMAP/PCA step failed: {exc}")
            return False

    @timer_wrap
    def build_and_export_training_data(
        self,
        output_filename: Optional[str] = None,
        format: str = "csv",
        drop_zero_variance: bool = True,
        near_zero_threshold: float = 1e-10,
    ) -> Tuple[bool, Optional[str]]:
        try:
            helper = DuckDBHelper(self.db_path)
            df_full = helper.get_full_dataset_dataframe()
            helper.close()

            if df_full is None or df_full.empty:
                logger.error("No dataset/features available to build training data")
                return False, None

            if "dataset_type" in df_full.columns:
                df_full["target"] = (df_full["dataset_type"].astype(str).str.upper() == "PRIMARY").astype(int)
            else:
                logger.warning("`dataset_type` column missing; defaulting target=0")
                df_full["target"] = 0

            engineered_cols = [c for c in df_full.columns if c.startswith("LEIDEN_") or c.startswith("UMAP_")]
            base_numeric = [c for c in [
                "total_tokens",
                "avg_tokens_per_chunk",
                "total_char_length",
                "clean_text_length",
            ] if c in df_full.columns]
            feature_cols = list(dict.fromkeys(base_numeric + engineered_cols))

            if not feature_cols:
                logger.error("No feature columns found to export")
                return False, None

            if drop_zero_variance:
                feature_variances = df_full[feature_cols].var(numeric_only=True)
                zero_var_features = feature_variances[feature_variances == 0].index.tolist()
                near_zero_var_features = feature_variances[feature_variances < near_zero_threshold].index.tolist()
                logger.info(
                    f"Variance analysis — total: {len(feature_cols)}, zero: {len(zero_var_features)}, near-zero(<{near_zero_threshold}): {len(near_zero_var_features)}"
                )
                if zero_var_features:
                    logger.info(f"Zero-variance features to remove: {zero_var_features}")
                    feature_cols = [c for c in feature_cols if c not in zero_var_features]
                if near_zero_var_features and len(near_zero_var_features) > len(zero_var_features):
                    logger.info(f"Near-zero variance features: {[c for c in near_zero_var_features if c not in zero_var_features]}")

            keep_cols = [c for c in (feature_cols + ["target", "document_id"]) if c in df_full.columns]
            df_out = df_full[keep_cols].copy()

            if output_filename is None:
                output_filename = os.path.join(self.output_dir, f"train_data.{ 'parquet' if format.lower()=='parquet' else 'csv'}")

            out_path = Path(output_filename)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == "parquet":
                try:
                    df_out.to_parquet(out_path, index=False)
                except Exception as exc:
                    logger.error(f"Parquet export failed ({exc}); falling back to CSV")
                    out_path = out_path.with_suffix('.csv')
                    df_out.to_csv(out_path, index=False)
            else:
                df_out.to_csv(out_path, index=False)

            logger.info(f"✅ Training data saved to {out_path}")
            return True, str(out_path)
        except Exception as exc:
            logger.error(f"Failed to build/export training data: {exc}")
            return False, None

    @timer_wrap
    def run_all(self, export_format: str = "csv") -> Tuple[bool, Optional[str]]:
        steps = [
            ("documents", self.run_documents),
            ("citations", self.run_citations),
            ("chunking", self.run_chunk_and_embed),
            ("datasets", self.run_construct_datasets),
            ("clustering", self.run_clustering),
            ("dimensionality_reduction", self.run_dimensionality_reduction),
            ("globals", self.run_global_umap_pca),
        ]
        for name, fn in steps:
            ok = fn()
            if not ok:
                logger.error(f"Stopping pipeline — step `{name}` failed")
                return False, None
        return self.build_and_export_training_data(format=export_format)

@timer_wrap
def main():
    pdf_paths = os.listdir(os.path.join(project_root, "Data/train/PDF"))
    pdf_paths = [os.path.join("Data/train/PDF", pdf) for pdf in pdf_paths if pdf.endswith(".pdf")]
    processor = PreprocessTrainingData(pdf_paths=pdf_paths, force_citations=True)
    success, path = processor.run_all(export_format="csv")
    if success:
        logger.info(f"Training data saved to {path}")
    else:
        logger.error("Failed to build training data")
        raise SystemExit(1)

if __name__ == "__main__":
    main()


