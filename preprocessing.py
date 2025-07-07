# preprocessing.py

"""
This script is used to run all preprocessing steps.
It will contain a class that will run preprocessing steps in sequence.
These steps will be:
1) Pre-Chunking EDA (scripts/run_prechunking_eda.py)
2) PDF -> XML conversion (scripts/run_doc_conversion.py)
3) Document Parsing (scripts/run_full_doc_parsing.py)
4) Semantic Chunking (scripts/run_chunking_pipeline.py)
5) Create Vector Embeddings 
6) Chunk-level EDA
7) QC
8) Export Artifacts for training loop
9) Generate Report
"""

import sys, os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from src.helpers import initialize_logging, timer_wrap

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

@timer_wrap
class PreprocessingPipeline:
    """
    Run preprocessing pipeline to generate training loop artifacts.
    """
    logger.info("Initializing PreprocessingPipeline...")
    def __init__(self, data_dir: str = "Data"):
        self.data_dir = data_dir
        self.pre_chunking_eda_report = None
        self.doc_conversion_report = None
        self.document_parsing_report = None
        self.semantic_chunking_report = None
        self.vector_embeddings_report = None
        self.chunk_level_eda_report = None
        self.qc_report = None
        self.export_artifacts_report = None
        self.preprocesssing_report = None

    def _generate_report(self):
        """
        Generate a report of the preprocessing pipeline.
        This report will be used to track the progress of the pipeline.
        It will be saved to a file called "preprocessing_report.md".
        """
        pass

    def pre_chunking_eda(self):
        from scripts.run_prechunking_eda import run_prechunking_eda
        reporter = run_prechunking_eda(
            data_dir="Data",
            output_format="all",
            save_reports=True,
            show_plots=False
        )
        self.pre_chunking_eda_report = reporter.generate_json_summary()
        return self.pre_chunking_eda_report

    def doc_conversion(self):
        from scripts.run_pdf_to_xml_conversion import run_pdf_to_xml_conversion
        reporter = run_pdf_to_xml_conversion(
            data_dir="Data",
            output_format="all",
            save_reports=True
        )
        # Generate custom reports
        self.doc_conversion_report = reporter.generate_json_summary()
        return self.doc_conversion_report

    def document_parsing(self):
        pass

    def semantic_chunking(self):
        pass

    def create_vector_embeddings(self):
        pass

    def chunk_level_eda(self):
        pass

    def qc(self):
        pass

    def export_artifacts(self):
        pass

    def run_all(self):
        self.pre_chunking_eda()
        self.doc_conversion()
        self.document_parsing()
        self.semantic_chunking()
        self.create_vector_embeddings()
        self.chunk_level_eda()
        self.qc()
        self.export_artifacts()
        self._generate_report()