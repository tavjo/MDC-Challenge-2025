# src/get_document_objects.py

import os, sys, warnings
from pathlib import Path
from typing import List, Optional
import numpy as np
import requests

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Project root: {project_root}")
sys.path.append(project_root)

from src.helpers import timer_wrap, initialize_logging
from api.utils.duckdb_utils import DuckDBHelper
from src.models import BulkParseRequest

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"


@timer_wrap
def get_document_object(pdf_path: str):
    response = requests.get(f"http://localhost:3000/parse_doc?pdf_path={pdf_path}")
    return response.json()

def build_payload(pdf_paths: List[str], subset: bool = False, subset_size: int = 20, export_file: Optional[str] = None, export_path: Optional[str] = None, max_workers: int = 8):
    payload = BulkParseRequest(
        pdf_paths=pdf_paths,
        export_file=export_file,
        export_path=export_path,
        subset=subset,
        subset_size=subset_size,
        max_workers=max_workers
    )
    return payload

@timer_wrap
def get_document_objects(pdf_paths: List[str], subset: bool = False, subset_size: int = 20, export_file: Optional[str] = None, export_path: Optional[str] = None, max_workers: int = 8):
    # Suppress pdfminer warnings about CropBox/MediaBox
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")
    if subset:
        np.random.seed(42)
        pdf_paths = np.random.choice(pdf_paths, subset_size, replace=False)
        logger.info(f"Subsetting to {subset_size} PDFs")
    payload = build_payload(pdf_paths=pdf_paths, subset=subset, subset_size=subset_size, export_file=export_file, export_path=export_path, max_workers=max_workers)
    response = requests.post(f"http://localhost:3000/bulk_parse_docs", json=payload.model_dump(exclude_none=True))
    return response.json()


@timer_wrap
def main():
    pdf_paths = os.listdir(os.path.join(project_root, "Data/train/PDF"))
    pdf_paths = [os.path.join("Data/train/PDF", pdf) for pdf in pdf_paths if pdf.endswith(".pdf")]
    response = get_document_objects(pdf_paths=pdf_paths, subset=True,
                                   subset_size=5)
    logger.info(f"Response: {response}")
    # # save to duckdb
    # db_helper = DuckDBHelper(DEFAULT_DUCKDB_PATH)
    # db_helper.batch_upsert_documents(response)
    # db_helper.close()

    # bulk_response = get_document_objects(pdf_paths=pdf_paths, subset=False)
    # logger.info(f"Bulk response: {bulk_response["message"]}")

if __name__ == "__main__":
    main()