# src/get_document_objects.py

import os, sys, warnings
from pathlib import Path
from typing import List
import numpy as np
import requests

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Project root: {project_root}")
sys.path.append(project_root)

from src.helpers import timer_wrap, initialize_logging

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

@timer_wrap
def get_document_object(pdf_path: str):
    response = requests.get(f"http://localhost:3000/parse_doc?pdf_path={pdf_path}")
    return response.json()

@timer_wrap
def get_document_objects(pdf_paths: List[str], subset: bool = False, subset_size: int = 20):
    # Suppress pdfminer warnings about CropBox/MediaBox
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")
    documents = []
    if subset:
        np.random.seed(42)
        pdf_paths = np.random.choice(pdf_paths, subset_size, replace=False)
        logger.info(f"Subsetting to {subset_size} PDFs")
    response = requests.get(f"http://localhost:3000/bulk_parse_docs?pdf_paths={pdf_paths}")
    return response.json()


@timer_wrap
def main():
    pdf_paths = os.listdir(os.path.join(project_root, "Data/train/PDF"))
    pdf_paths = [os.path.join("Data/train/PDF", pdf) for pdf in pdf_paths if pdf.endswith(".pdf")]
    response = get_document_object(pdf_path=pdf_paths[0])
    logger.info(f"Response: {response["message"]}")

    # bulk_response = get_document_objects(pdf_paths=pdf_paths, subset=False)
    # logger.info(f"Bulk response: {bulk_response["message"]}")

if __name__ == "__main__":
    main()