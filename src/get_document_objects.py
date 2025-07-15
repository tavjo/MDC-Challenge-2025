# src/get_document_objects.py

import os, sys, warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import List
import numpy as np

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Project root: {project_root}")
sys.path.append(project_root)

from src.helpers import timer_wrap, initialize_logging, parallel_processing_decorator, compute_file_hash, num_tokens, export_docs
from src.models import Document
from src.extract_pdf_text_unstructured import load_pdf_pages

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

@timer_wrap
def build_document_object(pdf_path: str):
    if os.path.exists(pdf_path):
        logger.info(f"Building document object for {pdf_path}")
    else:
        logger.error(f"PDF file does not exist: {pdf_path}")
        return None
    article_id = Path(pdf_path).stem
    # Suppress pdfminer warnings about CropBox/MediaBox
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")
    pages = load_pdf_pages(pdf_path)
    total_char_length = sum(len(page) for page in pages)
    parsed_timestamp = datetime.now(timezone.utc).isoformat()
    # calculate total tokens
    total_tokens = sum([num_tokens(page) for page in pages])
    document = Document(
        doi=article_id,
        full_text=pages,
        total_char_length=total_char_length,
        parsed_timestamp=parsed_timestamp,
        file_hash=compute_file_hash(pdf_path),
        file_path=pdf_path,
        n_pages = len(pages),
        total_tokens=total_tokens
    )
    return document

@timer_wrap
# @parallel_processing_decorator(batch_param_name="pdf_paths", batch_size=5, max_workers=8, flatten=False)
def build_document_objects(pdf_paths: List[str], subset: bool = False, subset_size: int = 20) -> List[Document]:
    # Suppress pdfminer warnings about CropBox/MediaBox
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")
    documents = []
    if subset:
        np.random.seed(42)
        pdf_paths = np.random.choice(pdf_paths, subset_size, replace=False)
        logger.info(f"Subsetting to {subset_size} PDFs")
    for pdf_path in pdf_paths:
        document = build_document_object(pdf_path)
        documents.append(document)
    return documents


@timer_wrap
def main():
    pdf_paths = os.listdir(os.path.join(project_root, "Data/train/PDF"))
    pdf_paths = [os.path.join("Data/train/PDF", pdf) for pdf in pdf_paths if pdf.endswith(".pdf")]
    documents = build_document_objects(pdf_paths=pdf_paths, subset=False)
    export_docs(documents)

if __name__ == "__main__":
    main()