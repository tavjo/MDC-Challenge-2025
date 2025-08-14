import os, sys, warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import List
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.helpers import timer_wrap, initialize_logging, compute_file_hash, num_tokens
from src.models import Document

# Prefer the unstructured-based extractor; fall back to the lightweight extractor if unavailable
# _EXTRACTOR_SOURCE = None
# try:
from src.extract_pdf_text_unstructured import load_pdf_pages, extract_xml_text  # type: ignore
from src.extract_pdf_text_light import load_pdf_pages as load_pdf_pages_light  # type: ignore
#     _EXTRACTOR_SOURCE = "unstructured"
# except Exception:
#     from src.extract_pdf_text_light import load_pdf_pages, extract_xml_text  # type: ignore
#     _EXTRACTOR_SOURCE = "light"

filename = os.path.basename(__file__)
logger = initialize_logging(filename)
# logger.info(f"Using '{_EXTRACTOR_SOURCE}' text extractor backend in document_parsing_service")

@timer_wrap
def build_document_object(pdf_path: str, strategy: str = "hi_res"):
    if os.path.exists(pdf_path):
        logger.debug(f"Building document object for {pdf_path}")
    else:
        logger.error(f"PDF file does not exist: {pdf_path}")
        return None
    # identify the document type
    doc_type = Path(pdf_path).suffix.lower()
    article_id = Path(pdf_path).stem
    # Suppress pdfminer warnings about CropBox/MediaBox
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")
    if doc_type == ".pdf":
        pages = []
        # 1) Try lightweight extractor first
        try:
            pages = load_pdf_pages_light(pdf_path)
            if not pages:
                logger.warning(f"Lightweight extractor returned 0 pages for {pdf_path}")
        except Exception as e:
            logger.warning(f"Lightweight extractor failed for {pdf_path}: {e}")
            pages = []

        # 2) Fallback to unstructured with fast strategy
        if not pages:
            try:
                pages = load_pdf_pages(pdf_path, strategy="fast")
                if not pages:
                    logger.warning(f"Unstructured extractor (strategy='fast') returned 0 pages for {pdf_path}")
            except Exception as e:
                logger.warning(f"Unstructured extractor (strategy='fast') failed for {pdf_path}: {e}")
                pages = []

        # 3) Final fallback to unstructured with hi_res strategy
        if not pages:
            try:
                pages = load_pdf_pages(pdf_path, strategy="hi_res")
                if not pages:
                    logger.error(f"Unstructured extractor (strategy='hi_res') returned 0 pages for {pdf_path}")
            except Exception as e:
                logger.error(f"Unstructured extractor (strategy='hi_res') failed for {pdf_path}: {e}")
                pages = []

        if not pages:
            logger.error(f"All extraction strategies yielded no pages for {pdf_path}")
            return None

        total_char_length = sum(len(page) for page in pages)
        # calculate total tokens
        total_tokens = sum([num_tokens(page) for page in pages])
        document = Document(
            doi=article_id,
            full_text=pages,
            total_char_length=total_char_length,
            parsed_timestamp=datetime.now(timezone.utc).isoformat(),
            file_hash=compute_file_hash(pdf_path),
            file_path=pdf_path,
            n_pages = len(pages),
            total_tokens=total_tokens
        )
    elif doc_type == ".xml":
        full_text = extract_xml_text(pdf_path)
        total_char_length = len(full_text)
        total_tokens = num_tokens(full_text)
        document = Document(
            doi=article_id,
            full_text=full_text,
            total_char_length=total_char_length,
            parsed_timestamp=datetime.now(timezone.utc).isoformat(),
            file_hash=compute_file_hash(pdf_path),
            file_path=pdf_path,
            total_tokens=total_tokens
        )
    else:
        logger.error(f"Unsupported document type: {doc_type}")
        return None
    return document

@timer_wrap
def build_document_objects(
    pdf_paths: List[str],
    subset: bool = False,
    subset_size: int = 20,
    max_workers: int = 8,
    strategy: str = "hi_res",
    progress: bool = True,
) -> List[Document]:
    # Suppress pdfminer warnings about CropBox/MediaBox
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")
    
    if subset:
        np.random.seed(42)
        pdf_paths = np.random.choice(pdf_paths, subset_size, replace=False).tolist()
        logger.info(f"Subsetting to {subset_size} PDFs")
    
    # Determine optimal number of workers
    workers = min(max_workers, len(pdf_paths)) if pdf_paths else 1
    logger.info(f"Processing {len(pdf_paths)} PDFs with {workers} workers")
    
    # Process documents in parallel
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {
            exe.submit(build_document_object, pdf_path, strategy=strategy): pdf_path
            for pdf_path in pdf_paths
        }
        documents: List[Document] = []

        iterator = as_completed(futures)
        if progress:
            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Parsing documents",
                unit="doc",
                dynamic_ncols=True,
                leave=True,
            )

        for future in iterator:
            try:
                document = future.result()
                if document is not None:
                    documents.append(document)
                else:
                    pdf_path = futures[future]
                    logger.warning(f"Failed to build document object for {pdf_path}")
            except Exception as e:
                pdf_path = futures[future]
                logger.error(f"Exception processing {pdf_path}: {str(e)}")
    
    logger.info(f"Successfully processed {len(documents)} out of {len(pdf_paths)} PDFs")
    return documents