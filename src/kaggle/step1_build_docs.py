import os
import sys
import logging
import importlib.util
import warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Union, Optional

# Allow importing sibling kaggle helpers/models when used as a standalone script
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

try:
    from src.kaggle.helpers import timer_wrap, compute_file_hash, num_tokens  # type: ignore
    from src.kaggle.models import Document  # type: ignore
except Exception:
    # Fallback when running from within the `src/kaggle` directory as a loose script
    from helpers import timer_wrap, compute_file_hash, num_tokens  # type: ignore
    from models import Document  # type: ignore


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logger = logging.getLogger("build_docs")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ----------------------------------------------------------------------------
# Text cleaning utilities (ported from unstructured-based extractor)
# ----------------------------------------------------------------------------
import re
import unicodedata

# 1) Break-point hyphens followed by a line-break and another word char
_HYPHEN_LINEBREAK_RE = re.compile(r"(?<=\w)-\s*(?:\r?\n|\r)+\s*(?=\w)")
# 2) Remaining bare new-lines (collapse to single space)
_NEWLINES_RE = re.compile(r"\s*(?:\r?\n|\r)+\s*")
# 3) Soft hyphen (U+00AD)
_SOFT_HYPHEN_RE = re.compile("\u00AD")
# 4) Collapse >=2 spaces / tabs into one
_MULTISPACE_RE = re.compile(r"[ \t\u00A0]{2,}")
# 5) Simple ligature map
_LIGATURE_MAP = str.maketrans({
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
})


def clean_page(text: str) -> str:
    """Normalize PDF-extracted text for downstream NLP."""
    text = _HYPHEN_LINEBREAK_RE.sub("", text)
    text = _SOFT_HYPHEN_RE.sub("", text)
    text = text.translate(_LIGATURE_MAP)
    text = _NEWLINES_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    text = unicodedata.normalize("NFKD", text)
    return text


# ----------------------------------------------------------------------------
# Optional dependency availability
# ----------------------------------------------------------------------------
_HAVE_UNSTRUCTURED = importlib.util.find_spec("unstructured.partition.pdf") is not None
_HAVE_TESSERACT = importlib.util.find_spec("pytesseract") is not None
_HAVE_PYMUPDF = importlib.util.find_spec("fitz") is not None
_HAVE_PDF2IMAGE = importlib.util.find_spec("pdf2image") is not None


# ----------------------------------------------------------------------------
# Unstructured-based PDF/XML extraction (inlined)
# ----------------------------------------------------------------------------
if _HAVE_UNSTRUCTURED:
    # Unstructured's PDF partitioner returns a list of Elements with page metadata
    from unstructured.partition.pdf import partition_pdf  # type: ignore
    from unstructured.documents.elements import Element  # type: ignore
else:
    class Element:  # minimal stub for typing when unstructured is absent
        pass


def extract_pdf_text(
    pdf_path: str,
    *,
    include_page_breaks: bool = True,
    return_elements: bool = False,
    strategy: str = "fast",
) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[Element]]]:
    """Parse a PDF with unstructured and return page-wise text.

    Returns either a list of {"page_number": int, "text": str} dicts,
    or (pages, elements) when return_elements=True.
    """
    if not _HAVE_UNSTRUCTURED:
        raise RuntimeError("unstructured is not available in this environment.")

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info("Parsing PDF %s with unstructured.partition_pdf (strategy=%s)", path, strategy)

    elements: List[Element] = partition_pdf(
        str(path), include_page_breaks=include_page_breaks, strategy=strategy
    )

    pages_acc: Dict[int, List[str]] = {}
    for el in elements:
        pn = el.metadata.page_number or 1
        el.text = clean_page(el.text)
        pages_acc.setdefault(pn, []).append(el.text.strip())

    pages: List[Dict[str, str]] = [
        {"page_number": pn, "text": "\n\n".join(blocks)} for pn, blocks in sorted(pages_acc.items())
    ]

    logger.info("Extracted %d pages (%d elements) from %s.", len(pages), len(elements), path)

    if return_elements:
        return pages, elements
    return pages


def load_pdf_pages(pdf_path: str, strategy: str = "fast") -> List[str]:
    """Load text pages from a PDF file using unstructured or OCR fallback."""
    logger.info("Loading PDF file: %s", pdf_path)
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")

    # Force OCR route if requested
    if strategy.lower() in {"ocr", "ocr_only"}:
        logger.info("Using OCR fallback (strategy=%s) for %s", strategy, pdf_path)
        return ocr_pdf_pages(str(path))

    # Try unstructured first when available
    if _HAVE_UNSTRUCTURED:
        try:
            pages_json = extract_pdf_text(str(path), return_elements=False, strategy=strategy)
            pages: List[str] = [p["text"] for p in pages_json]
            # If empty/near-empty, fallback to OCR
            if not pages or all(len(p.strip()) == 0 for p in pages):
                logger.warning("Unstructured returned empty pages; falling back to OCR for %s", pdf_path)
                if _HAVE_TESSERACT:
                    return ocr_pdf_pages(str(path))
            logger.info("Loaded %d pages from %s", len(pages_json), pdf_path)
            return pages
        except Exception as e:
            logger.warning("Unstructured failed for %s: %s; attempting OCR fallback", pdf_path, e)
            if _HAVE_TESSERACT:
                return ocr_pdf_pages(str(path))
            raise

    # If unstructured is not available, try OCR directly
    if _HAVE_TESSERACT:
        logger.info("Unstructured not available; using OCR for %s", pdf_path)
        return ocr_pdf_pages(str(path))

    raise RuntimeError("No available PDF extraction backends (need unstructured or OCR)")


def extract_xml_text(xml_path: str, encoding: Optional[str] = None) -> str:
    """Extract raw text content from an XML file and normalize it."""
    from xml.etree import ElementTree as ET

    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    with open(path, "rb") as f:
        data = f.read()

    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        logger.error("Failed to parse XML %s: %s", path, e)
        raise

    text_fragments = list(root.itertext())
    raw_text = " ".join(fragment.strip() for fragment in text_fragments if fragment and fragment.strip())
    normalized = clean_page(raw_text)
    return normalized


# ----------------------------------------------------------------------------
# OCR fallback via Tesseract (render with PyMuPDF or pdf2image)
# ----------------------------------------------------------------------------
def _render_pages_with_fitz(pdf_path: Path, dpi: int = 300):
    if not _HAVE_PYMUPDF:
        raise RuntimeError("PyMuPDF not available")
    import io
    from PIL import Image  # Pillow is typically available in Kaggle
    import fitz  # type: ignore

    images = []
    zoom = dpi / 72.0
    with fitz.open(str(pdf_path)) as doc:
        if doc.needs_pass and not doc.authenticate(""):
            raise PermissionError(f"Encrypted PDF requires a password: {pdf_path}")
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)
    return images


def _render_pages_with_pdf2image(pdf_path: Path, dpi: int = 300):
    if not _HAVE_PDF2IMAGE:
        raise RuntimeError("pdf2image not available")
    from pdf2image import convert_from_path  # type: ignore
    return convert_from_path(str(pdf_path), dpi=dpi)


def _ocr_pages_from_images(images, lang: str = "eng") -> List[str]:
    import pytesseract  # type: ignore
    pages: List[str] = []
    for img in images:
        text = pytesseract.image_to_string(img, lang=lang) or ""
        pages.append(clean_page(text))
    return pages


def ocr_pdf_pages(pdf_path: str, dpi: int = 300, lang: str = "eng") -> List[str]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not _HAVE_TESSERACT:
        raise RuntimeError("pytesseract not available for OCR fallback")

    # Prefer PyMuPDF rendering; else fallback to pdf2image
    if _HAVE_PYMUPDF:
        images = _render_pages_with_fitz(path, dpi=dpi)
    elif _HAVE_PDF2IMAGE:
        images = _render_pages_with_pdf2image(path, dpi=dpi)
    else:
        raise RuntimeError("No PDF-to-image renderer available (need PyMuPDF or pdf2image)")

    return _ocr_pages_from_images(images, lang=lang)

# ----------------------------------------------------------------------------
# Document object builders (based on api/services/document_parsing_service.py)
# ----------------------------------------------------------------------------
@timer_wrap
def build_document_object(pdf_path: str, strategy: str = "fast") -> Optional[Document]:
    if os.path.exists(pdf_path):
        logger.info("Building document object for %s", pdf_path)
    else:
        logger.error("PDF file does not exist: %s", pdf_path)
        return None

    doc_type = Path(pdf_path).suffix.lower()
    article_id = Path(pdf_path).stem

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")

    if doc_type == ".pdf":
        pages = load_pdf_pages(pdf_path, strategy=strategy)
        total_char_length = sum(len(page) for page in pages)
        # Token counting may rely on tiktoken; provide a safe fallback if unavailable
        try:
            total_tokens = sum(num_tokens(page) for page in pages)
        except Exception:
            total_tokens = sum(len(page.split()) for page in pages)
        document = Document(
            doi=article_id,
            full_text=pages,
            total_char_length=total_char_length,
            parsed_timestamp=datetime.now(timezone.utc).isoformat(),
            file_hash=compute_file_hash(Path(pdf_path)),
            file_path=pdf_path,
            n_pages=len(pages),
            total_tokens=total_tokens,
        )
    elif doc_type == ".xml":
        full_text = extract_xml_text(pdf_path)
        total_char_length = len(full_text)
        try:
            total_tokens = num_tokens(full_text)
        except Exception:
            total_tokens = len(full_text.split())
        document = Document(
            doi=article_id,
            full_text=full_text,
            total_char_length=total_char_length,
            parsed_timestamp=datetime.now(timezone.utc).isoformat(),
            file_hash=compute_file_hash(Path(pdf_path)),
            file_path=pdf_path,
            total_tokens=total_tokens,
        )
    else:
        logger.error("Unsupported document type: %s", doc_type)
        return None

    return document


@timer_wrap
def build_document_objects(
    pdf_paths: List[str],
    subset: bool = False,
    subset_size: int = 20,
    max_workers: int = 8,
    strategy: str = "fast",
) -> List[Document]:
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")

    if subset and pdf_paths:
        np.random.seed(42)
        pdf_paths = np.random.choice(pdf_paths, subset_size, replace=False).tolist()
        logger.info("Subsetting to %d PDFs", subset_size)

    workers = min(max_workers, len(pdf_paths)) if pdf_paths else 1
    logger.info("Processing %d PDFs with %d workers", len(pdf_paths), workers)

    documents: List[Document] = []
    if not pdf_paths:
        return documents

    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(build_document_object, pdf_path, strategy=strategy): pdf_path for pdf_path in pdf_paths}
        for future in as_completed(futures):
            pdf_path = futures[future]
            try:
                document = future.result()
                if document is not None:
                    documents.append(document)
                else:
                    logger.warning("Failed to build document object for %s", pdf_path)
            except Exception as e:
                logger.error("Exception processing %s: %s", pdf_path, str(e))

    logger.info("Successfully processed %d out of %d PDFs", len(documents), len(pdf_paths))
    return documents


__all__ = [
    "build_document_object",
    "build_document_objects",
    "extract_pdf_text",
    "extract_xml_text",
    "load_pdf_pages",
    "ocr_pdf_pages",
]

