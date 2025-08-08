import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import warnings
import io
import re
import unicodedata
import importlib.util

# Detect availability of optional dependencies without importing them at module import time
_HAVE_PYMUPDF = importlib.util.find_spec("fitz") is not None
_HAVE_PDFMINER = importlib.util.find_spec("pdfminer.high_level") is not None and importlib.util.find_spec("pdfminer.layout") is not None

# Project root and helpers import (mirror structure of the unstructured-based script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.helpers import initialize_logging, timer_wrap

filename = os.path.basename(__file__)
logger = initialize_logging(log_file=filename)


# --- pre-compiled regexes (kept identical to unstructured variant) ---
# 1.  Break-point hyphens followed by a line-break and another word char
_HYPHEN_LINEBREAK_RE = re.compile(r"(?<=\w)-\s*(?:\r?\n|\r)+\s*(?=\w)")

# 2.  Remaining bare new-lines (collapse to single space)
_NEWLINES_RE = re.compile(r"\s*(?:\r?\n|\r)+\s*")

# 3.  Soft hyphen (U+00AD)
_SOFT_HYPHEN_RE = re.compile("\u00AD")

# 4.  Collapse ≥2 spaces / tabs into one
_MULTISPACE_RE = re.compile(r"[ \t\u00A0]{2,}")

# 5.  Simple ligature map
_LIGATURE_MAP = str.maketrans({
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
})


def clean_page(text: str) -> str:
    """Normalizes PDF-extracted text for downstream NLP.

    Mirrors the cleaning in `extract_pdf_text_unstructured.py` for consistency.
    """
    # 1. join   exam- \n ple  → example
    text = _HYPHEN_LINEBREAK_RE.sub("", text)

    # 2. strip soft-hyphens
    text = _SOFT_HYPHEN_RE.sub("", text)

    # 3. replace ligatures
    text = text.translate(_LIGATURE_MAP)

    # 4. convert remaining line breaks → space
    text = _NEWLINES_RE.sub(" ", text)

    # 4b. delete any “dash + spaces” that remain inside a token
    text = re.sub(r'-\s+(?=\w)', '', text)

    # 4c. delete spaces after a dash within a token
    text = re.sub(r'(?<=-)\s+(?=\w)', '', text)

    # 5. squeeze multiple spaces / tabs
    text = _MULTISPACE_RE.sub(" ", text).strip()

    # 6. Unicode normalisation
    text = unicodedata.normalize("NFKD", text)
    return text


def _extract_with_pymupdf(pdf_path: Path) -> List[Dict[str, Union[int, str]]]:
    """Extract page-wise text using PyMuPDF.

    Returns a list of {"page_number": int, "text": str} dicts.
    """
    if not _HAVE_PYMUPDF:
        raise RuntimeError("PyMuPDF not available")

    pages: List[Dict[str, Union[int, str]]] = []

    # Import locally to avoid linter unresolved-import warnings when dependency is absent
    import fitz  # type: ignore

    # Open and authenticate if needed
    with fitz.open(str(pdf_path)) as doc:
        if doc.needs_pass:
            # Try empty password first (common for view-protected, copy-allowed docs)
            if not doc.authenticate(""):
                raise PermissionError(f"Encrypted PDF requires a password: {pdf_path}")

        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            # "text" is a good balance of fidelity and simplicity
            raw_text: str = page.get_text("text") or ""
            cleaned = clean_page(raw_text)
            pages.append({"page_number": page_index + 1, "text": cleaned})

    return pages


def _extract_with_pdfminer(pdf_path: Path) -> List[Dict[str, Union[int, str]]]:
    """Extract page-wise text using pdfminer.six.

    pdfminer’s high_level API writes to a file-like object; we do one pass
    for all pages and then split on form feed "\x0c" which separates pages.
    """
    if not _HAVE_PDFMINER:
        raise RuntimeError("pdfminer.six not available")

    # Import locally to avoid linter unresolved-import warnings when dependency is absent
    from pdfminer.high_level import extract_text_to_fp  # type: ignore
    from pdfminer.layout import LAParams  # type: ignore

    output = io.StringIO()
    laparams = LAParams(
        all_texts=True,  # keep as much text as possible
        line_margin=0.2,
        char_margin=2.0,
        word_margin=0.1,
    )
    with open(pdf_path, "rb") as fp:
        extract_text_to_fp(
            fp,
            outfp=output,
            laparams=laparams,
            output_type="text",
            codec=None,
        )
    full_text = output.getvalue()
    # pdfminer separates pages with form feed characters
    raw_pages = full_text.split("\x0c")
    pages: List[Dict[str, Union[int, str]]] = []
    for i, raw in enumerate(raw_pages, start=1):
        if not raw.strip():
            continue
        pages.append({"page_number": i, "text": clean_page(raw)})
    return pages


def extract_pdf_text(
    pdf_path: str,
    *,
    include_page_breaks: bool = True,
    return_elements: bool = False,
    strategy: str = "fast",  # accepted for signature compatibility; no OCR here
    include_metadata: bool = True,
) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List]]:
    """Parse pdf_path and return page-wise text using lightweight deps.

    Backwards-compatible signature with the unstructured-based variant.

    - Prefers PyMuPDF; falls back to pdfminer.six.
    - No OCR modes here; if strategy == "ocr_only" we log a warning and continue.

    Returns
    -------
    list | (list, list)
        When return_elements is False (default):
            List[{"page_number": int, "text": str}]
        When True: (pages, elements) where elements is an empty list for
            compatibility (we do not provide element-level metadata here).
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if strategy.lower() == "ocr_only":
        logger.warning("strategy='ocr_only' requested, but OCR is not supported in this lightweight parser. Proceeding with text extraction only.")

    logger.info("Parsing PDF %s (lightweight parser; PyMuPDF=%s, pdfminer=%s)…", path, _HAVE_PYMUPDF, _HAVE_PDFMINER)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")

        pages: Optional[List[Dict[str, Union[int, str]]]] = None
        errors: List[str] = []

        # Try PyMuPDF first
        if _HAVE_PYMUPDF:
            try:
                pages = _extract_with_pymupdf(path)
            except Exception as e:
                errors.append(f"PyMuPDF failed: {e}")
                logger.warning("PyMuPDF extraction failed for %s: %s", path, e)

        # Fallback to pdfminer if needed
        if pages is None and _HAVE_PDFMINER:
            try:
                pages = _extract_with_pdfminer(path)
            except Exception as e:
                errors.append(f"pdfminer failed: {e}")
                logger.warning("pdfminer extraction failed for %s: %s", path, e)

        if pages is None:
            error_detail = "; ".join(errors) if errors else "no available backends"
            raise RuntimeError(f"Unable to extract text from {pdf_path}: {error_detail}")

    logger.info("Extracted %d pages from %s.", len(pages), path)

    # Coerce to expected return type (drop typing detail of int vs str for page_number)
    coalesced_pages: List[Dict[str, str]] = [
        {"page_number": int(p["page_number"]), "text": str(p["text"]) }  # type: ignore[index]
        for p in pages
    ]

    if return_elements:
        return coalesced_pages, []
    return coalesced_pages


def load_pdf_pages(pdf_path: str) -> List[str]:
    """Load text pages from a PDF file.

    Mirrors the behavior of the unstructured-based variant but uses the
    lightweight extractor defined above.
    """
    logger.info(f"Loading PDF file: {pdf_path}")
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")
    try:
        pages_json = extract_pdf_text(str(path), return_elements=False)
        pages: List[str] = [p["text"] for p in pages_json]
        logger.info(f"Loaded %d pages from %s", len(pages_json), pdf_path)
        return pages
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        raise e


@timer_wrap
# @parallel_processing_decorator(batch_param_name="pdf_files", batch_size=50, max_workers=8, flatten=True)
def get_all_pdfs(pdf_files: List[str], pdf_dir: str, subset: bool = False, subset_size: int = 20):
    """Load and save multiple PDFs as a dictionary keyed by stem name.

    Same signature as the unstructured-based version for easy drop-in.
    """
    if subset:
        pdf_files = pdf_files[:subset_size]
        logger.info(f"Subsetting to {subset_size} PDFs")
    pdf_dicts: Dict[str, List[str]] = {}
    for pdf_file in pdf_files:
        article_id = Path(pdf_file).stem
        pdf_path = os.path.join(pdf_dir, pdf_file)
        pages = load_pdf_pages(pdf_path)
        pdf_dicts[article_id] = pages
    return pdf_dicts


def save_pdf_dicts(pdf_dicts: Dict[str, List[str]], pdf_dir: str):
    """Save the PDF dictionary to a pickle file named all_pdfs.pkl in pdf_dir."""
    import pickle
    out_path = os.path.join(pdf_dir, "all_pdfs.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(pdf_dicts, f)
    logger.info(f"Loaded and saved {len(pdf_dicts)} PDFs to {out_path}")


def extract_xml_text(xml_path: str, encoding: Optional[str] = None) -> str:
    """Extract raw text content from an XML file as cleanly as possible.

    - Uses xml.etree.ElementTree to parse the document
    - Concatenates all text nodes (itertext)
    - Applies the same normalization as `clean_page` for consistency
    """
    from xml.etree import ElementTree as ET

    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    # Read as bytes to let ET infer encoding from the XML header if present
    with open(path, "rb") as f:
        data = f.read()

    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML %s: %s", path, e)
        raise

    text_fragments = list(root.itertext())
    raw_text = " ".join(fragment.strip() for fragment in text_fragments if fragment and fragment.strip())
    normalized = clean_page(raw_text)
    return normalized


if __name__ == "__main__":
    # Simple smoke test similar to the unstructured variant's __main__
    project_root_local = project_root
    pdf_dir = os.path.join(project_root_local, "Data/train/PDF")
    if os.path.isdir(pdf_dir):
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
        pdf_dicts = get_all_pdfs(pdf_files=pdf_files, pdf_dir=pdf_dir)
        # save_pdf_dicts(pdf_dicts=pdf_dicts, pdf_dir=pdf_dir)
    else:
        logger.info("PDF directory not found at %s; skipping demo run.", pdf_dir)


__all__ = [
    "extract_pdf_text",
    "extract_xml_text",
]

