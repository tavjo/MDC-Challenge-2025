import os, sys
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import warnings

try:
    # Unstructured's PDF partitioner returns a list of Elements that already
    # carry page numbers and layout information.
    from unstructured.partition.pdf import partition_pdf
    from unstructured.documents.elements import Element
except ImportError as e:
    raise ImportError(
        "The unstructured library is required for PDF parsing.\n"
        "Install it with: pip install \"unstructured[all]\""
    ) from e

import pickle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.helpers import initialize_logging, timer_wrap

filename = os.path.basename(__file__)
logger = initialize_logging(log_file = filename)

import re
import unicodedata

# --- pre-compiled regexes ---
# 1.  Break-point hyphens followed by a line-break and another word char
_HYPHEN_LINEBREAK_RE = re.compile(r"(?<=\w)-\s*(?:\r?\n|\r)+\s*(?=\w)")

# 2.  Remaining bare new-lines (collapse to single space)
_NEWLINES_RE = re.compile(r"\s*(?:\r?\n|\r)+\s*")

# 3.  Soft hyphen (U+00AD) – appears invisibly in many PDFs
_SOFT_HYPHEN_RE = re.compile("\u00AD")

# 4.  Collapse ≥2  spaces / tabs into one
_MULTISPACE_RE = re.compile(r"[ \t\u00A0]{2,}")

# 5.  Simple ligature map (extend as needed)
_LIGATURE_MAP = str.maketrans({
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
})

@timer_wrap
def clean_page(text: str) -> str:
    # 1. join   exam- \n ple  → example      (already present)
    text = _HYPHEN_LINEBREAK_RE.sub("", text)

    # 2. strip soft-hyphens                       (already present)
    text = _SOFT_HYPHEN_RE.sub("", text)

    # 3. replace ligatures                        (already present)
    text = text.translate(_LIGATURE_MAP)

    # 4. convert all remaining line breaks → space
    text = _NEWLINES_RE.sub(" ", text)
    
    # 4 b. **NEW**  delete any “dash + spaces” that remain inside a token  
    #               e.g. “bca- d9957” → “bcad9957”
    text = re.sub(r'-\s+(?=\w)', '', text)

    # 4 c. NEW – delete any spaces *inside* a token that follows a dash
    #(handles “…bca- d9957…”  →  “…bcad9957…”)
    text = re.sub(r'(?<=-)\s+(?=\w)', '', text)

    # 5. squeeze multiple spaces / tabs
    text = _MULTISPACE_RE.sub(" ", text).strip()

    # 6. Unicode normalisation
    text = unicodedata.normalize("NFKD", text)
    return text

@timer_wrap
def extract_pdf_text(
    pdf_path: str,
    *,
    include_page_breaks: bool = True,
    return_elements: bool = False,
    strategy: str = "fast",  # "fast", "hi_res", "ocr_only"
    include_metadata: bool = True,
) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[Element]]]:
    """Parse *pdf_path* with **unstructured** and return page-wise text.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file to parse.
    include_page_breaks : bool, optional
        Whether to tell ``partition_pdf`` to insert explicit page breaks.
    return_elements : bool, optional
        If ``True``, also return the raw ``Element`` list so that 
        callers can
        compute table/figure counts, language detection, etc.  
        Default is
        ``False`` for backward compatibility.
        Partitioning strategy: "fast", "hi_res", or "ocr_only".
    include_metadata : bool, optional
        Whether to include metadata in the output.
    
    Returns
    -------
    list | (list, list)
        * When *return_elements* is ``False`` (default):
          ``List({'page_number': int, 'text': str})``
        * When *True*: a tuple ``(pages, elements)`` where *pages* 
        is the same
          list described above and *elements* is the unmodified list 
          of
          :class:`unstructured.documents.elements.Element` objects.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    def _run(strategy_name: str):
        logger.info("Parsing PDF %s with unstructured.partition_pdf (strategy: %s)…", path, strategy_name)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")
            # Enable table structure inference when running high-fidelity modes, or if explicitly requested
            infer_tables_env = os.getenv("UNSTRUCTURED_INFER_TABLES", "auto").lower()
            if infer_tables_env in ("1", "true", "yes"):  # force on
                infer_table_structure = True
            elif infer_tables_env in ("0", "false", "no"):  # force off
                infer_table_structure = False
            else:  # auto
                infer_table_structure = strategy_name in ("hi_res", "ocr_only")

            elems: List[Element] = partition_pdf(
                str(path),
                include_page_breaks=include_page_breaks,
                strategy=strategy_name,
                include_metadata=include_metadata,
                infer_table_structure=infer_table_structure,
            )
        pages_acc: Dict[int, List[str]] = {}
        for el in elems:
            pn = el.metadata.page_number or 1
            el.text = clean_page(el.text)
            pages_acc.setdefault(pn, []).append(el.text.strip())
        pages_local = [
            {"page_number": pn, "text": "\n\n".join(blocks)}
            for pn, blocks in sorted(pages_acc.items())
        ]
        return pages_local, elems

    # Try requested strategy first, then fall back to hi_res, then ocr_only if no pages
    strategies = [strategy]
    for fallback in ("hi_res", "ocr_only"):
        if fallback not in strategies:
            strategies.append(fallback)

    last_elements: Optional[List[Element]] = None
    for strat in strategies:
        pages, elements = _run(strat)
        last_elements = elements
        logger.info("Extracted %d pages (%d elements) from %s using %s.", len(pages), len(elements), path, strat)
        if len(pages) > 0:
            if return_elements:
                return pages, elements
            return pages

    # If still empty, return empty consistent structure
    logger.warning("All strategies failed to yield pages for %s; returning empty result.", path)
    if return_elements:
        return [], (last_elements or [])
    return []

@timer_wrap
def load_pdf_pages(pdf_path: str) -> List[str]:
    """
    Load Document pages from a PDF file.
    """
    logger.info(f"Loading PDF file: {pdf_path}")
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # --------------------------------------------------
    # 1) Parse PDF with fallback strategies
    # --------------------------------------------------
    logger.info(f"Extracting text from {pdf_path}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")
    try:
        # Allow forcing a specific starting strategy via env var for hard cases
        start_strategy = os.getenv("PDF_PARSE_STRATEGY", "fast")
        pages_json = extract_pdf_text(str(path), return_elements=False, strategy=start_strategy)
        if not pages_json:
            logger.warning("Fast strategy returned no pages for %s; trying hi_res.", pdf_path)
            pages_json = extract_pdf_text(str(path), return_elements=False, strategy="hi_res")
        if not pages_json:
            logger.warning("hi_res returned no pages for %s; trying ocr_only.", pdf_path)
            pages_json = extract_pdf_text(str(path), return_elements=False, strategy="ocr_only")
        pages: List[str] = [p["text"] for p in pages_json]
        num_pages = len(pages_json)
        logger.info(f"Loaded {num_pages} pages from {pdf_path}")
        return pages
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        raise e

@timer_wrap
def get_all_pdfs(pdf_files: List[str], pdf_dir: str, subset: bool = False, subset_size: int = 20):
    """
    Load and save all PDFs as a single pickle file in a directory.
    """
    if subset:
        pdf_files = pdf_files[:subset_size]
        logger.info(f"Subsetting to {subset_size} PDFs")
    pdf_dicts = {}
    for pdf_file in pdf_files:
        article_id = Path(pdf_file).stem
        pdf_path = os.path.join(pdf_dir, pdf_file)
        pages = load_pdf_pages(pdf_path)
        pdf_dicts[article_id] = pages
    return pdf_dicts

@timer_wrap
def save_pdf_dicts(pdf_dicts: Dict[str, List[str]], pdf_dir: str):
    """
    Save the PDF dictionary to a pickle file.
    """
    with open(os.path.join(pdf_dir, "all_pdfs.pkl"), "wb") as f:
        pickle.dump(pdf_dicts, f)
    logger.info(f"Loaded and saved {len(pdf_dicts)} PDFs to {os.path.join(pdf_dir, 'all_pdfs.pkl')}")

if __name__ == "__main__":
    # pdf_path = os.path.join(project_root, "Data/train/PDF/10.1002_2017jc013030.pdf")
    # page_dicts = extract_pdf_text(pdf_path)
    # print(page_dicts)
    pdf_dir = os.path.join(project_root, "Data/train/PDF")
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    pdf_dicts = get_all_pdfs(pdf_files=pdf_files, pdf_dir=pdf_dir)
    save_pdf_dicts(pdf_dicts=pdf_dicts, pdf_dir=pdf_dir)

@timer_wrap
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
        logger.error(f"Failed to parse XML %s: {e}")
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

__all__ = ["extract_pdf_text", "extract_xml_text"]
