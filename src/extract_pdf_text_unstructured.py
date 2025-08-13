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


def clean_page(text: str) -> str:
    """
    Tidy a raw PDF-extracted page so that downstream regexes
    (e.g., citation patterns) match reliably.

    Steps:
    1.  Join words split with hyphen at line-end.
    2.  Remove discretionary/soft hyphens.
    3.  Replace ligature glyphs with ASCII equivalents.
    4.  Collapse bare new-lines into a single space.
    5.  Collapse runs of spaces/tabs.
    6.  NFKD normalisation (helps “weird” accents).
    """
    # 1. join   exam- \n ple  → example
    text = _HYPHEN_LINEBREAK_RE.sub("", text)

    # 2. strip the invisible soft hyphen
    text = _SOFT_HYPHEN_RE.sub("", text)

    # 3. replace common ligatures
    text = text.translate(_LIGATURE_MAP)

    # 4. convert remaining line breaks to spaces
    text = _NEWLINES_RE.sub(" ", text)

    # 5. squeeze multiple consecutive spaces/tabs/non-breaking spaces
    text = _MULTISPACE_RE.sub(" ", text).strip()

    # 6. Unicode normalisation – makes “ﬃ”→“ffi” consistent, etc.
    text = unicodedata.normalize("NFKD", text)

    return text


def extract_pdf_text(
    pdf_path: str,
    *,
    include_page_breaks: bool = True,
    return_elements: bool = False,
    strategy: str = "hi_res"
) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[Element]]]:
    """Parse *pdf_path* with **unstructured** and return page-wise text.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file to parse.
    include_page_breaks : bool, optional
        Whether to tell ``partition_pdf`` to insert explicit page breaks.
    return_elements : bool, optional
        If ``True``, also return the raw ``Element`` list so that callers can
        compute table/figure counts, language detection, etc.  Default is
        ``False`` for backward compatibility.

    Returns
    -------
    list | (list, list)
        * When *return_elements* is ``False`` (default):
          ``List[{'page_number': int, 'text': str}]``
        * When *True*: a tuple ``(pages, elements)`` where *pages* is the same
          list described above and *elements* is the unmodified list of
          :class:`unstructured.documents.elements.Element` objects.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info("Parsing PDF %s with unstructured.partition_pdf with strategy %s …", path, strategy)

    # Single pass over the PDF
    elements: List[Element] = partition_pdf(str(path), include_page_breaks=include_page_breaks, strategy=strategy)

    # Group texts by page number
    pages_acc: Dict[int, List[str]] = {}
    for el in elements:
        pn = el.metadata.page_number or 1
        el.text = clean_page(el.text)
        pages_acc.setdefault(pn, []).append(el.text.strip())

    pages = [
        {"page_number": pn, "text": "\n\n".join(blocks)}
        for pn, blocks in sorted(pages_acc.items())
    ]

    logger.info("Extracted %d pages (%d elements) from %s.", len(pages), len(elements), path)

    if return_elements:
        return pages, elements
    return pages


def load_pdf_pages(pdf_path: str, strategy: str = "hi_res") -> List[str]:
    """
    Load Document pages from a PDF file.
    """
    logger.info(f"Loading PDF file: {pdf_path}")
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # --------------------------------------------------
    # 1) Parse PDF once and get both pages + elements
    # --------------------------------------------------
    logger.info(f"Extracting text from {pdf_path}")
    try:
        pages_json = extract_pdf_text(str(path), return_elements=False, strategy=strategy)
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
