# Parse PDFs & XML files into pydantic `Document` objects and store into DuckDB for downstream steps
import os, sys, logging, importlib.util, warnings, re, unicodedata
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Union, Optional

# Allow importing sibling kaggle helpers/models when used as a standalone script
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from helpers import timer_wrap, compute_file_hash, num_tokens, ensure_dir  # type: ignore
from models import Document  # type: ignore

# --- GPU-capable Unstructured + EasyOCR + spawn-based multi-GPU executor -----
logger = logging.getLogger("build_docs")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ----------------------------- text cleaning ---------------------------------
_HYPHEN_LINEBREAK_RE = re.compile(r"(?<=\w)-\s*(?:\r?\n|\r)+\s*(?=\w)")
_NEWLINES_RE        = re.compile(r"\s*(?:\r?\n|\r)+\s*")
_SOFT_HYPHEN_RE     = re.compile("\u00AD")
_MULTISPACE_RE      = re.compile(r"[ \t\u00A0]{2,}")
_LIGATURE_MAP       = str.maketrans({"ﬀ":"ff","ﬁ":"fi","ﬂ":"fl","ﬃ":"ffi","ﬄ":"ffl"})

def clean_page(text: str) -> str:
    text = _HYPHEN_LINEBREAK_RE.sub("", text)
    text = _SOFT_HYPHEN_RE.sub("", text)
    text = text.translate(_LIGATURE_MAP)
    text = _NEWLINES_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return unicodedata.normalize("NFKD", text)

# ------------------------ optional deps discovery ----------------------------
_HAVE_UNSTRUCTURED = importlib.util.find_spec("unstructured.partition.pdf") is not None
_HAVE_TESSERACT    = importlib.util.find_spec("pytesseract") is not None
_HAVE_PYMUPDF      = importlib.util.find_spec("fitz") is not None
_HAVE_PDFMINER     = (importlib.util.find_spec("pdfminer.high_level") is not None and
                      importlib.util.find_spec("pdfminer.layout") is not None)
_HAVE_PDF2IMAGE    = importlib.util.find_spec("pdf2image") is not None
_HAVE_EASYOCR      = importlib.util.find_spec("easyocr") is not None
_HAVE_TORCH        = importlib.util.find_spec("torch") is not None
_HAVE_ORT          = importlib.util.find_spec("onnxruntime") is not None

if _HAVE_UNSTRUCTURED:
    from unstructured.partition.pdf import partition_pdf  # type: ignore
    from unstructured.documents.elements import Element   # type: ignore
else:
    class Element: pass  # stub

# ----------------------------- diagnostics -----------------------------------
def _log_gpu_preflight():
    if _HAVE_ORT:
        try:
            import onnxruntime as ort  # type: ignore
            logger.info("ONNX Runtime providers: %s", ort.get_available_providers())
        except Exception as e:
            logger.warning("ONNX preflight failed: %s", e)
    if _HAVE_TORCH:
        try:
            import torch  # type: ignore
            logger.info("Torch CUDA is_available=%s, devices=%d",
                        torch.cuda.is_available(), torch.cuda.device_count())
        except Exception as e:
            logger.warning("Torch preflight failed: %s", e)

# --------------------------- LIGHT extractors --------------------------------
def _extract_with_pymupdf(pdf_path: Path) -> List[str]:
    """Lightweight text extraction via PyMuPDF (fast, no ML). Returns per-page text."""
    if not _HAVE_PYMUPDF:
        return []
    import fitz  # type: ignore
    pages: List[str] = []
    with fitz.open(str(pdf_path)) as doc:
        if doc.needs_pass and not doc.authenticate(""):
            raise PermissionError(f"Encrypted PDF requires a password: {pdf_path}")
        for i in range(doc.page_count):
            # "text" is usually best for readable content; fall back to "blocks" if needed
            txt = doc.load_page(i).get_text("text") or ""
            txt = clean_page(txt)
            if txt.strip():
                pages.append(txt)
            else:
                pages.append("")  # keep page count alignment
    return pages

def _extract_with_pdfminer(pdf_path: Path) -> List[str]:
    """Lightweight text extraction via pdfminer.six. Returns per-page text."""
    if not _HAVE_PDFMINER:
        return []
    from pdfminer.high_level import extract_pages  # type: ignore
    from pdfminer.layout import LTTextContainer  # type: ignore
    pages: List[str] = []
    for page_layout in extract_pages(str(pdf_path)):
        chunks = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                chunks.append(element.get_text())
        txt = clean_page(" ".join(chunks))
        pages.append(txt)
    return pages

def extract_pdf_text_light(pdf_path: str) -> List[str]:
    """
    Try light/CPU parsers first (PyMuPDF, then pdfminer).
    Returns a list[str] (one entry per page). Empty list if no text extracted.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Suppress noisy warnings from some parsers
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CropBox.*MediaBox.*")
        warnings.filterwarnings("ignore", category=UserWarning)

        # 1) PyMuPDF
        try:
            if _HAVE_PYMUPDF:
                logger.info("Light parse with PyMuPDF: %s", path.name)
                pages = _extract_with_pymupdf(path)
                # consider success if any non-empty page exists
                if any(p.strip() for p in pages):
                    return pages
        except Exception as e:
            logger.warning("PyMuPDF light extract failed: %s", e)

        # 2) pdfminer.six
        try:
            if _HAVE_PDFMINER:
                logger.info("Light parse with pdfminer.six: %s", path.name)
                pages = _extract_with_pdfminer(path)
                if any(p.strip() for p in pages):
                    return pages
        except Exception as e:
            logger.warning("pdfminer light extract failed: %s", e)

    return []

# --------------------------- Unstructured parse ------------------------------
def extract_pdf_text(
    pdf_path: str,
    *,
    include_page_breaks: bool = True,
    return_elements: bool = False,
    strategy: str = "fast",
) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[Element]]]:
    """Parse a PDF with unstructured and return page-wise text (optionally elements)."""
    if not _HAVE_UNSTRUCTURED:
        raise RuntimeError("unstructured is not available in this environment.")
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Force GPU providers to be visible before model init
    _log_gpu_preflight()

    logger.info("Parsing %s with partition_pdf(strategy=%s)", path.name, strategy)

    # hi_res uses an ONNX layout model; make sure to ask for table structure.
    # You can pick a model via UNSTRUCTURED_HI_RES_MODEL_NAME env var.
    elements: List[Element] = partition_pdf(
        filename=str(path),
        strategy=strategy,
        include_page_breaks=include_page_breaks,
        infer_table_structure=True if strategy.lower() == "hi_res" else False,
        # Accept env override; default stays with Unstructured's layout model.
        hi_res_model_name=os.environ.get("UNSTRUCTURED_HI_RES_MODEL_NAME", None),
    )

    pages_acc: Dict[int, List[str]] = {}
    for el in elements:
        pn = getattr(el.metadata, "page_number", None) or 1
        txt = clean_page(getattr(el, "text", "") or "")
        if txt:
            pages_acc.setdefault(pn, []).append(txt)

    pages: List[Dict[str, str]] = [
        {"page_number": pn, "text": "\n\n".join(blocks)}
        for pn, blocks in sorted(pages_acc.items())
    ]
    if return_elements:
        return pages, elements
    return pages

# ------------------------------ XML extract ----------------------------------
def extract_xml_text(xml_path: str, encoding: Optional[str] = None) -> str:
    from xml.etree import ElementTree as ET
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    raw = " ".join(fragment.strip() for fragment in ET.fromstring(path.read_bytes()).itertext()
                   if fragment and fragment.strip())
    return clean_page(raw)

# ------------------------------- OCR helpers ---------------------------------
def _render_pages_with_fitz(pdf_path: Path, dpi: int = 250):
    if not _HAVE_PYMUPDF:
        raise RuntimeError("PyMuPDF not available")
    import io
    from PIL import Image
    import fitz  # type: ignore
    imgs, zoom = [], dpi / 72.0
    with fitz.open(str(pdf_path)) as doc:
        if doc.needs_pass and not doc.authenticate(""):
            raise PermissionError(f"Encrypted PDF requires a password: {pdf_path}")
        for i in range(doc.page_count):
            pix = doc.load_page(i).get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            imgs.append(Image.open(io.BytesIO(pix.tobytes("png"))))
    return imgs

def _render_pages_with_pdf2image(pdf_path: Path, dpi: int = 250):
    if not _HAVE_PDF2IMAGE:
        raise RuntimeError("pdf2image not available")
    from pdf2image import convert_from_path  # type: ignore
    return convert_from_path(str(pdf_path), dpi=dpi)

def _ocr_pages_from_images(images, lang: str = "eng") -> List[str]:
    """Prefer GPU EasyOCR; fallback to pytesseract if unavailable."""
    pages: List[str] = []
    if _HAVE_EASYOCR and _HAVE_TORCH:
        import numpy as np, torch  # type: ignore
        import easyocr              # type: ignore
        try:
            reader = easyocr.Reader([lang[:2]], gpu=torch.cuda.is_available())
            for img in images:
                lines = reader.readtext(np.array(img), detail=0, paragraph=True)
                pages.append(clean_page(" ".join(lines)))
            return pages
        except Exception as e:
            logger.warning("EasyOCR failed (%s); falling back to Tesseract.", e)
    if not _HAVE_TESSERACT:
        raise RuntimeError("No OCR engine available: install easyocr or pytesseract.")
    import pytesseract  # type: ignore
    for img in images:
        pages.append(clean_page(pytesseract.image_to_string(img, lang=lang) or ""))
    return pages

def ocr_pdf_pages(pdf_path: str, dpi: int = 250, lang: str = "eng") -> List[str]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    images = (_render_pages_with_fitz(path, dpi=dpi)
              if _HAVE_PYMUPDF else _render_pages_with_pdf2image(path, dpi=dpi))
    return _ocr_pages_from_images(images, lang=lang)

# ------------------------ top-level loading function -------------------------
def _nonempty(pages: List[str]) -> bool:
    return bool(pages) and any(p.strip() for p in pages)

def load_pdf_pages(pdf_path: str, strategy: str = "fast") -> List[str]:
    """
    Gradual fallback order:
      1) Light parsers (PyMuPDF → pdfminer)  [always attempted first]
      2) Unstructured 'fast'
      3) Unstructured 'hi_res'  (GPU-capable; ONNX/EasyOCR path)
      4) OCR fallback
    If strategy == 'ocr' or 'ocr_only', jump straight to OCR.
    """
    logger.info("Loading PDF file: %s", pdf_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CropBox missing from /Page")
        warnings.filterwarnings("ignore", category=UserWarning)

    # Explicit OCR path if requested
    if strategy.lower() in {"ocr", "ocr_only"}:
        logger.info("Strategy '%s' requested; using OCR directly.", strategy)
        return ocr_pdf_pages(pdf_path)

    # 1) LIGHT (always first)
    try:
        pages = extract_pdf_text_light(pdf_path)
        if _nonempty(pages):
            logger.info("Light extraction succeeded (%d pages).", len(pages))
            return pages
        else:
            logger.warning("Light extraction returned no usable text; escalating.")
    except Exception as e:
        logger.warning("Light extraction failed (%s); escalating.", e)

    # 2) UNSTRUCTURED FAST
    if _HAVE_UNSTRUCTURED:
        try:
            pages_json_fast = extract_pdf_text(pdf_path, return_elements=False, strategy="fast")
            pages = [p["text"] for p in pages_json_fast]
            if _nonempty(pages):
                logger.info("Unstructured fast succeeded (%d pages).", len(pages))
                return pages
            else:
                logger.warning("Unstructured fast empty; escalating to hi_res.")
        except Exception as e:
            logger.warning("Unstructured fast failed (%s); escalating to hi_res.", e)
    else:
        logger.info("Unstructured not available; skipping to OCR if hi_res not possible.")

    # 3) UNSTRUCTURED HI_RES (GPU-capable)
    if _HAVE_UNSTRUCTURED:
        try:
            _log_gpu_preflight()
            pages_json_hi = extract_pdf_text(pdf_path, return_elements=False, strategy="hi_res")
            pages = [p["text"] for p in pages_json_hi]
            if _nonempty(pages):
                logger.info("Unstructured hi_res succeeded (%d pages).", len(pages))
                return pages
            else:
                logger.warning("Unstructured hi_res empty; falling back to OCR.")
        except Exception as e:
            logger.warning("Unstructured hi_res failed (%s); falling back to OCR.", e)

    # 4) OCR (EasyOCR GPU if available → pytesseract)
    logger.warning("Falling back to OCR for %s", pdf_path)
    return ocr_pdf_pages(pdf_path)

# --------------------------- Document constructors ---------------------------
# NOTE: relies on your existing timer_wrap, compute_file_hash, num_tokens, Document
@timer_wrap
def build_document_object(pdf_path: str, strategy: str = "fast") -> Optional[Document]:
    if not os.path.exists(pdf_path):
        logger.error("PDF file does not exist: %s", pdf_path)
        return None

    doc_type = Path(pdf_path).suffix.lower()
    article_id = Path(pdf_path).stem

    if doc_type == ".pdf":
        pages = load_pdf_pages(pdf_path, strategy=strategy)
        total_char_length = sum(len(p) for p in pages)
        try:
            total_tokens = sum(num_tokens(p) for p in pages)
        except Exception:
            total_tokens = sum(len(p.split()) for p in pages)
        return Document(
            doi=article_id, full_text=pages, total_char_length=total_char_length,
            parsed_timestamp=datetime.now(timezone.utc).isoformat(),
            file_hash=compute_file_hash(Path(pdf_path)), file_path=pdf_path,
            n_pages=len(pages), total_tokens=total_tokens,
        )

    if doc_type == ".xml":
        full_text = extract_xml_text(pdf_path)
        total_char_length = len(full_text)
        try:
            total_tokens = num_tokens(full_text)
        except Exception:
            total_tokens = len(full_text.split())
        return Document(
            doi=article_id, full_text=full_text, total_char_length=total_char_length,
            parsed_timestamp=datetime.now(timezone.utc).isoformat(),
            file_hash=compute_file_hash(Path(pdf_path)), file_path=pdf_path,
            total_tokens=total_tokens,
        )

    logger.error("Unsupported document type: %s", doc_type)
    return None

# -------------------------- GPU worker (top-level) ---------------------------
def _build_document_object_on_gpu(args) -> Optional[Document]:
    """Picklable entrypoint for ProcessPoolExecutor. args=(pdf_path, gpu_id, strategy)"""
    pdf_path, gpu_id, strategy = args
    # Pin to specific GPU; if vLLM is on GPU 0, pass gpu_id=1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _log_gpu_preflight()
    # If PyTorch is present, pick the (now) local device 0
    if _HAVE_TORCH:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
        except Exception as e:
            logger.warning("torch.set_device failed on GPU %s: %s", gpu_id, e)
    return build_document_object(pdf_path, strategy=strategy)

# ----------------------------- multi-file driver -----------------------------
@timer_wrap
def build_document_objects(
    pdf_paths: List[str],
    subset: bool = False,
    subset_size: int = 20,
    max_workers: int = 8,
    strategy: str = "fast",
) -> List[Document]:
    import numpy as np, itertools, multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

    if subset and pdf_paths:
        np.random.seed(42)
        pdf_paths = np.random.choice(pdf_paths, subset_size, replace=False).tolist()

    if not pdf_paths:
        return []

    # Decide GPU assignment:
    # - By default, use GPU IDs from env PDF_GPU_IDS (e.g., "1" or "0,1").
    # - If not set and strategy is hi_res, try to use all visible GPUs.
    gpu_ids_env = os.environ.get("PDF_GPU_IDS")
    gpu_ids = None
    if strategy.lower() == "hi_res":
        if gpu_ids_env:
            gpu_ids = [int(x) for x in gpu_ids_env.split(",") if x.strip().isdigit()]
        elif _HAVE_TORCH:
            import torch  # type: ignore
            if torch.cuda.is_available():
                gpu_ids = list(range(min(2, torch.cuda.device_count())))  # up to two GPUs

    documents: List[Document] = []

    if gpu_ids:
        # Use spawn to keep CUDA happy inside subprocesses
        ctx = mp.get_context("spawn")
        logger.info("Using GPU workers on ids=%s", gpu_ids)
        tasks = [(p, gpu_ids[i % len(gpu_ids)], strategy) for i, p in enumerate(pdf_paths)]
        with ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=ctx) as ex:
            futs = [ex.submit(_build_document_object_on_gpu, t) for t in tasks]
            for f in as_completed(futs):
                try:
                    d = f.result()
                    if d is not None:
                        documents.append(d)
                except Exception as e:
                    logger.error("GPU worker exception: %s", e)
    else:
        # CPU threads (fast path, or no GPUs free)
        workers = min(max_workers, len(pdf_paths))
        logger.info("Using CPU ThreadPool with %d workers", workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(build_document_object, p, strategy=strategy): p for p in pdf_paths}
            for f in as_completed(futs):
                try:
                    d = f.result()
                    if d is not None:
                        documents.append(d)
                except Exception as e:
                    logger.error("Thread worker exception: %s", e)

    logger.info("Successfully processed %d / %d PDFs", len(documents), len(pdf_paths))
    return documents

__all__ = [
    "build_document_object", "build_document_objects",
    "extract_pdf_text", "extract_xml_text", "load_pdf_pages", "ocr_pdf_pages",
]
# --- END PATCH ---------------------------------------------------------------
