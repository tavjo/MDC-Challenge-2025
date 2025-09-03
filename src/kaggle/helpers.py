# General Helpers
# src/kaggle/helpers.py

"""
Helper functions for the Kaggle competition.
"""

from typing import List, Optional
import os, sys
import time
import inspect
from functools import wraps
import logging
import numpy as np
from pathlib import Path
import hashlib
import re
from functools import lru_cache


baml_wrapper = '/kaggle/input/baml-components/src/baml_wrapper'
sys.path.append(str(Path(baml_wrapper).parent))
from baml_wrapper import extract_cites
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
from models import CitationEntity, DatasetType


def initialize_logging(filename:str= "kaggle-mdc") -> logging.Logger:
    """
    Initialize logging for a given filename.
    """
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    # Remove any file handlers to ensure no log files are created
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
    # Add a stdout stream handler if not already present
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    # Avoid double logging when root logger is configured in notebooks
    logger.propagate = False
    return logger

logger = initialize_logging()

def ensure_dir(path) -> Path:
    """
    Create a directory if it doesn't exist. Returns a Path.
    Expands '~' and env vars.
    """
    p = Path(os.path.expanduser(os.path.expandvars(str(path))))
    p.mkdir(parents=True, exist_ok=True)
    return p

def ensure_dirs(*paths) -> list[Path]:
    """
    Create multiple directories if they don't exist. Returns list[Path].
    """
    return [ensure_dir(p) for p in paths]

def timer_wrap(func):
    if inspect.iscoroutinefunction(func):
        # Handle async functions
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            print(f"Executing {func.__name__}...")
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Function {func.__name__} took {elapsed_time:.4f} seconds to complete.")
            return result
        return async_wrapper
    elif inspect.isasyncgenfunction(func):
        # Handle async generator functions
        @wraps(func)
        async def async_gen_wrapper(*args, **kwargs):
            print(f"Executing {func.__name__} (async generator)...")
            start_time = time.time()
            async for item in func(*args, **kwargs):
                yield item
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Async generator {func.__name__} took {elapsed_time:.4f} seconds to complete.")
        return async_gen_wrapper
    else:
        # Handle sync functions
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            print(f"Executing {func.__name__}...")
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Function {func.__name__} took {elapsed_time:.4f} seconds to complete.")
            return result
        return sync_wrapper

# helpers.py

# --- offline-safe token counting (no network attempts) ---
# --- offline-safe token counting that RESPECTS `model` ---
import os, hashlib, pathlib

# Known base encodings we support offline
_BASES = {
    "cl100k_base": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    "o200k_base":  "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
}

def _cache_has(enc_name: str) -> bool:
    cache_dir = os.getenv("TIKTOKEN_CACHE_DIR")
    if not cache_dir or enc_name not in _BASES:
        return False
    h = hashlib.sha1(_BASES[enc_name].encode()).hexdigest()
    return pathlib.Path(cache_dir, h).exists()

def _resolve_model_to_encoding(model: str) -> str:
    """Pure-offline best effort: explicit base → use as-is; else map common models."""
    # 1) If caller passed a base name, use it directly
    if model in _BASES:
        return model
    # 2) Heuristics for modern models when tables lag:
    if not enc:
        # Most “o* / 4o / 4.1” models use o200k_base; classic chat/embeddings use cl100k_base
        lower = model.lower()
        if lower.startswith(("gpt-4o", "o4", "4.1", "o1")):
            enc = "o200k_base"
        else:
            enc = "cl100k_base"
    return enc

def get_encoding_name_for_model(model: str = "cl100k_base") -> str:
    """Helper: what we *intend* to use (even if cache missing)."""
    return _resolve_model_to_encoding(model)

_OFFLINE_ENC = {}  # cache of loaded encodings by name

def _get_offline_encoding(enc_name: str):
    """Return a tiktoken Encoding if the cache has the file; otherwise None."""
    if enc_name in _OFFLINE_ENC:
        return _OFFLINE_ENC[enc_name]
    try:
        import tiktoken
        if _cache_has(enc_name):
            _OFFLINE_ENC[enc_name] = tiktoken.get_encoding(enc_name)  # no network when cached
            return _OFFLINE_ENC[enc_name]
    except Exception:
        pass
    return None

def num_tokens(text: str, model: str = "o200k_base") -> int:
    enc_name = get_encoding_name_for_model(model)
    enc = _get_offline_encoding(enc_name)
    if enc:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    # Offline fallback (≈ bytes/4)
    return max(1, (len(text.encode("utf-8")) + 3) // 4)

def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of XML file for debugging reference."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


def sliding_window_chunks(text: str, window_size: int = 300, overlap: int = 30) -> List[str]:
    """
    Split the input text into sliding window chunks based on word count.
    """
    logger.info(f"Creating chunks with window size {window_size} and overlap {overlap}")
    # Normalize whitespace and split into words
    words = text.replace('\n', ' ').split()
    chunks = []
    start = 0
    total_words = len(words)
    # Create chunks with specified overlap
    while start < total_words:
        end = min(start + window_size, total_words)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        # Move start by window_size minus overlap
        start += window_size - overlap
    # Merge any chunks smaller than twice the overlap into the previous chunk
    min_size = overlap * 2
    refined_chunks: List[str] = []
    for ch in chunks:
        word_count = len(ch.split())
        if refined_chunks and word_count < min_size:
            refined_chunks[-1] += " " + ch
        else:
            refined_chunks.append(ch)
    chunks = refined_chunks
    logger.info(f"Successfully created {len(chunks)} chunks after merging small fragments")
    return chunks

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between A [N,D] and B [M,D]."""
    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A @ B.T  # [N, M]

# --- Embedding with local BGE-small ---
def load_bge_model(local_dir: str | Path):
    """
    Load a local, pre-downloaded BGE-small v1.5 SentenceTransformers model (offline-friendly).
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(local_dir))  # local path works offline
    return model

def embed_texts(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode texts with L2-normalized embeddings for cosine similarity.
    """
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,  # recommended for BGE
        show_progress_bar=True,
    )
    return emb

def extract_entities_baml(doc: List[str], doc_id: str) -> List[CitationEntity]:
    """
    Extract citation entities from the document text using the BAML client.
    """
    logger.info(f"Extracting citation entities using BAML client for {doc_id}.")
    citations = extract_cites(doc)
    if citations:
        citation_entities = [
            CitationEntity.model_validate({**entity.model_dump(mode="json"), "document_id": doc_id})
            for entity in citations
        ]
        return citation_entities
    else:
        logger.warning("No citation entities found using BAML client.")
        return []

# -----------------------------------------------------------------------------
# Citation matching helper (parity with src/get_citation_entities.py)
# -----------------------------------------------------------------------------

def clean_text_for_urls(text: str) -> str:
    import re
    """
    Clean text to normalize URLs that may be broken up by spaces.
    """
    # remove doi prefix
    text = re.sub('https://doi.org', '', text)
    text = re.sub('https://dx.doi.org', '', text)
    text = re.sub('http://dx.doi.org', '', text)
    text = re.sub('http://doi.org', '', text)
    text = re.sub('https://doi.org', '', text)
    # Fix common URL breakage patterns
    # Replace "dx.doi. org" with "dx.doi.org" (space after dot)
    text = re.sub(r'dx\.doi\.\s+org', 'dx.doi.org', text)
    # Replace "doi. org" with "doi.org" (space after dot)
    text = re.sub(r'doi\.\s+org', 'doi.org', text)
    # Replace "http://dx.doi. org" with "http://dx.doi.org"
    text = re.sub(r'http://dx\.doi\.\s+org', 'http://dx.doi.org', text)
    # Replace "https://dx.doi. org" with "https://dx.doi.org"
    text = re.sub(r'https://dx\.doi\.\s+org', 'https://dx.doi.org', text)
    # Replace "http://doi. org" with "http://doi.org"
    text = re.sub(r'http://doi\.\s+org', 'http://doi.org', text)
    # Replace "https://doi. org" with "https://doi.org"
    text = re.sub(r'https://doi\.\s+org', 'https://doi.org', text)
    
    return text

def normalise(page:str) -> str:
    page = page.replace('-\n', '')   # undo soft-hyphen splits
    page = page.replace('\n', ' ')
    return clean_text_for_urls(page)



def preprocess_text(text):
    import re
    # from nltk.corpus import stopwords
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Lowercase
    text = text.replace("\n", " ")
    # stop_words = set(stopwords.words("english"))
    # text = " ".join(word for word in text.split() if word not in stop_words)
    return text

def _alnum_collapse(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())

@lru_cache(maxsize=8192)
def _generate_id_variants(dataset_id: str) -> List[str]:
    variants = set()
    raw = dataset_id.strip()
    variants.add(raw)
    cleaned = clean_text_for_urls(raw)
    variants.add(cleaned)
    host_stripped = re.sub(r"(?i)^https?://(?:dx\.)?doi\.org/", "", cleaned)
    variants.add(host_stripped)
    no_ver = re.sub(r"(?i)\.v\d+$", "", host_stripped)
    variants.add(no_ver)
    no_tbl = re.sub(r"(?i)/t\d+$", "", no_ver)
    variants.add(no_tbl)
    no_trailing_dotnum = re.sub(r"(?i)\.(\d+)$", "", no_tbl)
    variants.add(no_trailing_dotnum)
    # E-GEOD ↔ GSE synonym
    if re.match(r"(?i)^E-GEOD-\d+$", raw):
        variants.add(re.sub(r"(?i)^E-GEOD-", "GSE", raw))

    # Accession-style split variants
    m = re.match(r"(?i)^(PXD|SRR|ERR)(\d+)$", raw)
    if m:
        prefix, digits = m.group(1).upper(), m.group(2)
        variants.add(f"{prefix}-{digits}")
        variants.add(f"{prefix} {digits}")

    m = re.match(r"(?i)^E-?PROT-?(\d+)$", raw)
    if m:
        digits = m.group(1)
        variants.add(f"E-PROT-{digits}")
        variants.add(f"E PROT {digits}")
        variants.add(f"EPROT{digits}")

    m = re.match(r"(?i)^(ENS)([A-Z]+)(\d+)$", raw)
    if m:
        prefix, letters, digits = m.group(1).upper(), m.group(2).upper(), m.group(3)
        variants.add(f"{prefix}-{letters}-{digits}")
        variants.add(f"{prefix} {letters} {digits}")

    # Generic short accession split
    g = re.match(r"(?i)^([A-Z0-9]{2,6})(\d{3,})$", raw)
    if g:
        head, tail = g.groups()
        variants.add(f"{head}-{tail}")
        variants.add(f"{head} {tail}")

    return [v for v in variants if v]


def _join_chars_with_seps(text: str) -> str:
    pieces = [re.escape(c) for c in text]
    return r"[\s\u00A0\-\u2010-\u2015_,.]*".join(pieces)


@lru_cache(maxsize=16384)
def _build_accession_pattern(ds_id: str) -> Optional[re.Pattern]:
    s = ds_id.strip()
    flags = re.IGNORECASE

    # PXD / SRR / ERR
    m = re.match(r"(?i)^(PXD|SRR|ERR)(\d+)$", s)
    if m:
        prefix, digits = m.group(1).upper(), m.group(2)
        prefix_pat = _join_chars_with_seps(prefix)
        digit_pat = rf"(?:\d[\s\u00A0\-\u2010-\u2015_,.]*){{{len(digits)}}}"
        pat = rf"\b{prefix_pat}[\s\u00A0\-\u2010-\u2015]*{digit_pat}\b"
        return re.compile(pat, flags=flags)

    # E-PROT-#
    m = re.match(r"(?i)^E-?PROT-?(\d+)$", s)
    if m:
        digits = m.group(1)
        e_pat = _join_chars_with_seps("E")
        prot_pat = _join_chars_with_seps("PROT")
        digit_pat = rf"(?:\d[\s\u00A0\-\u2010-\u2015_,.]*){{{len(digits)}}}"
        seps = r"[\s\u00A0\-\u2010-\u2015]*"
        pat = rf"\b{e_pat}{seps}{prot_pat}{seps}-?{seps}{digit_pat}\b"
        return re.compile(pat, flags=flags)

    # ENS[A-Z]+#
    m = re.match(r"(?i)^(ENS)([A-Z]+)(\d+)$", s)
    if m:
        prefix, letters, digits = m.group(1).upper(), m.group(2).upper(), m.group(3)
        prefix_pat = _join_chars_with_seps(prefix)
        letters_pat = r"[\s\u00A0\-\u2010-\u2015_,.]*".join(re.escape(ch) for ch in letters)
        digit_pat = rf"(?:\d[\s\u00A0\-\u2010-\u2015_,.]*){{{len(digits)}}}"
        pat = rf"\b{prefix_pat}[\s\u00A0\-\u2010-\u2015]*{letters_pat}[\s\u00A0\-\u2010-\u2015]*{digit_pat}\b"
        return re.compile(pat, flags=flags)

    # UniProt 6 or 10 chars
    uniprot_re = re.compile(
        r"(?ix)^(?: (?:[OPQ][0-9][A-Z0-9]{3}[0-9]) | (?:[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]) )(?:[A-Z0-9]{3}[0-9])?$"
    )
    if uniprot_re.match(s):
        per_char = _join_chars_with_seps(s)
        pat = rf"\b{per_char}\b"
        return re.compile(pat, flags=flags)

    # Generic accession: short head + digits
    if re.match(r"(?i)^[A-Z0-9]{2,6}\d{3,}$", s):
        per_char = _join_chars_with_seps(s)
        pat = rf"\b{per_char}\b"
        return re.compile(pat, flags=flags)

    return None


@lru_cache(maxsize=16384)
def _make_pattern(ds_id: str) -> re.Pattern:
    acc = _build_accession_pattern(ds_id)
    if acc is not None:
        return acc
    ds_id_norm = clean_text_for_urls(ds_id).strip()
    tokens = [t for t in re.split(r"[-_/:.\s]+", ds_id_norm) if t]
    if not tokens:
        tokens = [ds_id_norm]
    sep = r"[\s\u00A0\-\u2010-\u2015_/:.,]*"
    pat = sep.join(re.escape(t) for t in tokens)
    if re.fullmatch(r"[A-Z]{1,6}\d+[A-Z0-9]*", ds_id, re.I):
        pat = rf"\b{pat}\b"
    return re.compile(pat, flags=re.IGNORECASE)


def _build_text_views(text: str) -> dict:
    raw = text
    norm = normalise(text)
    prep = preprocess_text(text)
    prep_norm = preprocess_text(norm)
    alnum = _alnum_collapse(text)
    return {"raw": raw, "norm": norm, "prep": prep, "prep_norm": prep_norm, "alnum": alnum}


def _find_direct_string_matches(citation: str, text_views: dict) -> List[str]:
    cit_norm = clean_text_for_urls(citation).lower()
    lower_cit = re.escape(cit_norm)
    matches: List[str] = []
    matches.extend(re.findall(lower_cit, text_views["raw"].lower()))
    matches.extend(re.findall(lower_cit, text_views["norm"].lower()))
    matches.extend(re.findall(lower_cit, text_views["prep"]))
    return matches

# NEW: detect hyphenated accession ranges containing the target ID (e.g., MK838495–MK838499)
def _id_in_hyphenated_range(citation: str, text_views: dict) -> bool:
    import re
    cit = clean_text_for_urls(citation).strip()
    m = re.match(r"(?i)^\s*([A-Z]{2,10})[\s\u00A0\-\u2010-\u2015_]*([0-9]{3,})\s*$", cit)
    if not m:
        return False
    prefix, digits = m.group(1).upper(), m.group(2)
    target = int(digits)

    prefix_pat = _join_chars_with_seps(prefix)
    sep = r"[\s\u00A0\-\u2010-\u2015_,.]*"
    dash = r"[\-\u2010-\u2015]"

    # Search in views that preserve dashes
    rng = re.compile(
        rf"\b{prefix_pat}{sep}(\d{{3,}}){sep}{dash}{sep}(?:{prefix_pat}{sep})?(\d+)\b",
        re.IGNORECASE,
    )
    for key in ("raw", "norm"):
        page = text_views[key]
        for start_digits, end_digits in rng.findall(page):
            try:
                start_num = int(start_digits)
            except ValueError:
                continue
            if len(end_digits) < len(start_digits):
                # Expand partial RHS like 95 -> 838495 using LHS prefix
                end_full = int(start_digits[: len(start_digits) - len(end_digits)] + end_digits)
            else:
                try:
                    end_full = int(end_digits)
                except ValueError:
                    continue
            lo, hi = (start_num, end_full) if start_num <= end_full else (end_full, start_num)
            if lo <= target <= hi:
                return True
    return False


def _id_in_shared_prefix_list(citation: str, text_views: dict) -> bool:
    """Detect lists where a single textual prefix applies to multiple numeric IDs.

    Example: "CCDC 1951650 {..}, 1951651 {..}, 1951652 {..}, and 1951653 {..}"
    For citation "CCDC 1951652", return True if 1951652 appears in the list.
    """
    import re
    cit = clean_text_for_urls(citation).strip()
    m = re.match(r"(?i)^\s*([A-Z]{2,10})[\s\u00A0\-\u2010-\u2015_]*([0-9]{3,})\s*$", cit)
    if not m:
        return False
    prefix, digits = m.group(1).upper(), m.group(2)
    target_str = digits

    prefix_pat = _join_chars_with_seps(prefix)
    sep = r"[\s\u00A0\-\u2010-\u2015_,.]*"
    anchor = re.compile(rf"\b{prefix_pat}{sep}(\d{{3,}})\b", re.IGNORECASE)

    # Search in views that preserve punctuation/case best
    for key in ("raw", "norm"):
        page = text_views[key]
        for match in anchor.finditer(page):
            # Include the first number right after the prefix
            if match.group(1) == target_str:
                return True
            # Look ahead for nearby numbers in the same comma/and-separated list
            tail = page[match.end(): match.end() + 300]
            # Trim at likely list terminators
            stop = re.search(r"[.;](?:\s|$)", tail)
            if stop:
                tail = tail[: stop.start()]
            # Find numbers of the same length as the target (avoid short numbers in braces)
            same_len_nums = re.findall(rf"\b\d{{{len(target_str)}}}\b", tail)
            if target_str in same_len_nums:
                return True
    return False


def find_citation_matches(citation: str, text: str) -> List[str]:
    """Return list of matches for a citation within text using tolerant logic.

    Attempts, in order:
    - Accession/variant-aware regex over raw, normalised, preprocessed, prep(norm)
    - Direct substring over raw/normalised/preprocessed
    - Collapsed alphanumeric substring fallback (returns synthetic single match)
    """
    text_views = _build_text_views(text)

    # 1) regex over variants and views
    for variant in _generate_id_variants(citation):
        pattern = _make_pattern(variant)
        for key in ("raw", "norm", "prep", "prep_norm"):
            found = pattern.findall(text_views[key])
            if found:
                return found

    # 1b) hyphenated numeric ranges like MK838495–MK838499 or CCDC 1951652–1951653
    if _id_in_hyphenated_range(citation, text_views):
        return [citation]

    # 1c) shared-prefix lists like: CCDC 1951650, 1951651, 1951652, 1951653
    if _id_in_shared_prefix_list(citation, text_views):
        return [citation]

    # 2) direct substring fallback (robust text normalisation)
    direct = _find_direct_string_matches(citation, text_views)
    if direct:
        return direct

    # 3) collapsed alnum fallback for long IDs/DOIs
    id_alnum = _alnum_collapse(citation)
    id_alnum_host_stripped = _alnum_collapse(re.sub(r"(?i)^https?://(?:dx\.)?doi\.org/", "", citation))
    if citation.startswith("10.") or len(id_alnum) >= 12 or id_alnum_host_stripped:
        page_alnum = text_views["alnum"]
        if id_alnum and id_alnum in page_alnum:
            return [citation]
        if id_alnum_host_stripped and id_alnum_host_stripped in page_alnum:
            return [citation]

    return []