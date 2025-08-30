import logging
import os, sys, time, hashlib, json
from functools import lru_cache
from functools import wraps
import inspect
from typing import List, Any, Optional
import regex as re
from pydantic import BaseModel
from typing_extensions import get_type_hints
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
# import nltk
# # Ensure the NLTK stopwords corpus is available
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords', quiet=True)


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models import Document

# Ensure the logs directory exists
logs_dir = os.path.join(project_root, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Ensure the log file exists
helpers_log_file = os.path.join(logs_dir, 'helpers.log')
if not os.path.exists(helpers_log_file):
    with open(helpers_log_file, 'w') as f:
        pass # Create an empty file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(helpers_log_file, 'a')
    ]
)
logger = logging.getLogger(__name__)

def initialize_logging(log_file: str, project_root: str = project_root):
    # Configure logging
    os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)

    # Set up logging configuration
    log_filename = log_file.split(".")[0]
    log_file = os.path.join(project_root, 'logs', f"{log_filename}.log")
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized for {log_file}")
    logger.propagate = False
    return logger

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

def batch_list(lst, batch_size):
    """Yield successive batches from the list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def _all_iterables(items: List[Any]) -> bool:
    """Return *True* if every element is a list/tuple (but **not** a str)."""
    return all(isinstance(x, (list, tuple)) for x in items)

def flatten_results(func, results: list):
    """
    Flattens a list of results from parallel batches into a single output.
    
    For fields that are lists, the results are concatenated.
    For non-list fields, the first non-None and non-empty string value is taken.

    Rules applied in order:

    1. **Nested iterables** – If *every* element in ``results`` is a list/tuple,
       concatenate them into a single flat list.
    2. **Dict / Pydantic BaseModel** – If elements are dicts or BaseModels,
       merge them field‑wise: list‑typed fields are concatenated, scalar fields
       take the first non‑null/non‑empty value encountered.
    3. **Fallback** – For anything else, return ``results`` unchanged.
    
    Parameters:
        func: The original function (to extract its return type annotation)
        results: A list of results from the decorated function.
        
    Returns:
        A single merged result matching the expected output type.
    """
    # ------------------------------------------------------------------
    # Guard clauses & trivial cases
    # ------------------------------------------------------------------
    if not results:
        return None

    # ------------------------------------------------------------------
    # 1) Flatten one‑level nested lists produced by batched calls
    # ------------------------------------------------------------------
    if _all_iterables(results):
        flattened: List[Any] = []
        for batch in results:
            flattened.extend(batch)  # type: ignore[arg-type]
        logger.debug("Flattened %d nested batches into a single list.", len(results))
        return flattened

    # ------------------------------------------------------------------
    # 2) Merge dictionaries or Pydantic models
    # ------------------------------------------------------------------
    first = results[0]

    if isinstance(first, dict) or isinstance(first, BaseModel):
        # Convert everything to dict for uniform processing
        if isinstance(first, dict):
            dicts = results  # type: ignore[assignment]
            build_obj = lambda d: d  # noqa: E731
        else:
            dicts = [r.model_dump() for r in results]
            ret_type = get_type_hints(func).get("return", type(first))
            build_obj = (
                lambda d: ret_type(**d) if isinstance(ret_type, type) else first.__class__(**d)  # noqa: E731
            )

        merged: dict = {}
        for key in dicts[0].keys():
            sample_val = dicts[0][key]
            if isinstance(sample_val, list):
                # Concatenate list fields across batches
                merged[key] = [item for d in dicts for item in d.get(key, [])]
            else:
                # Take the first truthy scalar value
                merged[key] = next((d.get(key) for d in dicts if d.get(key) not in (None, "")), None)

        logger.debug("Merged %d dict/BaseModel batches into one object.", len(results))
        return build_obj(merged)

    # ------------------------------------------------------------------
    # 3) Fallback – leave result list untouched
    # ------------------------------------------------------------------
    logger.debug("Results contain heterogeneous or unsupported types; returning unchanged.")
    return results


# create generic parallel processing wrapper function
def parallel_processing_decorator(
        batch_size=10, 
        max_workers=5,  # Reduced concurrency from 5 to 3 by default.
        batch_param_name=None, 
        timeout=120,
        flatten=False,
    ):
    """
    Decorator to process function inputs in parallel using ThreadPoolExecutor.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine the argument to be batched.
            try:
                if batch_param_name:
                    if batch_param_name not in kwargs:
                        raise ValueError(f"Parameter '{batch_param_name}' not found in function arguments")
                    obj_list = kwargs[batch_param_name]
                else:
                    if len(args) == 0:
                        raise ValueError("No positional arguments provided to the function")
                    obj_list = args[0]
                if len(obj_list) == 0:
                    raise ValueError("No objects available to process")
                logger.info(f"Running {func.__name__} on {len(obj_list)} objects in batches of {batch_size}")

                results = []
                batches = list(batch_list(obj_list, batch_size))

                # Process batches in parallel using a ThreadPoolExecutor.
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_batch = {}
                    for batch in batches:
                        if batch_param_name:
                            # Instead of directly submitting func, we submit run_with_retries.
                            future = executor.submit(func, *args, **{batch_param_name: batch, **kwargs})
                        else:
                            future = executor.submit(func, batch, *args[1:], **kwargs)
                        future_to_batch[future] = batch

                    for future in as_completed(future_to_batch, timeout=timeout):
                        try:
                            result = future.result()
                            results.extend(result if isinstance(result, list) else [result])
                        except TimeoutError:
                            logger.warning("Processing batch timed out")
                        except Exception as e:
                            logger.error(f"Error processing batch: {e}")
                    if flatten:
                        try:
                            flat = flatten_results(func, results)
                            logger.info(f"Flattened results.")
                            return flat
                        except Exception as e:
                            logger.error(f"Error flattening results: {e}")
                            return results
                logger.info(f"Successfully executed {func.__name__} on {len(results)} objects")
                return results
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator

def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of XML file for debugging reference."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


# def num_tokens(text: str, model: str = "gpt-4o-mini") -> int:
#     import tiktoken
#     encoding = tiktoken.encoding_for_model(model)
#     return len(encoding.encode(text))

# helpers.py

_OFFLINE_ENC = None

def _get_offline_encoding():
    # Prefer fully offline, bundled encoding
    global _OFFLINE_ENC
    if _OFFLINE_ENC is not None:
        return _OFFLINE_ENC
    try:
        import tiktoken
        # cl100k_base is bundled; o200k_base may not be
        for name in ("cl100k_base", "o200k_base"):
            try:
                _OFFLINE_ENC = tiktoken.get_encoding(name)
                return _OFFLINE_ENC
            except Exception:
                pass
    except Exception:
        pass
    return None

def num_tokens(text: str, model: str = "o200k_base") -> int:
    # 1) Try bundled offline encoding immediately (no network)
    enc = _get_offline_encoding()
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # 2) If caller insists on a specific model or o200k_base is pre-cached
    try:
        import tiktoken
        if model in {"cl100k_base", "o200k_base"}:
            enc = tiktoken.get_encoding(model)
        else:
            # May fetch if not cached; avoided by default
            enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # 3) Fully offline fallback
        return max(1, (len(text.encode("utf-8")) + 3) // 4)

@timer_wrap
def export_docs(documents: List[Document], output_file: str = "documents.json", output_dir: Path = Path(os.path.join(project_root, "Data/train/"))) -> None:
    """
    Export the list of CitationEntity objects as plain dicts,
    avoiding double-encoded JSON strings.
    """
    outfile = str(output_dir / output_file)
    logger.info(f"Exporting {len(documents)} documents to {outfile}")
    # Use model_dump() to get a JSON-serializable dict directly
    to_export = [document.model_dump() for document in documents]
    with open(outfile, "w") as f:
        json.dump(to_export, f, indent=4)


def load_docs(input_file: str = os.path.join(project_root, "Data/train/documents.json")) -> List[Document]:
    """
    Load citation entities from a JSON file of dicts,
    reconstructing each via Pydantic’s model_validate.
    """
    logger.info(f"Loading citation entities from {input_file}")
    with open(input_file, "r") as f:
        raw = json.load(f)  # List[dict]
    # Validate and instantiate each model in one step
    return [Document.model_validate(item) for item in raw]

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



def get_embedding(text, model="text-embedding-3-small"):
    from openai import OpenAI
    client = OpenAI()
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding


class CustomOpenAIEmbedding:
    """Custom embedding class that wraps the get_embedding function for llama_index compatibility."""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text string."""
        return get_embedding(text, self.model)
    
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple text strings."""
        return [get_embedding(text, self.model) for text in texts]

def adjust_window_size(sentences: List[str], window_size: int, buffer: int) -> int:
    """
    This function adjusts the window size to account for long sentences. It calculates the new window size by rounding up the max sentence length to the nearest 100 and then adding the buffer. That is because the sliding_window_chunks function stalls if there is sentence with less than N difference between the current window size and the sentence length.

    Args:
        sentences (List[str]): A list of sentences
        window_size (int): The current window size
        buffer (int): The buffer to add to the window size. This is calculated as 3 times the overlap.
    Returns:
        int: The adjusted window size
    """
    import math
    # Check if any sentence is longer than the current window size
    max_sentence_length = max(len(sentence.split()) for sentence in sentences)
    # round up to nearest 100
    new_window_size = math.ceil(max_sentence_length / 100) * 100 + buffer
    logger.info(f"Adjusted window_size from {window_size} to {new_window_size} due to a sentence with {max_sentence_length} words")
    return new_window_size

@timer_wrap
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
    
# -----------------------------------------------------------------------------
# Citation matching helper (parity with src/get_citation_entities.py)
# -----------------------------------------------------------------------------

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
    
    