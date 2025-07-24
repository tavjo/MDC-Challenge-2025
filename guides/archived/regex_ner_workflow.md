Below is a **code-aware, drop-in upgrade path** that starts from the two functions you shared (`_get_known_entities`, `_get_unknown_entities`) and walks all the way to a cleaner list of `CitationEntity` objects. Everything is organised so that the output of one step feeds directly into the next, and it can run on a 2019 Intel MBP with no GPU.

---

## 0  Resources to add to your repo

| Resource                               | Where/How to add                                                                                                                                           |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Bioregistry dump** (`registry.json`) | Download once per run from `https://bioregistry.io/registry.json`; save in `resources/bioregistry/`.                                                       |
| **`patterns.yaml`**                    | Tiny YAML file you curate that maps each *prefix* you want <br>(`GEO`, `SRR`, `ERR`, `DRR`, `MODEL`, etc.) to a refined regex body (see Step 2).           |
| **Stop-word list**                     | `nltk` stop-words (`pip install nltk && nltk.download('stopwords')`).                                                                                      |
| **scispaCy small model**               | `pip install scispacy && pip install https://huggingface.co/allenai/scispacy_models/resolve/main/en_core_sci_sm-0.5.3.tar.gz`; loads in CPU RAM (\~75 MB). |
| **Lookup TSVs**                        | `resources/lookups/{geo.tsv, ena_run.tsv, biomodels.tsv, …}` — one accession ID per line; create once with NCBI/E-utilities scripts.                       |

The rest uses only Python std-lib (`json`, `pathlib`, `re`) and your existing Pydantic `CitationEntity`.

---

## 1  Pattern build step (runs at import time)

```python
# src/patterns.py
import json, yaml, re, pathlib, datetime, requests
from typing import Dict

PAT_DIR = pathlib.Path("resources/bioregistry")
PAT_DIR.mkdir(parents=True, exist_ok=True)
REG_FILE = PAT_DIR / f"registry_{datetime.date.today()}.json"

if not REG_FILE.exists():
    REG_FILE.write_bytes(requests.get(
        "https://bioregistry.io/registry.json", timeout=30).content)

_bioreg = json.loads(REG_FILE.read_text())

# --- your curated overrides / additions ---
_core = yaml.safe_load(open("resources/patterns.yaml"))

def _tighten(body: str) -> str:
    """Add global guards: >=1 upper, >=1 digit, len>=5"""
    return rf"(?=[A-Z]*\d)(?=\w*[A-Z])[A-Z0-9_-]{{5,}}(?:{body})?"

PATTERNS: Dict[str, re.Pattern] = {}
for prefix, body in _core.items():
    # use your override if present, else take Bioregistry pattern
    src = body or _bioreg[prefix]["pattern"]
    PATTERNS[prefix] = re.compile(rf"\b{_tighten(src)}\b")
```

*One-liner guard* removes everyday words immediately.

---

## 2  Refactor `_get_unknown_entities`

```python
STOP = set(nltk.corpus.stopwords.words("english"))
TRIGGERS = {
    "dataset", "accession", "deposited", "released",
    "retrieved", "GEO", "SRA", "BioProject", "archive"
}

def _passes_lexical_filters(token: str) -> bool:
    if token.lower() in STOP: 
        return False
    if token.isdigit() and len(token) <= 4:   # likely a year
        return False
    if not any(c.isdigit() for c in token):
        return False
    return True
```

Replace your loop body with:

```python
for idx, page in enumerate(pages):
    page_num = idx + 1
    for name, pat in PATTERNS.items():
        for m in pat.finditer(page):
            candidate = m.group(0)
            if not _passes_lexical_filters(candidate):
                continue
            window = page[max(0, m.start()-250): m.end()+250]
            if not any(t in window for t in TRIGGERS):
                continue
            # --- optional NER sanity check ---
            if self.use_ner:
                if not self._looks_like_dataset(window):
                    continue
            entity_pages.setdefault(candidate, set()).add(page_num)
```

Where

```python
def _looks_like_dataset(self, context: str) -> bool:
    doc = self._nlp(context)          # scispaCy small model
    return any(ent.label_ in {"RESOURCE", "DATASET"} for ent in doc.ents)
```

and initialise once:

```python
self._nlp = spacy.load("en_core_sci_sm", disable=["parser","tagger"])
self.use_ner = True
```

---

## 3  Lookup validation (optional but fast)

```python
def _in_lookup(candidate: str) -> bool:
    prefix = candidate.split(':', 1)[0] if ':' in candidate else candidate[:3]
    tsv_file = pathlib.Path(f"resources/lookups/{prefix.lower()}.tsv")
    if tsv_file.exists():
        if not hasattr(_in_lookup, "_cache"):
            _in_lookup._cache = {}
        if prefix not in _in_lookup._cache:
            _in_lookup._cache[prefix] = set(tsv_file.read_text().splitlines())
        return candidate in _in_lookup._cache[prefix]
    return True   # no lookup → trust regex
```

Call this right before you add `candidate` to `entity_pages`.

---

## 4  Known-ID path (`_get_known_entities`) — only one tweak

Replace your raw pattern construction line with:

```python
pattern = PATTERNS.get(dataset_id.split(':',1)[0])
if pattern is None:
    pattern = re.compile(rf"\b{re.escape(dataset_id)}\b")
```

so that your **training scans** benefit from the same guard logic (length, digits, etc.) and page-tracking stays consistent.

---

## 5  Unit tests & metrics

Create `tests/test_extractor.py`:

```python
def test_year_filtered():
    hits = extract("The study (2018) deposited SRR1234567 ...")
    assert "2018" not in hits and "SRR1234567" in hits

def test_stopword_filtered():
    hits = extract("Data from SRR777777 are available; from this ...")
    assert "from" not in hits
```

Add a pytest fixture that loads 5 labelled PDFs and asserts **precision ≥ 0.80**.

---

## 6  CLI / notebook driver

```bash
python -m src.extract \
    --pdf-dir Data/train/PDF \
    --labels Data/train/train_labels.csv \
    --pattern-file resources/patterns.yaml \
    --save-json outputs/citation_entities_known.json
```

---

## 7  Logging & traceability

* Add the SHA-256 of `patterns.yaml` and the Bioregistry file name to the JSON metadata for each run—helps you reproduce results later.
* At `INFO` level log:

  * `raw_matches`, `after_lexical`, `after_trigger`, `after_ner`, `after_lookup`.
  * time per article (so you can prove sub-9-hour runtime for Kaggle rules ).

---

### How outputs flow

1. **Pattern build (Step 1)** → `PATTERNS` dict
2. **Unknown-ID scanning (Step 2)** → `entity_pages{ID → pages}`
3. **Lookup pruning (Step 3)** → cleaned `entity_pages`
4. **Page-tracking & dedup (existing code)** → list `CitationEntity`
5. **Training path (Step 4 tweak)** → uniform precision with known IDs

Every step uses only resources from **0** and feeds a strictly smaller candidate list into the next, maximising precision before the list is written to disk.

---

#### Ready to implement?

If any of the above pieces do not align with your file layout or class names, just let me know which ones—everything is modular and easy to rename or relocate.
