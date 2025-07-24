## Detailed Implementation Plan for Entity Extraction Upgrade

### 📋 Implementation Checklist

#### Phase 1: Create New Pattern Management System
- [x] **Step 1.1**: Create `src/patterns.py` - Master pattern assembly system
- [x] **Step 1.2**: Create `artifacts/citation_patterns.yaml` - Curated pattern overrides

#### Phase 2: Add Lexical Filtering
- [x] **Step 2.1**: Update `src/get_citation_entities.py` - Add imports and constants
- [x] **Step 2.2**: Update `_get_unknown_entities` method - Add lexical filtering
- [x] **Step 2.3**: Update `_get_known_entities` method - Use new pattern system

#### Phase 3: Add Optional NER Validation
- [x] **Step 3.1**: Update `CitationEntityExtractor.__init__` - Add NER support
- [x] **Step 3.2**: Add `_looks_like_dataset` method - NER validation
- [x] **Step 3.3**: Add NER check to `_get_unknown_entities` - Integration

#### Phase 4: Add Lookup Validation (Optional)
- [x] **Step 4.1**: Create `_in_lookup` function - Lookup validation
- [x] **Step 4.2**: Add lookup check to `_get_unknown_entities` - Integration

#### Phase 5: Update Pattern Integration
- [x] **Step 5.1**: Update imports in `get_citation_entities.py` - Use new patterns
- [x] **Step 5.2**: Update pattern usage throughout - Replace old references

#### Phase 6: Add Logging and Metrics
- [x] **Step 6.1**: Add metrics tracking to `_get_unknown_entities` - Performance monitoring

#### Phase 7: Create Required Directories and Files
- [x] **Step 7.1**: Create directory structure - `artifacts/bioregistry/`, `artifacts/lookups/`
- [x] **Step 7.2**: Update `pyproject.toml` dependencies - Add nltk, scispacy

#### Phase 8: Testing and Validation
- [x] **Step 8.1**: Create basic tests - `tests/test_extractor.py`
- [ ] **Step 8.2**: Test year filtering - Verify years are filtered out
- [ ] **Step 8.3**: Test stopword filtering - Verify stopwords are filtered out
- [ ] **Step 8.4**: Test underscore regression - Verify SRR_123456 matches
- [ ] **Step 8.5**: Test precision guard - Verify only valid entities extracted

#### Final Validation
- [ ] **Integration Test**: Run full pipeline with new system
- [ ] **Performance Test**: Verify no significant performance regression
- [ ] **Documentation**: Update any relevant documentation
- [ ] **Deployment**: Deploy to production environment

---

### **Phase 1: Create New Pattern Management System**

**Step 1.1: Create `src/patterns.py`**
```python
# New file: src/patterns.py
"""
Assemble the master PATTERNS dict:
1.  start with DEFAULT_PATTERNS from your existing code
2.  overlay any curated tweaks in citation_patterns.yaml  (optional)
3.  extend with selected Bioregistry regexes
4.  wrap everything with a global '>=1 digit, >=1 uppercase, len>=5' guard
"""
from __future__ import annotations
import json, re, requests, pathlib, yaml, datetime
from typing import Dict

# -------- 1.1  your in-repo defaults  -----------------
from src.update_patterns import DEFAULT_PATTERNS      # <- your current dict

# -------- 1.2  optional manual overrides  -------------
OVERRIDE_FILE = pathlib.Path("artifacts/citation_patterns.yaml")
overrides: Dict[str,str] = yaml.safe_load(OVERRIDE_FILE.read_text()) if OVERRIDE_FILE.exists() else {}

# -------- 1.3  Bioregistry slice  ---------------------
BIOREG_DIR = pathlib.Path("artifacts/bioregistry")
BIOREG_DIR.mkdir(parents=True, exist_ok=True)
BIOREG_JSON = BIOREG_DIR / f"registry_{datetime.date.today()}.json"

# Download with fallback to cached copy
if not BIOREG_JSON.exists():
    try:
        BIOREG_JSON.write_bytes(requests.get("https://bioregistry.io/registry.json", timeout=30).content)
    except requests.exceptions.RequestException as e:
        # Fallback to most recent cached copy
        cached_files = list(BIOREG_DIR.glob("registry_*.json"))
        if cached_files:
            most_recent = max(cached_files, key=lambda f: f.stat().st_mtime)
            import shutil
            shutil.copy2(most_recent, BIOREG_JSON)
        else:
            raise FileNotFoundError(f"Bioregistry download failed and no cached copy found: {e}")

bioreg = json.loads(BIOREG_JSON.read_text())

# keep only prefixes you *don't* already define
bioreg_subset = {
    k.upper(): v["pattern"]           # 'pattern' field documented in Bioregistry schema
    for k, v in bioreg.items()
    if k.upper() not in DEFAULT_PATTERNS and k.upper() not in overrides
}

# -------- 1.4  helper to add global guard  ------------
def _tighten(pattern_body: str) -> str:
    # ≥1 digit, ≥1 uppercase, min len 5 - use lookahead to avoid forcing preamble
    # Note: \b treats underscore as word boundary, so we use explicit boundary check
    return rf"(?=[A-Z]*\d)(?=[A-Z0-9_-]*[A-Z])[A-Z0-9_-]{{5,}}(?:{pattern_body})?"

# -------- 1.5  compile master dict with precedence ----
PATTERNS: Dict[str, re.Pattern] = {}

#   (a) defaults first  -> lowest priority (use as-is, don't re-wrap)
for key, pat in DEFAULT_PATTERNS.items():
    PATTERNS[key.upper()] = pat  # Keep compiled pattern as-is

#   (b) YAML overrides  -> overwrite defaults (apply _tighten to raw strings)
for key, body in overrides.items():
    PATTERNS[key.upper()] = re.compile(rf"(?<![A-Za-z0-9_]){_tighten(body)}(?![A-Za-z0-9_])", re.I)

#   (c) Bioregistry extras -> only if new key (apply _tighten to raw strings)
for key, body in bioreg_subset.items():
    PATTERNS[key.upper()] = re.compile(rf"(?<![A-Za-z0-9_]){_tighten(body)}(?![A-Za-z0-9_])", re.I)
```

**Step 1.2: Create `artifacts/citation_patterns.yaml`**
```yaml
# New file: artifacts/citation_patterns.yaml
# Your curated pattern overrides - start with a few key ones
GEO: "GSE\\d{3,6}"
SRA_Run: "SRR\\d{5,}"
ENA_Run: "ERR\\d{6,}"
DDBJ_Run: "DRR\\d{6,}"
```

### **Phase 2: Add Lexical Filtering**

**Step 2.1: Update `src/get_citation_entities.py` - Add imports and constants**
```python
# Add to existing imports
import nltk
import re
from nltk.corpus import stopwords

# Download stopwords on startup (only if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Add after existing constants
STOP = set(nltk.corpus.stopwords.words("english"))
TRIGGERS = {
    "dataset", "accession", "deposited", "released",
    "retrieved", "GEO", "SRA", "BioProject", "archive", 
    "stored", "repository", "downloaded", "accession"
}

def _passes_lexical_filters(token: str) -> bool:
    # Normalize token: strip trailing punctuation and convert to lowercase
    tok = re.sub(r"\W+$", "", token).lower()
    
    if tok in STOP: 
        return False
    if tok.isdigit() and len(tok) <= 4:   # likely a year
        return False
    if not any(c.isdigit() for c in tok):
        return False
    return True
```

**Step 2.2: Update `_get_unknown_entities` method**
Replace the existing loop body in `_get_unknown_entities` with:

```python
for idx, page in enumerate(pages):
    page_num = idx + 1
    logger.info(f"Processing page {page_num} of {num_pages}")
    
    for name, pat in PATTERNS.items():
        for m in pat.finditer(page):
            candidate = m.group(0)
            if not _passes_lexical_filters(candidate):
                continue
            window = page[max(0, m.start()-250): m.end()+250]
            if not any(t in window.lower() for t in TRIGGERS):
                continue
            
            if candidate not in entity_pages:
                entity_pages[candidate] = set()
            entity_pages[candidate].add(page_num)
```

**Step 2.3: Update `_get_known_entities` method**
Replace the pattern construction line with:

```python
pattern = PATTERNS.get(dataset_id.split(':',1)[0])
if pattern is None:
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(dataset_id)}(?![A-Za-z0-9_])")
```

### **Phase 3: Add Optional NER Validation**

**Step 3.1: Add NER support to `CitationEntityExtractor.__init__`**
```python
def __init__(self, 
             data_dir: str = "Data", 
             known_entities: bool = True,
             labels_file: Optional[str] = "train_labels.csv", 
             draw_subset: bool = False,
             subset_size: Optional[int] = None,
             use_ner: bool = False):  # Add this parameter
    # ... existing code ...
    self.use_ner = use_ner
    if self.use_ner:
        import spacy
        self._nlp = spacy.load("en_core_sci_sm", disable=["parser","tagger"])
```

**Step 3.2: Add NER validation method**
```python
def _looks_like_dataset(self, context: str) -> bool:
    doc = self._nlp(context)
    return any(ent.label_ in {"RESOURCE", "DATASET"} for ent in doc.ents)
```

**Step 3.3: Add NER check to `_get_unknown_entities`**
Add this check right after the trigger check:

```python
# --- optional NER sanity check ---
if self.use_ner:
    if not self._looks_like_dataset(window):
        continue
```

### **Phase 4: Add Lookup Validation (Optional)**

**Step 4.1: Create lookup validation function**
```python
def _in_lookup(candidate: str) -> bool:
    # Extract prefix using regex to handle cases like "MODEL1234567890" correctly
    m = re.match(r"([A-Z]{2,5})", candidate)
    prefix = m.group(1) if m else None
    
    if not prefix:
        return True  # no valid prefix found → trust regex
    
    tsv_file = pathlib.Path(f"artifacts/lookups/{prefix.lower()}.tsv")
    if tsv_file.exists():
        if not hasattr(_in_lookup, "_cache"):
            _in_lookup._cache = {}
        if prefix not in _in_lookup._cache:
            _in_lookup._cache[prefix] = set(tsv_file.read_text().splitlines())
        return candidate in _in_lookup._cache[prefix]
    else:
        # Warn user about missing lookup file to avoid silent failures
        logger.warning(f"Lookup file missing for prefix '{prefix}': {tsv_file}")
        return True   # no lookup → trust regex
```

**Step 4.2: Add lookup check to `_get_unknown_entities`**
Add this check right before adding to `entity_pages`:

```python
if not _in_lookup(candidate):
    continue
```

### **Phase 5: Update Pattern Integration**

**Step 5.1: Update imports in `get_citation_entities.py`**
```python
# Replace the existing import
# from src.update_patterns import ENTITY_PATTERNS
from src.patterns import PATTERNS
```

**Step 5.2: Update pattern usage**
Replace every `self.patterns` with the new shared constant *once* in __init__, not scattered inside methods; easier to test.

### **Phase 6: Add Logging and Metrics**

**Step 6.1: Add metrics tracking to `_get_unknown_entities`**
```python
def _get_unknown_entities(self, pages: List[str], article_id: str) -> List[CitationEntity]:
    # Add at the beginning
    raw_matches = 0
    after_lexical = 0
    after_trigger = 0
    after_ner = 0
    after_lookup = 0
    
    # ... existing code with counters ...
    for idx, page in enumerate(pages):
        page_num = idx + 1
        logger.info(f"Processing page {page_num} of {num_pages}")
        
        for name, pat in PATTERNS.items():
            for m in pat.finditer(page):
                raw_matches += 1
                candidate = m.group(0)
                if not _passes_lexical_filters(candidate):
                    continue
                after_lexical += 1
                window = page[max(0, m.start()-250): m.end()+250]
                if not any(t in window.lower() for t in TRIGGERS):
                    continue
                after_trigger += 1
                
                # --- optional NER sanity check ---
                if self.use_ner:
                    if not self._looks_like_dataset(window):
                        continue
                    after_ner += 1
                
                # --- optional lookup validation ---
                if not _in_lookup(candidate):
                    continue
                after_lookup += 1
                
                if candidate not in entity_pages:
                    entity_pages[candidate] = set()
                entity_pages[candidate].add(page_num)
    
    # Add at the end
    logger.info(f"Entity extraction metrics for {article_id}: "
                f"raw={raw_matches}, lexical={after_lexical}, "
                f"trigger={after_trigger}, ner={after_ner}, lookup={after_lookup}")
```

### **Phase 7: Create Required Directories and Files**

**Step 7.1: Create directory structure**
```bash
mkdir -p artifacts/bioregistry
mkdir -p artifacts/lookups
```

**Step 7.2: Update `pyproject.toml` dependencies**
Add these to your dependencies:
```toml
nltk = "^3.8"
scispacy = "^0.5.3"
```

### **Phase 8: Testing and Validation**

**IMPORTANT:** NER sanity check
- Loading en_core_sci_sm (~75 MB) on CPU is fine on a 64 GB MBP (startup ≈ 2 s; inference ≈ 35 ms/paragraph) 
**Remember to disable it in unit tests unless you ship the model – or tests will fetch from S3 every run.**

**Step 8.1: Create basic test**
```python
# tests/test_extractor.py
def test_year_filtered():
    extractor = CitationEntityExtractor(known_entities=False)
    # Test that years are filtered out
    
def test_stopword_filtered():
    extractor = CitationEntityExtractor(known_entities=False)
    # Test that stopwords are filtered out

def test_underscore_regression():
    """Regression test for underscores - verify SRR_123456 matches when paper uses underscore"""
    extractor = CitationEntityExtractor(known_entities=False)
    test_text = "The dataset was deposited as SRR_123456 in the repository"
    entities = extractor._get_unknown_entities([test_text], "test_article")
    assert any("SRR_123456" in entity.text for entity in entities)

def test_precision_guard():
    """Precision guard - feed paragraph with year and assert only SRR ID survives"""
    extractor = CitationEntityExtractor(known_entities=False)
    test_text = "From 2018 we deposited SRR123456 in the repository"
    entities = extractor._get_unknown_entities([test_text], "test_article")
    # Should only extract SRR123456, not 2018
    entity_texts = [entity.text for entity in entities]
    assert "SRR123456" in entity_texts
    assert "2018" not in entity_texts
```

### **Implementation Order:**

1. **Start with Phase 1** - Create the new pattern system
2. **Phase 2** - Add lexical filtering (biggest impact on precision)
3. **Phase 3** - Add NER validation (optional, can be disabled)
4. **Phase 4** - Add lookup validation (optional)
5. **Phase 5** - Update pattern integration
6. **Phase 6** - Add logging
7. **Phase 7** - Create directories
8. **Phase 8** - Test

### **Key Benefits of This Approach:**

1. **Minimal Code Changes**: Your existing `CitationEntity` model and main extraction logic remain unchanged
2. **Backward Compatible**: All existing functionality continues to work
3. **Gradual Rollout**: You can enable features one by one (NER, lookups)
4. **Performance Optimized**: Uses the same parallel processing you already have
5. **Maintains Your Architecture**: Works with your existing file structure and logging

### **Files That Need Changes:**
- `src/get_citation_entities.py` (main changes)
- `pyproject.toml` (add dependencies)
- New files: `src/patterns.py`, `artifacts/citation_patterns.yaml`

### **Files That Stay the Same:**
- `src/models.py` (no changes needed)
- `src/update_patterns.py` (can be kept for backward compatibility)
- All other existing files