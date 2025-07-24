## **Critical Discovery: XML Format Differences** 🚨
- **124 Step 4 files** (from PDF conversion): Use **TEI format** with `<div type="...">` sections
- **400 existing files**: Use **JATS format** with `<sec id="...">` sections  
- **⚠️ CRITICAL**: JATS files often have namespaces requiring namespace-aware XPath queries

---

# **Step 5 Implementation Checklist: Document Parsing & Section Extraction**

## **❗ Pre-Implementation Validation**

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | **Verify XML format differences** | ✅ Confirmed | TEI vs JATS formats require different parsers |
| 2 | **Check inventory completeness** | ✅ Ready | `Data/document_inventory.csv` contains 524 total files |
| 3 | **Validate Step 4 output** | ✅ Complete | 124 files successfully converted via Grobid |
| 4 | **Review schema mismatch flags** | 🟥 **CRITICAL** | Need namespace-aware parsing for JATS |

---

## **🔧 Minimal Dependencies Setup**

| # | Task | Command/Action | Status |
|---|------|----------------|--------|
| 1 | **Install XML parsing** | `uv add lxml>=5.1` | ✅ Available |
| 3 | **Install regex support** | `uv add regex>=2023.12` | ✅ Available |

---

## **📁 File Structure Setup**

| # | Task | Location | Status |
|---|------|----------|--------|
| 1 | **Create output directory** | `Data/train/parsed/` | ✅ Complete |
| 2 | **Create source file** | `src/document_parser.py` | ✅ Complete |
| 3 | **Create format detection helper** | `src/xml_format_detector.py` | ✅ Complete |
| 4 | **Create section mapper** | `src/section_mapping.py` | ✅ Complete |

---

## **🔍 Step 0: Inputs & Expected Outputs**

| # | Task | Expected Result | Status |
|---|------|-----------------|--------|
| 1 | **Load document inventory** | `Data/document_inventory.csv` (524 files) | ✅ Available |
| 2 | **Verify XML directory** | `Data/train/XML/` with 524 files | ✅ Available |
| 3 | **Define output structure** | `parsed_documents.pkl` + `parsed_documents_summary.csv` | ⏳ |

---

## **🔬 🟥 CRITICAL: XML Format Detection & Namespace-Aware Parsing**

| # | Task | Implementation | Status |
|---|------|----------------|--------|
| 1 | **🟥 Robust format detector** | Parse root element and inspect `root.nsmap` | ✅ Complete |
| 2 | **🟥 Namespace-aware JATS parser** | Register namespace alias and use `'.//j:sec'` XPath | ✅ Complete |
| 3 | **🟧 Abstract extraction** | Extract `<abstract>` from `<article-meta>` before body sections | ✅ Complete |
| 4 | **🟧 Recursive section parsing** | Handle nested `<sec>` elements with depth tracking | ✅ Complete |

### 🟥 CRITICAL: Improved Format Detection Script:
```python
def detect_xml_format(xml_path):
    """Detect XML format by parsing root element and inspecting namespaces"""
    try:
        parser = etree.XMLParser(ns_clean=True, recover=True)
        tree = etree.parse(xml_path, parser)
        root = tree.getroot()
        
        # Check namespace map for JATS patterns
        nsmap = root.nsmap or {}
        for uri in nsmap.values():
            if uri and ('jats.nlm.nih.gov' in uri or '/JATS' in uri or uri.startswith('http://jats.')):
                return 'JATS'
        
        # Check for TEI namespace
        if any('tei-c.org' in str(uri) for uri in nsmap.values() if uri):
            return 'TEI'
            
        # Fallback to root tag inspection
        if root.tag.endswith('TEI'):
            return 'TEI'
        elif root.tag.endswith('article'):
            return 'JATS'
            
        return 'UNKNOWN'
        
    except Exception as e:
        print(f"Error detecting format for {xml_path}: {e}")
        return 'ERROR'
```

---

## **📋 Step 1: Load Inventory**

| # | Task | Code Implementation | Status |
|---|------|-------------------|--------|
| 1 | **Load inventory CSV** | `pd.read_csv("Data/document_inventory.csv")` | ✅ Complete |
| 2 | **Validate source types** | Check `source` column for 'grobid' vs None | ✅ Complete |
| 3 | **Count by format** | TEI: 124, JATS: 400 (confirmed) | ✅ Complete |

---

## **🔄 Step 2: Document Parsing Dispatcher**

| # | Task | Implementation Details | Status |
|---|------|------------------------|--------|
| 1 | **Create main dispatcher** | Route to TEI vs JATS parser based on format | ✅ Complete |
| 2 | **TEI namespace setup** | `TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}` | ✅ Complete |
| 3 | **🟥 JATS namespace setup** | Dynamic namespace registration from `root.nsmap` | ✅ Complete |
| 4 | **🟩 Enhanced error handling** | Log `root.nsmap` and failed XPath for debugging | ✅ Complete |

### Core Dispatcher Function:
```python
def parse_document(file_path, source_type):
    format_type = detect_xml_format(file_path)
    
    if format_type == 'TEI':
        return parse_tei_xml(file_path)
    elif format_type == 'JATS':
        return parse_jats_xml(file_path)
    else:
        # Log diagnostic info for unknown formats
        try:
            tree = etree.parse(file_path)
            root = tree.getroot()
            print(f"Unknown format - nsmap: {root.nsmap}, root tag: {root.tag}")
        except Exception as e:
            print(f"Parse error for {file_path}: {e}")
        return parse_fallback(file_path)
```

---

## **🧩 Step 2.1: TEI Parser Implementation**

| # | Task | XPath/Logic | Status |
|---|------|-------------|--------|
| 1 | **Parse TEI structure** | `root.xpath('.//tei:div[@type]', namespaces=TEI_NS)` | ✅ Complete |
| 2 | **Map section types** | Create canonical mapping for `@type` values | ✅ Complete |
| 3 | **🟩 Clean text extraction** | Strip XML tags before measuring length | ✅ Complete |
| 4 | **Record metadata** | Store order, type, character length | ✅ Complete |

### TEI Section Mapping:
```python
TEI_SECTION_MAPPING = {
    'abstract': 'abstract',
    'introduction': 'introduction', 
    'background': 'introduction',
    'method': 'methods',
    'methods': 'methods',
    'materials': 'methods',
    'result': 'results',
    'results': 'results',
    'discussion': 'discussion',
    'conclusion': 'conclusion',
    'availability': 'data_availability',
    'data_availability': 'data_availability',
    # Add more mappings as needed
}
```

---

## **🧩 Step 2.2: 🟥 CRITICAL Enhanced JATS Parser Implementation**

| # | Task | XPath/Logic | Status |
|---|------|-------------|--------|
| 1 | **🟥 Namespace-aware parsing** | Dynamic namespace registration and `'.//j:sec'` queries | ⏳ |
| 2 | **🟧 Abstract extraction** | Extract `<abstract>` from `<article-meta>` first | ⏳ |
| 3 | **🟧 Recursive section handling** | Process nested `<sec>` elements with depth tracking | ⏳ |
| 4 | **🟨 Enhanced section mapping** | Use `@sec-type` attribute + case-insensitive title matching | ⏳ |

### 🟥 CRITICAL: Namespace-Aware JATS Parser:
```python
def parse_jats_xml(xml_path):
    """Parse JATS XML with proper namespace handling"""
    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()
        
        # Register namespaces dynamically
        nsmap = root.nsmap or {}
        jats_ns = None
        for prefix, uri in nsmap.items():
            if uri and ('jats.nlm.nih.gov' in uri or '/JATS' in uri):
                jats_ns = uri
                break
        
        # Set up namespace dictionary
        if jats_ns:
            ns = {'j': jats_ns}
            sec_xpath = './/j:sec'
            abstract_xpath = './/j:article-meta/j:abstract'
        else:
            ns = {}
            sec_xpath = './/sec'
            abstract_xpath = './/article-meta/abstract'
        
        sections = []
        
        # 🟧 Extract abstract from article-meta first
        abstracts = root.xpath(abstract_xpath, namespaces=ns)
        for abstract in abstracts:
            text = extract_text_content(abstract)
            if text.strip():
                sections.append({
                    'type': 'abstract',
                    'text': text,
                    'order': 0,
                    'sec_level': 0
                })
        
        # 🟧 Process body sections recursively
        body_sections = root.xpath(sec_xpath, namespaces=ns)
        for sec in body_sections:
            process_section_recursive(sec, sections, ns, level=1)
            
        return sections
        
    except Exception as e:
        print(f"JATS parsing error for {xml_path}: {e}")
        print(f"Root nsmap: {root.nsmap if 'root' in locals() else 'Unknown'}")
        return []

def process_section_recursive(sec, sections, ns, level=1):
    """🟧 Recursively process nested sections"""
    # Extract section type and title
    sec_type = sec.get('sec-type', '').lower()
    title_elem = sec.find('title') if not ns else sec.find(f'{{{ns.get("j", "")}}}title')
    title = title_elem.text if title_elem is not None else ''
    
    # Map to canonical type
    canonical_type = map_jats_section_type(sec_type, title)
    
    # Extract text content (excluding nested sections)
    text = extract_section_text(sec, ns)
    
    if text.strip():
        sections.append({
            'type': canonical_type,
            'text': text,
            'order': len(sections),
            'sec_level': level,
            'original_type': sec_type,
            'title': title
        })
    
    # Process nested sections
    nested_xpath = 'sec' if not ns else f'{{{ns.get("j", "")}}}sec'
    for nested_sec in sec.findall(nested_xpath):
        process_section_recursive(nested_sec, sections, ns, level + 1)
```

### 🟨 Enhanced JATS Section Mapping:
```python
import re

JATS_TITLE_PATTERNS = {
    'abstract': r'abstract',
    'introduction': r'(background|introduction)',
    'methods': r'(methods?|methodology|materials?\s+and\s+methods?)',
    'results': r'results?',
    'discussion': r'discussion',
    'conclusion': r'conclus(ion|ions)',
    'data_availability': r'(data\s+availability|data\s+&\s+code\s+availability|code\s+availability)',
    'acknowledgments': r'acknowledgm(ent|ents)',
    'references': r'references?',
}

def map_jats_section_type(sec_type, title):
    """🟨 Map JATS section using sec-type attribute and title patterns"""
    # First try sec-type attribute
    if sec_type:
        for canonical, pattern in JATS_TITLE_PATTERNS.items():
            if re.search(pattern, sec_type, re.IGNORECASE):
                return canonical
    
    # Then try title text
    if title:
        for canonical, pattern in JATS_TITLE_PATTERNS.items():
            if re.search(pattern, title, re.IGNORECASE):
                return canonical
    
    return 'other'
```

---

## **✅ Step 3: Document Validation**

| # | Task | Validation Logic | Status |
|---|------|------------------|--------|
| 1 | **Section count check** | `len(sections) > 0` | ⏳ |
| 2 | **Key sections present** | Check for methods AND (results OR data_availability) | ⏳ |
| 3 | **🟩 Clean text length check** | Strip XML tags before measuring: `len(clean_text) > 1000` | ⏳ |
| 4 | **🟩 Enhanced error logging** | Record failures with namespace info and error context | ⏳ |

---

## **💾 Step 4: Create & Save Corpus**

| # | Task | Output Format | Status |
|---|------|---------------|--------|
| 1 | **Create parsed dict** | `{doi: {full_text, sections, section_labels, ...}}` | ⏳ |
| 2 | **Save pickle file** | `Data/train/parsed/parsed_documents.pkl` | ⏳ |
| 3 | **Generate summary CSV** | Stats per document with validation flags | ⏳ |
| 4 | **🟩 Optional raw XML hash** | Store XML hash for debugging reference | ⏳ |
| 5 | **Ensure directory exists** | `Path("Data/train/parsed").mkdir(parents=True, exist_ok=True)` | ⏳ |

---

## **📊 Step 5: Quality Reporting**

| # | Task | Metrics to Track | Status |
|---|------|------------------|--------|
| 1 | **🟨 Adjusted success rate** | `≥90%` success rate (accounts for editorials/reviews) | ⏳ |
| 2 | **Format breakdown** | TEI vs JATS success rates | ⏳ |
| 3 | **🟨 Article-type aware coverage** | Track methods/results coverage by article type | ⏳ |
| 4 | **Data availability** | % with data_availability sections | ⏳ |
| 5 | **🟧 Nested section stats** | Track section depth and hierarchy coverage | ⏳ |

---

## **🔗 Step 6: Hand-off Contract Validation**

| # | Task | Contract Guarantee | Status |
|---|------|-------------------|--------|
| 1 | **Verify pickle structure** | Correct nested dict format | ⏳ |
| 2 | **Check section standardization** | All sections use canonical labels | ⏳ |
| 3 | **Validate text extraction** | No XML tags in extracted text | ⏳ |
| 4 | **Ensure ID preservation** | All DOIs preserved correctly | ⏳ |
| 5 | **🟧 Verify section hierarchy** | Section levels and nesting preserved | ⏳ |

---

## **🧪 Testing & Validation Scripts**

| # | Task | Script Purpose | Status |
|---|------|----------------|--------|
| 1 | **🟥 Format detection test** | Test namespace-aware detection on sample files | ⏳ |
| 2 | **🟥 Namespace XPath test** | Verify JATS parsing with various namespace patterns | ⏳ |
| 3 | **🟧 Abstract extraction test** | Validate abstract extraction from article-meta | ⏳ |
| 4 | **🟧 Nested section test** | Test recursive section parsing | ⏳ |
| 5 | **End-to-end test** | Full pipeline on 10 sample files | ⏳ |

### 🟥 CRITICAL: Enhanced Test Script:
```python
# Create this as: tests/test_xml_formats.py
def test_namespace_aware_parsing():
    """Test critical namespace-aware parsing functionality"""
    
    # Test format detection
    tei_sample = "Data/train/XML/10.1002_ecs2.1280.xml"  # From Step 4
    jats_sample = "Data/train/XML/10.1186_s13321-015-0110-6.xml"  # Existing
    
    assert detect_xml_format(tei_sample) == 'TEI'
    assert detect_xml_format(jats_sample) == 'JATS'
    
    # Test namespace XPath queries
    jats_tree = etree.parse(jats_sample)
    jats_root = jats_tree.getroot()
    nsmap = jats_root.nsmap
    
    # Should find sections with proper namespace
    if nsmap:
        jats_ns = next((uri for uri in nsmap.values() if 'jats' in str(uri).lower()), None)
        if jats_ns:
            ns = {'j': jats_ns}
            sections = jats_root.xpath('.//j:sec', namespaces=ns)
            assert len(sections) > 0, "Namespace-aware XPath should find sections"
    
    print("✅ Critical namespace-aware parsing working correctly")
```

---

## **🎯 Success Criteria**

- [ ] **🟥 524 files processed** with namespace-aware parsing (100% of available XML files)
- [ ] **🟥 Zero silent failures** from namespace issues (proper XPath queries)
- [ ] **🟧 Abstract extraction** working for JATS files (from article-meta)
- [ ] **🟧 Nested section support** (hierarchical section parsing)
- [ ] **🟨 ≥90% success rate** (adjusted for editorial/review articles)
- [ ] **🟨 Enhanced section mapping** (sec-type + case-insensitive title matching)
- [ ] **Clean text extraction** (no XML artifacts)
- [ ] **Ready for Step 6** (chunking pipeline can consume output)

---

## **Priority Legend**
- 🟥 **Critical**: Blocks correct parsing; fix before running pipeline
- 🟧 **High**: Strongly recommended for reliable coverage  
- 🟨 **Medium**: Improves robustness/metrics
- 🟩 **Low**: Nice-to-have / cleanup

This enhanced checklist addresses the critical namespace issues that would cause silent failures in JATS parsing, adds proper abstract extraction, and implements recursive section handling for better coverage.