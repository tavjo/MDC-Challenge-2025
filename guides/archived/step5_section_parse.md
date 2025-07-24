Below is an implementation-ready blueprint for **Step 5 – Document Parsing & Section Extraction**.
It starts by flagging the only items in the master guide that could cause confusion, then lists the *minimal* dependencies for this step, and finally walks through the workflow in fine detail—from loading the inventory produced in Step 4 to exporting a clean, section-aware corpus for downstream chunking. All tasks stay strictly within the remit of “parse documents and extract sections”.

---

## ❗ Things to flag before proceeding

| # | Observation                                                                                                                                                          | Why it could be a problem                                                                      | Suggested fix                                                                                                                                                 |
| - | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | `Section` is defined as a `dataclass`, but later code (e.g. `refine_chunk_types`) treats `section_primary` as an **attribute of `ChunkMetadata`**, not of `Section`. | Schema mismatch could break type-checking and serialization.                                   | Either: (a) move `section_primary` into `ChunkMetadata`, **or** (b) switch the whole object model to Pydantic (recommended for consistency with other steps). |
| 2 | `page_start = i+1` is used for XML sections although TEI does **not** preserve page numbers – this value is meaningless.                                             | Downstream code that relies on page numbers (e.g. ordering, plotting) will mis-order sections. | Replace `page_start`/`page_end` with a simple **sequence index** when the source is XML.                                                                      |
| 3 | The Step-5 draft already contains *chunking* utilities (`create_section_aware_chunks`, etc.).                                                                        | Chunking belongs to Step 6; including it here violates the single-responsibility design.       | Move all chunking-related functions to Step 6 so that Step 5 finishes with a **parsed + validated corpus**, nothing more.                                     |
| 4 | A TODO comment says “`# TODO: Should be pydantic class but look into dataclass`”.                                                                                    | Leaving it unresolved creates ambiguity for maintainers.                                       | Pick one: stay with lightweight `dataclass` **or** migrate now to `pydantic.BaseModel`; document the choice.                                                  |

---

## Dependencies for Step 5 only

| Purpose                          | Library                                            | Why                                                  |
| -------------------------------- | -------------------------------------------------- | ---------------------------------------------------- |
| Load inventory & write summaries | `pandas >=2.2`                                     | fast CSV/Parquet IO                                  |
| Parse TEI/JATS XML               | `lxml >=5.1` **or** stdlib `xml.etree.ElementTree` | `lxml` is faster & XPath-aware � TEI uses namespaces |
| Structured logs & progress bars  | `tqdm >=4.66`                                      | nice-to-have for long runs                           |
| Robust section/ID regex          | `regex >=2023.12`                                  | better Unicode & overlapped matches                  |
| Optional text fallback           | `pdfplumber >=0.10` (already installed in Step 1)  | only needed when XML is missing                      |

> **No vector, ML, or chunking packages are required in this step**.

Create a `requirements_step5.txt` if you want to isolate these:

```
pandas>=2.2
lxml>=5.1          # or comment out if using ElementTree
tqdm>=4.66
regex>=2023.12
```

---

## Step-by-Step Workflow

### Step 0 – Inputs & expected outputs

| Asset                    | Location                                                                                              | Notes                                                                                |
| ------------------------ | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Document inventory CSV   | `Data/train/XML/document_inventory_step4.csv` (produced by Step 4)                                    | Columns include `doi`, `filename_stem`, `file_path`, `source_type` (“xml” or “text”) |
| Full-text XML files      | `Data/train/XML/`                                                                                     | 124 files, all created or verified in Step 4                                         |
| **Outputs of this step** | 1. `parsed_documents.pkl` — pickled dict<br>2. `parsed_documents_summary.csv` — quick stats per paper | Both land in `Data/train/parsed/`                                                    |

### Step 1 – Load inventory

```python
inv = pd.read_csv("Data/train/XML/document_inventory_step4.csv")
assert inv['source_type'].isin({'xml', 'text'}).all()
```

*Why?* Centralising the file list avoids walking the directory tree again.

### Step 2 – Iterate over documents with a single parsing dispatcher

```python
from pathlib import Path
from lxml import etree           # change to ElementTree if you prefer

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}          # :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}

def parse_doc(row):
    path = Path(row.file_path)
    if row.source_type == "xml":
        return parse_tei(path)
    else:
        return parse_plain_text(path)     # rare fallback
```

#### 2.1 Parsing TEI/JATS XML (`parse_tei`)

1. **Read XML** and register namespace (`parser = etree.XMLParser(recover=True)`).

2. **Select all `<div type="…">`** elements (TEI) or `<sec>` (JATS) with XPath.
   Example XPath for TEI:

   ```python
   sections = root.xpath('.//tei:div[@type]', namespaces=TEI_NS)
   ```

   *Source*: TEI section tags documented in GROBID TEI output.

3. **Map raw types ➜ canonical labels**

   | Raw `@type`                         | Canonical          |
   | ----------------------------------- | ------------------ |
   | `abstract`                          | abstract           |
   | `introduction` or `background`      | introduction       |
   | `method`, `methods`, `materials`    | methods            |
   | `result`, `results`                 | results            |
   | `discussion`                        | discussion         |
   | `conclusion`                        | conclusion         |
   | `availability`, `data_availability` | data\_availability |
   | `reference*`                        | references         |
   | (else)                              | other              |

   The mapping reflects typical IMRaD structure.

4. **Extract plain text** inside each `<p>` descendant, join with double newlines for readability.

5. **Record metadata** per section:

   ```python
   SectionInfo = dict(
       order=i,                    # sequential index
       canonical_type=canon,
       char_len=len(section_text),
   )
   ```

6. **Collect full document text** (`'\n\n'.join(all_section_texts)`).

#### 2.2 Parsing fallback plain-text files (`parse_plain_text`)

If Step 4 produced a text-only fallback (source =`pdfplumber`):

1. Split on newline and use **regex anchors** to detect headings:

   ```python
   HEADINGS = {
       r'^\s*ABSTRACT\s*$': 'abstract',
       r'^\s*INTRODUCTION\s*$': 'introduction',
       # …
   }
   ```

   These headings mirror standard journal layout.

2. Iterate line-by-line, start a new section whenever a heading matches.

3. Store same metadata structure; `order` increments whenever a new heading is found.

### Step 3 – Validate per-document extraction

For each parsed doc:

| Check                 | Target                                             | How                          |
| --------------------- | -------------------------------------------------- | ---------------------------- |
| ≥ 1 section extracted | True                                               | `len(sections)>0`            |
| Key sections present  | `methods` AND `results` **or** `data_availability` | helps downstream recall      |
| Full-text length >    | 1 000 characters                                   | guards against truncated XML |

Failures go into `parsing_failed` with an error string; they stay in the dict but are excluded from downstream steps.

### Step 4 – Create & save the corpus

```python
import pickle, json, pandas as pd, os

out_dir = Path("Data/train/parsed")
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir/"parsed_documents.pkl", "wb") as f:
    pickle.dump(parsed_docs, f)

summary = pd.DataFrame([
    {
      "doi": d["doi"],
      "n_sections": len(d["sections"]),
      "has_data_availability": "data_availability" in d["section_labels"],
      "fulltext_len": len(d["full_text"]),
      "parsing_failed": d["parsing_failed"]
    }
    for d in parsed_docs.values()
])
summary.to_csv(out_dir/"parsed_documents_summary.csv", index=False)
```

### Step 5 – Quality report (console OK)

```python
ok = summary[~summary.parsing_failed]
print(f"✅ Parsed {len(ok)}/{len(summary)} documents")
print(f"Median sections per paper: {ok.n_sections.median():.1f}")
print(f"Data-availability section present in {ok.has_data_availability.mean():.1%} of papers")
```

### Step 6 – Hand-off contract to Step 6 (chunking)

*Guarantees provided to downstream code*:

| Key                    | Description                                                       |
| ---------------------- | ----------------------------------------------------------------- |
| `parsed_documents.pkl` | `{doi: {full_text:str, sections:list[dict], section_order:dict}}` |
| No PDF reading         | Everything is already text; chunker needn’t touch XML             |
| All IDs intact         | No masking or chunking done yet (handled later)                   |

---

## Appendix – Reference regex patterns (ready for Step 6)

| Entity       | Pattern                                           | Source                                  |
| ------------ | ------------------------------------------------- | --------------------------------------- |
| DOI          | `10.\d{4,9}/[-._;()/:A-Z0-9]+` (case-insensitive) | Crossref official regex                 |
| GEO (GSE)    | `GSE\d{3,6}`                                      | GEO docs & community examples           |
| ArrayExpress | `E-[A-Z]+-\d+`                                    | EMBL-EBI docs                           |
| PDB          | `\b[0-9][A-Za-z0-9]{3}\b`                         | PDB format description (see wwPDB site) |

These are **not** used during Step 5 but will be essential for citation-integrity checks in later steps.

---

### Why this design is robust

* **Namespace-aware XPath** avoids brittle string searches and leverages the TEI model used by GROBID.
* **Canonical section mapping** normalises the kaleidoscope of publisher tags into a fixed set, simplifying retrieval heuristics later.
* **Fallback regex headings** keep recall high even if some PDFs only yielded plain text.
* All validation metrics are cheap to compute and catch the most common parsing failures (empty XML, half-scraped documents, missing key sections).

With this in place, you can proceed to Step 6 (semantic chunking) confident that every article is represented as a clean, section-aware text object.
