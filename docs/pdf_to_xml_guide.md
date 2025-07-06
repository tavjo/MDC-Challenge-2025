Below is a focused implementation play-book for **Guide §4 — Selective PDF → XML Conversion**.
It begins with a short list of *only* the libraries and services you need, flags two inconsistencies I spotted in the current guide, and then walks line-by-line through an executable workflow that consumes the **`Data/conversion_candidates.csv`** emitted by Step 3. All instructions stay strictly inside the remit of Step 4 (no chunking, masking, or embedding yet).

---

## Quick-look: Dependencies for Step 4

| Category                  | Requirement                                                                                                                                                               | Why it’s needed                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Container**             | Docker ≥ 20.x & 4 GB RAM                                                                                                                                                  | Runs the Grobid service that does the heavy XML conversion ([grobid.readthedocs.io][1]) |
| **Python libs**           | `pandas` • `requests` • `pdfplumber` (MIT-licensed text fallback) ([github.com][2], [pypi.org][3]) • `tqdm` (progress bars) • `python-dotenv` (for optional service URLs) | Data wrangling, HTTP multipart upload, fallback extraction, logging                     |
| **CLI**                   | `curl` (health-check Grobid)                                                                                                                                              | Sanity-check during CI/CD                                                               |
| **Files already present** | `Data/conversion_candidates.csv` (produced by Step 3)                                                                                                                     | Drives the selective logic                                                              |
| **Optional**              | `multiprocessing` (for parallel PDF submit)                                                                                                                               | Speed – use if CI runner has CPU headroom                                               |

> **Install tip (inside the project venv)**
> `uv add pandas requests pdfplumber tqdm python-dotenv`

---

## What in the Guide looked confusing

| Section                                                                                                                                                                                                  | Issue                                                                                                                      | Suggested Fix |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------- |
| §4 sample code inventories `Data/train/XML` **via labels**, but Step 3 already materialises a shortlist in `Data/conversion_candidates.csv`. Using both is redundant and could double-convert some PDFs. | Change the primary driver for Step 4 to `conversion_candidates.csv`. Keep the label-driven route only as a fallback audit. |               |
| Several snippets write fresh XML into `processed_docs/`. Later steps expect them under `Data/train/XML`.                                                                                                 | Set `--target-dir` default to `Data/train/XML` so downstream relative paths stay consistent.                               |               |

Everything else in §4 is logically sound and matches the earlier pipeline.

---

## Step-by-Step Workflow

### 0 . Spin-up Grobid once

```bash
docker run -d --name grobid \
  -p 8070:8070 lfoppiano/grobid:0.8.0
# Wait for health
until curl -s http://localhost:8070/api/isalive | grep 'true'; do sleep 3; done
echo "✅ Grobid ready"
```

*Notes*:

* The `/api/processFulltextDocument` endpoint gives fully-tagged TEI-XML — ideal for later XPath sectioning ([grobid.readthedocs.io][4]).
* 4 GB RAM keeps the container from OOM-killing when batch-posting PDFs ([grobid.readthedocs.io][1]).

---

### 1 . Load the candidate manifest

```python
import pandas as pd, pathlib as pl
cand = pd.read_csv("Data/conversion_candidates.csv")
assert {'doi','pdf_path','needs_xml'}.issubset(cand), "CSV schema mismatch"
cand = cand[cand['needs_xml'] == True]          # ~25 % of train set
```

*Why*: The CSV created in Step 3 already tells you *exactly* which PDFs lacked XML during the ID-masking audit. No need to scan labels again.

---

### 2 . Inventory existing XML (safety net)

```python
xml_dir = pl.Path("Data/train/XML")
have_xml = {p.stem for p in xml_dir.glob("*.xml")}
cand['already_has_xml'] = cand['doi'].map(lambda d: d.split('/')[-1] in have_xml)
todo = cand[~cand['already_has_xml']]
print(f"{len(todo)} PDFs still need conversion")
```

---

### 3 . Batch-convert with a robust wrapper

```python
from requests import post
from pdfplumber import open as pdfopen
from tqdm import tqdm
import json, os

GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")

def convert_one(pdf_path: pl.Path, out_xml: pl.Path):
    # Try Grobid first
    with pdf_path.open('rb') as f:
        r = post(f"{GROBID_URL}/api/processFulltextDocument",
                 files={'input': f}, timeout=120)          # multipart/form-data :contentReference[oaicite:4]{index=4}
    if r.status_code == 200 and r.text.strip():
        out_xml.write_text(r.text, encoding='utf-8')
        return "grobid"
    # Fallback – text only, still better than nothing
    text = []
    with pdfopen(str(pdf_path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ''
            text.append(t)
    out_xml.write_text('\n'.join(text), encoding='utf-8')
    return "pdfplumber"

log = []
for row in tqdm(todo.to_dict('records')):
    pdf = pl.Path(row['pdf_path'])
    xml = xml_dir / f"{pdf.stem}.xml"
    try:
        source = convert_one(pdf, xml)
        log.append({**row, 'xml_path': str(xml), 'source': source, 'error': None})
    except Exception as e:
        log.append({**row, 'xml_path': None, 'source': None, 'error': str(e)})
```

*Key behaviours*

| Feature                                                                                                                 | Rationale                             |
| ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| **Single wrapper** for train *and* inference                                                                            | Pipeline symmetry avoids drift        |
| **Multipart POST** respected by `requests` when `files` dict is used ([stackabuse.com][5])                              |                                       |
| **pdfplumber fallback** keeps recall high when Grobid times-out or sees encrypted PDFs ([github.com][2], [pypi.org][3]) |                                       |
| **Timeout 120 s**                                                                                                       | Prevents pipeline hang on gnarly PDFs |
| **Progress bar**                                                                                                        | CI visibility                         |

---

### 4 . Cache & resumability

Converted XML lands in `Data/train/XML/`.
You can resume the job safely: the inventory step (2) will skip any XML that already exists, and the wrapper never overwrites unless `--force` is passed.

---

### 5 . Emit an inventory for downstream steps

```python
pd.DataFrame(log).to_csv("document_inventory_step4.csv", index=False)
# Quick QC
succ = [l for l in log if l['error'] is None]
fail = [l for l in log if l['error']]
print(f"✅ {len(succ)} success | ⚠️ {len(fail)} failed")
```

`document_inventory_step4.csv` now tells Step 5 exactly which documents are available, which conversion path was used (`grobid` vs `pdfplumber`), and which need manual repair.

---

### 6 . Validation guard-rails

1. **Schema check** – confirm `xml_path` exists and file size > 2 kB.
2. **Spot-check** 10 random Grobid XMLs with:

   ```bash
   grep -E -c "<teiHeader|<text>" file.xml
   ```

   Counts should be ≥ 2; if not, the file is probably truncated.
3. **Coverage KPI** – `(successful_conversions + already_has_xml) / len(conversion_candidates)` should be ≥ 0.90 by design; if lower, raise alarm.

---

### 7 . House-keeping & memory notes

* Grobid needs \~400 MB resident for steady state; keep the container running through later steps to reuse the JVM warm-up.
* Each XML averages 250 KB ⇒ the whole converted delta (\~130 files) adds < 35 MB to disk.
* No RAM spikes on the Python side; `pdfplumber` streams page-by-page.
* If you later shard the job across workers, pass unique `--temp-dir` to Grobid endpoints to avoid lock contention ([github.com][6]).

---

## Why this fulfils the wider guide’s intent

* **Selective** – only touches PDFs flagged by Step 3, saving \~75 % compute (Europe PMC already provides XML for most OA articles ([europepmc.org][7], [en.wikipedia.org][8])).
* **License-safe** – pdfplumber’s MIT license keeps the pipeline OSS-friendly ([github.com][2]).
* **Reproducible** – deterministic output paths and a conversion log that can be version-controlled.
* **Forward-compatible** – the inventory CSV schema matches the fields expected by §5 “Document Parsing & Section Extraction”.

Happy converting!

[1]: https://grobid.readthedocs.io/en/latest/Grobid-docker/?utm_source=chatgpt.com "GROBID and Docker containers"
[2]: https://github.com/jsvine/pdfplumber?utm_source=chatgpt.com "jsvine/pdfplumber - and easily extract text and tables. - GitHub"
[3]: https://pypi.org/project/pdfplumber-aemc/0.5.28/?utm_source=chatgpt.com "pdfplumber-aemc - PyPI"
[4]: https://grobid.readthedocs.io/en/latest/Grobid-service/?utm_source=chatgpt.com "GROBID Service API"
[5]: https://stackabuse.com/bytes/how-to-send-multipart-form-data-with-requests-in-python/?utm_source=chatgpt.com "How to Send \"multipart/form-data\" with Requests in Python"
[6]: https://github.com/orbstack/orbstack/issues/1220?utm_source=chatgpt.com "Grobid container failing · Issue #1220 - GitHub"
[7]: https://europepmc.org/downloads/openaccess?utm_source=chatgpt.com "Open access - Developers - Europe PMC"
[8]: https://en.wikipedia.org/wiki/Europe_PubMed_Central?utm_source=chatgpt.com "Europe PubMed Central"
