# Memo: Data Citation Extraction Troubleshooting

Date: 2025-08-10

## Issue summary
- Goal: match all known dataset citations (non-missing `type`) from `Data/train_labels.csv` in parsed PDF texts and store them in DuckDB.
- Expected: 719 total labeled citations across 524 docs (excluding `Missing`).
- Problem observed: Extraction plateaued at 569/719 (150 missing). Root causes included:
  - A subset of `documents.full_text` saved as empty (n_pages = 0) due to parsing gaps.
  - Insufficient normalization/variant matching for DOIs (host prefixes, version/table suffixes, trailing dot numbers), leading to misses for Dryad, PANGAEA, Zenodo, etc.

## What I changed / did
1) Robust PDF parsing fallback
- File: `src/extract_pdf_text_unstructured.py`
- Change: If `strategy='fast'` yields 0 pages, retry with `hi_res`, then `ocr_only`.
- Action: Reparsed only documents with `n_pages=0` and upserted. 47 docs repaired.

2) Audit script fix + reporting
- File: `scripts/audit_citations.py`
- Change: Deduplicate citations by hashable columns only (`document_id`, `data_citation`) to avoid ndarray hashing error.
- Output reports written to: `reports/citation_audit/`
  - `found.csv`, `missing.csv`, `extras.csv`, `missing_by_document.csv`, `missing_by_type.csv`

3) Better citation variant handling
- File: `src/get_citation_entities.py`
- Changes:
  - Pattern builder uses `clean_text_for_urls` and flexible separator class.
  - Variant generator now strips:
    - DOI host (`https://doi.org/`, `https://dx.doi.org/`, etc.)
    - Version suffix `.vN`
    - Table suffix `/tN`
    - Trailing dot-number (e.g., `dryad.5q1sb.1` -> `dryad.5q1sb`)
    - E-GEOD → GSE synonym
  - Added alnum-collapsed fallback for both raw and host-stripped DOIs.

4) Accession-aware matching (added)
- File: `src/get_citation_entities.py`
- Changes:
  - Introduced `_build_accession_pattern` handling: `PXD\d+`, `SRR\d+`, `ERR\d+`, `E-PROT-\d+`, `ENS[A-Z]+\d+`, UniProt (6/10-char), and a generic short-head + digits pattern.
  - Tolerant separators within IDs: spaces, hyphens, underscores, commas (to cope with PDF token splits).
  - Generated split variants for accession-like labels (e.g., `PXD 12345`, `ENS MMUT 000...`).

## Current status (after each step)
- Before any fixes: 569 found / 719 expected (150 missing).
- After PDF reparsing fallback + re-extraction: 662 found (57 missing).
- After DOI variant expansion + re-extraction: 676 found (43 missing).
- After accession-aware regex + targeted reparsing: 681 found (38 missing).

## Remaining misses (themes)
- The 38 remaining are concentrated where the labeled IDs do not show up in parsed body text (likely figures/tables/images not captured by text-only parsing):
  - `10.1371_journal.pcbi.1011828`: E-PROT-### series.
  - `10.1080_21645515.2023.2189598`: ENSMMUT..., `F6RVJ3`.
  - `10.7554_eLife.63194`: SRR/ERR accessions.
- DOI hosts and suffix normalization already handled; regex tolerance now includes underscores/commas. Further gains require better capture of tables/captions or OCR.

Refer to latest misses:
- File: `reports/citation_audit/missing.csv`
- Type summary: `reports/citation_audit/missing_by_type.csv`
- Doc hotspots: `reports/citation_audit/missing_by_document.csv`

## Where things are
- Labels: `Data/train_labels.csv`
- Audit script: `scripts/audit_citations.py`
- Reports: `reports/citation_audit/` (overwritten on each run)
- Extraction code:
  - Known-entity extraction: `src/get_citation_entities.py`
  - PDF parsing: `src/extract_pdf_text_unstructured.py`
- Database: `artifacts/mdc_challenge.db` (DuckDB)

## How to reproduce quickly
- Activate venv and run audit:
  - `python scripts/audit_citations.py`
- Re-extract known entities into DuckDB:
  - `python src/get_citation_entities.py`
- Then re-run audit to see updated counts.

## Next steps (to reach 719/719)
1) Enhance parsing for hard cases (no supplementary files available):
   - Force OCR-only passes on hotspot PDFs.
   - Enable table/caption-aware parsing using high-resolution element extraction; concatenate table cell text and captions into `full_text`.
   - Reparse and upsert `documents.full_text` for the top-miss DOIs from `missing_by_document.csv`.
2) Re-run extraction and audit to measure impact. Iterate on parsing heuristics for any stragglers.

## Notes
- Keep an eye on documents with previously `n_pages=0`; the fallback fixed many, but any new additions should use the same parse strategy.
- The audit compares labels vs. extracted rows on `(document_id, data_citation)` after basic normalization.