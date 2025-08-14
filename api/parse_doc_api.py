# api/parse_doc_api.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import re
import csv
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api.services.document_parsing_service import build_document_objects, build_document_object
from api.utils.duckdb_utils import get_duckdb_helper
from src.models import BulkParseRequest, Document
from src.helpers import export_docs, initialize_logging, num_tokens, compute_file_hash, timer_wrap
from src.extract_pdf_text_unstructured import load_pdf_pages as load_pdf_pages_unstructured

logger = initialize_logging("parse_doc_api")

app = FastAPI(
    title="Document Parsing Microservice",
    description="API for parsing pdfs with unstructured and storing them into DuckDB",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Remove module-level DuckDBHelper instantiation to defer until requests
# DUCKDB_HELPER = get_duckdb_helper(os.path.join(project_root, "artifacts", "mdc_challenge.db"))
DB_PATH = os.path.join(project_root, "artifacts", "mdc_challenge.db")

# Parse a single document
@app.get("/parse_doc")
async def parse_doc(pdf_path: str):
    helper = get_duckdb_helper(DB_PATH)
    try:
        document = build_document_object(pdf_path=pdf_path)
        helper.store_document(document)
        return {"message": "Document parsed and stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        helper.close()

# Parse multiple documents at once
@app.post("/bulk_parse_docs")
async def parse_docs(payload: BulkParseRequest):
    pdf_paths = payload.pdf_paths
    export_file = payload.export_file
    export_path = payload.export_path
    subset = payload.subset
    subset_size = payload.subset_size
    max_workers = payload.max_workers
    strategy = payload.strategy
    params = {
        "pdf_paths": pdf_paths,
        "subset": subset,
        "subset_size": subset_size,
        "max_workers": max_workers,
        "strategy": strategy
    }
    helper = get_duckdb_helper(DB_PATH)
    try:
        documents = build_document_objects(**params)
        if not documents:
            logger.error(f"No document objects built: {pdf_paths}")
            raise HTTPException(status_code=400, detail="No document objects built")
        success = helper.batch_upsert_documents(documents)
        if not success:
            logger.error(f"Failed to store documents: {documents}")
            raise HTTPException(status_code=500, detail="Failed to store documents")
        # also export as a json file
        if export_file:
            export_docs(documents, output_file=export_file, output_dir=export_path)
        elif export_path:
            export_docs(documents, output_dir=export_path)
        elif export_file and export_path:
            export_docs(documents, output_file=export_file, output_dir=export_path)
        else:
            export_docs(documents)
        return {"message": f"{len(documents)} Documents parsed and stored successfully"}
    except Exception as e:
        logger.error(f"Error parsing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        helper.close()

@app.get("/health")
async def health_check():
    helper = get_duckdb_helper(DB_PATH)
    try:
        conn = helper.engine
        # Test basic connection
        conn.execute("SELECT 1")
        # Test that we can access the documents table
        conn.execute("SELECT COUNT(*) FROM documents")
        duckdb_ok = True
        duckdb_error = None
    except Exception as e:
        logger.error(f"DuckDB health check failed: {str(e)}")
        duckdb_ok = False
        duckdb_error = str(e)
    finally:
        helper.close()
    status = "healthy" if duckdb_ok else "unhealthy"
    response = {
        "status": status,
        "duckdb_connected": duckdb_ok,
        "service": "document_parsing",
        "db_path": DB_PATH
    }
    if not duckdb_ok:
        response["error"] = duckdb_error
    return response


class RepairRequest(BaseModel):
    document_ids: Optional[List[str]] = None
    max_docs: Optional[int] = None
    dry_run: bool = False
    pdf_dir: Optional[str] = None  # optional override if PDFs are elsewhere


def _candidate_pages_from_text(pages: List[str]) -> List[int]:
    """Identify likely citation pages using simple regex heuristics.

    - Keyword anchors: doi.org, Data availability, Supplementary, Table, common repos
    - ID-like patterns: SRR/ERR/PRJ/GSE/E-GEOD/PXD/ENS*/RRID/UniProt-like
    - Always include the last ~35% of pages
    """
    if not pages:
        return []

    keyword_re = re.compile(
        r"(?i)(doi\.org|data availability|data accessibility|supplementary|appendix|table|zenodo|dryad|pangaea|mendeley|GEO|SRA|PRIDE)"
    )
    id_like_re = re.compile(
        r"(?i)(SRR\d+|ERR\d+|PRJ\w*\d+|GSE\d+|E-GEOD-\d+|PXD\d+|ENS[A-Z]+\d+|RRID:?\s*[A-Z0-9_\-]+|\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b)"
    )

    hits: List[int] = []
    total_pages = len(pages)
    tail_start = max(1, int(total_pages * 0.65))
    for idx, text in enumerate(pages, start=1):
        if not text:
            continue
        if idx >= tail_start or keyword_re.search(text) or id_like_re.search(text):
            hits.append(idx)
    return sorted(set(hits))


@app.post("/repair_docs_ocr")
@timer_wrap
async def repair_docs_ocr(payload: RepairRequest) -> Dict[str, Any]:
    """Guess likely citation pages and reparse them with Unstructured OCR-only.

    Behaviour:
    - If `document_ids` not provided, read `reports/citation_audit/missing.csv` for targets.
    - Use existing `documents.full_text` to choose candidate pages via regex.
    - Parse entire PDF with `strategy='ocr_only'` and splice OCR text into selected pages.
    - Update text-related fields in DuckDB for each repaired document.
    """
    helper = get_duckdb_helper(DB_PATH)
    try:
        # 1) Determine target documents
        target_docs: List[str] = []
        if payload.document_ids:
            target_docs = list(dict.fromkeys(payload.document_ids))
        else:
            missing_path = os.path.join(project_root, "reports", "citation_audit", "missing.csv")
            if os.path.exists(missing_path):
                with open(missing_path, newline="") as f:
                    r = csv.DictReader(f)
                    target_docs = sorted({row["document_id"] for row in r if row.get("document_id")})
            else:
                return {"success": False, "error": "missing.csv not found and no document_ids provided"}

        if payload.max_docs is not None:
            target_docs = target_docs[: payload.max_docs]

        results: Dict[str, Any] = {"processed": [], "skipped": [], "dry_run": payload.dry_run}

        for doc_id in target_docs:
            # Fetch document row
            res = helper.engine.execute(
                "SELECT doi, full_text, n_pages, file_path FROM documents WHERE doi = ?",
                [doc_id],
            )
            row = res.fetchone()
            if not row:
                results["skipped"].append({"document_id": doc_id, "reason": "not_in_db"})
                continue

            cols = [d[0] for d in res.description]
            doc_row = dict(zip(cols, row))
            full_text: List[str] = doc_row.get("full_text") or []
            file_path = doc_row.get("file_path")

            if not file_path or not os.path.exists(file_path):
                results["skipped"].append({"document_id": doc_id, "reason": "file_not_found", "file_path": file_path})
                continue

            # Candidate pages
            candidate_pages = _candidate_pages_from_text(full_text)
            if not candidate_pages:
                logger.warning(f"No candidate pages found for {doc_id}; using last 20% of pages")
                n = max(1, len(full_text))
                tail = max(3, min(6, int(n * 0.2)))
                candidate_pages = list(range(max(1, n - tail + 1), n + 1))

            if payload.dry_run:
                results["processed"].append({
                    "document_id": doc_id,
                    "candidate_pages": candidate_pages,
                    "action": "dry_run"
                })
                continue

            # OCR-only full parse
            try:
                ocr_pages = load_pdf_pages_unstructured(file_path, strategy="ocr_only")
            except Exception as e:
                results["skipped"].append({"document_id": doc_id, "reason": f"ocr_failed: {e}"})
                continue

            if not ocr_pages:
                results["skipped"].append({"document_id": doc_id, "reason": "ocr_empty"})
                continue

            n_existing = len(full_text)
            n_ocr = len(ocr_pages)
            n_min = min(n_existing, n_ocr) if n_existing else n_ocr

            if n_existing == 0:
                # No existing pages; use OCR pages but still limit to candidate pages selection
                new_full_text = list(ocr_pages)
            else:
                # Replace only candidate pages within bounds
                new_full_text = list(full_text[:n_min])
                safe_candidates = [p for p in candidate_pages if 1 <= p <= n_min]
                for p in safe_candidates:
                    new_full_text[p - 1] = ocr_pages[p - 1]

            # Recompute stats
            total_char_length = sum(len(p or "") for p in new_full_text)
            total_tokens = sum(num_tokens(p or "") for p in new_full_text)

            # Build Document object from current DB row to update
            res2 = helper.engine.execute("SELECT * FROM documents WHERE doi = ?", [doc_id])
            row2 = res2.fetchone()
            doc_cols = [d[0] for d in res2.description]
            row2_dict = dict(zip(doc_cols, row2))
            doc_obj = Document.from_duckdb_row(row2_dict)
            doc_obj.full_text = new_full_text
            doc_obj.total_char_length = total_char_length
            doc_obj.total_tokens = total_tokens
            doc_obj.n_pages = len(new_full_text)
            # refresh parsed_timestamp to reflect repair time
            try:
                from datetime import datetime, timezone
                doc_obj.parsed_timestamp = datetime.now(timezone.utc).isoformat()
            except Exception:
                pass
            try:
                doc_obj.file_hash = compute_file_hash(file_path)
            except Exception:
                pass

            ok = helper.force_update_document_text(doc_obj)
            if not ok:
                # Try fallback that preserves citations by snapshot/delete/restore
                logger.warning(f"Failed to update document text for {doc_id}; trying fallback")
                ok = helper.update_document_text_preserving_citations(doc_obj)
            results["processed"].append({
                "document_id": doc_id,
                "updated": bool(ok),
                "candidate_pages": candidate_pages,
                "n_pages": len(new_full_text),
            })

        return {"success": True, **results}
    except Exception as e:
        logger.error(f"repair_docs_ocr failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        helper.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
