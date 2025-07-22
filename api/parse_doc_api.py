# api/parse_doc_api.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, sys
from pathlib import Path
from typing import Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api.services.document_parsing_service import build_document_objects, build_document_object
from api.utils.duckdb_utils import get_duckdb_helper
from src.helpers import export_docs, initialize_logging

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
@app.get("/bulk_parse_docs")
async def parse_docs(pdf_paths: List[str], export_file: Optional[str] = None, export_path: Optional[str] = None):
    helper = get_duckdb_helper(DB_PATH)
    try:
        documents = build_document_objects(pdf_paths=pdf_paths, subset=False)
        if not documents:
            raise HTTPException(status_code=400, detail="No document objects built")
        helper.store_documents(documents)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
