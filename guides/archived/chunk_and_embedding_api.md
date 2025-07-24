## 📋 **Comprehensive Plan for API and Docker Integration with DuckDB and ChromaDB**

### **🎯 Phase 1: DuckDB Integration & Data Migration**

**1.1 DuckDB Schema Design (Aligned with Original Design)**
```sql
-- Documents table (matches src.models.Document exactly)
CREATE TABLE documents (
    doi VARCHAR PRIMARY KEY,
    has_dataset_citation BOOLEAN,
    full_text LIST<VARCHAR>, 
    total_char_length INTEGER,
    parsed_timestamp TIMESTAMP,
    total_chunks INTEGER,
    total_tokens INTEGER,
    avg_tokens_per_chunk REAL,
    file_hash VARCHAR,
    file_path VARCHAR,
    citation_entities LIST<VARCHAR>,  --Note: "flatten" each citation entity instance into a string with method then rehydrate
    n_pages INTEGER
);

-- Citations table (normalized, as suggested in original design)
CREATE TABLE citations (
    data_citation VARCHAR NOT NULL,
    document_id VARCHAR NOT NULL,
    pages LIST<INTEGER>,  -- Fall back to JSON for DuckDB <= 0.10
    evidence LIST<VARCHAR>,  -- Fall back to JSON for DuckDB <= 0.10
    PRIMARY KEY data_citation,
    FOREIGN KEY (doc_id) REFERENCES documents(doi)
);

-- Chunks table (matches src.models.Chunk, stores citation IDs as strings)
CREATE TABLE chunks (
    chunk_id VARCHAR PRIMARY KEY,
    document_id VARCHAR NOT NULL,  -- Note: using document_id to match Chunk model
    chunk_text TEXT NOT NULL, --Note: actual field name in pydantic class is "text' but changed to `chunk_text` here
    score REAL,  -- similarity score
    chunk_metadata STRUCT(
        created_at  TIMESTAMP,
        previous_chunk_id VARCHAR,
        next_chunk_id VARCHAR,
        token_count INTEGER
        citation_entities LIST<VARCHAR> --Note: "flatten" each citation entity instance into a string with method then rehydrate
    ),
    PRIMARY KEY chunk_id,
    FOREIGN KEY (document_id) REFERENCES documents(doi)
);

-- NOTE: Foreign key enforcement is off by default in DuckDB.
-- Run `SET FOREIGN_KEYS=ON;` on connection to enable it.

-- NOTE: Chunk stats can be generated on-the-fly (e.g., SELECT a, LEN(b) ...).
-- A separate table is not needed as DuckDB can compute these cheaply.

-- NOTE: NO embeddings table - ChromaDB is the sole source of truth for embeddings
```

**1.2 Data Migration Strategy**
- Migrate `Data/train/documents_with_known_entities.json` → `documents` table (using Document model fields)
- Migrate `Data/citation_entities_known.json` → `citation_entities` table (using CitationEntity model fields)
- Store chunking outputs in DuckDB instead of files (using Chunk model structure)

### **🔌 API Endpoints Design (Aligned with Original Microservice Design)**

**Core Principle: Simple endpoints that trigger the chunking & embedding pipeline**

**1. Main Pipeline Endpoint:**
```python
POST /run_semantic_chunking
# Triggers the semantic chunking pipeline as defined in run_semantic_chunking.py
# Reads from DuckDB, processes documents, stores results in DuckDB + ChromaDB

Request Body: None or minimal config
{
  "config_path": "configs/chunking.yaml"  # Optional, defaults to standard config
}

Response: ChunkingResult model
{
  "success": true,
  "total_documents": 100,
  "total_unique_datasets": 250,
  "total_chunks": 1500,
  "total_tokens": 300000,
  "avg_tokens_per_chunk": 200.0,
  "validation_passed": true,
  "pipeline_completed_at": "2024-01-10T12:00:00Z",
  "entity_retention": 100.0
}
```

**2. Document Processing Endpoint (if needed for specific documents):**
```python
POST /chunk/documents
# Process specific documents through the chunking pipeline

Request Body: List[Document]  # Using existing Document model
[
  {
    "doi": "10.1002/example1",
    "full_text": ["page1 text", "page2 text"],
    "total_char_length": 5000,
    # ... other Document fields
  }
]

Response: ChunkingResult model
```

**3. Health Check Endpoint:**
```python
GET /health
# Simple health check for the microservice

Response:
{
  "status": "healthy",
  "duckdb_connected": true,
  "chromadb_connected": true,
  "embedding_model": "text-embedding-3-small"
}
```

**Note:** Following the original design principle of creating a focused microservice for the chunking & embedding step, we avoid adding unnecessary endpoints like database queries, separate embedding endpoints, or complex pipeline configurations. The service should do one thing well: run the semantic chunking pipeline.

### **🏗️ Architecture (Aligned with Microservice Design)**

```
┌─────────────────────────────────────────────────────────────┐
│                    API Service Layer                        │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Server (uvicorn)                                   │
│  ├── /run_semantic_chunking (POST) - Main pipeline trigger  │
│  ├── /chunk/documents (POST) - Process specific documents   │      
│  └── /health (GET) - Health check                           │
├─────────────────────────────────────────────────────────────┤
│                    Business Logic Layer                     │
├─────────────────────────────────────────────────────────────┤
│  src/run_semantic_chunking.py (MODIFIED for DuckDB I/O)     │
│  src/semantic_chunking.py (UNCHANGED)                       │
│  src/helpers.py (UNCHANGED)                                 │
│  src/models.py (UNCHANGED - single source of truth)         │
├─────────────────────────────────────────────────────────────┤
│                    Data Layer                               │
├─────────────────────────────────────────────────────────────┤
│  DuckDB (Structured Data)                                   │
│  ├── documents table                                        │
│  ├── citations table                                        │
│  ├── chunks table                                           │
│  ChromaDB (Vector Storage Only)                             │
│  ├── Embeddings with chunk_id                               │
│  └── Metadata (doc_id, citation_ids)                        │
└─────────────────────────────────────────────────────────────┘
```

### **📁 Minimal File Structure (Following Original Design)**

```
api/
├── __init__.py
├── main.py              # FastAPI application with minimal endpoints
└── duckdb_utils.py      # Helper functions for DuckDB operations

# Modified existing files:
src/
├── run_semantic_chunking.py  # Modified to read/write from DuckDB
└── models.py                 # Add helper methods if needed (to_api_dict, from_db_row)

# No new model files, no complex directory structure
# Reuse existing code as much as possible
```

### **🔧 Implementation Steps (Aligned with Original Design)**

**Phase 1: DuckDB Schema & Data Migration** ✅ **COMPLETED**
1. ✅ Read all associated scripts and files thoroughly:
    - `src/models.py`
    - `chunk_and_embedding_api.md` (read this document for instructions on creating DuckDB helper functions as well as the database schema). **This very important.**
2. ✅ Create DuckDB schema initialization script (`api/duckdb_schema.py`)
3. ✅ Write simple migration script for existing JSON files (`api/migrate_data.py`):
   - `Data/train/documents_with_known_entities.json` → `documents` table (524 documents migrated)
   - Extract `Data/citation_entities_known.json` to `citations` table (487 citations migrated)
4. ✅ Test data migration locally by creating test script in `tests` directory (`tests/test_duckdb_migration.py`)

**Migration Results:**
- 524 documents successfully migrated
- 487 citations successfully migrated  
- 95 documents have citation relationships
- All data integrity constraints working properly
- Schema validation passed for all tables

**Phase 2: Modify run_semantic_chunking.py** ✅ **COMPLETED**
1. ✅ Read all associated scripts and files thoroughly:
    - `src/models.py`
    - `api/duckdb_schema.py`
    - `semantic_chunking.py` (read this script for functions on semantic chunking, loading and extracting chunks from ChromaDB which includes embedding) **This very important.**
    - `run_semantic_chunking.py`
    - `chunk_and_embedding_api.md` (read this document for instructions on how to modify the `run_semantic_chunking.py` script for loading inputs from DuckDB and saving outputs to DuckDB).
2. ✅ Add `load_input_data_from_duckdb()` function
3. ✅ Add `save_chunks_to_duckdb()` function
4. ✅ Update main pipeline to use DuckDB I/O
5. ⏳ Test modified pipeline with DuckDB by creating test script in `tests` directory (PENDING - will test in Docker environment)

**Phase 2 Implementation Results:**
- Successfully modified `src/run_semantic_chunking.py` to support DuckDB I/O
- Added `load_input_data_from_duckdb()` function that loads documents and citations from DuckDB
- Added `save_chunks_to_duckdb()` function that saves chunks to DuckDB 
- Updated main pipeline function `run_semantic_chunking_pipeline()` with:
  - New `use_duckdb` parameter (default: True)
  - New `db_path` parameter for database location
  - Conditional logic to use DuckDB or JSON files for data loading
  - Automatic chunk saving to DuckDB after ChromaDB persistence
- Created comprehensive test script `tests/test_run_semantic_chunking_duckdb.py` (ready for Docker testing)
- Pipeline now supports hybrid storage: DuckDB for structured data, ChromaDB for embeddings
- Maintains backward compatibility with existing JSON-based workflow

**Phase 3: Create Minimal API** ✅ **COMPLETED**
1. ✅ Read all associated scripts and files thoroughly:
    - `src/models.py`
    - `api/duckdb_schema.py`
    - `semantic_chunking.py` (read this script for functions on semantic chunking, loading and extracting chunks from ChromaDB which includes embedding) **This very important.**
    - `run_semantic_chunking.py`
    - `chunk_and_embedding_api.md` (read this document for instructions on  DuckDB helper functions).
2. ✅ Create `api/` directory with:
   - `main.py` - FastAPI app with 3 endpoints
   - `duckdb_utils.py` - Helper functions
3. ⏳ **PENDING** Test API endpoints by creating test script in `tests` directory

**Phase 3 Implementation Results:**
- Successfully created FastAPI application with 3 endpoints in `api/main.py`:
  - `POST /run_semantic_chunking` - Main pipeline trigger with query parameters
  - `POST /chunk/documents` - Process specific documents through pipeline
  - `GET /health` - Health check with database connection testing
- Created comprehensive DuckDB helper class in `api/duckdb_utils.py` with:
  - Database connection management and schema initialization
  - CRUD operations for documents, citations, and chunks
  - Query helpers and database statistics
  - Error handling and connection testing
- Created thorough test suite in `tests/test_api_endpoints.py` with:
  - Unit tests for all 3 endpoints
  - Error handling tests
  - Database integration tests
  - Manual testing capability for development
- **NOTE: Test script created but tests have not been run yet - testing is PENDING**
- API follows the minimal microservice design principle from the original plan
- All endpoints use existing Pydantic models from `src/models.py`
- Database operations use DuckDB for structured data and ChromaDB for embeddings
- Proper error handling and logging throughout

**Phase 4: Handle Offline Embeddings** ✅ **COMPLETED**
1. ✅ Read all associated scripts and files thoroughly:
    - `semantic_chunking.py` (read this script for functions on semantic chunking, loading and extracting chunks from ChromaDB which includes embedding) **This very important.**
    - `run_semantic_chunking.py`
    - `src.helpers.py`
    - `chunk_and_embedding_api.md` (read this document for instructions on which local model to download).
2. ✅ Add SentenceTransformer support to embedding logic in `semantic_chunking.py` but do NOT disable or delete current OpenAIEmbeddings functionalities.
3. ✅ Download and cache model for offline use
4. ✅ Update configuration inputs to use offline model when specified by user
5. ✅ Update `tests/test_run_semantic_chunking_duckdb.py` to include tests for local embedding model selection

**Phase 4 Implementation Results:**
- Successfully added `OfflineEmbedder` class that uses SentenceTransformers for local embeddings
- Enhanced `_build_embedder` function to support both OpenAI and offline models with automatic model selection
- Added `bge-small-en-v1.5` as the default offline model with support for multiple offline models
- Implemented model caching and download functionality with cache directory management
- Created offline-specific configuration file `configs/chunking_offline.yaml`
- Added comprehensive test suite for offline model functionality including:
  - Model download and caching tests
  - Offline embedder initialization and batch processing tests
  - Full pipeline testing with offline models
  - Model selection logic verification
- Updated project dependencies to include `sentence-transformers>=3.0.0` and related packages
- Maintained backward compatibility with existing OpenAI embedding functionality
- Added CLI support for downloading offline models via `python src/semantic_chunking.py download [model_name]`

**Phase 5: Docker Integration**
1. Read all associated scripts and files thoroughly:
    - `docker-compose.yml`
    - `api/main.py`
    - `chunk_and_embedding_api.md` (read this document for instructions on which dependencies to install (listed below)).
2. Create minimal Dockerfile:
   ```dockerfile
   FROM python:3.11-slim
   
   # Install only required dependencies
   RUN pip install uv

   uv add fastapi uvicorn duckdb chromadb sentence-transformers llama-index llama-index-core llama-index-embeddings-openai uvicorn openai python-dotenv tiktoken pandas pytest pydantic 
   
   # Copy necessary files
   COPY src/ /app/src/
   COPY api/ /app/api/
   COPY configs/ /app/configs/
   
   # Pre-download embedding model(actually let's leave this out for now)
   RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('bge-small-en-v1.5')"
   
   WORKDIR /app
   CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

**Dependencies Summary:**
- `duckdb` - For database operations
- `sentence-transformers` - For offline embeddings
- `uvicorn` - For running FastAPI (if not already installed)
- All other dependencies already exist in the project

### ** Pydantic Model Usage (Aligned with Original Design)**

**Core Principle: Reuse existing Pydantic models from `src/models.py` throughout the pipeline and API**

As stated in the original design:
- "We will strive to **reuse the same Pydantic models (`src.models.py`) throughout the pipeline and API**"
- "If we encounter any **minor incompatibility**... we will **not create a brand-new model**"

**1. Direct Usage of Existing Models:**

```python
# In API endpoints, directly use models from src/models.py
from src.models import Document, Chunk, CitationEntity, ChunkingResult, Dataset
from src.models import FirstClassifierInput, SecondClassifierInput

# For API request/response handling with FastAPI
from fastapi import FastAPI, HTTPException
from typing import List, Optional

app = FastAPI()

@app.post("/chunk/documents", response_model=ChunkingResult)
async def chunk_documents(documents: List[Document], chunk_size: int = 200):
    """Direct use of Document model for input, ChunkingResult for output"""
    # Implementation here
    pass

@app.post("/chunks", response_model=List[Chunk])
async def create_chunks(chunks: List[Chunk]):
    """Direct use of Chunk model"""
    # Implementation here
    pass
```

**2. Handling API-Specific Needs Without New Models:**

```python
# If API needs slightly different fields, use helper methods on existing models
# Example: Adding a method to existing models in src/models.py

class Document(BaseModel):
    # ... existing fields ...
    
    def to_api_dict(self, exclude_internal: bool = False) -> dict:
        """Convert to API-friendly dictionary, optionally excluding internal fields"""
        data = self.model_dump()
        if exclude_internal:
            data.pop('file_hash', None)
            data.pop('file_path', None)
        return data
```

**3. API Parameter Handling Without New Models:**

```python
# For API-specific parameters, use function parameters or query params
from fastapi import Query

@app.post("/embed/chunks")
async def embed_chunks(
    chunks: List[Chunk],  # Use existing Chunk model
    batch_size: int = Query(100, description="Batch size for processing"),
    model_name: Optional[str] = Query(None, description="Embedding model to use")
):
    """Handle API parameters without creating new request models"""
    # Implementation here
    pass

# For complex requests, use dictionaries with validation
@app.post("/chunk/pipeline")
async def run_pipeline(
    documents: List[Document],
    config: dict = {
        "chunk_size": 200,
        "chunk_overlap": 20,
        "save_to_chroma": True
    }
):
    """Use existing models with config dict for parameters"""
    # Validate config as needed
    chunk_size = config.get("chunk_size", 200)
    # Implementation here
    pass
```

**Note:** This approach maintains structural consistency and avoids model drift, as emphasized in the original design. Any transformations needed for API responses are handled through methods on the existing models rather than creating parallel model hierarchies.

### ** Implementation Approach (Aligned with Original Design)**

**Key Principle: Modify existing code minimally, reuse as much as possible**

**1. Modifying `src/run_semantic_chunking.py` for DuckDB I/O:**

```python
# Modified sections of run_semantic_chunking.py

import duckdb
from src.models import Document, CitationEntity, Chunk

def load_input_data_from_duckdb(db_path: str = "artifacts/mdc_challenge.db"):
    """Load documents and citations from DuckDB instead of JSON files"""
    conn = duckdb.connect(db_path)
    
    # Load documents
    doc_rows = conn.execute("SELECT * FROM documents").fetchall()
    documents = []
    for row in doc_rows:
        row_dict = dict(row)
        document = Document.from_duckdb_row(row=row_dict)
        documents.append(document)
    
    # Load citations if stored separately
    citation_rows = conn.execute("SELECT * FROM citations").fetchall()
    citations = []
    for row in citation_rows:
        row_dict = dict(row)
        citation = CitationEntity.from_duckdb_row(row=row_dict)
        citations.append(citation)
    
    conn.close()
    return documents, citations

import duckdb, pyarrow as pa

def insert_chunks(conn: duckdb.DuckDBPyConnection, chunks: list[Chunk]) -> None:
    tbl = pa.Table.from_pylist([c.to_duckdb_row() for c in chunks])
    conn.register("tmp_chunks", tbl)
    conn.execute("""
        INSERT INTO chunks SELECT * FROM tmp_chunks
    """)

def load_chunk(conn: duckdb.DuckDBPyConnection, id_: int) -> Chunk | None:
    rec = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", [id_]).fetchone()
    return None if rec is None else Chunk.from_duckdb_row(rec)

# Update the main pipeline function
def run_semantic_chunking_pipeline(config_path: Optional[str] = None):
    """Modified to use DuckDB for I/O"""
    # Load from DuckDB instead of JSON
    documents, citations_by_doc = load_input_data_from_duckdb()
    
    # ... existing chunking logic ...
    
    # Save to DuckDB instead of JSON/CSV
    insert_chunks(chunks)
    
    # Continue with ChromaDB for embeddings (unchanged)
    save_chunk_objs_to_chroma(chunks, collection_name, embedder)
    
    return ChunkingResult(...)
```

**2. Simple API Implementation (`api/main.py`):**

```python
from fastapi import FastAPI, HTTPException
from typing import List, Optional
import sys
sys.path.append("..")  # Add parent directory to path

from src.models import Document, ChunkingResult
from src.run_semantic_chunking import run_semantic_chunking_pipeline
import duckdb

app = FastAPI(title="Chunking & Embedding Microservice")

@app.post("/run_semantic_chunking", response_model=ChunkingResult)
async def run_pipeline(config_path: Optional[str] = None):
    """Trigger the semantic chunking pipeline"""
    try:
        result = run_semantic_chunking_pipeline(config_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chunk/documents", response_model=ChunkingResult)
async def chunk_specific_documents(documents: List[Document]):
    """Process specific documents through the pipeline"""
    # Save documents to DuckDB temporarily
    conn = duckdb.connect("artifacts/mdc_challenge.db")
    
    for doc in documents:
        # Insert document to DuckDB
        conn.execute("""
            INSERT OR REPLACE INTO documents 
            (doi, has_dataset_citation, full_text, total_char_length, 
             parsed_timestamp, file_hash, file_path, n_pages)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc.doi,
            doc.has_dataset_citation,
            doc.full_text,
            doc.total_char_length,
            doc.parsed_timestamp,
            doc.file_hash,
            doc.file_path,
            doc.n_pages
        ))
    
    conn.commit()
    conn.close()
    
    # Run pipeline
    return run_semantic_chunking_pipeline()

@app.get("/health")
async def health_check():
    """Simple health check, including a write test for permissions."""
    try:
        # Check DuckDB connection and write permissions
        conn = duckdb.connect("artifacts/mdc_challenge.db")
        # Test write permissions with a rollback to avoid side effects
        conn.execute("CREATE TABLE IF NOT EXISTS health_check (i INTEGER);")
        conn.execute("BEGIN; INSERT INTO health_check VALUES (1); ROLLBACK;")
        conn.close()
        duckdb_ok = True
    except Exception:
        duckdb_ok = False
    
    return {
        "status": "healthy" if duckdb_ok else "unhealthy",
        "duckdb_connected": duckdb_ok,
        "chromadb_connected": True,  # Assume ChromaDB is OK
        "embedding_model": "text-embedding-3-small"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**3. DuckDB Helper Functions (`api/duckdb_utils.py`):**
```python
from pathlib import Path
from typing import List
import duckdb
from src.models import Document, CitationEntity, Chunk, DocumentRow, CitationEntityRow

class DuckDBHelper:
    def __init__(self, db_path: str = "artifacts/mdc_challenge.db"):
        self.db_path = db_path
        self.engine = self._create_engine()
        self._initialize_schema()
    
    def _create_engine(self):
        """
        Create DuckDB connection and engine.
        NOTE: DuckDB operates in single-writer mode. This means only one process
        can have a write transaction at a time. Concurrent calls to endpoints
        that write to the DB (e.g., /run_semantic_chunking) from multiple
        API workers will lead to a TransactionException.
        """
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(db_path))
    
    def _initialize_schema(self):
        """Initialize database schema if it doesn't exist."""
        # Create documents table
        self.engine.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doi VARCHAR PRIMARY KEY,
                has_dataset_citation BOOLEAN,
                full_text LIST<VARCHAR>,
                total_char_length INTEGER,
                parsed_timestamp TIMESTAMP,
                total_chunks INTEGER,
                total_tokens INTEGER,
                avg_tokens_per_chunk REAL,
                file_hash VARCHAR,
                file_path VARCHAR,
                citation_entities LIST<VARCHAR>,
                n_pages INTEGER
            )
        """)
        
        # Create citation_entities table
        self.engine.execute("""
            CREATE TABLE IF NOT EXISTS citations (
                data_citation VARCHAR NOT NULL,
                doc_id VARCHAR NOT NULL,
                pages LIST<INTEGER>,
                evidence LIST<VARCHAR>,
                PRIMARY KEY (data_citation, doc_id),
                FOREIGN KEY (doc_id) REFERENCES documents(doi)
            )
        """)
        
        # Create chunks table
        self.engine.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id VARCHAR PRIMARY KEY,
                document_id VARCHAR NOT NULL,
                chunk_text TEXT NOT NULL,
                score REAL,
                chunk_metadata STRUCT(
                    created_at TIMESTAMP,
                    previous_chunk_id VARCHAR,
                    next_chunk_id VARCHAR,
                    token_count INTEGER,
                    citation_entities LIST<VARCHAR>
                ),
                FOREIGN KEY (document_id) REFERENCES documents(doi)
            )
        """)
        
        # NOTE: chunk_stats is no longer a persisted table.
        # Stats can be generated on-the-fly from the `chunks` table.
    
    def get_documents_by_query(self, query: str, limit: int = 100) -> List[Document]:
        """Get documents using SQL query."""
        try:
            result = self.engine.execute(f"{query} LIMIT {limit}")
            rows = result.fetchall()
            documents = []
            for row in rows:
                row_dict = dict(zip([desc[0] for desc in result.description], row))
                documents.append(Document.from_duckdb_row(row_dict))
            return documents
        except Exception as e:
            raise ValueError(f"Database query failed: {str(e)}")
    
    def get_citation_entities_by_doc_id(self, doc_id: str) -> List[CitationEntity]:
        """Get citation entities for a specific document."""
        result = self.engine.execute(
            "SELECT * FROM citations WHERE doc_id = ?",
            [doc_id]
        )
        rows = result.fetchall()
        entities = []
        for row in rows:
            row_dict = dict(zip([desc[0] for desc in result.description], row))
            entities.append(CitationEntity.from_duckdb_row(row_dict))
        return entities
    
    def store_chunks(self, chunks: List[Chunk]) -> bool:
        """Store chunks in DuckDB."""
        try:
            for chunk in chunks:
                chunk_row = chunk.to_duckdb_row()
                self.engine.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, document_id, chunk_text, score, chunk_metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    chunk_row["chunk_id"],
                    chunk_row["document_id"],
                    chunk_row["chunk_text"],
                    chunk_row["score"],
                    chunk_row["chunk_metadata"],
                ))
            return True
        except Exception as e:
            raise ValueError(f"Failed to store chunks: {str(e)}")
```

**`api/services/embedding_service.py`:** 
**Actually for this, just use existing functions in `src.semantic_chunking` that both creates the embeddings & automatically upsert and/or persist embeddings to Chromadb. These functions are: 
`save_chunk_objs_to_chroma`, `save_chunks_to_chroma`, `save_chunk_to_chroma`, `save_chunk_obj_to_chroma`**


### **🔄 Data Flow with DuckDB + ChromaDB Hybrid Approach**

```
┌─────────────────────────────────────────────────────────────┐
│                    API Request                              │
├─────────────────────────────────────────────────────────────┤
│  POST /chunk/documents                                      │
│  {                                                          │
│    "document_ids": ["10.1002/example1", "10.1002/example2"],│
│    "chunk_size": 200                                        │
│  }                                                          │
├─────────────────────────────────────────────────────────────┤
│                    DuckDB Service                           │
├─────────────────────────────────────────────────────────────┤
│  1. Fetch documents from DuckDB                             │
│  2. Convert to Document objects using model_validate        │
│  3. Pass to chunking service                                │
├─────────────────────────────────────────────────────────────┤
│                    Chunking Service                         │
├─────────────────────────────────────────────────────────────┤
│  1. Create chunks using existing semantic_chunking logic    │
│  2. Store chunk metadata in DuckDB (text, IDs, stats)       │
│  3. Store embeddings ONLY in ChromaDB (with chunk_id)       │
├─────────────────────────────────────────────────────────────┤
│                    ChromaDB Service                         │
├─────────────────────────────────────────────────────────────┤
│  1. Upsert chunk embeddings with metadata                   │
│  2. Store chunk_id, document_id, citation IDs as metadata   │
│  3. Enable fast similarity search                           │
├─────────────────────────────────────────────────────────────┤
│                    Response (ChunkingResult model)          │
├─────────────────────────────────────────────────────────────┤
│  {                                                          │
│    "success": true,                                         │
│    "total_documents": 2,                                    │
│    "total_chunks": 150,                                     │
│    "entity_retention": 100.0                                │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

### **📊 Updated Usage Examples**

**DuckDB Integration Usage:**
```bash
# Query documents from database
curl -X POST "http://localhost:8000/db/query" \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "SELECT doi, title FROM documents WHERE token_count > 1000",
    "limit": 50
  }'

# Chunk documents using database query
curl -X POST "http://localhost:8000/chunk/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "SELECT * FROM documents WHERE doi LIKE '\''10.1002%'\''",
    "chunk_size": 200,
    "chunk_overlap": 20
  }'

# Embed chunks and store in database
curl -X POST "http://localhost:8000/embed/chunks" \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": ["chunk1", "chunk2", "chunk3"],
    "batch_size": 100,
    "return_metadata": true
  }'
```

**Embedding API Usage:**
```bash
# Single text embedding
curl -X POST "http://localhost:8000/embed/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a sample text for embedding.",
    "return_metadata": true
  }'

# Bulk chunk embedding
curl -X POST "http://localhost:8000/embed/chunks" \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      "First chunk of text...",
      "Second chunk of text...",
      "Third chunk of text..."
    ],
    "batch_size": 50,
    "return_metadata": true
  }'

# Combined workflow: chunk then embed
curl -X POST "http://localhost:8000/chunk/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Long document text...", "chunk_size": 200}' | \
jq -r '.chunks' | \
curl -X POST "http://localhost:8000/embed/chunks" \
  -H "Content-Type: application/json" \
  -d '{"chunks": $chunks, "batch_size": 100}'
```

### ** Updated Security & Configuration**

**Environment Variables:**
- `OPENAI_API_KEY` - Required for embeddings
- `EMBEDDING_MODEL` - Default embedding model (default: text-embedding-3-small)
- `EMBEDDING_BATCH_SIZE` - Default batch size for bulk operations (default: 100)
- `EMBEDDING_RATE_LIMIT` - Rate limiting for OpenAI API calls
- `DUCKDB_PATH` - Path to DuckDB database file (default: artifacts/mdc_challenge.db)
- `CHUNKING_CONFIG` - Path to chunking configuration file

**Configuration Management:**
- Extend existing `configs/chunking.yaml` with embedding settings
- Add API-specific embedding configuration
- Support for different embedding models and providers
- Database connection pooling and configuration

### **📈 Offline Capability & Embedding Model Selection**

**Critical Requirement: The container must run entirely offline**

As noted in the original design:
- "In our case, we're using the OpenAI API for embeddings, which in a truly offline scenario would be an issue"
- "We actually need to switch to a local embedding model or use precomputed embeddings"

**Offline Embedding Solution:**
```python
# Modified embedding approach for offline execution
from sentence_transformers import SentenceTransformer

class OfflineEmbedder:
    def __init__(self, model_name="bge-small-en-v1.5"):
        # Use a small, efficient model for offline embeddings
        self.model = SentenceTransformer(model_name)
    
    def get_text_embedding(self, text: str):
        # Generate embedding locally
        return self.model.encode(text).tolist()

# Update _build_embedder in semantic_chunking.py to use offline model
def _build_embedder(embed_model: str):
    if embed_model == "offline":
        return OfflineEmbedder()
    else:
        # Original OpenAI embedder for development
        return CustomOpenAIEmbedding(model_name=embed_model)
```

**Docker Considerations:**
- Pre-download the SentenceTransformer model during image build
- Include model files in the Docker image
- No external API calls at runtime

### **🔍 Integration with Existing Code**

The API will leverage your existing infrastructure while maintaining separation:

1. **`src/helpers.py`** - Reuse `get_embedding()` and `CustomOpenAIEmbedding`
2. **`src/semantic_chunking.py`** - Reuse `_build_embedder()` and `_load_cfg()`
3. **`configs/chunking.yaml`** - Extend with embedding-specific configurations
4. **`src/models.py`** - UNCHANGED, all new models in separate files

### ** Migration Strategy**

**Step 1: Create DuckDB Infrastructure**
1. Create database schema
2. Implement migration scripts
3. Add database connection management

**Step 2: Migrate Existing Data**
1. Migrate `documents_with_known_entities.json`
2. Migrate `citation_entities_known.json`
3. Validate data integrity

**Step 3: Update Existing Functions**
1. Modify `run_semantic_chunking_pipeline()` to use DuckDB
2. Update chunk storage to use database instead of files
3. Add database result storage

**Step 4: Create API Layer**
1. Implement endpoints with DuckDB integration
2. Add query validation and security
3. Create comprehensive error handling

## **Summary: Aligned Design Principles**

This updated plan follows the design from `Chunking_and_Embedding_MicroService.md`:

1. **Preserves Pydantic Models**: Reuses existing models from `src/models.py` without creating parallel hierarchies
2. **Maintains Classifier Compatibility**: Ensures output remains compatible with `FirstClassifierInput` and `SecondClassifierInput`
3. **Hybrid Storage Strategy**: DuckDB for structured data, ChromaDB exclusively for embeddings
4. **Minimal API Surface**: Simple microservice focused on running the chunking pipeline
5. **Offline Capability**: Uses SentenceTransformers for offline embedding generation
6. **Lightweight Implementation**: Modifies existing code rather than creating extensive new structures

Key differences from the initial divergent plan:
- No separate API models or complex directory structures
- No database query endpoints or unnecessary features
- No embedding storage in DuckDB
- Focus on modifying `run_semantic_chunking.py` rather than creating new services
- Minimal Docker image with only essential dependencies

### **📦 Dependencies Summary**

**New dependencies needed:**
```bash
# Activate virtual environment
source .venv/bin/activate 

# Essential new dependencies
uv add duckdb                    # Database operations
uv add sentence-transformers     # Offline embeddings
# uvicorn should already be installed with FastAPI
```

**Existing dependencies to leverage:**
- `fastapi` - Web framework (already in project)
- `pydantic` - Data validation (already in project)
- `chromadb` - Vector database (already in project)
- All other existing dependencies remain unchanged