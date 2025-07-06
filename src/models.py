from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class Section(BaseModel):
    page_start: int
    page_end: Optional[int] = None
    section_type: Optional[str] = None  # "methods", "results", "data_availability", etc.
    subsections: Optional[List[str]] = []

class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str  # DOI from step 5
    section_type: Optional[str] = None        # Primary section containing this chunk
    section_order: Optional[int] = None       # Order within document sections
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    chunk_type: Optional[str] = None          # "body", "header", "caption"
    conversion_source: Optional[str] = None   # "grobid", "pdfplumber", "existing_xml"
    token_count: Optional[int] = None
    citation_entities: Optional[List[str]] = []  # Entities found in this chunk

class Chunk(BaseModel):
    chunk_id: str
    text: str
    score: Optional[float] = None       # similarity score (added later)
    chunk_metadata: ChunkMetadata
    
    def __str__(self):
        return f"Chunk({self.chunk_id[:8]}..., {len(self.text)} chars, {self.chunk_metadata.section_type})"

class ChunkingResult(BaseModel):
    """Result of the chunking pipeline"""
    chunks: List[Chunk]
    total_documents: int
    total_chunks: int
    metadata: Dict[str, any] = Field(default_factory=dict) 

class PreprocessingReport(BaseModel):
    """Report of the preprocessing pipeline"""
    pass