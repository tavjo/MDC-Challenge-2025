from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal, Any
from datetime import datetime
import pandas as pd

class Section(BaseModel):
    # page_start: int
    # page_end: int
    text: str
    section_type: str  # "methods", "results", "data_availability","unknown" etc.
    # section_label: str
    subsections: Optional[List[str]] = []
    order: int
    char_length: int
    sec_level: int
    original_type: str

class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str  # DOI from step 5
    section_type: Optional[str] = None        # Primary section containing this chunk
    section_order: Optional[int] = None       # Order within document sections
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    chunk_type: Optional[str] = None          # "body", "header", "caption"
    format_type: Literal["TEI", "JATS", "UNKNOWN"] = Field(..., description="XML Format of the document")
    conversion_source: Optional[Literal["GROBID", "PDFPLUMBER"]] = Field(None, description="Source of the document")
    token_count: Optional[int] = None
    citation_entities: Optional[List[str]] = []  # Entities found in this chunk

class Document(BaseModel):
    doi: str = Field(..., description="DOI or unique identifier of the document")
    has_dataset_citation: Optional[bool] = Field(None, description="Whether the document has 1 or more dataset citation")
    full_text: str = Field(..., description="Full text of the document")
    section_labels: List[str] = Field(..., description="List of section labels in the document")
    sections: List[Section] = Field(..., description="List of sections in the document")
    section_count: int = Field(..., description="Number of sections in the document")
    # section_order: List[int] = Field(..., description="Order of sections in the document")
    total_char_length: int = Field(..., description="Total number of characters in the document")
    clean_text_length: int = Field(..., description="Total number of characters in the document after cleaning")
    format_type: Literal["TEI", "JATS", "UNKNOWN"] = Field(..., description="XML Format of the document")
    source_type: Optional[str] = Field(None, description="Source type of the document")
    conversion_source: Optional[Literal["GROBID", "PDFPLUMBER"]] = Field(None, description="Source of the document")
    # sections_with_text: int = Field(..., description="Number of sections with text in the document")
    parsed_timestamp: str = Field(..., description="Timestamp of when the document was parsed")
    validated: bool = Field(..., description="Whether the document has been validated")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks in the document")
    total_tokens: Optional[int] = Field(None, description="Total number of tokens in the document")
    avg_tokens_per_chunk: Optional[float] = Field(None, description="Average number of tokens per chunk in the document")
    xml_hash: str = Field(..., description="Hash of the XML file")
    file_path: str = Field(..., description="Path to the XML file")

class Chunk(BaseModel):
    chunk_id: str
    text: str
    score: Optional[float] = None       # similarity score (added later)
    chunk_metadata: ChunkMetadata
    
    def __str__(self):
        return f"Chunk({self.chunk_id[:8]}..., {len(self.text)} chars, {self.chunk_metadata.section_type})"

class ChunkingResult(BaseModel):
    """Result of the chunking pipeline"""
    success: bool = Field(..., description="Whether the chunking pipeline completed successfully.")
    total_documents: int = Field(..., description="Total number of documents processed.")
    total_unique_datasets: int = Field(..., description="Total number of unique entities (dataset citations) found.")
    total_chunks: int = Field(..., description="Total number of chunks created.")
    total_tokens: int = Field(..., description="Total number of tokens in all chunks.")
    avg_tokens_per_chunk: float = Field(..., description="Average number of tokens per chunk.")
    validation_passed: bool = Field(..., description="Whether the chunking pipeline passed validation.")
    pipeline_completed_at: str = Field(..., description="Timestamp of when the pipeline completed or failed.")
    entity_retention: float = Field(..., description="Percentage of dataset IDs retained after chunking; aim is 100%.")
    output_path: Optional[str] = Field(None, description="Path to the output files.")
    output_files: Optional[List[str]] = Field(None, description="List of output file paths.")
    lost_entities: Optional[Dict[str, Any]]= Field(None, description="List of dataset IDs that were lost during chunking; aim is 0. Keep for sanity check.") 
    error: Optional[str] = Field(None, description="Error message if pipeline failed.")

class Dataset(BaseModel):
    """Dataset Citation Extracted from Document text"""
    dataset_id: str = Field(..., description="Dataset ID")
    doc_id: str = Field(..., description="DOI in which the dataset citation was found")
    total_tokens: int = Field(..., description="Total number of tokens in all chunks.")
    avg_tokens_per_chunk: float = Field(..., description="Average number of tokens per chunk.")
    dataset_url: Optional[str] = Field(None, description="Dataset URL")
    total_char_length: int = Field(..., description="Total number of characters")
    clean_text_length: int = Field(..., description="Total number of characters after cleaning")
    format_type: Literal["TEI", "JATS", "UNKNOWN"] = Field(..., description="XML Format of the document")
    conversion_source: Optional[Literal["GROBID", "PDFPLUMBER"]] = Field(None, description="Source of the document")
    dataset_type: Optional[str] = Field(None, description="Dataset Type: main target of the classification task")

class FirstClassifierInput(BaseModel):
    """Input for classifier to determine if a document has a dataset citation"""
    doc: Document = Field(..., description="Document to be classified")
    embeddings: List[float] = Field(..., description="Embeddings of the document")
    UMAP_1: Optional[float] = Field(None, description="UMAP 1 dimension")
    UMAP_2: Optional[float] = Field(None, description="UMAP 2 dimension")
    PC_1: Optional[float] = Field(None, description="PC 1 dimension")
    PC_2: Optional[float] = Field(None, description="PC 2 dimension")


class SecondClassifierInput(BaseModel):
    """Input for classifer that predicts the type of dataset citation"""
    dataset: Dataset = Field(..., description="Dataset to be classified")
    embeddings: List[float] = Field(..., description="Embeddings of the dataset")
    UMAP_1: Optional[float] = Field(None, description="UMAP 1 dimension")
    UMAP_2: Optional[float] = Field(None, description="UMAP 2 dimension")
    PC_1: Optional[float] = Field(None, description="PC 1 dimension")
    PC_2: Optional[float] = Field(None, description="PC 2 dimension")

class PreprocessingReport(BaseModel):
    """Report of the preprocessing pipeline"""
    pass