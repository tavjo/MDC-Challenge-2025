from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal, Any, Union
from datetime import datetime
import pandas as pd

class Section(BaseModel):
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    text: str
    section_type: str  # "methods", "results", "data_availability","unknown" etc.
    subsections: Optional[List[str]] = []
    order: int
    char_length: int
    sec_level: Optional[int] = None
    original_type: Optional[str] = None

class CitationEntity(BaseModel):
    data_citation: str = Field(..., description="Data citation from text")
    doc_id: str = Field(..., description="DOI of the document where the data citation is found")
    pages: Optional[List[int]] = Field(None, description="List of page numbers where the data citation is mentioned.")
    evidence: Optional[List[str]] = Field(None, description="List of evidence from the text for the data citation")


class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str  # DOI from step 5
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    token_count: int = Field(..., description="Number of tokens in the chunk")
    citation_entities: Optional[List[CitationEntity]] = Field(None, description="Citation entities found in this chunk")  # Entities found in this chunk

class Chunk(BaseModel):
    chunk_id: str
    text: str
    score: Optional[float] = None       # similarity score (added later)
    chunk_metadata: ChunkMetadata
    
    # def __str__(self):
    #     return f"Chunk({self.chunk_id[:8]}..., {len(self.text)} chars, {self.chunk_metadata.section_type})"

class Document(BaseModel):
    doi: str = Field(..., description="DOI or unique identifier of the document")
    has_dataset_citation: Optional[bool] = Field(None, description="Whether the document has 1 or more dataset citation")
    full_text: Union[str, List[str]] = Field(..., description="Full text of the document")
    total_char_length: int = Field(..., description="Total number of characters in the document")
    # clean_text_length: int = Field(..., description="Total number of characters in the document after cleaning")
    parsed_timestamp: str = Field(..., description="Timestamp of when the document was parsed")
    # validated: bool = Field(..., description="Whether the document has been validated")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks in the document")
    total_tokens: Optional[int] = Field(None, description="Total number of tokens in the document")
    avg_tokens_per_chunk: Optional[float] = Field(None, description="Average number of tokens per chunk in the document")
    file_hash: str = Field(..., description="Hash of the document file")
    file_path: str = Field(..., description="Path to the document file")
    citation_entities: Optional[List[CitationEntity]] = Field(None, description="List of citation entities found in the document")
    n_pages: Optional[int] = Field(None, description="Number of pages in the document")

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
    total_char_length: int = Field(..., description="Total number of characters")
    clean_text_length: int = Field(..., description="Total number of characters after cleaning")
    cluster: Optional[str] = Field(None, description="Cluster of the dataset citation")
    dataset_type: Optional[Literal["PRIMARY", "SECONDARY"]] = Field(None, description="Dataset Type: main target of the classification task")
    text: str = Field(..., description= "Text in the document where the dataset citation is found") 

class FirstClassifierInput(BaseModel):
    """Input for classifier to determine if a document has a dataset citation"""
    doc: Document = Field(..., description="Document to be classified")
    embeddings: List[float] = Field(..., description="Embeddings of the text within the document potentially containing dataset citations")
    UMAP_1: Optional[float] = Field(None, description="UMAP 1 dimension")
    UMAP_2: Optional[float] = Field(None, description="UMAP 2 dimension")
    PC_1: Optional[float] = Field(None, description="PC 1 dimension")
    PC_2: Optional[float] = Field(None, description="PC 2 dimension")
    has_data_citation: bool = Field(..., description="Whether the document has a dataset citation: first classifer target variable")

class SecondClassifierInput(BaseModel):
    """Input for classifer that predicts the type of dataset citation"""
    dataset: Dataset = Field(..., description="Dataset to be classified")
    embeddings: List[float] = Field(..., description="Embeddings of the text containing the dataset citation")
    UMAP_1: float = Field(None, description="UMAP 1 dimension")
    UMAP_2: float = Field(None, description="UMAP 2 dimension")
    PC_1: float = Field(None, description="PC 1 dimension")
    PC_2: Optional[float] = Field(None, description="PC 2 dimension")
    Cluster: Optional[str] = Field(None, description="Cluster of the dataset citation")
    is_primary: bool = Field(..., description="Whether the dataset citation is a primary dataset citation. Since there are only 2 classes at this stage, anything else is a secondary dataset citation.")
    # Allow arbitrary additional fields per nf-core samplesheet flexibility
    class Config:
        extra = "allow"

class PreprocessingReport(BaseModel):
    """Report of the preprocessing pipeline"""
    pass