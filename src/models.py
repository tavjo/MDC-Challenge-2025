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
    document_id: str = Field(..., description="DOI of the document where the data citation is found")
    pages: Optional[List[int]] = Field(None, description="List of page numbers where the data citation is mentioned.")
    evidence: Optional[List[str]] = Field(None, description="List of evidence from the text for the data citation")

    def to_string(self) -> str:
        """
        Flatten the citation entity into a string.
        """
        pages_str = ",".join(map(str, self.pages)) if self.pages else ""
        evidence_str = ",".join(self.evidence) if self.evidence else ""
        return f"{self.data_citation}|{self.document_id}|{pages_str}|{evidence_str}"

    @classmethod
    def from_string(cls, citation_entity_str: str) -> "CitationEntity":
        """
        Rehydrate the citation entity from a string.
        """
        parts = citation_entity_str.split("|", 3)
        data_citation, document_id, pages_str, evidence_str = (
            parts[0],
            parts[1],
            parts[2] if len(parts) > 2 else "",
            parts[3] if len(parts) > 3 else "",
        )

        pages = [int(p) for p in pages_str.split(",") if p] if pages_str else None
        evidence = evidence_str.split(",") if evidence_str else None

        return cls(
            data_citation=data_citation,
            document_id=document_id,
            pages=pages,
            evidence=evidence,
        )
    
    def to_duckdb_row(self) -> Dict[str, Any]:
        """
        Convert the citation entity into a dictionary that can be inserted into DuckDB.
        """
        return {
            "data_citation": self.data_citation,
            "document_id": self.document_id,
            "pages": self.pages,
            "evidence": self.evidence,
        }
    
    @classmethod
    def from_duckdb_row(cls, row: Dict[str, Any]) -> "CitationEntity":
        """
        Imports the citation entity from a DuckDB row.
        """
        return cls(
            data_citation=row["data_citation"],
            document_id=row["document_id"],
            pages=row["pages"],
            evidence=row["evidence"]
        )

class ChunkMetadata(BaseModel):
    chunk_id: str
    # document_id: str  # DOI from step 5
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    token_count: int = Field(..., description="Number of tokens in the chunk")
    citation_entities: Optional[List[CitationEntity]] = Field(None, description="Citation entities found in this chunk")  # Entities found in this chunk

class Chunk(BaseModel):
    chunk_id: str
    document_id: str #NEW: moved from ChunkMetadata to Chunk
    text: str
    score: Optional[float] = None       # similarity score (added later)
    chunk_metadata: ChunkMetadata

    def to_duckdb_row(self) -> Dict[str, Any]:
        """
        Convert the chunk and associated metadata into a dictionary that can be inserted into DuckDB.
        """
        citation_entities = [ce.to_string() for ce in self.chunk_metadata.citation_entities] if self.chunk_metadata.citation_entities else []
        chunk_metadata_dict = {
                "previous_chunk_id": self.chunk_metadata.previous_chunk_id,
                "next_chunk_id": self.chunk_metadata.next_chunk_id,
                "token_count": self.chunk_metadata.token_count,
                "citation_entities": citation_entities,
                "created_at": datetime.now().isoformat(),
            }
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_text": self.text,
            "score": self.score,
            "chunk_metadata": chunk_metadata_dict
        }

    @classmethod
    def from_duckdb_row(cls, row: Dict[str, Any]) -> "Chunk":
        """
        Rehydrate the chunk from a DuckDB row.
        """
        citation_entities = [CitationEntity.from_string(ce_str) for ce_str in row["chunk_metadata"]["citation_entities"] if ce_str]
        return cls(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            text=row["chunk_text"],
            score=row["score"],
            chunk_metadata = ChunkMetadata(
                chunk_id=row["chunk_id"],
                previous_chunk_id=row["chunk_metadata"]["previous_chunk_id"],
                next_chunk_id=row["chunk_metadata"]["next_chunk_id"],
                token_count=row["chunk_metadata"]["token_count"],
                citation_entities=citation_entities
            )
        )


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

    def to_duckdb_row(self) -> Dict[str, Any]:
        """
        Convert the document and associated metadata into a dictionary that can be inserted into DuckDB.
        """
        citation_entities = [ce.to_string() for ce in self.citation_entities] if self.citation_entities else []
        return {
            "doi": self.doi,
            "has_dataset_citation": self.has_dataset_citation,
            "full_text": self.full_text,
            "total_char_length": self.total_char_length,
            "parsed_timestamp": self.parsed_timestamp,
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_chunk": self.avg_tokens_per_chunk,
            "file_hash": self.file_hash,
            "file_path": self.file_path,
            "citation_entities": citation_entities,
            "n_pages": self.n_pages,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

    @classmethod
    def from_duckdb_row(cls, row: Dict[str, Any]) -> "Document":
        """
        Rehydrate the document from a DuckDB row.
        """
        citation_entities = [CitationEntity.from_string(ce_str) for ce_str in row["citation_entities"] if ce_str]
        return cls(
            doi=row["doi"],
            has_dataset_citation=row["has_dataset_citation"],
            full_text=row["full_text"],
            total_char_length=row["total_char_length"],
            parsed_timestamp=str(row["parsed_timestamp"]),
            total_chunks=row["total_chunks"],
            total_tokens=row["total_tokens"],
            avg_tokens_per_chunk=row["avg_tokens_per_chunk"],
            file_hash=row["file_hash"],
            file_path=row["file_path"],
            citation_entities=citation_entities,
            n_pages=row["n_pages"]
        )


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
    lost_entities: Optional[Dict[str, Any]] = Field(None, description="List of dataset IDs that were lost; aim is 0.")
    error: Optional[str] = Field(None, description="Error message if pipeline failed.")

# New: per-document result model
class DocumentChunkingResult(BaseModel):
    document_id: str
    success: bool
    error: Optional[str] = None

    # pipeline parameters
    chunk_size: int
    chunk_overlap: int
    cfg_path: str
    collection_name: str

    # citation stats
    pre_chunk_total_citations: int
    post_chunk_total_citations: int
    validation_passed: bool
    entity_retention: float
    lost_entities: Optional[Dict[str, Any]] = None

    # chunk stats
    total_chunks: int
    total_tokens: int
    avg_tokens_per_chunk: float

    # outputs
    output_path: Optional[str] = None
    output_files: Optional[List[str]] = None

    # timing
    pipeline_started_at: str
    pipeline_completed_at: str

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

class ChunkingPipelinePayload(BaseModel):
    """Payload for the chunking API"""
    chunk_size: Optional[int] = Field(300, description="Target chunk size in tokens")
    chunk_overlap: Optional[int] = Field(2, description="Overlap between chunks in tokens")
    collection_name: str = Field(..., description="Name of the collection in ChromaDB")
    local_model: Optional[bool] = Field(False, description="Whether to use the local model")
    cfg_path: str = Field(..., description="Path to the configuration file")
    db_path: str = Field(..., description="Path to the DuckDB database")
    subset: Optional[bool] = Field(False, description="Whether to use a subset of the data")
    subset_size: Optional[int] = Field(5, description="Size of the subset")
    output_dir: Optional[str] = Field(None, description="Path to the output directory")
    output_files: Optional[List[str]] = Field(None, description="List of output file paths")
    output_path: Optional[str] = Field(None, description="Path to the output file")
    max_workers: Optional[int] = Field(1, description="Number of parallel worker threads")

class EmbeddingPayload(BaseModel):
    """Payload for the embedding API"""
    text: str = Field(..., description="Text to be embedded")
    collection_name: str = Field(..., description="Name of the collection in ChromaDB")
    local_model: Optional[bool] = Field(False, description="Whether to use the local model")
    cfg_path: Optional[str] = Field(None, description="Path to the configuration file")
    model_name: Optional[str] = Field(None, description="Name of the model used for embedding. If not provided, will be inferred from default local_model selection")

class EmbeddingResult(BaseModel):
    """Result of the embedding step"""
    success: bool = Field(..., description="Whether the embedding pipeline completed successfully.")
    error: Optional[str] = Field(None, description="Error message if pipeline failed.")
    embeddings: Optional[List[float]] = Field(None, description="Embeddings of the text")
    model_name: Optional[str] = Field(None, description="Name of the model used for embedding")
    collection_name: Optional[str] = Field(None, description="Name of the collection in ChromaDB")
    id: str = Field(..., description="ID of the embedding in ChromaDB if not associated with a chunk")

class RetrievalPayload(BaseModel):
    """Payload for the retrieval API"""
    query_texts: Dict[str, List[str]] = Field(..., description="Dictionary of dataset ID to query texts to search for")
    doc_id_map: Dict[str, str] = Field({}, description="Dictionary of dataset ID to document ID")
    collection_name: str = Field(..., description="Name of the collection in ChromaDB")
    k: int = Field(3, description="Number of chunks to retrieve")
    cfg_path: Optional[str] = Field(None, description="Path to the configuration file")
    model_name: Optional[str] = Field(None, description="Name of the model used for embedding or the path to the local model")
    symbolic_boost: Optional[float] = Field(0.15, description="Symbolic boost for the retrieval")
    use_fusion_scoring: Optional[bool] = Field(True, description="Whether to use fusion scoring")
    analyze_chunk_text: Optional[bool] = Field(False, description="Whether to analyze the chunk text")
    max_workers: Optional[int] = Field(1, description="Number of parallel worker threads")

class RetrievalResult(BaseModel):
    """Single query retrieval result"""
    collection_name: Optional[str] = Field(None, description="Name of the collection in ChromaDB")
    success: bool = Field(..., description="Whether the retrieval pipeline completed successfully.")
    error: Optional[str] = Field(None, description="Error message if pipeline failed.")
    k: int = Field(3, description="Number of chunks retrieved")
    chunk_ids: List[str] = Field([], description="List of chunk IDs retrieved if any")
    median_score: Optional[float] = Field(None, description="Median score of the retrieved chunks")
    max_score: Optional[float] = Field(None, description="Maximum score of the retrieved chunks")
    retrieval_time: Optional[float] = Field(None, description="Time taken to retrieve the chunks")

class BatchRetrievalResult(BaseModel):
    """Batch retrieval result"""
    collection_name: Optional[str] = Field(None, description="Name of the collection in ChromaDB")
    total_queries: int = Field(..., description="Total number of queries")
    success: bool = Field(..., description="Whether the retrieval pipeline completed successfully.")
    error: Optional[str] = Field(None, description="Error message if pipeline failed.")
    k: int = Field(3, description="Number of chunks retrieved")
    chunk_ids: dict[str, List[str]] = Field({}, description="Dictionary of dataset ID to chunk IDs retrieved if any")
    median_score: Optional[float] = Field(None, description="Median score of the retrieved chunks")
    max_score: Optional[float] = Field(None, description="Maximum score of the retrieved chunks")
    avg_retrieval_time: Optional[float] = Field(None, description="Time taken to retrieve the chunks")
    max_retrieval_time: Optional[float] = Field(None, description="Maximum time taken to retrieve the chunks")
    total_failed_queries: Optional[int] = Field(None, description="Total number of failed queries")