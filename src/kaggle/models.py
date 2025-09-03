# pydantic models
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal, Any, Union
from enum import Enum
from datetime import datetime


class RetrievalWeights(BaseModel):
    """Weights for combining retrieval signals in hybrid scoring.

    Defaults mirror the previous hard-coded constants used in retrieval_module.
    """

    prototype: float = Field(0.70, description="Weight for prototype affinity scores")
    das: float = Field(0.10, description="Weight for DAS prior (lexical cues)")
    rrf: float = Field(0.10, description="Weight for RRF fused sparse+dense scores")
    regex: float = Field(0.05, description="Weight for specific accession-style regex matches")
    doi_repo: float = Field(0.05, description="Weight for repository DOI prior gated by context")
    ref_penalty: float = Field(0.12, description="Reserved/unused; kept for completeness")


class BoostConfig(BaseModel):
    """Configuration for hybrid retrieval flow (prototype-first + RRF + bounded priors + MMR).

    - TOPK_PER_SIGNAL = mmr_top_k * signal_k_multiplier
    - Prototypes (if provided) drive primary candidates; RRF (BM25+dense) is secondary
    - Small, bounded priors (DAS terms, specific accession regex, repository DOIs) are added
    """

    # RRF rank fusion constant
    rrf_k: int = Field(
        20,
        description=(
            "RRF constant k; controls contribution of lower-ranked items across rankers."
        ),
    )

    # New: control candidate pool multiplier for signal pools
    signal_k_multiplier: int = Field(
        3,
        ge=1,
        description=(
            "Multiplier used to derive TOPK_PER_SIGNAL from mmr_top_k (TOPK_PER_SIGNAL = mmr_top_k * signal_k_multiplier)."
        ),
    )

    # Final diversification
    mmr_lambda: float = Field(
        0.85,
        description=(
            "MMR diversity-control parameter λ in [0,1]; higher favors relevance, lower favors diversity."
        ),
    )
    mmr_top_k: int = Field(
        15,
        description=(
            "Final number of items to select using MMR re-ranking from the boosted candidate pool."
        ),
    )
    # New: aggregation over prototypes when computing per-chunk prototype affinity
    prototype_top_m: int = Field(
        1,
        ge=1,
        description=(
            "Number of top prototype similarities to average per chunk (1 reduces to max)."
        ),
    )

    # Retrieval weights previously hard-coded in retrieval_module
    retrieval_weights: RetrievalWeights = Field(
        default_factory=RetrievalWeights,
        description="Weights for combining retrieval signals in hybrid scoring.",
    )

class DatasetType(str, Enum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"

class CitationEntity(BaseModel):
    data_citation: str = Field(..., description="Data citation from text")
    document_id: str = Field(..., description="DOI of the document where the data citation is found")
    pages: Optional[List[int]] = Field(None, description="List of page numbers where the data citation is mentioned.")
    evidence: Optional[List[str]] = Field(None, description="List of evidence from the text for the data citation")
    dataset_type: Optional[DatasetType] = Field(None, description="Type of dataset citation")

    def to_string(self) -> str:
        """
        Flatten the citation entity into a string.
        """
        pages_str = ",".join(map(str, self.pages)) if self.pages else ""
        evidence_str = ",".join(self.evidence) if self.evidence else ""
        ds = (
            self.dataset_type.value if isinstance(self.dataset_type, Enum) else self.dataset_type
        ) if self.dataset_type else ""
        return f"{self.data_citation}|{self.document_id}|{pages_str}|{evidence_str}|{ds}"

    @classmethod
    def from_string(cls, citation_entity_str: str) -> "CitationEntity":
        """
        Rehydrate the citation entity from a string.
        """
        parts = citation_entity_str.split("|")
        data_citation = parts[0] if len(parts) > 0 else ""
        document_id = parts[1] if len(parts) > 1 else ""
        pages_str = parts[2] if len(parts) > 2 else ""
        evidence_str = parts[3] if len(parts) > 3 else ""
        ds_str = parts[4] if len(parts) > 4 else None

        pages = [int(p) for p in pages_str.split(",") if p] if pages_str else None
        evidence = evidence_str.split(",") if evidence_str else None
        ds_val = None if not ds_str else ds_str

        return cls(
            data_citation=data_citation,
            document_id=document_id,
            pages=pages,
            evidence=evidence,
            dataset_type=ds_val,
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
            "dataset_type": (
                self.dataset_type.value if isinstance(self.dataset_type, Enum) else self.dataset_type
            ) if self.dataset_type else None,
        }
    
    @classmethod
    def from_duckdb_row(cls, row: Dict[str, Any]) -> "CitationEntity":
        """
        Imports the citation entity from a DuckDB row.
        """
        ds = row.get("dataset_type")
        ds_val = None if ds in (None, "") else ds
        return cls(
            data_citation=row["data_citation"],
            document_id=row["document_id"],
            pages=row["pages"],
            evidence=row["evidence"],
            dataset_type=ds_val,
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
class Dataset(BaseModel):
    """Dataset Citation Extracted from Document text"""
    dataset_id: str = Field(..., description="Dataset ID")
    document_id: str = Field(..., description="DOI in which the dataset citation was found")
    total_tokens: int = Field(..., description="Total number of tokens in all chunks.")
    avg_tokens_per_chunk: float = Field(..., description="Average number of tokens per chunk.")
    total_char_length: int = Field(..., description="Total number of characters")
    clean_text_length: int = Field(..., description="Total number of characters after cleaning")
    cluster: Optional[str] = Field(None, description="Cluster of the dataset citation")
    dataset_type: Optional[Literal["PRIMARY", "SECONDARY"]] = Field(None, description="Dataset Type: main target of the classification task")
    text: str = Field(..., description= "Text in the document where the dataset citation is found")

    # save to DuckDB
    def to_duckdb_row(self) -> Dict[str, Any]:
        """
        Convert the dataset into a dictionary that can be inserted into DuckDB.
        """
        return {
            "dataset_id": self.dataset_id,
            "document_id": self.document_id,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_chunk": self.avg_tokens_per_chunk,
            "total_char_length": self.total_char_length,
            "clean_text_length": self.clean_text_length,
            "cluster": self.cluster,
            "dataset_type": self.dataset_type,
            "text": self.text,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    @classmethod
    def from_duckdb_row(cls, row: Dict[str, Any]) -> "Dataset":
        """
        Rehydrate the dataset from a DuckDB row.
        """
        return cls(
            dataset_id=row["dataset_id"],
            document_id=row["document_id"],
            total_tokens=row["total_tokens"],
            avg_tokens_per_chunk=row["avg_tokens_per_chunk"],
            total_char_length=row["total_char_length"],
            clean_text_length=row["clean_text_length"],
            cluster=row["cluster"],
            dataset_type=row["dataset_type"],
            text=row["text"]
        )

class EngineeredFeatures(BaseModel):
    """Engineered features for the dataset citation classification task"""
    dataset_id: str = Field(..., description="Dataset ID")
    document_id: str = Field(..., description="DOI of document in which the dataset citation was found")
    UMAP_1: Optional[float] = Field(None, description="UMAP 1 dimension")
    UMAP_2: Optional[float] = Field(None, description="UMAP 2 dimension")
    LEIDEN_1: Optional[float] = Field(None, description="LEIDEN 1 PC1 loadings")
    # Allow arbitrary additional fields based on the number leiden clusters
    model_config = {"extra": "allow"}  # capture any other feature_*
    
    def to_eav_rows(self) -> List[Dict[str, Any]]:
        """Return a list[dict] ready for bulk INSERT into engineered_feature_values."""
        base = self.model_dump()            
        now = datetime.now().isoformat()
        return [
            {
                "dataset_id":  base["dataset_id"],
                "document_id": base["document_id"],
                "feature_name": k,
                "feature_value": v,
                "created_at": now,
                "updated_at": now,
            }
            for k, v in base.items()
            if k not in {"dataset_id", "document_id"}
        ]

    @classmethod
    def from_eav_rows(cls, rows: List[Dict[str, Any]]) -> "EngineeredFeatures":
        """Hydrate a model from SELECT * WHERE dataset_id=… AND document_id=…"""
        merged: Dict[str, Any] = {}
        for r in rows:
            merged.setdefault("dataset_id", r["dataset_id"])
            merged.setdefault("document_id", r["document_id"])
            merged[r["feature_name"]] = r["feature_value"]
        return cls.model_validate(merged)