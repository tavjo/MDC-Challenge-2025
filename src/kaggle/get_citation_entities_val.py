# src/get_citation_entities.py
"""
This script is used to get the citation entities from the documents.
It will use the train_labels.csv file to get the citation entities for each document.
It will output a json file with a list of citation entities for each document.
During inference, since there will be no train_labels.csv, we will create another function to extract citation entities from the document text using the regex patterns from `src/update_patterns.py`
This script is currently only meant to operate on PDF files.
"""

# import re
import regex as re
from typing import List, Optional, Dict, Set
from pathlib import Path
import pandas as pd
import numpy as np
import sys, os, json
# import nltk
# from nltk.corpus import stopwords
import pathlib
from dotenv import load_dotenv

load_dotenv()

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Project root: {project_root}")
sys.path.append(project_root)

from src.helpers import timer_wrap, initialize_logging
from src.models import CitationEntity, Document, ValRetrievalPayload, BatchRetrievalResult, Chunk
from src.baml_client import b as baml
from api.utils.duckdb_utils import DuckDBHelper
import requests

DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"
DEFAULT_CHROMA_CONFIG = "configs/chunking.yaml"

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

DEFAULT_GLOBALS_PATH = "../../artifacts/feature_decomposition.pkl"


@timer_wrap
class UnknownCitationEntityExtractor:
    def __init__(self, 
                 data_dir: str = "Data", 
                 draw_subset: bool = False,
                 subset_size: Optional[int] = None,
                 db_path: str = DEFAULT_DUCKDB_PATH,
                 globals_path: str = "globals.pkl",
                 k: int = 5,
                 max_workers: int = 8,
                 cfg_path: str = DEFAULT_CHROMA_CONFIG,
                 model_name: str = "BAAI/bge-small-en-v1.5",
                 symbolic_boost: float = 0.15,
                 use_fusion_scoring: bool = True,
                 analyze_chunk_text: bool = False,
                 base_url: str = "http://localhost:8000"
                 ):  # Add this parameter
        logger.info(f"Initializing CitationEntityExtractor with data directory: {data_dir}")
        self.data_dir = os.path.join(project_root, data_dir)
        self.db_path = db_path
        self.files_dir = os.path.join(self.data_dir,"train","PDF") # assumes similar directory structure as me. 
        self.pdf_files : List[str] = [os.path.join(self.files_dir, f) for f in os.listdir(self.files_dir) if f.endswith('.pdf')]
        logger.info("No known entities provided. Using regex patterns to extract citation entities.")
        self.labels_path = None
        self.labels_df = pd.DataFrame(columns = ["article_id", "dataset_id"])
        self.article_ids = [Path(file).stem for file in self.pdf_files]
        self.citation_entities: List[CitationEntity] = []
        self.docs = self._load_doc_pages()
        self.base_url = base_url
        self.full_url = f"{self.base_url}/batch_retrieve_val"
        self.k = k
        self.max_workers = max_workers
        self.cfg_path = cfg_path
        self.model_name = model_name
        self.symbolic_boost = symbolic_boost
        self.use_fusion_scoring = use_fusion_scoring
        self.analyze_chunk_text = analyze_chunk_text
        self.globals_path = globals_path
        if draw_subset:
            self.subset_size = subset_size if subset_size is not None else 20
            logger.info(f"Drawing subset of {self.subset_size} files from {len(self.pdf_files)} files.")
            # set seed
            np.random.seed(42)
            self.subset_ids = np.random.choice(self.pdf_files, self.subset_size, replace=False)
            self.pdf_files = self.subset_ids
            self.docs = np.random.choice(self.docs, self.subset_size, replace=False)
    

    def _load_doc_pages(self) -> List[Document]:
        """
        load all document pages into memory once.
        """
        db_helper = DuckDBHelper(self.db_path)
        docs = db_helper.get_all_documents()
        db_helper.close()
        return docs
    
    def _extract_entities_baml(self, doc: List[str], doc_id: str) -> List[CitationEntity]:
        """
        Extract citation entities from the document text using the BAML client.
        """
        logger.info(f"Extracting citation entities using BAML client for {doc_id}.")
        citations = baml.ExtractCitation(doc)
        if citations:
            entities = [{
                "data_citation": entity.data_citation,
                "document_id": doc_id,
                "evidence": entity.evidence,
            } for entity in citations]
            citation_entities = [CitationEntity.model_validate(entity) for entity in entities]
            self.citation_entities.extend(citation_entities)
            return citation_entities
        else:
            logger.warning("No citation entities found using BAML client.")
            return []
    
    def _load_query_embeddings(self) -> pd.DataFrame:
        """
        Load query embeddings from a pickle file
        """
        import pickle
        with open(self.globals_path, "rb") as f:
            globals_df = pickle.load(f)
        return globals_df
    
    def retrieve_context(self) -> List[Chunk]:
        # Call the batch retrieval API endpoint with Pydantic payload
        doc_ids = [doc.doi for doc in self.docs]
        url = self.full_url
        query_embeddings = {}
        globals = self._load_query_embeddings()
        # turn into numpy array
        query_embeddings = np.array(globals.values())
        for doc_id in doc_ids:
            query_embeddings[doc_id] = globals
        # Determine default max_workers as half the CPU count, minimum 1
        cpu_count = os.cpu_count() or 1
        default_workers = max(1, cpu_count // 2)
        payload_obj = ValRetrievalPayload(
            query_embeddings=query_embeddings,
            doc_ids=doc_ids,
            collection_name="mdc_val_data",
            k=5,
            max_workers=default_workers,
            cfg_path=DEFAULT_CHROMA_CONFIG,
            model_name="BAAI/bge-small-en-v1.5",
            symbolic_boost=0.15,
            use_fusion_scoring=True,
            analyze_chunk_text=False
        )
        payload_data = payload_obj.model_dump(exclude_none=True)
        response = requests.post(url, json=payload_data)
        if response.status_code != 200:
            print(f"Error: API call failed with status {response.status_code}: {response.text}")
            return
        results = BatchRetrievalResult.model_validate(response.json())
        chunk_ids = [chunk_id for chunk_id in results.chunk_ids.values()]
        # get chunks from duckdb
        chunks = self._get_chunks(chunk_ids)
        return chunks
    
    def _get_chunks(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Get chunks from the database
        """
        db_helper = DuckDBHelper(self.db_path)
        chunks = db_helper.get_chunks_by_chunk_ids(chunk_ids)
        db_helper.close()
        return chunks
    
    def bulk_baml_extraction(self) -> List[CitationEntity]:
        """
        Extract citations from all documents using the BAML client.
        """
        chunks = self.retrieve_context()
        for chunk in chunks:
            self._extract_entities_baml(chunk.text, chunk.doc_id)
        return self.citation_entities
    
    def extract_entities(self) -> List[CitationEntity]:
        """
        Extract citation entities from the document text using the known entities from the labels file or the regex patterns.
        """
        return self.bulk_baml_extraction()
        
    def save_entities(self) -> None:
        """
        Save the list of CitationEntity objects to a DuckDB table.
        """
        db_helper = DuckDBHelper(self.db_path)
        db_helper.store_citations_batch(self.citation_entities)
        db_helper.close()
        logger.info(f"Saved {len(self.citation_entities)} entities to {self.db_path}")


    def load_entities(self) -> List[CitationEntity]:
        """
        Load citation entities from a JSON file of dicts,
        reconstructing each via Pydantic’s model_validate.
        """
        db_helper = DuckDBHelper( self.db_path)
        citations = db_helper.get_all_citation_entities()
        db_helper.close()
        return [CitationEntity.model_validate(citation) for citation in citations]

@timer_wrap
def main():
    extractor = UnknownCitationEntityExtractor(
        data_dir="Data", 
        draw_subset=True, 
        subset_size=20, 
        db_path=DEFAULT_DUCKDB_PATH, 
        globals_path="globals.pkl", 
        k=5, 
        max_workers=8, 
        cfg_path=DEFAULT_CHROMA_CONFIG, 
        model_name="BAAI/bge-small-en-v1.5", 
        symbolic_boost=0.15, 
        use_fusion_scoring=True, 
        analyze_chunk_text=False,
        base_url="http://localhost:8000"
    )
    extractor.extract_entities()
    extractor.save_entities()
    logger.info(f"Saved {len(extractor.citation_entities)} entities to {extractor.db_path}")


if __name__ == "__main__":
    main()