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
import os


from pathlib import Path
import sys

# Allow importing sibling kaggle helpers/models when used as a standalone script
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from src.kaggle.helpers import timer_wrap, initialize_logging, extract_entities_baml
from src.kaggle.models import CitationEntity, Document, Chunk
from src.kaggle.duckdb import get_duckdb_helper
from src.kaggle.get_citation_context import run_hybrid_retrieval_on_document

DEFAULT_DUCKDB_PATH = "/kaggle/temp/mdc.duckdb"

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

DEFAULT_GLOBALS_PATH = "feature_decomposition.pkl"


@timer_wrap
class UnknownCitationEntityExtractor:
    def __init__(self, 
                 draw_subset: bool = False,
                 subset_size: Optional[int] = None,
                 db_path: str = DEFAULT_DUCKDB_PATH,
                 globals_path: str = DEFAULT_GLOBALS_PATH,
                 k: int = 15,
                 max_workers: int = 8
                 ):  # Add this parameter
        logger.info(f"Initializing CitationEntityExtractor")
        self.db_path = db_path
        self.citation_entities: List[CitationEntity] = []
        self.docs = self._load_doc_pages()
        self.k = k
        self.max_workers = max_workers
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
        db_helper = get_duckdb_helper(self.db_path)
        docs = db_helper.get_all_documents()
        db_helper.close()
        return docs
    
    def _load_query_embeddings(self) -> pd.DataFrame:
        """
        Load query embeddings from a pickle file
        """
        import pickle
        with open(self.globals_path, "rb") as f:
            globals_df = pickle.load(f)
        return globals_df
    
    def _retrieve_context(self, doc: str,prototypes: bool = False) -> List[Chunk]:
        """
        Retrieve relevant context from documents using the hybrid retrieval model.
        """
        if prototypes:
            prototypes = self._load_query_embeddings()
        else:
            prototypes = None
        return run_hybrid_retrieval_on_document(doc_id = doc.id, model_dir = self.model_name, top_k = self.k, prototypes=prototypes, db_path=self.db_path)
    
    def retrieve_context(self) -> List[Chunk]:
        """
        Retrieve context from all documents using the hybrid retrieval model.
        """
        chunks = []
        for doc in self.docs:
            chunks.extend(self._retrieve_context(doc.text, prototypes=True))
        return chunks
    
    def bulk_baml_extraction(self) -> List[CitationEntity]:
        """
        Extract citations from all documents using the BAML client.
        """
        chunks = self.retrieve_context()
        for chunk in chunks:
            cites = extract_entities_baml(chunk.text, chunk.document_id)
            self.citation_entities.extend(cites)
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
        db_helper = get_duckdb_helper(self.db_path)
        db_helper.store_citations_batch(self.citation_entities)
        db_helper.close()
        logger.info(f"Saved {len(self.citation_entities)} entities to {self.db_path}")


    def load_entities(self) -> List[CitationEntity]:
        """
        Load citation entities from a JSON file of dicts,
        reconstructing each via Pydantic’s model_validate.
        """
        db_helper = get_duckdb_helper( self.db_path)
        citations = db_helper.get_all_citation_entities()
        db_helper.close()
        return [CitationEntity.model_validate(citation) for citation in citations]

@timer_wrap
def main():
    extractor = UnknownCitationEntityExtractor(
        draw_subset=False, 
        subset_size=None, 
        db_path=DEFAULT_DUCKDB_PATH, 
        globals_path=DEFAULT_GLOBALS_PATH, 
        k=5, 
        max_workers=8
        )
    extractor.extract_entities()
    extractor.save_entities()
    logger.info(f"Saved {len(extractor.citation_entities)} entities to {extractor.db_path}")


if __name__ == "__main__":
    main()