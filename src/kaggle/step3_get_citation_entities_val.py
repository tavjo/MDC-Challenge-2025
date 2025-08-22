# src/get_citation_entities.py
"""
This script is used to get the citation entities from the documents.
It will use the train_labels.csv file to get the citation entities for each document.
It will output a json file with a list of citation entities for each document.
During inference, since there will be no train_labels.csv, we will create another function to extract citation entities from the document text using the regex patterns from `src/update_patterns.py`
This script is currently only meant to operate on PDF files.
"""

# import re
from typing import List, Optional
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

try:
    from src.kaggle.helpers import timer_wrap, initialize_logging, extract_entities_baml
    from src.kaggle.models import CitationEntity, Document, Chunk
    from src.kaggle.duckdb_utils import get_duckdb_helper
    from src.kaggle.get_citation_context import run_hybrid_retrieval_on_document
except Exception:
    from .helpers import timer_wrap, initialize_logging, extract_entities_baml
    from .models import CitationEntity, Document, Chunk
    from .duckdb_utils import get_duckdb_helper
    from .get_citation_context import run_hybrid_retrieval_on_document


logger = initialize_logging()

base_tmp = "/kaggle/temp/"


artifacts = os.path.join(base_tmp, "artifacts")
DEFAULT_DUCKDB = os.path.join(artifacts, "mdc_challenge.db")

global_prototypes = "/kaggle/input/rf-model-metadata-files/feature_decomposition.pkl"
ds_prototypes = "/kaggle/input/rf-model-metadata-files/prototypes.pkl"

@timer_wrap
class UnknownCitationEntityExtractor:
    def __init__(self, 
                 model,
                 draw_subset: bool = False,
                 subset_size: Optional[int] = None,
                 db_path: str = DEFAULT_DUCKDB,
                 globals_path: Optional[str] = global_prototypes,
                 prototypes: Optional[pd.DataFrame] = None,
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
        self.model = model
        self.prototypes=prototypes
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
        if globals_df:
            globals_df = globals_df.T
        # also load the ds_prototypes
        ds_prototypes_path = self.globals_path.replace("feature_decomposition.pkl", "prototypes.pkl")
        if os.path.exists(ds_prototypes_path):
            with open(ds_prototypes_path, "rb") as f:
                ds_prototypes = pickle.load(f)
            if ds_prototypes:
                ds_prototypes = ds_prototypes
        else:
            ds_prototypes = None
        # combined the two dataframes
        if ds_prototypes is not None:
            combined = pd.concat([globals_df, ds_prototypes])
        else:
            combined = globals_df
        return combined
    
    def _retrieve_context(self, doc_id: str) -> List[Chunk]:
        """
        Retrieve relevant context from documents using the hybrid retrieval model.
        """
        if self.prototypes.empty and self.globals_path is not None:
            prototypes = self._load_query_embeddings()
        elif self.prototypes.empty and not self.globals_path:
            print("Neither prototype dataframe nor path to prototypes provided. Retrieving context without prototypes.")
            prototypes = None
        else:
            prototypes = self.prototypes
        myres = run_hybrid_retrieval_on_document(doc_id = doc_id, 
                                                model = self.model, 
                                                top_k = self.k,
                                                prototypes=prototypes,
                                                db_path=self.db_path,
                                                prototype_top_m = 3
                                                )
        cids = [cid for cid,pre in myres]
        return self._get_top_chunks(cids)
    
    def _get_top_chunks(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Retrieve top chunks by chunk IDs
        """
        db_helper = get_duckdb_helper(self.db_path)
        chunks = db_helper.get_chunks_by_chunk_ids(chunk_ids)
        return chunks

    def retrieve_context(self) -> List[Chunk]:
        """
        Retrieve context from all documents using the hybrid retrieval model.
        """
        chunks = []
        for doc in self.docs:
            chunks.extend(self._retrieve_context(doc_id = doc.doi))
        return chunks
    
    def bulk_baml_extraction(self) -> List[CitationEntity]:
        """
        Extract citations from all documents using the BAML client.
        """
        chunks = self.retrieve_context()
        doc_ids = [ck.document_id for ck in chunks]
        doc_ids = list(set(doc_ids))
        doc_chunk_texts = {}
        for doc_id in doc_ids:
            doc_chunk_texts[doc_id] = [ck.text for ck in chunks if ck.document_id == doc_id]
        for doc, text in doc_chunk_texts.items():
            cites = extract_entities_baml(text, doc)
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
    def run_all(self) -> List[CitationEntity]:
        logger.info("Extracting citation entities")
        cites = self.extract_entities()
        logger.info(f"Saving {len(cites)} citation entities to DuckDb")
        self.save_entities()
        logger.info(f"Saved {len(self.citation_entities)} entities to {self.db_path}")
        return cites

# @timer_wrap
# def main():
#     print("Initializing UnknownCitationEntityExtractor")
#     extractor = UnknownCitationEntityExtractor(
#         draw_subset=False, 
#         subset_size=None, 
#         db_path=DEFAULT_DUCKDB, 
#         globals_path=None,
#         prototypes = prot,
#         k=10, 
#         max_workers=8,
#         model = embedder
#         )
#     print("Extracting citation entities")
#     cites = extractor.extract_entities()
#     print(f"Saving {len(cites)} citation entities to DuckDb")
#     extractor.save_entities()
#     print(f"Saved {len(extractor.citation_entities)} entities to {extractor.db_path}")


# if __name__ == "__main__":
#     main()