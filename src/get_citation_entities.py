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

from src.helpers import timer_wrap, initialize_logging, preprocess_text, normalise, clean_text_for_urls
from src.models import CitationEntity, Document
from src.baml_client import b as baml
from api.utils.duckdb_utils import DuckDBHelper

# Use a project-relative path so Docker doesn't try to write to root '/'
DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"


filename = os.path.basename(__file__)
logger = initialize_logging(filename)


@timer_wrap
class CitationEntityExtractor:
    def __init__(self, 
                 data_dir: str = "Data", 
                 known_entities: bool = True,
                 labels_file: Optional[str] = "train_labels.csv", 
                 draw_subset: bool = False,
                 subset_size: Optional[int] = None,
                 db_path: str = DEFAULT_DUCKDB_PATH
                #  use_ner: bool = True
                 ):  # Add this parameter
        logger.info(f"Initializing CitationEntityExtractor with data directory: {data_dir}")
        self.data_dir = os.path.join(project_root, data_dir)
        self.db_path = db_path
        self.known_entities = known_entities
        self.files_dir = os.path.join(self.data_dir,"train","PDF") # assumes similar directory structure as me. 
        self.pdf_files : List[str] = [os.path.join(self.files_dir, f) for f in os.listdir(self.files_dir) if f.endswith('.pdf')]
        if self.known_entities:
            logger.info(f"Using known entities from: {labels_file}")
            self.labels_path = os.path.join(self.data_dir, labels_file)
            logger.info(f"Loading labels from: {self.labels_path}")
            self.labels_df = pd.read_csv(self.labels_path)
            logger.info(f"Loaded {len(self.labels_df)} labels")
            self.missing_entities = None
            self.article_ids = self.labels_df["article_id"].unique()
        else:
            logger.info("No known entities provided. Using regex patterns to extract citation entities.")
            self.labels_path = None
            self.labels_df = pd.DataFrame(columns = ["article_id", "dataset_id"])
            self.article_ids = [Path(file).stem for file in self.pdf_files]

        
        self.citation_entities: List[CitationEntity] = []
        self.docs = self._load_doc_pages()
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
    
    def _make_pattern(self, ds_id: str) -> re.Pattern:
        """Build a tolerant regex for a dataset ID.

        - Normalises common URL prefixes (doi.org, dx.doi.org)
        - Treats separators [-_/:.\s] as interchangeable/flexible
        - Keeps alphanumeric token order intact
        - Adds word boundaries for simple accession-like IDs
        """
        ds_id_norm = clean_text_for_urls(ds_id).strip()
        # Split on common separators, keep alnum tokens
        tokens = [t for t in re.split(r"[-_/:.\s]+", ds_id_norm) if t]
        if not tokens:
            tokens = [ds_id_norm]
        # Join tokens with a flexible separator class (including optional spaces)
        sep = r"[-_/:.\s]*"
        pat = sep.join(re.escape(t) for t in tokens)

        # Word boundaries for simple accessions (avoid overmatching inside words)
        if re.fullmatch(r"[A-Z]{1,6}\d+[A-Z0-9]*", ds_id, re.I):
            pat = rf"\b{pat}\b"

        return re.compile(pat, flags=re.IGNORECASE)
    
    def _generate_id_variants(self, dataset_id: str) -> List[str]:
        """Produce tolerant variants for a labeled dataset_id.

        - Strip DOI host, optional ".vN" suffix, optional "/tN" suffix
        - Keep original as first candidate
        - Map E-GEOD-12345 -> GSE12345 synonym
        """
        variants: Set[str] = set()
        raw = dataset_id.strip()
        norm = clean_text_for_urls(raw)
        variants.add(raw)
        variants.add(norm)
        no_ver = re.sub(r"(?i)\.v\d+$", "", norm)
        variants.add(no_ver)
        no_tbl = re.sub(r"(?i)/t\d+$", "", no_ver)
        variants.add(no_tbl)
        # E-GEOD ↔ GSE
        if re.match(r"(?i)^E-GEOD-\d+$", raw):
            variants.add(re.sub(r"(?i)^E-GEOD-", "GSE", raw))
        return [v for v in variants if v]

    @staticmethod
    def _alnum_collapse(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())
    
    # def _make_pattern(self, dataset_id: str) -> re.Pattern:
    #     """Create pattern for dataset citation: substring match anywhere, case-insensitive."""
    #     pat = re.escape(dataset_id)
    #     return re.compile(pat, flags=re.IGNORECASE)

    def find_citation_in_text(self, citation: str, text: str) -> List[str]:
        """Robust dual matching: raw substring or preprocessed substring."""
        # Normalise citation id and page text consistently
        citation_norm = clean_text_for_urls(citation).lower()
        lower_cit = re.escape(citation_norm)
        # find all occurrences in raw and cleaned/normalised variants
        raw_matches = re.findall(lower_cit, text.lower())
        normalised_matches = re.findall(lower_cit, normalise(text).lower())
        preprocessed_matches = re.findall(lower_cit, preprocess_text(text))
        return raw_matches + normalised_matches + preprocessed_matches

    # @parallel_processing_decorator(batch_param_name="pages", batch_size=5, max_workers=8, flatten=True)
    def _get_known_entities(self, pages: List[str], article_id: str, dataset_ids: List[str]) -> List[CitationEntity]:
        """
        Get known entities from the document text using the known entities from the labels file.
        """
        num_pages = len(pages)
        # full_text = " ".join(pages)
        # Collect CitationEntity objects found for this document
        entities: List[CitationEntity] = []
        
        # Track pages per dataset_id
        dataset_pages = {}  # dataset_id -> set of pages where it appears
        
        for idx, page in enumerate(pages):
            page = page
            page_num = idx + 1
            logger.info(f"Processing page {page_num} of {num_pages} for document: {article_id}")
        # logger.info(f"Processing document: {article_id}")
        
            for dataset_id in dataset_ids:
                logger.info(f"Processing dataset: {dataset_id}")
                found = False
                for variant in self._generate_id_variants(dataset_id):
                    regex = self._make_pattern(variant)
                    # try against multiple text variants
                    matches = regex.findall(page)
                    if not matches:
                        matches = regex.findall(normalise(page))
                    if not matches:
                        matches = regex.findall(preprocess_text(page))
                    if matches:
                        dataset_pages.setdefault(dataset_id, set()).add(page_num)
                        found = True
                        break
                if not found:
                    # Aggressive fallback for long IDs (e.g., DOIs): collapsed alnum substring
                    id_alnum = self._alnum_collapse(dataset_id)
                    if dataset_id.startswith("10.") or len(id_alnum) >= 12:
                        page_alnum = self._alnum_collapse(page)
                        if id_alnum and id_alnum in page_alnum:
                            dataset_pages.setdefault(dataset_id, set()).add(page_num)
                            found = True
                if not found:
                    matches = self.find_citation_in_text(dataset_id, page)
                    if matches:
                        dataset_pages.setdefault(dataset_id, set()).add(page_num)
                        logger.info(f"Found {len(matches)} matches for {dataset_id} on page {page_num}")
        # Create one CitationEntity per unique dataset_id found (after processing all pages)
        for dataset_id, pages_set in dataset_pages.items():
            citation_entity = CitationEntity(
                data_citation=dataset_id,
                document_id=article_id,
                pages=sorted(list(pages_set))  # Convert set to sorted list
            )
            entities.append(citation_entity)
        
        logger.info(f"Found {len(entities)} citation entities for {article_id}")
        
        # Add to the main list
        self.citation_entities.extend(entities)
        
        return self.citation_entities


    def extract_known_entities(self) -> List[CitationEntity]:
        """
        Extract citation entities from the document text using the known entities from the labels file.
        """
        logger.info("Extracting known entities from the document text using the known entities from the labels file.")
        articles_ids = self.article_ids
        # create dictionary where keys are article IDs and values are lists of dataset IDs
        # maybe can group by article ID and then get the dataset IDs for each article
        # Filter out "Missing" values
        labels_dict = {}
        for article_id in articles_ids:
            dataset_ids = self.labels_df[
                (self.labels_df["article_id"] == article_id) & 
                (self.labels_df["type"] != "Missing")
            ]["dataset_id"].tolist()
            labels_dict[article_id] = dataset_ids
        # check if there are any pdf files
        if len(self.docs) == 0:
            logger.warning("No documents found. Returning empty list.")
            return []
        for doc in self.docs:
            logger.info(f"Processing document: {doc.doi}")
            article_id = doc.doi
            pages = doc.full_text
            if article_id not in articles_ids:
                logger.warning(f"Article ID {article_id} not found in the labels file. Skipping file: {article_id}")
                continue
            else:
                dataset_ids = labels_dict[article_id]
            self._get_known_entities(pages=pages, article_id=article_id, dataset_ids=dataset_ids)
            # self.citation_entities.extend(list(citation_entities))
        #final check: compare length of citation entities with expected count LIMITED to parsed docs
        parsed_doc_ids: Set[str] = {doc.doi for doc in self.docs}
        labels_in_parsed_docs = self.labels_df[
            (self.labels_df["article_id"].isin(parsed_doc_ids)) & (self.labels_df["type"] != "Missing")
        ]
        expected_within_parsed = len(labels_in_parsed_docs)
        if len(self.citation_entities) != expected_within_parsed:
            logger.warning(
                f"Extracted citations ({len(self.citation_entities)}) != expected within parsed docs ({expected_within_parsed})."
            )
            # Log high-signal diagnostics
            missing_docs = set(self.labels_df["article_id"].unique()) - parsed_doc_ids
            if missing_docs:
                logger.warning(
                    f"There are {len(missing_docs)} labeled article_ids not present in the documents table; "
                    "their citations cannot be extracted until those PDFs are parsed."
                )
            # check which dataset IDs are missing within parsed docs
            labeled_ds_in_parsed = set(labels_in_parsed_docs["dataset_id"].tolist())
            found_ds = {entity.data_citation for entity in self.citation_entities}
            self.missing_entities = labeled_ds_in_parsed - found_ds
            if self.missing_entities:
                logger.warning(f"Missing dataset IDs within parsed docs (n={len(self.missing_entities)}).")
        else:
            logger.info("Known Entity extraction complete. No missing entities within parsed docs.")
        return self.citation_entities
    
    def _extract_entities_baml(self, doc: Document) -> List[CitationEntity]:
        """
        Extract citation entities from the document text using the BAML client.
        """
        logger.info("Extracting citation entities using BAML client.")
        citations = baml.ExtractCitation(doc)
        if citations:
            citation_entities = [CitationEntity.model_validate(entity.model_dump()) for entity in citations.citation_entities]
            self.citation_entities.extend(citation_entities)
            return citation_entities
        else:
            logger.warning("No citation entities found using BAML client.")
            return []
    
    def bulk_baml_extraction(self) -> List[CitationEntity]:
        """
        Extract citations from all documents using the BAML client.
        """
        for doc in self.docs:
            self._extract_entities_baml(doc)
        return self.citation_entities
    
    def extract_entities(self) -> List[CitationEntity]:
        """
        Extract citation entities from the document text using the known entities from the labels file or the regex patterns.
        """
        if self.known_entities:
            return self.extract_known_entities()
        else:
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
    extractor = CitationEntityExtractor(data_dir="Data", known_entities=True, labels_file="train_labels.csv", draw_subset=False, db_path=DEFAULT_DUCKDB_PATH)
    extractor.extract_entities()
    extractor.save_entities()
    logger.info(f"Saved {len(extractor.citation_entities)} entities to {extractor.db_path}")


if __name__ == "__main__":
    main()



            
        


